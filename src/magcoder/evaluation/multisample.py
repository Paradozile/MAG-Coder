"""Multi-sample generation and pass@k evaluation."""

from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml

from magcoder.benchmark.package import BenchmarkPackage
from magcoder.benchmark.task import BenchmarkTask
from magcoder.evaluation.checkpoints import evaluate_checkpoints
from magcoder.evaluation.metrics import (
    checkpoint_accuracy,
    framework_passed,
    mandatory_success,
)
from magcoder.evaluation.passk import (
    estimate_pass_at_k,
    evaluate_success_criterion,
    observed_pass_at_k,
)
from magcoder.evaluation.passk_report import (
    write_failure_summary,
    write_passk_summary_reports,
)
from magcoder.evaluation.real_run import (
    RealAblationSetting,
    default_real_ablation_settings,
)
from magcoder.evaluation.scenario import ScenarioSpec
from magcoder.generation import (
    DEFAULT_GENERATION_CACHE_DIR,
    GenerationConfig,
    GenerationRequest,
    generate_with_provider,
    save_generation_result,
)
from magcoder.graphics import load_mag_from_graphical_artifact
from magcoder.models import MAGModel, load_mag_from_yaml
from magcoder.refinement import (
    analyze_python_source,
    check_consistency,
    constraints_with_interaction_contracts,
    extract_model_constraints,
    load_framework_spec_from_yaml,
    refine_code,
)
from magcoder.refinement.contracts import build_interaction_contracts
from magcoder.semantic import build_hybrid_input


@dataclass(slots=True)
class MultiSampleRunConfig:
    """Configuration for multi-sample pass@k evaluation."""

    run_id: str
    benchmark_path: str
    provider: str
    model_name: str
    output_dir: str
    temperature: float = 0.0
    api_base_url: str | None = None
    api_key_env: str | None = None
    timeout_seconds: int = 60
    top_p: float | None = None
    max_tokens: int | None = None
    stop: list[str] | None = None
    think: bool | None = None
    native_endpoint: str | None = None
    num_predict: int | None = None
    prompt_profile_id: str | None = None
    prompt_profile_path: str | None = None
    require_non_empty_code: bool = False
    base_seed: int | None = None
    num_samples: int = 1
    k_values: list[int] = field(default_factory=lambda: [1])
    input_modes: list[str] = field(
        default_factory=lambda: ["structured", "graphical"],
    )
    task_ids: list[str] = field(default_factory=list)
    ablation_ids: list[str] = field(
        default_factory=lambda: [
            "graph_nl_with_refinement",
            "graph_only_with_refinement",
            "graph_nl_without_refinement",
        ],
    )
    cache_enabled: bool = True
    resume: bool = True
    success_criterion: str = "mandatory_success"
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Render this config as plain Python containers."""

        return asdict(self)


@dataclass(slots=True)
class SampleResult:
    """Result metadata for one generated sample."""

    task_id: str
    input_mode: str
    ablation_id: str
    sample_index: int
    seed: int
    generation_result_path: str
    code_initial_path: str
    initial_report_path: str
    parse_success: bool
    framework_passed: bool
    mandatory_success: bool
    checkpoint_accuracy: float
    success: bool
    cache_hit: bool
    code_refined_path: str | None = None
    refined_report_path: str | None = None
    generation_error: str | None = None
    latency_seconds: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Render this sample result as plain Python containers."""

        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SampleResult:
        """Build a sample result from JSON data."""

        metadata = data.get("metadata", {})
        if not isinstance(metadata, dict):
            raise TypeError("SampleResult metadata must be a dict.")
        return cls(
            task_id=str(data["task_id"]),
            input_mode=str(data["input_mode"]),
            ablation_id=str(data["ablation_id"]),
            sample_index=int(data["sample_index"]),
            seed=int(data["seed"]),
            generation_result_path=str(data["generation_result_path"]),
            code_initial_path=str(data["code_initial_path"]),
            code_refined_path=_optional_str(data.get("code_refined_path")),
            initial_report_path=str(data["initial_report_path"]),
            refined_report_path=_optional_str(data.get("refined_report_path")),
            parse_success=bool(data["parse_success"]),
            framework_passed=bool(data["framework_passed"]),
            mandatory_success=bool(data["mandatory_success"]),
            checkpoint_accuracy=float(data["checkpoint_accuracy"]),
            success=bool(data["success"]),
            cache_hit=bool(data["cache_hit"]),
            generation_error=_optional_str(data.get("generation_error")),
            latency_seconds=(
                None
                if data.get("latency_seconds") is None
                else float(data["latency_seconds"])
            ),
            metadata=dict(metadata),
        )


@dataclass(slots=True)
class MultiSampleRunResult:
    """Result bundle for a multi-sample pass@k run."""

    run_id: str
    sample_results: list[SampleResult]
    summary_rows: list[dict[str, Any]]
    output_dir: str
    metadata: dict[str, Any] = field(default_factory=dict)


class MultiSamplePassKRunner:
    """Run multiple generation samples and aggregate pass@k results."""

    def run(
        self,
        package: BenchmarkPackage,
        config: MultiSampleRunConfig,
    ) -> MultiSampleRunResult:
        """Run multi-sample generation, analysis/refinement, and pass@k."""

        _validate_config(config)
        output_dir = Path(config.output_dir)
        samples_root = output_dir / "samples"
        output_dir.mkdir(parents=True, exist_ok=True)
        samples_root.mkdir(parents=True, exist_ok=True)
        _save_run_config(config, output_dir / "run_config.yaml")

        settings = _selected_settings(config.ablation_ids)
        cache_dir = Path(
            str(config.metadata.get("cache_dir", DEFAULT_GENERATION_CACHE_DIR))
        )
        sample_results: list[SampleResult] = []
        selected_task_ids = set(config.task_ids)
        for task in package.tasks:
            if selected_task_ids and task.task_id not in selected_task_ids:
                continue
            scenario = task.to_scenario_spec(
                base_dir=Path(package.root_dir) / "tasks" / task.task_id,
            )
            for input_mode in config.input_modes:
                if input_mode not in task.supported_input_modes:
                    continue
                for setting in settings:
                    for sample_index in range(config.num_samples):
                        sample = _run_or_resume_sample(
                            package=package,
                            task=task,
                            scenario=scenario,
                            setting=setting,
                            input_mode=input_mode,
                            sample_index=sample_index,
                            config=config,
                            samples_root=samples_root,
                            cache_dir=cache_dir,
                        )
                        sample_results.append(sample)

        sample_results_path = output_dir / "sample_results.jsonl"
        _write_sample_results_jsonl(sample_results, sample_results_path)
        summary_rows = build_passk_summary_rows(
            sample_results=sample_results,
            config=config,
        )
        write_passk_summary_reports(
            summary_rows,
            csv_path=output_dir / "passk_summary.csv",
            markdown_path=output_dir / "passk_summary.md",
        )
        write_failure_summary(
            _failure_rows(sample_results),
            output_dir / "failure_summary.md",
        )
        return MultiSampleRunResult(
            run_id=config.run_id,
            sample_results=sample_results,
            summary_rows=summary_rows,
            output_dir=str(output_dir),
            metadata={
                "benchmark_id": package.benchmark_id,
                "provider": config.provider,
                "model_name": config.model_name,
                "sample_results_path": str(sample_results_path),
                "cache_dir": str(cache_dir),
            },
        )


def build_passk_summary_rows(
    *,
    sample_results: list[SampleResult],
    config: MultiSampleRunConfig,
) -> list[dict[str, Any]]:
    """Aggregate sample results into pass@k summary rows."""

    grouped: dict[tuple[str, str, str], list[SampleResult]] = defaultdict(list)
    for sample in sample_results:
        grouped[(sample.task_id, sample.input_mode, sample.ablation_id)].append(sample)

    rows: list[dict[str, Any]] = []
    for (task_id, input_mode, ablation_id), samples in sorted(grouped.items()):
        ordered = sorted(samples, key=lambda sample: sample.sample_index)
        num_samples = len(ordered)
        num_correct = sum(1 for sample in ordered if sample.success)
        successes = [sample.success for sample in ordered]
        for k in config.k_values:
            rows.append(
                {
                    "run_id": config.run_id,
                    "provider": config.provider,
                    "model_name": config.model_name,
                    "task_id": task_id,
                    "input_mode": input_mode,
                    "ablation_id": ablation_id,
                    "num_samples": num_samples,
                    "num_correct": num_correct,
                    "k": k,
                    "estimated_pass_at_k": _format_float(
                        estimate_pass_at_k(num_samples, num_correct, k),
                    ),
                    "observed_pass_at_k": observed_pass_at_k(successes, k),
                    "success_criterion": config.success_criterion,
                    "avg_checkpoint_accuracy": _format_float(
                        _average([sample.checkpoint_accuracy for sample in ordered]),
                    ),
                    "framework_pass_rate": _format_float(
                        _rate(sample.framework_passed for sample in ordered),
                    ),
                    "parse_success_rate": _format_float(
                        _rate(sample.parse_success for sample in ordered),
                    ),
                    "generation_error_rate": _format_float(
                        _rate(
                            sample.generation_error is not None
                            for sample in ordered
                        ),
                    ),
                    "cache_hit_rate": _format_float(
                        _rate(sample.cache_hit for sample in ordered),
                    ),
                }
            )
    return rows


def _run_or_resume_sample(
    *,
    package: BenchmarkPackage,
    task: BenchmarkTask,
    scenario: ScenarioSpec,
    setting: RealAblationSetting,
    input_mode: str,
    sample_index: int,
    config: MultiSampleRunConfig,
    samples_root: Path,
    cache_dir: Path,
) -> SampleResult:
    sample_dir = (
        samples_root
        / task.task_id
        / input_mode
        / setting.ablation_id
        / f"sample_{sample_index:03d}"
    )
    result_path = sample_dir / "sample_result.json"
    if config.resume and result_path.exists():
        return SampleResult.from_dict(_load_json(result_path))

    sample_dir.mkdir(parents=True, exist_ok=True)
    seed = _sample_seed(config.base_seed, sample_index)
    try:
        sample = _run_sample(
            package=package,
            task=task,
            scenario=scenario,
            setting=setting,
            input_mode=input_mode,
            sample_index=sample_index,
            seed=seed,
            config=config,
            sample_dir=sample_dir,
            cache_dir=cache_dir,
        )
    except Exception as error:  # noqa: BLE001
        sample = _failed_sample_result(
            task=task,
            setting=setting,
            input_mode=input_mode,
            sample_index=sample_index,
            seed=seed,
            sample_dir=sample_dir,
            error=error,
        )
    _save_json(sample.to_dict(), result_path)
    return sample


def _run_sample(
    *,
    package: BenchmarkPackage,
    task: BenchmarkTask,
    scenario: ScenarioSpec,
    setting: RealAblationSetting,
    input_mode: str,
    sample_index: int,
    seed: int,
    config: MultiSampleRunConfig,
    sample_dir: Path,
    cache_dir: Path,
) -> SampleResult:
    mag = _load_mag_for_mode(scenario, input_mode)
    framework = load_framework_spec_from_yaml(scenario.framework_path)
    constraints = constraints_with_interaction_contracts(
        extract_model_constraints(mag),
        build_interaction_contracts(mag, framework),
    )
    hybrid_mode = "graph+nl" if setting.use_nl else "graph-only"
    hybrid_input = build_hybrid_input(mag, mode=hybrid_mode)
    prompt_text = hybrid_input.to_prompt_text(
        render_mode=str(config.metadata.get("prompt_render_mode", "default"))
    )
    request = GenerationRequest(
        hybrid_input=hybrid_input,
        config=GenerationConfig(
            provider=config.provider,
            model_name=config.model_name,
            api_base_url=config.api_base_url,
            api_key_env=config.api_key_env,
            timeout_seconds=config.timeout_seconds,
            temperature=config.temperature,
            top_p=config.top_p,
            max_tokens=config.max_tokens,
            stop=config.stop,
            think=config.think,
            native_endpoint=config.native_endpoint,
            num_predict=config.num_predict,
            seed=seed,
            prompt_variant=setting.ablation_id,
            prompt_profile_id=config.prompt_profile_id,
            prompt_profile_path=config.prompt_profile_path,
            require_non_empty_code=config.require_non_empty_code,
            input_mode=input_mode,
            cache_enabled=config.cache_enabled,
            dry_run=config.provider == "dry_run",
            mode=hybrid_mode,
            metadata={
                "run_id": config.run_id,
                "benchmark_id": package.benchmark_id,
                "task_id": task.task_id,
                "input_mode": input_mode,
                "ablation_id": setting.ablation_id,
                "sample_index": sample_index,
                "seed": seed,
            },
        ),
        prompt_text=prompt_text,
    )
    generation = generate_with_provider(request, cache_dir=cache_dir)
    generation_path = sample_dir / "generation_result.json"
    save_generation_result(generation, generation_path)

    code_initial_path = sample_dir / "code_initial.py"
    code_initial_path.write_text(generation.code_initial, encoding="utf-8")
    initial_analysis = analyze_python_source(generation.code_initial)
    initial_report = check_consistency(initial_analysis, constraints, framework)
    initial_report_path = sample_dir / "initial_report.json"
    _save_json(initial_report.to_dict(), initial_report_path)

    if setting.use_refinement:
        refined = refine_code(generation.code_initial, mag, framework)
        code_refined_path = sample_dir / "code_refined.py"
        code_refined_path.write_text(refined.code_refined, encoding="utf-8")
        refined_analysis = analyze_python_source(refined.code_refined)
        refined_report = refined.refined_report
        refined_report_path = sample_dir / "refined_report.json"
        _save_json(refined_report.to_dict(), refined_report_path)
        active_code = refined.code_refined
        active_analysis = refined_analysis
        active_report = refined_report
    else:
        code_refined_path = None
        refined_report_path = None
        active_code = generation.code_initial
        active_analysis = initial_analysis
        active_report = initial_report

    checkpoints = evaluate_checkpoints(
        code=active_code,
        analysis=active_analysis,
        report=active_report,
        constraints=constraints,
        framework=framework,
        scenario=scenario,
    )
    accuracy = checkpoint_accuracy(checkpoints)
    sample_mandatory_success = mandatory_success(
        checkpoints,
        task.mandatory_checkpoints,
    )
    sample_framework_passed = framework_passed(active_report)
    parse_success = not active_analysis.parse_errors
    success = evaluate_success_criterion(
        criterion=config.success_criterion,
        parse_success=parse_success,
        framework_passed=sample_framework_passed,
        mandatory_success=sample_mandatory_success,
        checkpoint_accuracy=accuracy,
    )
    return SampleResult(
        task_id=task.task_id,
        input_mode=input_mode,
        ablation_id=setting.ablation_id,
        sample_index=sample_index,
        seed=seed,
        generation_result_path=str(generation_path),
        code_initial_path=str(code_initial_path),
        code_refined_path=(
            None if code_refined_path is None else str(code_refined_path)
        ),
        initial_report_path=str(initial_report_path),
        refined_report_path=(
            None if refined_report_path is None else str(refined_report_path)
        ),
        parse_success=parse_success,
        framework_passed=sample_framework_passed,
        mandatory_success=sample_mandatory_success,
        checkpoint_accuracy=accuracy,
        success=success,
        cache_hit=bool(generation.metadata.get("cache_hit", False)),
        latency_seconds=float(generation.metadata.get("latency_seconds", 0.0)),
        metadata={
            "provider": config.provider,
            "model_name": config.model_name,
            "use_nl": setting.use_nl,
            "use_refinement": setting.use_refinement,
            "timeout_seconds": config.timeout_seconds,
            "max_tokens": config.max_tokens,
            "think": config.think,
            "native_endpoint": config.native_endpoint,
            "num_predict": config.num_predict,
            "prompt_profile_id": generation.metadata.get("prompt_profile_id"),
            "final_content_empty": generation.metadata.get("final_content_empty"),
            "reasoning_present": generation.metadata.get("reasoning_present"),
            "reasoning_may_contain_code": generation.metadata.get(
                "reasoning_may_contain_code"
            ),
            "extraction_strategy_used": generation.metadata.get(
                "extraction_strategy_used"
            ),
            "extracted_code_non_empty": generation.metadata.get(
                "extracted_code_non_empty"
            ),
            "prompt_hash": generation.metadata.get("prompt_hash"),
            "request_hash": generation.metadata.get("request_hash"),
            "cache_path": generation.metadata.get("cache_path"),
            "generation_error": generation.metadata.get("error"),
        },
    )


def _failed_sample_result(
    *,
    task: BenchmarkTask,
    setting: RealAblationSetting,
    input_mode: str,
    sample_index: int,
    seed: int,
    sample_dir: Path,
    error: Exception,
) -> SampleResult:
    return SampleResult(
        task_id=task.task_id,
        input_mode=input_mode,
        ablation_id=setting.ablation_id,
        sample_index=sample_index,
        seed=seed,
        generation_result_path=str(sample_dir / "generation_result.json"),
        code_initial_path=str(sample_dir / "code_initial.py"),
        code_refined_path=(
            str(sample_dir / "code_refined.py") if setting.use_refinement else None
        ),
        initial_report_path=str(sample_dir / "initial_report.json"),
        refined_report_path=(
            str(sample_dir / "refined_report.json")
            if setting.use_refinement
            else None
        ),
        parse_success=False,
        framework_passed=False,
        mandatory_success=False,
        checkpoint_accuracy=0.0,
        success=False,
        cache_hit=False,
        generation_error=f"{type(error).__name__}: {error}",
        latency_seconds=0.0,
        metadata={
            "use_nl": setting.use_nl,
            "use_refinement": setting.use_refinement,
        },
    )


def _selected_settings(ablation_ids: list[str]) -> list[RealAblationSetting]:
    settings_by_id = {
        setting.ablation_id: setting for setting in default_real_ablation_settings()
    }
    missing = [
        ablation_id
        for ablation_id in ablation_ids
        if ablation_id not in settings_by_id
    ]
    if missing:
        raise ValueError(f"Unknown ablation ids: {', '.join(missing)}")
    return [settings_by_id[ablation_id] for ablation_id in ablation_ids]


def _load_mag_for_mode(scenario: ScenarioSpec, input_mode: str) -> MAGModel:
    if input_mode == "structured":
        return load_mag_from_yaml(scenario.mag_path)
    if input_mode == "graphical":
        if (
            scenario.graphical_dot_path is None
            or scenario.graphical_manifest_path is None
        ):
            raise ValueError("Graphical input mode requires DOT and manifest paths.")
        return load_mag_from_graphical_artifact(
            scenario.graphical_dot_path,
            scenario.graphical_manifest_path,
        )
    raise ValueError(f"Unsupported input mode: {input_mode}")


def _sample_seed(base_seed: int | None, sample_index: int) -> int:
    if base_seed is None:
        return sample_index
    return base_seed + sample_index


def _validate_config(config: MultiSampleRunConfig) -> None:
    if config.num_samples <= 0:
        raise ValueError("num_samples must be positive.")
    if not config.k_values or any(k <= 0 for k in config.k_values):
        raise ValueError("k_values must contain positive integers.")
    evaluate_success_criterion(
        criterion=config.success_criterion,
        parse_success=False,
        framework_passed=False,
        mandatory_success=False,
        checkpoint_accuracy=0.0,
    )


def _write_sample_results_jsonl(samples: list[SampleResult], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        for sample in samples:
            file.write(json.dumps(sample.to_dict(), sort_keys=True) + "\n")


def _failure_rows(samples: list[SampleResult]) -> list[dict[str, Any]]:
    return [
        {
            "task_id": sample.task_id,
            "input_mode": sample.input_mode,
            "ablation_id": sample.ablation_id,
            "sample_index": sample.sample_index,
            "generation_error": sample.generation_error or "",
            "success": sample.success,
        }
        for sample in samples
        if sample.generation_error is not None or not sample.success
    ]


def _save_run_config(config: MultiSampleRunConfig, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        yaml.safe_dump(config.to_dict(), file, sort_keys=False)


def _save_json(data: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(data, file, indent=2, sort_keys=True)
        file.write("\n")


def _load_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as file:
        data = json.load(file)
    if not isinstance(data, dict):
        raise TypeError("Expected JSON object.")
    return data


def _average(values: list[float]) -> float:
    if not values:
        return 0.0
    return sum(values) / len(values)


def _rate(values: Any) -> float:
    items = list(values)
    if not items:
        return 0.0
    return sum(1 for value in items if value) / len(items)


def _format_float(value: float) -> str:
    return f"{value:.3f}"


def _optional_str(value: Any) -> str | None:
    if value is None:
        return None
    return str(value)
