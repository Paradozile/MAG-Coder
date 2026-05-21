"""Single-run real-provider ablation evaluation."""

from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml

from magcoder.benchmark.package import BenchmarkPackage
from magcoder.benchmark.task import BenchmarkTask
from magcoder.evaluation.checkpoints import CheckpointResult, evaluate_checkpoints
from magcoder.evaluation.metrics import (
    checkpoint_accuracy,
    framework_passed,
    framework_violation_count,
    mandatory_success,
)
from magcoder.evaluation.scenario import ScenarioSpec
from magcoder.generation import (
    DEFAULT_GENERATION_CACHE_DIR,
    GenerationConfig,
    GenerationRequest,
    GenerationResult,
    generate_with_provider,
    save_generation_result,
)
from magcoder.graphics import load_mag_from_graphical_artifact
from magcoder.models import MAGModel, load_mag_from_yaml
from magcoder.refinement import (
    ASTAnalysisResult,
    ConsistencyReport,
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
class RealRunConfig:
    """Configuration for a single-run real-provider ablation."""

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
    prompt_profile_id: str | None = None
    prompt_profile_path: str | None = None
    require_non_empty_code: bool = False
    seed: int | None = None
    cache_enabled: bool = True
    input_modes: list[str] = field(
        default_factory=lambda: ["structured", "graphical"],
    )
    ablation_ids: list[str] = field(
        default_factory=lambda: [
            "graph_nl_with_refinement",
            "graph_only_with_refinement",
            "graph_nl_without_refinement",
        ],
    )
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Render this config as plain Python containers."""

        return asdict(self)


@dataclass(slots=True)
class RealAblationSetting:
    """One real-run ablation setting, independent of input mode."""

    ablation_id: str
    use_nl: bool
    use_refinement: bool
    description: str

    def to_dict(self) -> dict[str, Any]:
        """Render this setting as plain Python containers."""

        return asdict(self)


@dataclass(slots=True)
class RealRunSampleResult:
    """Result for one task, input mode, and ablation setting."""

    run_id: str
    provider: str
    model_name: str
    task_id: str
    scenario_type: str
    input_mode: str
    ablation_id: str
    use_nl: bool
    use_refinement: bool
    prompt_text: str
    raw_response: str | None = None
    code_initial: str | None = None
    initial_analysis: dict[str, Any] | None = None
    initial_report: dict[str, Any] | None = None
    code_refined: str | None = None
    refined_analysis: dict[str, Any] | None = None
    refined_report: dict[str, Any] | None = None
    generator_metadata: dict[str, Any] = field(default_factory=dict)
    cache_hit: bool = False
    error: str | None = None
    checkpoint_results: list[dict[str, Any]] = field(default_factory=list)
    checkpoint_accuracy: float = 0.0
    framework_passed: bool = False
    framework_violations: int = 0
    mandatory_success: bool = False
    parse_success: bool = False
    latency_seconds: float = 0.0
    output_paths: dict[str, str] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Render this result as plain Python containers."""

        return asdict(self)


@dataclass(slots=True)
class RealRunResult:
    """Bundle for a complete single-run real-provider ablation."""

    run_id: str
    results: list[RealRunSampleResult]
    summary_rows: list[dict[str, Any]]
    output_dir: str
    metadata: dict[str, Any] = field(default_factory=dict)


class RealSingleRunAblationRunner:
    """Run one generation per Mini MAG-Bench task/input/ablation combination."""

    def run(
        self,
        package: BenchmarkPackage,
        config: RealRunConfig,
    ) -> RealRunResult:
        """Run the configured single-run ablation matrix."""

        output_dir = Path(config.output_dir)
        layout = _prepare_output_layout(output_dir)
        _save_run_config(config, output_dir / "run_config.yaml")

        settings = _selected_settings(config.ablation_ids)
        cache_dir = Path(
            str(config.metadata.get("cache_dir", DEFAULT_GENERATION_CACHE_DIR))
        )
        results: list[RealRunSampleResult] = []
        rows: list[dict[str, Any]] = []

        for task in package.tasks:
            scenario = task.to_scenario_spec(
                base_dir=Path(package.root_dir) / "tasks" / task.task_id,
            )
            for input_mode in config.input_modes:
                if input_mode not in task.supported_input_modes:
                    continue
                for setting in settings:
                    sample = _run_sample(
                        package=package,
                        task=task,
                        scenario=scenario,
                        setting=setting,
                        input_mode=input_mode,
                        config=config,
                        output_layout=layout,
                        cache_dir=cache_dir,
                    )
                    results.append(sample)
                    rows.append(real_run_summary_row(sample))

        write_real_run_summary_reports(
            rows,
            csv_path=output_dir / "real_ablation_summary.csv",
            markdown_path=output_dir / "real_ablation_summary.md",
        )
        return RealRunResult(
            run_id=config.run_id,
            results=results,
            summary_rows=rows,
            output_dir=str(output_dir),
            metadata={
                "benchmark_id": package.benchmark_id,
                "provider": config.provider,
                "model_name": config.model_name,
                "cache_dir": str(cache_dir),
            },
        )


def default_real_ablation_settings() -> list[RealAblationSetting]:
    """Return default Phase 6B single-run ablation settings."""

    return [
        RealAblationSetting(
            ablation_id="graph_nl_with_refinement",
            use_nl=True,
            use_refinement=True,
            description="graph+NL prompt with deterministic refinement.",
        ),
        RealAblationSetting(
            ablation_id="graph_only_with_refinement",
            use_nl=False,
            use_refinement=True,
            description="graph-only prompt with deterministic refinement.",
        ),
        RealAblationSetting(
            ablation_id="graph_nl_without_refinement",
            use_nl=True,
            use_refinement=False,
            description="graph+NL prompt without deterministic refinement.",
        ),
    ]


def real_run_summary_row(result: RealRunSampleResult) -> dict[str, Any]:
    """Build one summary row for a real-run sample."""

    return {
        "run_id": result.run_id,
        "provider": result.provider,
        "model_name": result.model_name,
        "task_id": result.task_id,
        "input_mode": result.input_mode,
        "ablation_id": result.ablation_id,
        "use_nl": result.use_nl,
        "use_refinement": result.use_refinement,
        "cache_hit": result.cache_hit,
        "generation_error": "" if result.error is None else result.error,
        "parse_success": result.parse_success,
        "checkpoint_accuracy": _format_float(result.checkpoint_accuracy),
        "framework_passed": result.framework_passed,
        "framework_violations": result.framework_violations,
        "mandatory_success": result.mandatory_success,
        "latency_seconds": _format_float(result.latency_seconds),
    }


def write_real_run_summary_reports(
    rows: list[dict[str, Any]],
    *,
    csv_path: str | Path,
    markdown_path: str | Path,
) -> None:
    """Write CSV and Markdown summary reports for a real single-run ablation."""

    _write_csv(rows, Path(csv_path))
    _write_markdown(rows, Path(markdown_path))


def _run_sample(
    *,
    package: BenchmarkPackage,
    task: BenchmarkTask,
    scenario: ScenarioSpec,
    setting: RealAblationSetting,
    input_mode: str,
    config: RealRunConfig,
    output_layout: dict[str, Path],
    cache_dir: Path,
) -> RealRunSampleResult:
    base_name = f"{task.task_id}_{input_mode}_{setting.ablation_id}"
    try:
        mag = _load_mag_for_mode(scenario, input_mode)
        framework = load_framework_spec_from_yaml(scenario.framework_path)
        constraints = constraints_with_interaction_contracts(
            extract_model_constraints(mag),
            build_interaction_contracts(mag, framework),
        )
        hybrid_mode = "graph+nl" if setting.use_nl else "graph-only"
        hybrid_input = build_hybrid_input(mag, mode=hybrid_mode)
        prompt_text = hybrid_input.to_prompt_text()
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
                seed=config.seed,
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
                },
            ),
            prompt_text=prompt_text,
        )
        generation = generate_with_provider(request, cache_dir=cache_dir)
        raw_path = output_layout["raw_generations"] / f"{base_name}.json"
        save_generation_result(generation, raw_path)

        initial_analysis = analyze_python_source(generation.code_initial)
        initial_report = check_consistency(initial_analysis, constraints, framework)
        report_path = output_layout["reports"] / f"{base_name}_initial_report.json"
        _save_json(initial_report.to_dict(), report_path)

        if setting.use_refinement:
            refined = refine_code(generation.code_initial, mag, framework)
            code_refined = refined.code_refined
            refined_analysis = analyze_python_source(code_refined)
            refined_report = refined.refined_report
            code_path = output_layout["refined_outputs"] / f"{base_name}.py"
            code_path.write_text(code_refined, encoding="utf-8")
            refined_report_path = (
                output_layout["reports"] / f"{base_name}_refined_report.json"
            )
            _save_json(refined_report.to_dict(), refined_report_path)
            active_code = code_refined
            active_analysis = refined_analysis
            active_report = refined_report
        else:
            code_refined = None
            refined_analysis = None
            refined_report = None
            code_path = output_layout["raw_generations"] / f"{base_name}.py"
            code_path.write_text(generation.code_initial, encoding="utf-8")
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
        result = _successful_sample_result(
            config=config,
            task=task,
            input_mode=input_mode,
            setting=setting,
            prompt_text=prompt_text,
            generation=generation,
            initial_analysis=initial_analysis,
            initial_report=initial_report,
            code_refined=code_refined,
            refined_analysis=refined_analysis,
            refined_report=refined_report,
            active_analysis=active_analysis,
            active_report=active_report,
            checkpoints=checkpoints,
            output_paths={
                "result": str(output_layout["results"] / f"{base_name}_result.json"),
                "raw_generation": str(raw_path),
                "initial_report": str(report_path),
                "code": str(code_path),
                **(
                    {}
                    if refined_report_path is None
                    else {"refined_report": str(refined_report_path)}
                ),
            },
        )
    except Exception as error:  # noqa: BLE001
        result = _failed_sample_result(
            config=config,
            task=task,
            input_mode=input_mode,
            setting=setting,
            prompt_text=_best_effort_prompt(scenario, input_mode, setting),
            error=error,
            output_path=output_layout["results"] / f"{base_name}_result.json",
        )

    result_path = output_layout["results"] / f"{base_name}_result.json"
    result.output_paths["result"] = str(result_path)
    _save_json(result.to_dict(), result_path)
    return result


def _successful_sample_result(
    *,
    config: RealRunConfig,
    task: BenchmarkTask,
    input_mode: str,
    setting: RealAblationSetting,
    prompt_text: str,
    generation: GenerationResult,
    initial_analysis: ASTAnalysisResult,
    initial_report: ConsistencyReport,
    code_refined: str | None,
    refined_analysis: ASTAnalysisResult | None,
    refined_report: ConsistencyReport | None,
    active_analysis: ASTAnalysisResult,
    active_report: ConsistencyReport,
    checkpoints: list[CheckpointResult],
    output_paths: dict[str, str],
) -> RealRunSampleResult:
    return RealRunSampleResult(
        run_id=config.run_id,
        provider=config.provider,
        model_name=config.model_name,
        task_id=task.task_id,
        scenario_type=task.scenario_type,
        input_mode=input_mode,
        ablation_id=setting.ablation_id,
        use_nl=setting.use_nl,
        use_refinement=setting.use_refinement,
        prompt_text=prompt_text,
        raw_response=generation.raw_response,
        code_initial=generation.code_initial,
        initial_analysis=initial_analysis.to_dict(),
        initial_report=initial_report.to_dict(),
        code_refined=code_refined,
        refined_analysis=(
            None if refined_analysis is None else refined_analysis.to_dict()
        ),
        refined_report=None if refined_report is None else refined_report.to_dict(),
        generator_metadata=generation.metadata,
        cache_hit=bool(generation.metadata.get("cache_hit", False)),
        checkpoint_results=[checkpoint.to_dict() for checkpoint in checkpoints],
        checkpoint_accuracy=checkpoint_accuracy(checkpoints),
        framework_passed=framework_passed(active_report),
        framework_violations=framework_violation_count(active_report),
        mandatory_success=mandatory_success(
            checkpoints,
            _mandatory_checkpoints_from_metadata(task),
        ),
        parse_success=not active_analysis.parse_errors,
        latency_seconds=float(generation.metadata.get("latency_seconds", 0.0)),
        output_paths=output_paths,
        metadata={"description": setting.description},
    )


def _failed_sample_result(
    *,
    config: RealRunConfig,
    task: BenchmarkTask,
    input_mode: str,
    setting: RealAblationSetting,
    prompt_text: str,
    error: Exception,
    output_path: Path,
) -> RealRunSampleResult:
    return RealRunSampleResult(
        run_id=config.run_id,
        provider=config.provider,
        model_name=config.model_name,
        task_id=task.task_id,
        scenario_type=task.scenario_type,
        input_mode=input_mode,
        ablation_id=setting.ablation_id,
        use_nl=setting.use_nl,
        use_refinement=setting.use_refinement,
        prompt_text=prompt_text,
        error=f"{type(error).__name__}: {error}",
        output_paths={"result": str(output_path)},
        metadata={"description": setting.description},
    )


def _selected_settings(ablation_ids: list[str]) -> list[RealAblationSetting]:
    settings_by_id = {
        setting.ablation_id: setting for setting in default_real_ablation_settings()
    }
    selected_ids = ablation_ids or list(settings_by_id)
    missing = [
        ablation_id
        for ablation_id in selected_ids
        if ablation_id not in settings_by_id
    ]
    if missing:
        raise ValueError(f"Unknown real-run ablation ids: {', '.join(missing)}")
    return [settings_by_id[ablation_id] for ablation_id in selected_ids]


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


def _best_effort_prompt(
    scenario: ScenarioSpec,
    input_mode: str,
    setting: RealAblationSetting,
) -> str:
    try:
        mag = _load_mag_for_mode(scenario, input_mode)
        hybrid_mode = "graph+nl" if setting.use_nl else "graph-only"
        return build_hybrid_input(mag, mode=hybrid_mode).to_prompt_text()
    except Exception:  # noqa: BLE001
        return ""


def _mandatory_checkpoints_from_metadata(task: BenchmarkTask) -> list[str]:
    return list(task.mandatory_checkpoints)


def _prepare_output_layout(output_dir: Path) -> dict[str, Path]:
    layout = {
        "results": output_dir / "results",
        "raw_generations": output_dir / "raw_generations",
        "refined_outputs": output_dir / "refined_outputs",
        "reports": output_dir / "reports",
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    for path in layout.values():
        path.mkdir(parents=True, exist_ok=True)
    return layout


def _save_run_config(config: RealRunConfig, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        yaml.safe_dump(config.to_dict(), file, sort_keys=False)


def _save_json(data: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(data, file, indent=2, sort_keys=True)
        file.write("\n")


def _write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=_fieldnames())
        writer.writeheader()
        writer.writerows(rows)


def _write_markdown(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    headers = _fieldnames()
    lines = [
        "# Real LLM Single-Run Ablation Summary",
        "",
        "This table compares graph+NL, graph-only, refinement on/off, "
        "structured input, graphical input, and failed cases for one generation "
        "sample per task/configuration.",
        "",
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(row[header]) for header in headers) + " |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _fieldnames() -> list[str]:
    return [
        "run_id",
        "provider",
        "model_name",
        "task_id",
        "input_mode",
        "ablation_id",
        "use_nl",
        "use_refinement",
        "cache_hit",
        "generation_error",
        "parse_success",
        "checkpoint_accuracy",
        "framework_passed",
        "framework_violations",
        "mandatory_success",
        "latency_seconds",
    ]


def _format_float(value: float) -> str:
    return f"{value:.3f}"
