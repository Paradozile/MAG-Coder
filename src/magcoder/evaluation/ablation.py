"""Ablation evaluation over Mini MAG-Bench packages."""

from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal

from magcoder.benchmark.package import BenchmarkPackage
from magcoder.benchmark.task import BenchmarkTask
from magcoder.evaluation.checkpoints import evaluate_checkpoints
from magcoder.evaluation.metrics import (
    checkpoint_accuracy,
    framework_passed,
    framework_violation_count,
    mandatory_success,
)
from magcoder.evaluation.runner import ScenarioEvaluationResult
from magcoder.evaluation.scenario import ScenarioSpec
from magcoder.generation import (
    GenerationConfig,
    GenerationRequest,
    generate_with_provider,
)
from magcoder.graphics import load_mag_from_graphical_artifact
from magcoder.models import MAGModel, load_mag_from_yaml
from magcoder.refinement import (
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

AblationInputMode = Literal["structured", "graphical"]


@dataclass(slots=True)
class AblationConfig:
    """Configuration for one ablation setting."""

    ablation_id: str
    input_mode: AblationInputMode
    use_nl: bool
    use_refinement: bool
    description: str
    provider: str = "fake"
    model_name: str = "fake-generator"
    api_base_url: str | None = None
    api_key_env: str | None = None
    cache_enabled: bool = False
    dry_run: bool = False
    generation_options: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Render this config as plain Python containers."""

        return asdict(self)


@dataclass(slots=True)
class AblationTaskResult:
    """One benchmark task evaluated under one ablation config."""

    benchmark_id: str
    task_id: str
    scenario_type: str
    ablation: AblationConfig
    result: ScenarioEvaluationResult
    prompt_text: str
    code_used: str
    framework_violations: int
    notes: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Render this ablation result as plain Python containers."""

        return {
            "benchmark_id": self.benchmark_id,
            "task_id": self.task_id,
            "scenario_type": self.scenario_type,
            "ablation": self.ablation.to_dict(),
            "result": self.result.to_dict(),
            "prompt_text": self.prompt_text,
            "code_used": self.code_used,
            "framework_violations": self.framework_violations,
            "notes": self.notes,
            "metadata": self.metadata,
        }


@dataclass(slots=True)
class AblationBenchmarkResult:
    """Bundle of all task/config ablation results."""

    benchmark_id: str
    results: list[AblationTaskResult]
    summary_rows: list[dict[str, Any]]
    output_dir: str
    metadata: dict[str, Any] = field(default_factory=dict)


class AblationEvaluationRunner:
    """Run Mini MAG-Bench tasks under ablation settings."""

    def run(
        self,
        package: BenchmarkPackage,
        *,
        output_dir: str | Path,
        configs: list[AblationConfig] | None = None,
    ) -> AblationBenchmarkResult:
        """Run every task/config pair and write JSON plus summary reports."""

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        active_configs = configs if configs is not None else default_ablation_configs()

        results: list[AblationTaskResult] = []
        rows: list[dict[str, Any]] = []
        for task in package.tasks:
            for config in active_configs:
                if config.input_mode not in task.supported_input_modes:
                    continue
                scenario = task.to_scenario_spec(
                    base_dir=Path(package.root_dir) / "tasks" / task.task_id,
                )
                result = _run_task_ablation(
                    benchmark_id=package.benchmark_id,
                    task=task,
                    scenario=scenario,
                    config=config,
                )
                results.append(result)
                rows.append(ablation_summary_row(result))
                _save_ablation_result(
                    result,
                    output_path
                    / f"{task.task_id}_{config.ablation_id}_result.json",
                )

        write_ablation_summary_reports(
            rows,
            csv_path=output_path / "ablation_summary.csv",
            markdown_path=output_path / "ablation_summary.md",
        )
        return AblationBenchmarkResult(
            benchmark_id=package.benchmark_id,
            results=results,
            summary_rows=rows,
            output_dir=str(output_path),
            metadata={"config_count": len(active_configs)},
        )


def default_ablation_configs() -> list[AblationConfig]:
    """Return the default Phase 5C ablation settings."""

    return [
        AblationConfig(
            ablation_id="full_structured",
            input_mode="structured",
            use_nl=True,
            use_refinement=True,
            description="Structured MAG input with NL descriptions and refinement.",
        ),
        AblationConfig(
            ablation_id="no_nl_structured",
            input_mode="structured",
            use_nl=False,
            use_refinement=True,
            description="Structured MAG input using graph-only hybrid semantics.",
        ),
        AblationConfig(
            ablation_id="no_refinement_structured",
            input_mode="structured",
            use_nl=True,
            use_refinement=False,
            description="Structured MAG input without deterministic refinement.",
        ),
        AblationConfig(
            ablation_id="full_graphical",
            input_mode="graphical",
            use_nl=True,
            use_refinement=True,
            description="Graphical DOT+manifest input with NL and refinement.",
        ),
        AblationConfig(
            ablation_id="no_nl_graphical",
            input_mode="graphical",
            use_nl=False,
            use_refinement=True,
            description="Graphical DOT+manifest input using graph-only semantics.",
        ),
        AblationConfig(
            ablation_id="no_refinement_graphical",
            input_mode="graphical",
            use_nl=True,
            use_refinement=False,
            description="Graphical DOT+manifest input without refinement.",
        ),
    ]


def ablation_summary_row(result: AblationTaskResult) -> dict[str, Any]:
    """Build one CSV/Markdown row from an ablation result."""

    return {
        "benchmark_id": result.benchmark_id,
        "task_id": result.task_id,
        "scenario_type": result.scenario_type,
        "ablation_id": result.ablation.ablation_id,
        "input_mode": result.ablation.input_mode,
        "use_nl": result.ablation.use_nl,
        "use_refinement": result.ablation.use_refinement,
        "checkpoint_accuracy": _format_float(result.result.checkpoint_accuracy),
        "framework_passed": result.result.framework_passed,
        "framework_violations": result.framework_violations,
        "mandatory_success": result.result.mandatory_success,
        "notes": "; ".join(result.notes),
    }


def write_ablation_summary_reports(
    rows: list[dict[str, Any]],
    *,
    csv_path: str | Path,
    markdown_path: str | Path,
) -> None:
    """Write ablation summary CSV and Markdown reports."""

    _write_csv(rows, Path(csv_path))
    _write_markdown(rows, Path(markdown_path))


def _run_task_ablation(
    *,
    benchmark_id: str,
    task: BenchmarkTask,
    scenario: ScenarioSpec,
    config: AblationConfig,
) -> AblationTaskResult:
    mag = _load_mag_for_mode(scenario, config.input_mode)
    framework = load_framework_spec_from_yaml(scenario.framework_path)
    hybrid_mode = "graph+nl" if config.use_nl else "graph-only"
    hybrid_input = build_hybrid_input(mag, mode=hybrid_mode)
    prompt_text = hybrid_input.to_prompt_text()
    generation = generate_with_provider(
        GenerationRequest(
            hybrid_input=hybrid_input,
            config=GenerationConfig(
                provider=config.provider,
                model_name=config.model_name,
                api_base_url=config.api_base_url,
                api_key_env=config.api_key_env,
                timeout_seconds=int(
                    config.generation_options.get("timeout_seconds", 60),
                ),
                temperature=float(config.generation_options.get("temperature", 0.0)),
                top_p=config.generation_options.get("top_p"),
                max_tokens=config.generation_options.get("max_tokens"),
                seed=config.generation_options.get("seed"),
                prompt_variant=config.ablation_id,
                input_mode=config.input_mode,
                cache_enabled=config.cache_enabled,
                dry_run=config.dry_run,
                mode=hybrid_mode,
                metadata={
                    "benchmark_id": benchmark_id,
                    "task_id": task.task_id,
                    "ablation_id": config.ablation_id,
                    "input_mode": config.input_mode,
                    "use_nl": config.use_nl,
                    "use_refinement": config.use_refinement,
                },
            ),
            prompt_text=prompt_text,
        )
    )
    constraints = constraints_with_interaction_contracts(
        extract_model_constraints(mag),
        build_interaction_contracts(mag, framework),
    )
    initial_analysis = analyze_python_source(generation.code_initial)
    initial_report = check_consistency(initial_analysis, constraints, framework)
    notes = _notes_for_config(config)

    if config.use_refinement:
        refined = refine_code(generation.code_initial, mag, framework)
        analysis = analyze_python_source(refined.code_refined)
        report = refined.refined_report
        code_used = refined.code_refined
        initial_report_for_result: ConsistencyReport | None = refined.initial_report
        refined_report_for_result: ConsistencyReport | None = refined.refined_report
        metadata = {
            "generation_metadata": generation.metadata,
            "hybrid_mode": hybrid_mode,
            "rewrite_actions": refined.rewrite_actions,
            "refinement_applied": True,
        }
    else:
        analysis = initial_analysis
        report = initial_report
        code_used = generation.code_initial
        initial_report_for_result = initial_report
        refined_report_for_result = None
        metadata = {
            "generation_metadata": generation.metadata,
            "hybrid_mode": hybrid_mode,
            "rewrite_actions": [],
            "refinement_applied": False,
        }

    checkpoints = evaluate_checkpoints(
        code=code_used,
        analysis=analysis,
        report=report,
        constraints=constraints,
        framework=framework,
        scenario=scenario,
    )
    evaluation = ScenarioEvaluationResult(
        scenario_id=scenario.scenario_id,
        input_mode=config.input_mode,
        method="full_pipeline" if config.use_refinement else "initial",
        checkpoint_results=checkpoints,
        checkpoint_accuracy=checkpoint_accuracy(checkpoints),
        mandatory_success=mandatory_success(
            checkpoints,
            scenario.mandatory_checkpoints,
        ),
        framework_passed=framework_passed(report),
        initial_report=initial_report_for_result,
        refined_report=refined_report_for_result,
        metadata={
            **metadata,
            "ablation_id": config.ablation_id,
            "description": config.description,
            "code_length": len(code_used),
            "prompt_text_length": len(prompt_text),
        },
    )
    return AblationTaskResult(
        benchmark_id=benchmark_id,
        task_id=task.task_id,
        scenario_type=task.scenario_type,
        ablation=config,
        result=evaluation,
        prompt_text=prompt_text,
        code_used=code_used,
        framework_violations=framework_violation_count(report),
        notes=notes,
        metadata={
            "mag_id": mag.id,
            "generator": generation.generator_name,
            "prompt_text_sha256": generation.metadata.get("prompt_text_sha256"),
        },
    )


def _load_mag_for_mode(
    scenario: ScenarioSpec,
    input_mode: AblationInputMode,
) -> MAGModel:
    if input_mode == "structured":
        return load_mag_from_yaml(scenario.mag_path)
    if scenario.graphical_dot_path is None or scenario.graphical_manifest_path is None:
        raise ValueError("Graphical input mode requires DOT and manifest paths.")
    return load_mag_from_graphical_artifact(
        scenario.graphical_dot_path,
        scenario.graphical_manifest_path,
    )


def _notes_for_config(config: AblationConfig) -> list[str]:
    notes: list[str] = []
    if config.use_nl:
        notes.append("hybrid_mode=graph+nl")
    else:
        notes.append("hybrid_mode=graph-only")
    if config.use_refinement:
        notes.append("refinement_applied")
    else:
        notes.append("refinement_not_applied")
    return notes


def _save_ablation_result(result: AblationTaskResult, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(result.to_dict(), file, indent=2, sort_keys=True)
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
        "# Ablation Evaluation Summary",
        "",
        "This table compares full pipeline, graph-only, no-refinement, "
        "structured, and graphical settings under the deterministic fake "
        "generator.",
        "",
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(row[header]) for header in headers) + " |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _fieldnames() -> list[str]:
    return [
        "benchmark_id",
        "task_id",
        "scenario_type",
        "ablation_id",
        "input_mode",
        "use_nl",
        "use_refinement",
        "checkpoint_accuracy",
        "framework_passed",
        "framework_violations",
        "mandatory_success",
        "notes",
    ]


def _format_float(value: float) -> str:
    return f"{value:.3f}"
