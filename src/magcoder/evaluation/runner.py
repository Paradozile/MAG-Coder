"""Scenario evaluation runner."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

from magcoder.evaluation.checkpoints import CheckpointResult, evaluate_checkpoints
from magcoder.evaluation.metrics import (
    checkpoint_accuracy,
    framework_passed,
    framework_violation_count,
    mandatory_success,
)
from magcoder.evaluation.scenario import ScenarioSpec
from magcoder.generation import FakeCodeGenerator, GenerationConfig, GenerationRequest
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

InputMode = Literal["structured", "graphical"]
EvaluationMethod = Literal["initial", "refined", "full_pipeline"]


@dataclass(slots=True)
class ScenarioEvaluationResult:
    """Evaluation result for one scenario/method/input-mode combination."""

    scenario_id: str
    input_mode: InputMode
    method: EvaluationMethod
    checkpoint_results: list[CheckpointResult]
    checkpoint_accuracy: float
    mandatory_success: bool
    framework_passed: bool
    initial_report: ConsistencyReport | None = None
    refined_report: ConsistencyReport | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Render this evaluation result as plain Python containers."""

        return {
            "scenario_id": self.scenario_id,
            "input_mode": self.input_mode,
            "method": self.method,
            "checkpoint_results": [
                checkpoint.to_dict() for checkpoint in self.checkpoint_results
            ],
            "checkpoint_accuracy": self.checkpoint_accuracy,
            "mandatory_success": self.mandatory_success,
            "framework_passed": self.framework_passed,
            "initial_report": (
                None if self.initial_report is None else self.initial_report.to_dict()
            ),
            "refined_report": (
                None if self.refined_report is None else self.refined_report.to_dict()
            ),
            "metadata": self.metadata,
        }


@dataclass(slots=True)
class ScenarioRunResult:
    """Initial/refined evaluation bundle for a scenario run."""

    scenario: ScenarioSpec
    input_mode: InputMode
    initial_result: ScenarioEvaluationResult
    refined_result: ScenarioEvaluationResult
    full_pipeline_result: ScenarioEvaluationResult
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Render this run result as plain Python containers."""

        return {
            "scenario": self.scenario.to_dict(),
            "input_mode": self.input_mode,
            "initial_result": self.initial_result.to_dict(),
            "refined_result": self.refined_result.to_dict(),
            "full_pipeline_result": self.full_pipeline_result.to_dict(),
            "metadata": self.metadata,
        }


class ScenarioEvaluationRunner:
    """Run scenario evaluation through generation, analysis, and refinement."""

    def run(
        self,
        scenario: ScenarioSpec,
        *,
        input_mode: InputMode,
        hybrid_mode: str = "graph+nl",
    ) -> ScenarioRunResult:
        """Run a structured or graphical scenario evaluation."""

        mag = _load_mag_for_mode(scenario, input_mode)
        framework = load_framework_spec_from_yaml(scenario.framework_path)
        hybrid_input = build_hybrid_input(mag, mode=hybrid_mode)
        prompt_text = hybrid_input.to_prompt_text()
        generation = FakeCodeGenerator().generate(
            GenerationRequest(
                hybrid_input=hybrid_input,
                config=GenerationConfig(
                    mode=hybrid_mode,
                    metadata={
                        "scenario_id": scenario.scenario_id,
                        "input_mode": input_mode,
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
        initial_report = check_consistency(
            initial_analysis,
            constraints,
            framework,
        )
        refined = refine_code(generation.code_initial, mag, framework)
        refined_analysis = analyze_python_source(refined.code_refined)

        initial_result = _build_evaluation_result(
            scenario=scenario,
            input_mode=input_mode,
            method="initial",
            code=generation.code_initial,
            analysis=initial_analysis,
            report=initial_report,
            constraints=constraints,
            framework=framework,
            initial_report=initial_report,
            refined_report=None,
            metadata={
                "generator": generation.generator_name,
                "method_label": "H-only initial generation",
                "framework_violation_count": framework_violation_count(
                    initial_report,
                ),
            },
        )
        refined_result = _build_evaluation_result(
            scenario=scenario,
            input_mode=input_mode,
            method="refined",
            code=refined.code_refined,
            analysis=refined_analysis,
            report=refined.refined_report,
            constraints=constraints,
            framework=framework,
            initial_report=refined.initial_report,
            refined_report=refined.refined_report,
            metadata={
                "method_label": "Full method with deterministic refinement",
                "rewrite_actions": refined.rewrite_actions,
                "framework_violation_count": framework_violation_count(
                    refined.refined_report,
                ),
            },
        )
        full_pipeline_result = _build_evaluation_result(
            scenario=scenario,
            input_mode=input_mode,
            method="full_pipeline",
            code=refined.code_refined,
            analysis=refined_analysis,
            report=refined.refined_report,
            constraints=constraints,
            framework=framework,
            initial_report=refined.initial_report,
            refined_report=refined.refined_report,
            metadata={
                "method_label": "Graph/structured MAG to refined code",
                "checkpoint_accuracy_initial": initial_result.checkpoint_accuracy,
                "checkpoint_accuracy_refined": refined_result.checkpoint_accuracy,
            },
        )
        return ScenarioRunResult(
            scenario=scenario,
            input_mode=input_mode,
            initial_result=initial_result,
            refined_result=refined_result,
            full_pipeline_result=full_pipeline_result,
            metadata={
                "mag_id": mag.id,
                "hybrid_mode": hybrid_mode,
                "prompt_text_length": len(prompt_text),
            },
        )


def _build_evaluation_result(
    *,
    scenario: ScenarioSpec,
    input_mode: InputMode,
    method: EvaluationMethod,
    code: str,
    analysis: Any,
    report: ConsistencyReport,
    constraints: Any,
    framework: Any,
    initial_report: ConsistencyReport | None,
    refined_report: ConsistencyReport | None,
    metadata: dict[str, Any],
) -> ScenarioEvaluationResult:
    checkpoints = evaluate_checkpoints(
        code=code,
        analysis=analysis,
        report=report,
        constraints=constraints,
        framework=framework,
        scenario=scenario,
    )
    accuracy = checkpoint_accuracy(checkpoints)
    return ScenarioEvaluationResult(
        scenario_id=scenario.scenario_id,
        input_mode=input_mode,
        method=method,
        checkpoint_results=checkpoints,
        checkpoint_accuracy=accuracy,
        mandatory_success=mandatory_success(
            checkpoints,
            scenario.mandatory_checkpoints,
        ),
        framework_passed=framework_passed(report),
        initial_report=initial_report,
        refined_report=refined_report,
        metadata={**metadata, "code_length": len(code)},
    )


def _load_mag_for_mode(scenario: ScenarioSpec, input_mode: InputMode) -> MAGModel:
    if input_mode == "structured":
        return load_mag_from_yaml(scenario.mag_path)
    if scenario.graphical_dot_path is None or scenario.graphical_manifest_path is None:
        raise ValueError("Graphical input mode requires DOT and manifest paths.")
    return load_mag_from_graphical_artifact(
        scenario.graphical_dot_path,
        scenario.graphical_manifest_path,
    )
