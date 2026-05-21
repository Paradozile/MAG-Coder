"""Contract-level evaluation adapter for Full MAG-Bench cases."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from magcoder.benchmark.full_runner.case_loader import FullBenchRunCase
from magcoder.evaluation.checkpoints import CheckpointResult, evaluate_checkpoints
from magcoder.evaluation.metrics import (
    checkpoint_accuracy,
    framework_passed,
    mandatory_success,
)
from magcoder.evaluation.scenario import ScenarioSpec
from magcoder.models import MAGModel
from magcoder.refinement import (
    ASTAnalysisResult,
    ConsistencyReport,
    FrameworkSpec,
    ModelConstraints,
)

FULL_BENCH_MANDATORY_CHECKPOINTS = [
    "execution.parseable",
    "entity.agent_classes",
    "entity.attributes",
    "interaction.methods",
    "condition.filter_guards",
    "framework.no_violations",
    "framework.message_api",
]


@dataclass(slots=True)
class FullBenchEvaluationResult:
    """Contract-level evaluation result for one generated sample."""

    case_id: str
    evaluation_level: str
    checkpoint_results: list[CheckpointResult]
    checkpoint_accuracy: float
    mandatory_success: bool
    framework_passed: bool
    parse_success: bool
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Render this result as JSON-compatible containers."""

        return {
            "case_id": self.case_id,
            "evaluation_level": self.evaluation_level,
            "checkpoint_results": [
                checkpoint.to_dict() for checkpoint in self.checkpoint_results
            ],
            "checkpoint_accuracy": self.checkpoint_accuracy,
            "mandatory_success": self.mandatory_success,
            "framework_passed": self.framework_passed,
            "parse_success": self.parse_success,
            "metadata": self.metadata,
        }


def evaluate_full_bench_code(
    *,
    case: FullBenchRunCase,
    code: str,
    analysis: ASTAnalysisResult,
    report: ConsistencyReport,
    constraints: ModelConstraints,
    framework: FrameworkSpec,
    mag: MAGModel,
) -> FullBenchEvaluationResult:
    """Evaluate generated code at contract level for one Full MAG-Bench case."""

    scenario = _scenario_from_case(case, mag)
    checkpoints = evaluate_checkpoints(
        code=code,
        analysis=analysis,
        report=report,
        constraints=constraints,
        framework=framework,
        scenario=scenario,
    )
    accuracy = checkpoint_accuracy(checkpoints)
    is_mandatory_success = mandatory_success(
        checkpoints,
        FULL_BENCH_MANDATORY_CHECKPOINTS,
    )
    return FullBenchEvaluationResult(
        case_id=case.case_id,
        evaluation_level="contract",
        checkpoint_results=checkpoints,
        checkpoint_accuracy=accuracy,
        mandatory_success=is_mandatory_success,
        framework_passed=framework_passed(report),
        parse_success=not analysis.parse_errors,
        metadata={
            "source_task_id": case.package_case.source_task_id,
            "expected_contract": case.expected_contract,
            "mandatory_checkpoints": list(FULL_BENCH_MANDATORY_CHECKPOINTS),
            "functional_tests_executed": False,
            "functional_test_hook_available": case.tests_py_path.exists(),
        },
    )


def _scenario_from_case(case: FullBenchRunCase, mag: MAGModel) -> ScenarioSpec:
    contract = case.expected_contract
    expected_edges = [
        {
            "edge_id": edge.id,
            "src_agent": edge.src_agent,
            "dst_agent": edge.dst_agent,
            "filter_id": edge.filter_id,
        }
        for edge in mag.edges
    ]
    return ScenarioSpec(
        scenario_id=case.case_id,
        name=case.case_id,
        description=str(case.objectified_task.get("objectified_prompt", "")),
        mag_path=str(case.mag_yaml_path),
        graphical_dot_path=str(case.dot_path),
        graphical_manifest_path=str(case.manifest_path),
        framework_path=str(case.framework_path),
        expected_agents=[str(agent) for agent in contract.get("agents", [])],
        expected_edges=expected_edges,
        expected_filters=[str(item) for item in contract.get("filters", [])],
        mandatory_checkpoints=list(FULL_BENCH_MANDATORY_CHECKPOINTS),
        metadata={
            "evaluation_level": "contract",
            "source_task_id": case.package_case.source_task_id,
            "case_metadata": asdict(case.package_case),
        },
    )
