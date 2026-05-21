"""Checkpoint evaluators for scenario-level evaluation."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Literal

from magcoder.evaluation.scenario import ScenarioSpec
from magcoder.refinement import (
    ASTAnalysisResult,
    ConsistencyReport,
    FrameworkSpec,
    ModelConstraints,
    constraints_with_interaction_contracts,
)
from magcoder.refinement.contracts import build_interaction_contracts_from_constraints

CheckpointCategory = Literal[
    "entity",
    "interaction",
    "condition",
    "framework",
    "execution",
]


@dataclass(slots=True)
class CheckpointResult:
    """Result for one scenario checkpoint."""

    checkpoint_id: str
    category: CheckpointCategory
    description: str
    passed: bool
    evidence: dict[str, Any] = field(default_factory=dict)
    message: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Render this checkpoint result as plain Python containers."""

        return asdict(self)


def evaluate_checkpoints(
    *,
    code: str,
    analysis: ASTAnalysisResult,
    report: ConsistencyReport,
    constraints: ModelConstraints,
    framework: FrameworkSpec,
    scenario: ScenarioSpec,
) -> list[CheckpointResult]:
    """Evaluate standard scenario checkpoints for generated/refined code."""

    if not constraints.expected_interaction_contracts:
        constraints = constraints_with_interaction_contracts(
            constraints,
            build_interaction_contracts_from_constraints(constraints, framework),
        )
    return [
        _agent_class_checkpoint(analysis, constraints, scenario),
        _attribute_checkpoint(analysis, constraints),
        _interaction_method_checkpoint(analysis, constraints, framework),
        _direction_checkpoint(code, constraints),
        _filter_guard_checkpoint(report, constraints),
        _framework_violation_checkpoint(report),
        _message_api_checkpoint(analysis, framework),
        _parse_checkpoint(analysis),
    ]


def _agent_class_checkpoint(
    analysis: ASTAnalysisResult,
    constraints: ModelConstraints,
    scenario: ScenarioSpec,
) -> CheckpointResult:
    expected = constraints.expected_agent_class_names
    actual = analysis.class_names
    missing = [class_name for class_name in expected if class_name not in actual]
    message = "all expected agent classes present" if not missing else "missing classes"
    return CheckpointResult(
        checkpoint_id="entity.agent_classes",
        category="entity",
        description="Expected agent classes exist in generated code.",
        passed=not missing,
        evidence={
            "expected": expected,
            "expected_from_scenario": scenario.expected_agents,
            "actual": actual,
            "missing": missing,
        },
        message=message,
    )


def _attribute_checkpoint(
    analysis: ASTAnalysisResult,
    constraints: ModelConstraints,
) -> CheckpointResult:
    missing: dict[str, list[str]] = {}
    for class_name, expected_attributes in constraints.expected_attributes.items():
        actual = set(analysis.attributes_by_class.get(class_name, []))
        class_missing = [
            attribute for attribute in expected_attributes if attribute not in actual
        ]
        if class_missing:
            missing[class_name] = class_missing
    message = (
        "all expected attributes present" if not missing else "missing attributes"
    )
    return CheckpointResult(
        checkpoint_id="entity.attributes",
        category="entity",
        description="Expected agent attributes exist in class initializers.",
        passed=not missing,
        evidence={
            "expected": constraints.expected_attributes,
            "actual": analysis.attributes_by_class,
            "missing": missing,
        },
        message=message,
    )


def _interaction_method_checkpoint(
    analysis: ASTAnalysisResult,
    constraints: ModelConstraints,
    framework: FrameworkSpec,
) -> CheckpointResult:
    available = set(analysis.function_names)
    for methods in analysis.methods_by_class.values():
        available.update(methods)
    missing: list[str] = []
    fallback_present: dict[str, list[str]] = {}
    expected: list[str] = []
    for contract in constraints.expected_interaction_contracts:
        expected.append(contract.required_method_name)
        if _contract_is_present(
            analysis,
            constraints,
            framework,
            contract.edge_id,
            contract.required_method_name,
        ):
            continue
        missing.append(contract.required_method_name)
        present_fallbacks = [
            method_name
            for method_name in contract.fallback_method_names
            if _contract_is_present(
                analysis,
                constraints,
                framework,
                contract.edge_id,
                method_name,
            )
        ]
        if present_fallbacks:
            fallback_present[contract.edge_id] = present_fallbacks
    message = (
        "interaction methods present" if not missing else "missing interactions"
    )
    return CheckpointResult(
        checkpoint_id="interaction.methods",
        category="interaction",
        description="Expected interaction code entry points exist.",
        passed=not missing,
        evidence={
            "expected_required_methods": expected,
            "required_contracts": [
                contract.to_dict()
                for contract in constraints.expected_interaction_contracts
            ],
            "fallback_present": fallback_present,
            "actual": sorted(available),
            "missing": missing,
            "framework_prefix": framework.interaction_method_prefix,
        },
        message=message,
    )


def _contract_is_present(
    analysis: ASTAnalysisResult,
    constraints: ModelConstraints,
    framework: FrameworkSpec,
    edge_id: str,
    method_name: str,
) -> bool:
    if framework.naming_conventions.get("interaction_method_style") == "edge_function":
        return method_name in analysis.function_names
    source_class_name = _source_class_name(constraints, edge_id)
    return method_name in analysis.methods_by_class.get(source_class_name, [])


def _source_class_name(constraints: ModelConstraints, edge_id: str) -> str:
    for edge in constraints.expected_interaction_edges:
        if edge.edge_id == edge_id:
            return edge.source_class_name
    return ""


def _direction_checkpoint(
    code: str,
    constraints: ModelConstraints,
) -> CheckpointResult:
    missing: list[str] = []
    expected: list[str] = []
    for edge in constraints.expected_interaction_edges:
        source = f"{edge.source_agent}.{edge.source_port}"
        target = f"{edge.target_agent}.{edge.target_port}"
        expected.append(f"{source} -> {target}")
        if source not in code or target not in code:
            missing.append(edge.edge_id)
    return CheckpointResult(
        checkpoint_id="interaction.direction",
        category="interaction",
        description="Source-target direction is preserved in code metadata.",
        passed=not missing,
        evidence={"expected": expected, "missing_edge_ids": missing},
        message="directions preserved" if not missing else "direction metadata missing",
    )


def _filter_guard_checkpoint(
    report: ConsistencyReport,
    constraints: ModelConstraints,
) -> CheckpointResult:
    expected = list(constraints.expected_guard_requirements)
    missing = report.missing_filter_guards
    return CheckpointResult(
        checkpoint_id="condition.filter_guards",
        category="condition",
        description="Filter guard exists or is safely represented.",
        passed=not missing,
        evidence={"expected_guard_edges": expected, "missing": missing},
        message="filter guards present" if not missing else "missing filter guards",
    )


def _framework_violation_checkpoint(report: ConsistencyReport) -> CheckpointResult:
    return CheckpointResult(
        checkpoint_id="framework.no_violations",
        category="framework",
        description="Phase 4A report has no framework violations.",
        passed=not report.framework_violations,
        evidence={"framework_violations": report.framework_violations},
        message=(
            "no framework violations"
            if not report.framework_violations
            else "framework violations remain"
        ),
    )


def _message_api_checkpoint(
    analysis: ASTAnalysisResult,
    framework: FrameworkSpec,
) -> CheckpointResult:
    if not framework.message_api_name:
        passed = True
    else:
        passed = framework.message_api_name in analysis.function_calls
    return CheckpointResult(
        checkpoint_id="framework.message_api",
        category="framework",
        description="Configured message API is used when required.",
        passed=passed,
        evidence={
            "message_api_name": framework.message_api_name,
            "function_calls": analysis.function_calls,
        },
        message="message API present" if passed else "message API missing",
    )


def _parse_checkpoint(analysis: ASTAnalysisResult) -> CheckpointResult:
    return CheckpointResult(
        checkpoint_id="execution.parseable",
        category="execution",
        description="Code can be parsed by Python ast.parse.",
        passed=not analysis.parse_errors,
        evidence={"parse_errors": analysis.parse_errors},
        message="parseable" if not analysis.parse_errors else "parse errors found",
    )
