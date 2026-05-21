"""Consistency checks for generated code before deterministic refinement."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from magcoder.refinement.ast_analyzer import ASTAnalysisResult
from magcoder.refinement.constraints import (
    ModelConstraints,
    constraints_with_interaction_contracts,
    expected_edge_function_name,
)
from magcoder.refinement.contracts import build_interaction_contracts_from_constraints
from magcoder.refinement.framework import FrameworkSpec


@dataclass(slots=True)
class ConsistencyReport:
    """Consistency report for generated code against model and framework inputs."""

    missing_agent_classes: list[str] = field(default_factory=list)
    missing_attributes: dict[str, list[str]] = field(default_factory=dict)
    missing_interaction_methods: list[str] = field(default_factory=list)
    missing_filter_guards: list[str] = field(default_factory=list)
    framework_violations: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    passed: bool = False

    def to_dict(self) -> dict[str, Any]:
        """Render the report as plain Python containers."""

        return asdict(self)


def check_consistency(
    analysis: ASTAnalysisResult,
    constraints: ModelConstraints,
    framework: FrameworkSpec,
) -> ConsistencyReport:
    """Check AST analysis against model-derived and framework constraints."""

    constraints = _ensure_interaction_contracts(constraints, framework)
    report = ConsistencyReport()
    report.warnings.extend(analysis.parse_errors)

    report.missing_agent_classes = [
        class_name
        for class_name in constraints.expected_agent_class_names
        if class_name not in analysis.class_names
    ]

    for class_name, expected_attributes in constraints.expected_attributes.items():
        actual_attributes = set(analysis.attributes_by_class.get(class_name, []))
        missing = [
            attribute_name
            for attribute_name in expected_attributes
            if attribute_name not in actual_attributes
        ]
        if missing:
            report.missing_attributes[class_name] = missing

    report.missing_interaction_methods = _missing_interaction_methods(
        analysis,
        constraints,
        framework,
    )
    report.missing_filter_guards = _missing_filter_guards(analysis, constraints)
    report.framework_violations = _framework_violations(
        analysis,
        constraints,
        framework,
    )
    report.warnings.extend(
        _fallback_method_warnings(analysis, constraints, framework)
    )
    report.passed = not (
        analysis.parse_errors
        or report.missing_agent_classes
        or report.missing_attributes
        or report.missing_interaction_methods
        or report.missing_filter_guards
        or report.framework_violations
    )
    return report


def _ensure_interaction_contracts(
    constraints: ModelConstraints,
    framework: FrameworkSpec,
) -> ModelConstraints:
    if constraints.expected_interaction_contracts:
        return constraints
    return constraints_with_interaction_contracts(
        constraints,
        build_interaction_contracts_from_constraints(constraints, framework),
    )


def save_consistency_report(report: ConsistencyReport, path: str | Path) -> None:
    """Save a consistency report as JSON."""

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file:
        json.dump(report.to_dict(), file, indent=2, sort_keys=True)
        file.write("\n")


def _missing_interaction_methods(
    analysis: ASTAnalysisResult,
    constraints: ModelConstraints,
    framework: FrameworkSpec,
) -> list[str]:
    missing: list[str] = []
    for contract in constraints.expected_interaction_contracts:
        if not _contract_is_present(
            analysis,
            constraints,
            framework,
            contract.edge_id,
            contract.required_method_name,
        ):
            missing.append(contract.required_method_name)
    return missing


def _missing_filter_guards(
    analysis: ASTAnalysisResult,
    constraints: ModelConstraints,
) -> list[str]:
    missing: list[str] = []
    conditions = " ".join(analysis.if_conditions)
    for edge_id, requirement in constraints.expected_guard_requirements.items():
        filter_name_fragment = requirement.filter_id.replace("-", "_")
        expression_present = any(
            expression in conditions for expression in requirement.expressions
        )
        filter_present = filter_name_fragment in conditions
        if not expression_present and not filter_present:
            missing.append(edge_id)
    return missing


def _framework_violations(
    analysis: ASTAnalysisResult,
    constraints: ModelConstraints,
    framework: FrameworkSpec,
) -> list[str]:
    violations: list[str] = []
    violations.extend(_required_method_violations(analysis, constraints, framework))
    violations.extend(_base_class_violations(analysis, constraints, framework))
    violations.extend(_interaction_naming_violations(analysis, constraints, framework))

    violations.extend(_message_api_violations(analysis, constraints, framework))

    if framework.state_access_style != "attribute":
        violations.append(
            f"state access style {framework.state_access_style!r} is not supported",
        )
    return violations


def _required_method_violations(
    analysis: ASTAnalysisResult,
    constraints: ModelConstraints,
    framework: FrameworkSpec,
) -> list[str]:
    violations: list[str] = []
    for class_name in constraints.expected_agent_class_names:
        methods = set(analysis.methods_by_class.get(class_name, []))
        for method_name in framework.required_agent_methods:
            if method_name not in methods:
                violations.append(
                    f"class {class_name!r} is missing method {method_name!r}",
                )
    return violations


def _base_class_violations(
    analysis: ASTAnalysisResult,
    constraints: ModelConstraints,
    framework: FrameworkSpec,
) -> list[str]:
    if framework.required_base_class is None:
        return []
    violations: list[str] = []
    for class_name in constraints.expected_agent_class_names:
        base_classes = analysis.base_classes_by_class.get(class_name, [])
        if framework.required_base_class not in base_classes:
            violations.append(
                f"class {class_name!r} does not inherit from "
                f"{framework.required_base_class!r}",
            )
    return violations


def _interaction_naming_violations(
    analysis: ASTAnalysisResult,
    constraints: ModelConstraints,
    framework: FrameworkSpec,
) -> list[str]:
    fallback_pass_allowed = bool(
        framework.naming_conventions.get(
            "allow_fallback_interaction_method_pass",
            False,
        )
    )
    violations: list[str] = []
    for contract in constraints.expected_interaction_contracts:
        if _contract_is_present(
            analysis,
            constraints,
            framework,
            contract.edge_id,
            contract.required_method_name,
        ):
            continue
        fallback_present = [
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
        if fallback_pass_allowed and fallback_present:
            continue
        violations.append(
            f"class for edge {contract.edge_id!r} is missing required "
            f"interaction method {contract.required_method_name!r}",
        )
    if constraints.expected_interaction_contracts:
        return violations

    style = framework.naming_conventions.get(
        "interaction_method_style",
        "target_method",
    )
    if style == "edge_function":
        available_functions = set(analysis.function_names)
        return [
            "missing interaction function "
            f"{expected_edge_function_name(edge.edge_id)!r}"
            for edge in constraints.expected_interaction_edges
            if expected_edge_function_name(edge.edge_id) not in available_functions
        ]

    return violations


def _message_api_violations(
    analysis: ASTAnalysisResult,
    constraints: ModelConstraints,
    framework: FrameworkSpec,
) -> list[str]:
    if not framework.message_api_name or not constraints.expected_interaction_edges:
        return []
    if not constraints.expected_interaction_contracts:
        if framework.message_api_name not in analysis.function_calls:
            return [f"message API {framework.message_api_name!r} is not called"]
        return []

    violations: list[str] = []
    for contract in constraints.expected_interaction_contracts:
        source_class = _source_class_name(constraints, contract.edge_id)
        method_name = contract.required_method_name
        if not _contract_is_present(
            analysis,
            constraints,
            framework,
            contract.edge_id,
            method_name,
        ):
            violations.append(
                f"class {source_class!r} method {method_name!r} for edge "
                f"{contract.edge_id!r} does not exist; cannot verify message API "
                f"{framework.message_api_name!r}"
            )
            continue
        calls = _contract_method_calls(
            analysis,
            framework,
            source_class,
            method_name,
        )
        if framework.message_api_name not in calls:
            violations.append(
                f"class {source_class!r} method {method_name!r} for edge "
                f"{contract.edge_id!r} is missing message API "
                f"{framework.message_api_name!r}"
            )
    return violations


def _available_method_names(analysis: ASTAnalysisResult) -> set[str]:
    available_names = set(analysis.function_names)
    for methods in analysis.methods_by_class.values():
        available_names.update(methods)
    return available_names


def _fallback_method_warnings(
    analysis: ASTAnalysisResult,
    constraints: ModelConstraints,
    framework: FrameworkSpec,
) -> list[str]:
    if framework.naming_conventions.get(
        "allow_fallback_interaction_method_pass",
        False,
    ):
        return []
    warnings: list[str] = []
    for contract in constraints.expected_interaction_contracts:
        if _contract_is_present(
            analysis,
            constraints,
            framework,
            contract.edge_id,
            contract.required_method_name,
        ):
            continue
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
            warnings.append(
                f"Edge {contract.edge_id!r} has fallback interaction method(s) "
                f"{present_fallbacks!r}, but requires "
                f"{contract.required_method_name!r}."
            )
    return warnings


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


def _contract_method_calls(
    analysis: ASTAnalysisResult,
    framework: FrameworkSpec,
    source_class: str,
    method_name: str,
) -> list[str]:
    if framework.naming_conventions.get("interaction_method_style") == "edge_function":
        return analysis.function_calls_by_function.get(method_name, [])
    return analysis.function_calls_by_class_method.get(source_class, {}).get(
        method_name,
        [],
    )
