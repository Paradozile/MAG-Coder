"""Rewrite planning for deterministic AST/code refinement."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from magcoder.refinement.checker import ConsistencyReport
from magcoder.refinement.constraints import (
    ModelConstraints,
    constraints_with_interaction_contracts,
)
from magcoder.refinement.contracts import build_interaction_contracts_from_constraints
from magcoder.refinement.framework import FrameworkSpec


@dataclass(slots=True)
class RewriteAction:
    """One deterministic refinement action."""

    action_type: str
    target: str
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Render this rewrite action as plain Python containers."""

        return asdict(self)


@dataclass(slots=True)
class RewritePlan:
    """A deterministic set of rewrite actions."""

    actions: list[RewriteAction] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Render this rewrite plan as plain Python containers."""

        return {
            "actions": [action.to_dict() for action in self.actions],
            "warnings": self.warnings,
            "metadata": self.metadata,
        }

    def action_labels(self) -> list[str]:
        """Return compact labels for persistence and CLI output."""

        return [
            f"{action.action_type}:{action.target}" for action in self.actions
        ]


def build_rewrite_plan(
    report: ConsistencyReport,
    constraints: ModelConstraints,
    framework: FrameworkSpec,
) -> RewritePlan:
    """Build deterministic rewrite actions from analysis results."""

    if not constraints.expected_interaction_contracts:
        constraints = constraints_with_interaction_contracts(
            constraints,
            build_interaction_contracts_from_constraints(constraints, framework),
        )
    plan = RewritePlan(
        metadata={
            "mag_id": constraints.mag_id,
            "framework": framework.name,
        },
    )
    available_missing_methods = set(report.missing_interaction_methods)
    contracts_by_edge = {
        contract.edge_id: contract
        for contract in constraints.expected_interaction_contracts
    }

    for edge in constraints.expected_interaction_edges:
        contract = contracts_by_edge.get(edge.edge_id)
        required_method = (
            contract.required_method_name if contract is not None else edge.edge_id
        )
        if required_method in available_missing_methods:
            plan.actions.append(
                RewriteAction(
                    action_type="add_required_edge_method",
                    target=edge.edge_id,
                    details={
                        "method_name": required_method,
                        "source_class": edge.source_class_name,
                    },
                )
            )

        if contract is not None:
            for fallback_name in contract.fallback_method_names:
                plan.actions.append(
                    RewriteAction(
                        action_type="preserve_fallback_method",
                        target=edge.edge_id,
                        details={
                            "method_name": fallback_name,
                            "source_class": edge.source_class_name,
                        },
                    )
                )

        if _needs_framework_interaction_method(report, required_method):
            plan.actions.append(
                RewriteAction(
                    action_type="normalize_interaction_method_name",
                    target=edge.edge_id,
                    details={
                        "method_name": required_method,
                        "source_class": edge.source_class_name,
                        "target_class": edge.target_class_name,
                    },
                )
            )

        guard = constraints.expected_guard_requirements.get(edge.edge_id)
        if guard is not None:
            plan.actions.append(
                RewriteAction(
                    action_type="insert_filter_guard",
                    target=edge.edge_id,
                    details={
                        "filter_id": guard.filter_id,
                        "expressions": guard.expressions,
                    },
                )
            )

        if framework.message_api_name and _needs_message_api_call(
            report,
            framework.message_api_name,
        ):
            plan.actions.append(
                RewriteAction(
                    action_type="insert_message_api_call",
                    target=edge.edge_id,
                    details={
                        "message_api_name": framework.message_api_name,
                        "source_class": edge.source_class_name,
                    },
                )
            )

    return _deduplicate_plan(plan)


def _needs_framework_interaction_method(
    report: ConsistencyReport,
    method_name: str,
) -> bool:
    return any(method_name in violation for violation in report.framework_violations)


def _needs_message_api_call(
    report: ConsistencyReport,
    message_api_name: str,
) -> bool:
    return any(
        message_api_name in violation for violation in report.framework_violations
    )


def _deduplicate_plan(plan: RewritePlan) -> RewritePlan:
    seen: set[tuple[str, str, str]] = set()
    actions: list[RewriteAction] = []
    for action in plan.actions:
        method_name = str(action.details.get("method_name", ""))
        key = (action.action_type, action.target, method_name)
        if key in seen:
            continue
        seen.add(key)
        actions.append(action)
    plan.actions = actions
    return plan
