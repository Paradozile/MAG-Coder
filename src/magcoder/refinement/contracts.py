"""Interaction method contracts shared by prompting, checking, and refinement."""

from __future__ import annotations

import keyword
import re
from dataclasses import asdict, dataclass, field
from typing import Any

from magcoder.models import MAGModel
from magcoder.refinement.constraints import ModelConstraints
from magcoder.refinement.framework import FrameworkSpec


@dataclass(slots=True)
class InteractionMethodContract:
    """Required method contract for one MAG interaction edge."""

    edge_id: str
    src_agent: str
    dst_agent: str
    src_port: str
    dst_port: str
    required_method_name: str
    fallback_method_names: list[str]
    filter_id: str | None
    filter_expressions: list[str]
    message_api_name: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Render this contract as plain Python containers."""

        return asdict(self)


def build_interaction_contracts(
    mag: MAGModel,
    framework: FrameworkSpec,
) -> list[InteractionMethodContract]:
    """Build shared interaction method contracts from MAG + framework."""

    mag.validate()
    filters_by_id = {
        attribute_filter.id: list(attribute_filter.expressions)
        for attribute_filter in mag.filters
    }
    strategy = _contract_strategy(mag=mag, framework=framework)
    contracts: list[InteractionMethodContract] = []
    for edge in mag.edges:
        edge_method = edge_method_name(edge.id)
        target_method = target_method_name(
            framework.interaction_method_prefix,
            edge.dst_agent,
        )
        required_method = (
            target_method if strategy == "target_agent" else edge_method
        )
        fallback_names = [
            method_name
            for method_name in [target_method, edge_method]
            if method_name != required_method
        ]
        contracts.append(
            InteractionMethodContract(
                edge_id=edge.id,
                src_agent=edge.src_agent,
                dst_agent=edge.dst_agent,
                src_port=edge.src_port,
                dst_port=edge.dst_port,
                required_method_name=required_method,
                fallback_method_names=fallback_names,
                filter_id=edge.filter_id,
                filter_expressions=(
                    [] if edge.filter_id is None else filters_by_id[edge.filter_id]
                ),
                message_api_name=framework.message_api_name,
                metadata={
                    "strategy": strategy,
                    "edge_method_name": edge_method,
                    "target_method_name": target_method,
                },
            )
        )
    return contracts


def build_interaction_contracts_from_constraints(
    constraints: ModelConstraints,
    framework: FrameworkSpec,
) -> list[InteractionMethodContract]:
    """Build contracts when only extracted model constraints are available."""

    strategy = _constraint_contract_strategy(
        constraints=constraints,
        framework=framework,
    )
    guards_by_edge = constraints.expected_guard_requirements
    contracts: list[InteractionMethodContract] = []
    for edge in constraints.expected_interaction_edges:
        edge_method = edge_method_name(edge.edge_id)
        target_method = target_method_name(
            framework.interaction_method_prefix,
            edge.target_agent,
        )
        required_method = (
            target_method if strategy == "target_agent" else edge_method
        )
        fallback_names = [
            method_name
            for method_name in [target_method, edge_method]
            if method_name != required_method
        ]
        guard = guards_by_edge.get(edge.edge_id)
        contracts.append(
            InteractionMethodContract(
                edge_id=edge.edge_id,
                src_agent=edge.source_agent,
                dst_agent=edge.target_agent,
                src_port=edge.source_port,
                dst_port=edge.target_port,
                required_method_name=required_method,
                fallback_method_names=fallback_names,
                filter_id=None if guard is None else guard.filter_id,
                filter_expressions=[] if guard is None else list(guard.expressions),
                message_api_name=framework.message_api_name,
                metadata={
                    "strategy": strategy,
                    "edge_method_name": edge_method,
                    "target_method_name": target_method,
                    "source": "ModelConstraints",
                },
            )
        )
    return contracts


def edge_method_name(edge_id: str) -> str:
    """Return edge-id strategy method name."""

    return f"interact_{to_snake_case(edge_id)}"


def target_method_name(prefix: str, target_agent: str) -> str:
    """Return target-agent strategy method name."""

    return f"{prefix}_{to_snake_case(target_agent)}"


def to_snake_case(value: str) -> str:
    """Convert an identifier-ish value to a safe snake_case name."""

    value = re.sub(r"(.)([A-Z][a-z]+)", r"\1_\2", value)
    value = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", value)
    value = re.sub(r"[^A-Za-z0-9_]+", "_", value).strip("_").lower()
    if not value:
        value = "value"
    if value[0].isdigit():
        value = f"value_{value}"
    if keyword.iskeyword(value):
        value = f"{value}_"
    return value


def _contract_strategy(*, mag: MAGModel, framework: FrameworkSpec) -> str:
    configured = (
        framework.naming_conventions.get("interaction_contract_strategy")
        or framework.naming_conventions.get("interaction_method_contract_strategy")
        or mag.metadata.get("interaction_contract_strategy")
        or mag.metadata.get("interaction_method_strategy")
    )
    if configured in {"target_agent", "edge_id"}:
        return str(configured)
    if (
        framework.naming_conventions.get("interaction_method_style") == "target_method"
        and framework.naming_conventions.get("allow_target_method_contract", False)
    ):
        # Keep a compatibility escape hatch for frameworks that explicitly want
        # target-agent methods and opt out of edge contracts.
        return "target_agent"
    return "edge_id"


def _constraint_contract_strategy(
    *,
    constraints: ModelConstraints,
    framework: FrameworkSpec,
) -> str:
    configured = (
        framework.naming_conventions.get("interaction_contract_strategy")
        or framework.naming_conventions.get("interaction_method_contract_strategy")
        or constraints.metadata.get("interaction_contract_strategy")
        or constraints.metadata.get("interaction_method_strategy")
    )
    if configured in {"target_agent", "edge_id"}:
        return str(configured)
    if (
        framework.naming_conventions.get("interaction_method_style") == "target_method"
        and framework.naming_conventions.get("allow_target_method_contract", False)
    ):
        return "target_agent"
    return "edge_id"
