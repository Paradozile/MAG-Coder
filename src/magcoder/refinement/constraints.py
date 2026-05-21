"""Model-derived constraints for refinement analysis."""

from __future__ import annotations

import keyword
import re
from dataclasses import asdict, dataclass, field
from typing import TYPE_CHECKING, Any

from magcoder.models import InteractionEdge, MAGModel

if TYPE_CHECKING:
    from magcoder.refinement.contracts import InteractionMethodContract


@dataclass(slots=True)
class InteractionConstraint:
    """Expected source-target structure for a MAG interaction edge."""

    edge_id: str
    source_agent: str
    source_class_name: str
    source_port: str
    target_agent: str
    target_class_name: str
    target_port: str
    filter_id: str | None
    direction: str


@dataclass(slots=True)
class GuardRequirement:
    """Expected guard condition for a filtered interaction edge."""

    edge_id: str
    filter_id: str
    expressions: list[str]


@dataclass(slots=True)
class ModelConstraints:
    """Constraints extracted from a MAG model for code consistency checks."""

    mag_id: str
    expected_agent_class_names: list[str]
    expected_attributes: dict[str, list[str]]
    expected_interaction_edges: list[InteractionConstraint]
    expected_filter_expressions: dict[str, list[str]]
    expected_source_target_direction: dict[str, str]
    expected_guard_requirements: dict[str, GuardRequirement]
    expected_interaction_contracts: list[InteractionMethodContract] = field(
        default_factory=list
    )
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Render constraints as plain Python containers."""

        return asdict(self)


def extract_model_constraints(mag: MAGModel) -> ModelConstraints:
    """Extract expected code-shape constraints from a validated MAG model."""

    mag.validate()
    class_names_by_agent = {
        agent.id: _to_class_name(agent.id) for agent in mag.agents
    }
    filters_by_id = {
        attribute_filter.id: list(attribute_filter.expressions)
        for attribute_filter in mag.filters
    }
    expected_attributes = {
        class_names_by_agent[agent.id]: [
            _to_snake_case(attribute.name) for attribute in agent.attributes
        ]
        for agent in mag.agents
    }
    edge_constraints = [
        _edge_constraint(edge, class_names_by_agent) for edge in mag.edges
    ]
    guard_requirements = {
        edge.edge_id: GuardRequirement(
            edge_id=edge.edge_id,
            filter_id=edge.filter_id,
            expressions=filters_by_id[edge.filter_id],
        )
        for edge in edge_constraints
        if edge.filter_id is not None
    }
    return ModelConstraints(
        mag_id=mag.id,
        expected_agent_class_names=list(class_names_by_agent.values()),
        expected_attributes=expected_attributes,
        expected_interaction_edges=edge_constraints,
        expected_filter_expressions=filters_by_id,
        expected_source_target_direction={
            edge.edge_id: edge.direction for edge in edge_constraints
        },
        expected_guard_requirements=guard_requirements,
        metadata={"source": "MAGModel"},
    )


def expected_edge_function_name(edge_id: str) -> str:
    """Return the Phase 3 fake-generator-style function name for an edge."""

    return f"interact_{_to_snake_case(edge_id)}"


def expected_target_method_name(prefix: str, target_agent: str) -> str:
    """Return a framework-style interaction method name for a target agent."""

    return f"{prefix}_{_to_snake_case(target_agent)}"


def constraints_with_interaction_contracts(
    constraints: ModelConstraints,
    contracts: list[InteractionMethodContract],
) -> ModelConstraints:
    """Return constraints carrying shared interaction method contracts."""

    constraints.expected_interaction_contracts = contracts
    return constraints


def _edge_constraint(
    edge: InteractionEdge,
    class_names_by_agent: dict[str, str],
) -> InteractionConstraint:
    return InteractionConstraint(
        edge_id=edge.id,
        source_agent=edge.src_agent,
        source_class_name=class_names_by_agent[edge.src_agent],
        source_port=edge.src_port,
        target_agent=edge.dst_agent,
        target_class_name=class_names_by_agent[edge.dst_agent],
        target_port=edge.dst_port,
        filter_id=edge.filter_id,
        direction=(
            f"{edge.src_agent}.{edge.src_port} -> "
            f"{edge.dst_agent}.{edge.dst_port}"
        ),
    )


def _to_class_name(value: str) -> str:
    words = re.findall(r"[A-Za-z0-9]+", value)
    if not words:
        return "GeneratedAgent"
    class_name = "".join(word[:1].upper() + word[1:] for word in words)
    if class_name[0].isdigit():
        return f"Agent{class_name}"
    if keyword.iskeyword(class_name):
        return f"{class_name}Agent"
    return class_name


def _to_snake_case(value: str) -> str:
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
