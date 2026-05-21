"""Core Multi-Agent Graph data model."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal

import yaml

PortDirection = Literal["input", "output"]


class MAGValidationError(ValueError):
    """Raised when a MAG model violates structural constraints."""


@dataclass(slots=True)
class Port:
    """An interaction endpoint exposed by an agent class."""

    name: str
    direction: PortDirection
    description: str | None = None

    def __post_init__(self) -> None:
        if self.direction not in ("input", "output"):
            msg = f"Port {self.name!r} has invalid direction {self.direction!r}."
            raise MAGValidationError(msg)


@dataclass(slots=True)
class AgentAttribute:
    """A named attribute available on an agent class."""

    name: str
    type: str | None = None
    description: str | None = None


@dataclass(slots=True)
class AgentClass:
    """A participating agent class in a MAG."""

    id: str
    ports: list[Port] = field(default_factory=list)
    attributes: list[AgentAttribute] = field(default_factory=list)
    nl_description: str | None = None


@dataclass(slots=True)
class AttributeFilter:
    """An attribute-based condition constraining an interaction edge."""

    id: str
    expressions: list[str] = field(default_factory=list)
    nl_description: str | None = None


@dataclass(slots=True)
class InteractionEdge:
    """A directed interaction from one agent port to another."""

    id: str
    src_agent: str
    src_port: str
    dst_agent: str
    dst_port: str
    filter_id: str | None = None
    nl_description: str | None = None


@dataclass(slots=True)
class MAGModel:
    """A minimal structured representation of MAG=(A,E,F)."""

    id: str
    agents: list[AgentClass] = field(default_factory=list)
    edges: list[InteractionEdge] = field(default_factory=list)
    filters: list[AttributeFilter] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        """Validate basic referential and port-direction constraints."""

        agent_ids = [agent.id for agent in self.agents]
        _ensure_unique(agent_ids, "agent ids")

        filter_ids = [attribute_filter.id for attribute_filter in self.filters]
        _ensure_unique(filter_ids, "filter ids")

        edge_ids = [edge.id for edge in self.edges]
        _ensure_unique(edge_ids, "edge ids")

        agents_by_id = {agent.id: agent for agent in self.agents}
        filters_by_id = {attribute_filter.id for attribute_filter in self.filters}

        ports_by_agent: dict[str, dict[str, Port]] = {}
        for agent in self.agents:
            port_names = [port.name for port in agent.ports]
            _ensure_unique(port_names, f"port names for agent {agent.id!r}")
            ports_by_agent[agent.id] = {port.name: port for port in agent.ports}

        for edge in self.edges:
            if edge.src_agent not in agents_by_id:
                msg = (
                    f"Edge {edge.id!r} references unknown src_agent "
                    f"{edge.src_agent!r}."
                )
                raise MAGValidationError(msg)
            if edge.dst_agent not in agents_by_id:
                msg = (
                    f"Edge {edge.id!r} references unknown dst_agent "
                    f"{edge.dst_agent!r}."
                )
                raise MAGValidationError(msg)

            src_ports = ports_by_agent[edge.src_agent]
            dst_ports = ports_by_agent[edge.dst_agent]

            if edge.src_port not in src_ports:
                msg = (
                    f"Edge {edge.id!r} references unknown src_port "
                    f"{edge.src_port!r} on agent {edge.src_agent!r}."
                )
                raise MAGValidationError(msg)
            if edge.dst_port not in dst_ports:
                msg = (
                    f"Edge {edge.id!r} references unknown dst_port "
                    f"{edge.dst_port!r} on agent {edge.dst_agent!r}."
                )
                raise MAGValidationError(msg)

            src_port = src_ports[edge.src_port]
            dst_port = dst_ports[edge.dst_port]
            if src_port.direction != "output":
                msg = (
                    f"Edge {edge.id!r} uses src_port {edge.src_port!r} with "
                    f"direction {src_port.direction!r}; expected 'output'."
                )
                raise MAGValidationError(msg)
            if dst_port.direction != "input":
                msg = (
                    f"Edge {edge.id!r} uses dst_port {edge.dst_port!r} with "
                    f"direction {dst_port.direction!r}; expected 'input'."
                )
                raise MAGValidationError(msg)

            if edge.filter_id is not None and edge.filter_id not in filters_by_id:
                msg = (
                    f"Edge {edge.id!r} references unknown filter_id "
                    f"{edge.filter_id!r}."
                )
                raise MAGValidationError(msg)

    def to_dict(self) -> dict[str, Any]:
        """Serialize this model to plain Python containers."""

        return asdict(self)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> MAGModel:
        """Build a MAG model from plain Python containers."""

        agents_data = data.get("agents", [])
        edges_data = data.get("edges", [])
        filters_data = data.get("filters", [])
        metadata_data = data.get("metadata", {})

        if not isinstance(agents_data, list):
            raise TypeError("MAG field 'agents' must be a list.")
        if not isinstance(edges_data, list):
            raise TypeError("MAG field 'edges' must be a list.")
        if not isinstance(filters_data, list):
            raise TypeError("MAG field 'filters' must be a list.")
        if not isinstance(metadata_data, dict):
            raise TypeError("MAG field 'metadata' must be a dict.")

        mag = cls(
            id=str(data["id"]),
            agents=[_agent_from_mapping(item) for item in agents_data],
            edges=[_edge_from_mapping(item) for item in edges_data],
            filters=[_filter_from_mapping(item) for item in filters_data],
            metadata=dict(metadata_data),
        )
        return mag


def load_mag_from_json(path: str | Path) -> MAGModel:
    """Load and validate a MAG model from a JSON file."""

    with Path(path).open(encoding="utf-8") as file:
        data = json.load(file)
    if not isinstance(data, Mapping):
        raise TypeError("MAG JSON root must be an object.")
    mag = MAGModel.from_dict(data)
    mag.validate()
    return mag


def save_mag_to_json(mag: MAGModel, path: str | Path) -> None:
    """Validate and save a MAG model to a JSON file."""

    mag.validate()
    with Path(path).open("w", encoding="utf-8") as file:
        json.dump(mag.to_dict(), file, indent=2)
        file.write("\n")


def load_mag_from_yaml(path: str | Path) -> MAGModel:
    """Load and validate a MAG model from a YAML file."""

    with Path(path).open(encoding="utf-8") as file:
        data = yaml.safe_load(file)
    if not isinstance(data, Mapping):
        raise TypeError("MAG YAML root must be an object.")
    mag = MAGModel.from_dict(data)
    mag.validate()
    return mag


def save_mag_to_yaml(mag: MAGModel, path: str | Path) -> None:
    """Validate and save a MAG model to a YAML file."""

    mag.validate()
    with Path(path).open("w", encoding="utf-8") as file:
        yaml.safe_dump(mag.to_dict(), file, sort_keys=False)


def _ensure_unique(values: list[str], label: str) -> None:
    seen: set[str] = set()
    duplicates: set[str] = set()
    for value in values:
        if value in seen:
            duplicates.add(value)
        seen.add(value)
    if duplicates:
        duplicate_list = ", ".join(sorted(repr(value) for value in duplicates))
        msg = f"Duplicate {label}: {duplicate_list}."
        raise MAGValidationError(msg)


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{label} must be an object.")
    return value


def _agent_from_mapping(value: Any) -> AgentClass:
    data = _mapping(value, "Agent")
    ports_data = data.get("ports", [])
    attributes_data = data.get("attributes", [])
    if not isinstance(ports_data, list):
        raise TypeError("Agent field 'ports' must be a list.")
    if not isinstance(attributes_data, list):
        raise TypeError("Agent field 'attributes' must be a list.")
    return AgentClass(
        id=str(data["id"]),
        ports=[_port_from_mapping(item) for item in ports_data],
        attributes=[
            _attribute_from_mapping(item) for item in attributes_data
        ],
        nl_description=_optional_str(data.get("nl_description")),
    )


def _port_from_mapping(value: Any) -> Port:
    data = _mapping(value, "Port")
    direction = data["direction"]
    if direction not in ("input", "output"):
        msg = f"Port {data['name']!r} has invalid direction {direction!r}."
        raise MAGValidationError(msg)
    return Port(
        name=str(data["name"]),
        direction=direction,
        description=_optional_str(data.get("description")),
    )


def _attribute_from_mapping(value: Any) -> AgentAttribute:
    data = _mapping(value, "Agent attribute")
    return AgentAttribute(
        name=str(data["name"]),
        type=_optional_str(data.get("type")),
        description=_optional_str(data.get("description")),
    )


def _filter_from_mapping(value: Any) -> AttributeFilter:
    data = _mapping(value, "Attribute filter")
    expressions = data.get("expressions", [])
    if not isinstance(expressions, list):
        raise TypeError("Attribute filter field 'expressions' must be a list.")
    return AttributeFilter(
        id=str(data["id"]),
        expressions=[str(expression) for expression in expressions],
        nl_description=_optional_str(data.get("nl_description")),
    )


def _edge_from_mapping(value: Any) -> InteractionEdge:
    data = _mapping(value, "Interaction edge")
    return InteractionEdge(
        id=str(data["id"]),
        src_agent=str(data["src_agent"]),
        src_port=str(data["src_port"]),
        dst_agent=str(data["dst_agent"]),
        dst_port=str(data["dst_port"]),
        filter_id=_optional_str(data.get("filter_id")),
        nl_description=_optional_str(data.get("nl_description")),
    )


def _optional_str(value: Any) -> str | None:
    if value is None:
        return None
    return str(value)
