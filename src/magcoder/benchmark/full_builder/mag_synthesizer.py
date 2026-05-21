"""Synthesize MAGModel instances from objectified benchmark tasks."""

from __future__ import annotations

from typing import Any

from magcoder.benchmark.full_builder.objectifier import ObjectifiedTask
from magcoder.models import (
    AgentAttribute,
    AgentClass,
    AttributeFilter,
    InteractionEdge,
    MAGModel,
    Port,
)


def synthesize_mag_from_objectified_task(task: ObjectifiedTask) -> MAGModel:
    """Convert an ObjectifiedTask into a validated MAGModel."""

    agents = [_agent_from_spec(agent_spec) for agent_spec in task.agent_classes]
    filters = [
        AttributeFilter(
            id=str(filter_spec["id"]),
            expressions=[str(item) for item in filter_spec.get("expressions", [])],
            nl_description=_optional_str(filter_spec.get("nl_description")),
        )
        for filter_spec in task.filters
    ]
    edges = [
        InteractionEdge(
            id=str(edge_spec["id"]),
            src_agent=str(edge_spec["src_agent"]),
            src_port=str(edge_spec["src_port"]),
            dst_agent=str(edge_spec["dst_agent"]),
            dst_port=str(edge_spec["dst_port"]),
            filter_id=_optional_str(edge_spec.get("filter_id")),
            nl_description=_optional_str(edge_spec.get("nl_description")),
        )
        for edge_spec in task.interaction_edges
    ]
    mag = MAGModel(
        id=task.task_id,
        agents=agents,
        edges=edges,
        filters=filters,
        metadata={
            "source_task_id": task.source_task_id,
            "objectified_task_id": task.task_id,
            "class_interface_spec": task.class_interface_spec,
            "expected_behavior": task.expected_behavior,
            "classification": task.classification,
            "required_method_names": [
                str(edge.get("required_method_name"))
                for edge in task.interaction_edges
                if edge.get("required_method_name") is not None
            ],
            "builder": "full_mag_bench_v1",
        },
    )
    mag.validate()
    return mag


def _agent_from_spec(agent_spec: dict[str, Any]) -> AgentClass:
    ports_spec = agent_spec.get("ports", {})
    if not isinstance(ports_spec, dict):
        raise TypeError("Objectified agent ports must be an object.")
    input_ports = [str(name) for name in ports_spec.get("input", [])]
    output_ports = [str(name) for name in ports_spec.get("output", [])]
    attributes = agent_spec.get("attributes", [])
    if not isinstance(attributes, list):
        raise TypeError("Objectified agent attributes must be a list.")
    return AgentClass(
        id=str(agent_spec["id"]),
        ports=[
            *[
                Port(
                    name=name,
                    direction="input",
                    description=f"Input port {name}.",
                )
                for name in input_ports
            ],
            *[
                Port(
                    name=name,
                    direction="output",
                    description=f"Output port {name}.",
                )
                for name in output_ports
            ],
        ],
        attributes=[
            AgentAttribute(
                name=str(attribute["name"]),
                type=_optional_str(attribute.get("type")),
                description=_optional_str(attribute.get("description")),
            )
            for attribute in attributes
        ],
        nl_description=_optional_str(agent_spec.get("nl_description")),
    )


def _optional_str(value: Any) -> str | None:
    if value is None:
        return None
    return str(value)
