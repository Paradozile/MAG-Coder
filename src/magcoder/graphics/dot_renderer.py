"""Render MAG models to standardized Graphviz DOT."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from magcoder.graphics.visual_style import (
    EDGE_ATTRIBUTES,
    GRAPH_ATTRIBUTES,
    NODE_ATTRIBUTES,
)
from magcoder.models import AgentClass, InteractionEdge, MAGModel


def render_mag_to_dot(mag: MAGModel) -> str:
    """Render a MAGModel as standardized DOT graph text."""

    mag.validate()
    graph_attributes = {**GRAPH_ATTRIBUTES, "label": f"MAG: {mag.id}"}
    lines = [
        f"digraph {_dot_identifier(mag.id)} {{",
        f"  graph [{_format_attributes(graph_attributes)}];",
        f"  node [{_format_attributes(NODE_ATTRIBUTES)}];",
        f"  edge [{_format_attributes(EDGE_ATTRIBUTES)}];",
        "",
    ]
    lines.extend(_render_agent_node(agent) for agent in mag.agents)
    lines.append("")
    lines.extend(_render_interaction_edge(edge) for edge in mag.edges)
    lines.append("}")
    return "\n".join(lines) + "\n"


def save_mag_dot(mag: MAGModel, path: str | Path) -> None:
    """Save a MAGModel as DOT graph text."""

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(render_mag_to_dot(mag), encoding="utf-8")


def _render_agent_node(agent: AgentClass) -> str:
    input_ports = [
        port.name for port in agent.ports if port.direction == "input"
    ]
    output_ports = [
        port.name for port in agent.ports if port.direction == "output"
    ]
    attributes = [attribute.name for attribute in agent.attributes]
    label_parts = [
        agent.id,
        f"inputs: {_join_or_none(input_ports)}",
        f"outputs: {_join_or_none(output_ports)}",
        f"attributes: {_join_or_none(attributes)}",
    ]
    node_attributes = {
        "label": "\n".join(label_parts),
        "agent_id": agent.id,
        "ports": _port_hint(agent),
        "tooltip": agent.nl_description or "",
    }
    return f'  {_dot_quote(agent.id)} [{_format_attributes(node_attributes)}];'


def _render_interaction_edge(edge: InteractionEdge) -> str:
    label_parts = [
        edge.id,
        f"{edge.src_port} -> {edge.dst_port}",
    ]
    if edge.filter_id is not None:
        label_parts.append(f"filter: {edge.filter_id}")
    edge_attributes = {
        "label": "\n".join(label_parts),
        "edge_id": edge.id,
        "src_port": edge.src_port,
        "dst_port": edge.dst_port,
        "filter_id": edge.filter_id or "",
        "tooltip": edge.nl_description or "",
    }
    return (
        f"  {_dot_quote(edge.src_agent)} -> {_dot_quote(edge.dst_agent)} "
        f"[{_format_attributes(edge_attributes)}];"
    )


def _format_attributes(attributes: dict[str, Any]) -> str:
    return ", ".join(
        f"{key}={_dot_quote(str(value))}" for key, value in attributes.items()
    )


def _dot_identifier(value: str) -> str:
    cleaned = "".join(char if char.isalnum() else "_" for char in value)
    if not cleaned:
        return "MAG"
    if cleaned[0].isdigit():
        return f"mag_{cleaned}"
    return cleaned


def _dot_quote(value: str) -> str:
    escaped = (
        value.replace("\\", "\\\\")
        .replace('"', '\\"')
        .replace("\n", "\\n")
    )
    return f'"{escaped}"'


def _join_or_none(values: list[str]) -> str:
    if not values:
        return "none"
    return ", ".join(values)


def _port_hint(agent: AgentClass) -> str:
    return "; ".join(f"{port.name}:{port.direction}" for port in agent.ports)
