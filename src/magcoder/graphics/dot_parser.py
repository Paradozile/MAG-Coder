"""Parse standardized DOT + manifest graphical MAG artifacts."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from magcoder.graphics.manifest import load_graphical_mag_manifest
from magcoder.models import (
    AgentAttribute,
    AgentClass,
    AttributeFilter,
    InteractionEdge,
    MAGModel,
    Port,
)

_NODE_RE = re.compile(r'^\s*"(?P<id>(?:\\.|[^"\\])*)"\s+\[')
_EDGE_RE = re.compile(
    r'^\s*"(?P<src>(?:\\.|[^"\\])*)"\s*->\s*'
    r'"(?P<dst>(?:\\.|[^"\\])*)"\s+\[(?P<attrs>.*)\];\s*$',
)
_ATTR_RE = re.compile(r'(?P<key>[A-Za-z_][A-Za-z0-9_]*)="(?P<value>(?:\\.|[^"\\])*)"')


def load_mag_from_graphical_artifact(
    dot_path: str | Path,
    manifest_path: str | Path,
) -> MAGModel:
    """Recover a MAGModel from standardized DOT plus its manifest."""

    dot_text = Path(dot_path).read_text(encoding="utf-8")
    manifest = load_graphical_mag_manifest(manifest_path)
    dot_nodes, dot_edges = _parse_standard_dot(dot_text)
    _verify_manifest_matches_dot(
        dot_nodes=dot_nodes,
        dot_edges=dot_edges,
        manifest_nodes=manifest.nodes,
        manifest_edges=manifest.edges,
    )
    mag = MAGModel(
        id=manifest.mag_id,
        agents=[_agent_from_node(node) for node in manifest.nodes],
        edges=[_edge_from_manifest(edge) for edge in manifest.edges],
        filters=[
            _filter_from_manifest(attribute_filter)
            for attribute_filter in manifest.filters
        ],
        metadata={
            "roundtrip_source": "graphical_mag_artifact",
            "dot_path": str(dot_path),
            "manifest_path": str(manifest_path),
            **manifest.metadata,
        },
    )
    mag.validate()
    return mag


def _parse_standard_dot(dot_text: str) -> tuple[set[str], list[dict[str, str]]]:
    nodes: set[str] = set()
    edges: list[dict[str, str]] = []
    for line in dot_text.splitlines():
        edge_match = _EDGE_RE.match(line)
        if edge_match is not None:
            attrs = _parse_attributes(edge_match.group("attrs"))
            edges.append(
                {
                    "src": _dot_unescape(edge_match.group("src")),
                    "dst": _dot_unescape(edge_match.group("dst")),
                    **attrs,
                }
            )
            continue

        if "->" in line:
            continue
        node_match = _NODE_RE.match(line)
        if node_match is not None:
            nodes.add(_dot_unescape(node_match.group("id")))
    return nodes, edges


def _verify_manifest_matches_dot(
    *,
    dot_nodes: set[str],
    dot_edges: list[dict[str, str]],
    manifest_nodes: list[dict[str, Any]],
    manifest_edges: list[dict[str, Any]],
) -> None:
    manifest_node_ids = {str(node["id"]) for node in manifest_nodes}
    if dot_nodes != manifest_node_ids:
        msg = (
            "DOT node ids do not match manifest nodes: "
            f"dot={sorted(dot_nodes)}, manifest={sorted(manifest_node_ids)}."
        )
        raise ValueError(msg)

    dot_edge_index = {
        (
            edge.get("edge_id", ""),
            edge["src"],
            edge["dst"],
        )
        for edge in dot_edges
    }
    manifest_edge_index = {
        (
            str(edge["id"]),
            str(edge["src_agent"]),
            str(edge["dst_agent"]),
        )
        for edge in manifest_edges
    }
    if dot_edge_index != manifest_edge_index:
        msg = (
            "DOT edges do not match manifest edges: "
            f"dot={sorted(dot_edge_index)}, manifest={sorted(manifest_edge_index)}."
        )
        raise ValueError(msg)


def _agent_from_node(node: dict[str, Any]) -> AgentClass:
    return AgentClass(
        id=str(node["id"]),
        ports=[_port_from_manifest(port) for port in _list_value(node, "ports")],
        attributes=[
            _attribute_from_manifest(attribute)
            for attribute in _list_value(node, "attributes")
        ],
        nl_description=_optional_str(node.get("nl_description")),
    )


def _edge_from_manifest(edge: dict[str, Any]) -> InteractionEdge:
    return InteractionEdge(
        id=str(edge["id"]),
        src_agent=str(edge["src_agent"]),
        src_port=str(edge["src_port"]),
        dst_agent=str(edge["dst_agent"]),
        dst_port=str(edge["dst_port"]),
        filter_id=_optional_str(edge.get("filter_id")),
        nl_description=_optional_str(edge.get("nl_description")),
    )


def _filter_from_manifest(attribute_filter: dict[str, Any]) -> AttributeFilter:
    expressions = _list_value(attribute_filter, "expressions")
    return AttributeFilter(
        id=str(attribute_filter["id"]),
        expressions=[str(expression) for expression in expressions],
        nl_description=_optional_str(attribute_filter.get("nl_description")),
    )


def _port_from_manifest(port: Any) -> Port:
    if not isinstance(port, dict):
        raise TypeError("Manifest port entries must be objects.")
    return Port(
        name=str(port["name"]),
        direction=port["direction"],
        description=_optional_str(port.get("description")),
    )


def _attribute_from_manifest(attribute: Any) -> AgentAttribute:
    if not isinstance(attribute, dict):
        raise TypeError("Manifest attribute entries must be objects.")
    return AgentAttribute(
        name=str(attribute["name"]),
        type=_optional_str(attribute.get("type")),
        description=_optional_str(attribute.get("description")),
    )


def _parse_attributes(attributes_text: str) -> dict[str, str]:
    return {
        match.group("key"): _dot_unescape(match.group("value"))
        for match in _ATTR_RE.finditer(attributes_text)
    }


def _dot_unescape(value: str) -> str:
    return (
        value.replace("\\n", "\n")
        .replace('\\"', '"')
        .replace("\\\\", "\\")
    )


def _list_value(data: dict[str, Any], key: str) -> list[Any]:
    value = data.get(key, [])
    if not isinstance(value, list):
        raise TypeError(f"Manifest field {key!r} must be a list.")
    return value


def _optional_str(value: Any) -> str | None:
    if value is None:
        return None
    return str(value)
