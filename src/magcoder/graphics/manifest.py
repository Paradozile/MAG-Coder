"""Graphical MAG manifest generation and storage."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from magcoder.models import MAGModel


@dataclass(slots=True)
class GraphicalMAGManifest:
    """Manifest linking graphical concrete syntax to MAG abstract syntax."""

    mag_id: str
    nodes: list[dict[str, Any]]
    edges: list[dict[str, Any]]
    filters: list[dict[str, Any]]
    layout: dict[str, Any] = field(default_factory=dict)
    nl_descriptions: dict[str, dict[str, str | None]] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Render the manifest as JSON-serializable containers."""

        return asdict(self)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> GraphicalMAGManifest:
        """Build a graphical MAG manifest from parsed JSON data."""

        nodes = _list_of_mappings(data.get("nodes", []), "nodes")
        edges = _list_of_mappings(data.get("edges", []), "edges")
        filters = _list_of_mappings(data.get("filters", []), "filters")
        layout = data.get("layout", {})
        nl_descriptions = data.get("nl_descriptions", {})
        metadata = data.get("metadata", {})

        if not isinstance(layout, dict):
            raise TypeError("Manifest field 'layout' must be a dict.")
        if not isinstance(nl_descriptions, dict):
            raise TypeError("Manifest field 'nl_descriptions' must be a dict.")
        if not isinstance(metadata, dict):
            raise TypeError("Manifest field 'metadata' must be a dict.")

        return cls(
            mag_id=str(data["mag_id"]),
            nodes=[dict(node) for node in nodes],
            edges=[dict(edge) for edge in edges],
            filters=[dict(attribute_filter) for attribute_filter in filters],
            layout=dict(layout),
            nl_descriptions=_nested_optional_str_dict(nl_descriptions),
            metadata=dict(metadata),
        )


def build_graphical_mag_manifest(
    mag: MAGModel,
    *,
    layout: dict[str, Any] | None = None,
    metadata: dict[str, Any] | None = None,
) -> GraphicalMAGManifest:
    """Build a manifest that can recover MAGModel semantics after rendering."""

    mag.validate()
    return GraphicalMAGManifest(
        mag_id=mag.id,
        nodes=[
            {
                "id": agent.id,
                "ports": [
                    {
                        "name": port.name,
                        "direction": port.direction,
                        "description": port.description,
                    }
                    for port in agent.ports
                ],
                "attributes": [
                    {
                        "name": attribute.name,
                        "type": attribute.type,
                        "description": attribute.description,
                    }
                    for attribute in agent.attributes
                ],
                "nl_description": agent.nl_description,
            }
            for agent in mag.agents
        ],
        edges=[
            {
                "id": edge.id,
                "src_agent": edge.src_agent,
                "src_port": edge.src_port,
                "dst_agent": edge.dst_agent,
                "dst_port": edge.dst_port,
                "filter_id": edge.filter_id,
                "nl_description": edge.nl_description,
            }
            for edge in mag.edges
        ],
        filters=[
            {
                "id": attribute_filter.id,
                "expressions": list(attribute_filter.expressions),
                "nl_description": attribute_filter.nl_description,
            }
            for attribute_filter in mag.filters
        ],
        layout=layout or {"engine": "dot", "rankdir": "LR"},
        nl_descriptions={
            "agents": {agent.id: agent.nl_description for agent in mag.agents},
            "edges": {edge.id: edge.nl_description for edge in mag.edges},
            "filters": {
                attribute_filter.id: attribute_filter.nl_description
                for attribute_filter in mag.filters
            },
        },
        metadata={
            "manifest_version": 1,
            "source": "MAGModel",
            **(metadata or {}),
        },
    )


def save_graphical_mag_manifest(
    manifest: GraphicalMAGManifest,
    path: str | Path,
) -> None:
    """Save a graphical MAG manifest as JSON."""

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file:
        json.dump(manifest.to_dict(), file, indent=2, sort_keys=True)
        file.write("\n")


def load_graphical_mag_manifest(path: str | Path) -> GraphicalMAGManifest:
    """Load a graphical MAG manifest from JSON."""

    with Path(path).open(encoding="utf-8") as file:
        data = json.load(file)
    if not isinstance(data, Mapping):
        raise TypeError("Manifest JSON root must be an object.")
    return GraphicalMAGManifest.from_dict(data)


def _list_of_mappings(value: Any, label: str) -> list[Mapping[str, Any]]:
    if not isinstance(value, list):
        raise TypeError(f"Manifest field {label!r} must be a list.")
    if not all(isinstance(item, Mapping) for item in value):
        raise TypeError(f"Manifest field {label!r} must contain objects.")
    return value


def _nested_optional_str_dict(
    value: dict[Any, Any],
) -> dict[str, dict[str, str | None]]:
    result: dict[str, dict[str, str | None]] = {}
    for outer_key, inner_value in value.items():
        if not isinstance(inner_value, dict):
            raise TypeError("Manifest field 'nl_descriptions' must contain dicts.")
        result[str(outer_key)] = {
            str(inner_key): None if item is None else str(item)
            for inner_key, item in inner_value.items()
        }
    return result
