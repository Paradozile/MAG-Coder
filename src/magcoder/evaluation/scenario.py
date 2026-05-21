"""Scenario specifications for evaluation runs."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass(slots=True)
class ScenarioSpec:
    """A representative ABMS scenario evaluation specification."""

    scenario_id: str
    name: str
    description: str
    mag_path: str
    framework_path: str
    expected_agents: list[str]
    expected_edges: list[dict[str, Any]]
    expected_filters: list[str]
    mandatory_checkpoints: list[str]
    graphical_dot_path: str | None = None
    graphical_manifest_path: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Render this scenario spec as plain Python containers."""

        return asdict(self)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> ScenarioSpec:
        """Build a scenario spec from parsed YAML data."""

        expected_agents = _list_of_strings(data.get("expected_agents", []))
        expected_edges = _list_of_dicts(data.get("expected_edges", []))
        expected_filters = _list_of_strings(data.get("expected_filters", []))
        mandatory_checkpoints = _list_of_strings(
            data.get("mandatory_checkpoints", []),
        )
        metadata = data.get("metadata", {})
        if not isinstance(metadata, dict):
            raise TypeError("Scenario field 'metadata' must be a dict.")

        return cls(
            scenario_id=str(data["scenario_id"]),
            name=str(data["name"]),
            description=str(data["description"]),
            mag_path=str(data["mag_path"]),
            graphical_dot_path=_optional_str(data.get("graphical_dot_path")),
            graphical_manifest_path=_optional_str(
                data.get("graphical_manifest_path"),
            ),
            framework_path=str(data["framework_path"]),
            expected_agents=expected_agents,
            expected_edges=expected_edges,
            expected_filters=expected_filters,
            mandatory_checkpoints=mandatory_checkpoints,
            metadata=dict(metadata),
        )


def load_scenario_spec(path: str | Path) -> ScenarioSpec:
    """Load a scenario spec from YAML."""

    with Path(path).open(encoding="utf-8") as file:
        data = yaml.safe_load(file)
    if not isinstance(data, Mapping):
        raise TypeError("Scenario YAML root must be an object.")
    return ScenarioSpec.from_dict(data)


def _list_of_strings(value: Any) -> list[str]:
    if not isinstance(value, list):
        raise TypeError("Expected a list.")
    return [str(item) for item in value]


def _list_of_dicts(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        raise TypeError("Expected a list.")
    if not all(isinstance(item, dict) for item in value):
        raise TypeError("Expected a list of objects.")
    return [dict(item) for item in value]


def _optional_str(value: Any) -> str | None:
    if value is None:
        return None
    return str(value)
