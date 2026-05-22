"""Benchmark task data model."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml

from magcoder.evaluation import ScenarioSpec


@dataclass(slots=True)
class BenchmarkTask:
    """One Mini MAG-Bench task."""

    task_id: str
    name: str
    description: str
    task_type: str
    scenario_type: str
    mag_path: str
    framework_path: str
    scenario_path: str
    expected_agents: list[str]
    expected_edges: list[dict[str, Any]]
    expected_filters: list[str]
    mandatory_checkpoints: list[str]
    supported_input_modes: list[str]
    graphical_dot_path: str | None = None
    graphical_manifest_path: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Render this task as plain Python containers."""

        return asdict(self)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> BenchmarkTask:
        """Build a benchmark task from parsed YAML data."""

        return cls(
            task_id=str(data["task_id"]),
            name=str(data["name"]),
            description=str(data["description"]),
            task_type=str(data["task_type"]),
            scenario_type=str(data["scenario_type"]),
            mag_path=str(data["mag_path"]),
            graphical_dot_path=_optional_str(data.get("graphical_dot_path")),
            graphical_manifest_path=_optional_str(
                data.get("graphical_manifest_path"),
            ),
            framework_path=str(data["framework_path"]),
            scenario_path=str(data["scenario_path"]),
            expected_agents=_list_of_strings(data.get("expected_agents", [])),
            expected_edges=_list_of_dicts(data.get("expected_edges", [])),
            expected_filters=_list_of_strings(data.get("expected_filters", [])),
            mandatory_checkpoints=_list_of_strings(
                data.get("mandatory_checkpoints", []),
            ),
            supported_input_modes=_list_of_strings(
                data.get("supported_input_modes", []),
            ),
            metadata=_dict_value(data.get("metadata", {}), "metadata"),
        )

    def to_scenario_spec(self, *, base_dir: str | Path) -> ScenarioSpec:
        """Convert this task into a ScenarioSpec with resolved paths."""

        root = Path(base_dir)
        return ScenarioSpec(
            scenario_id=self.task_id,
            name=self.name,
            description=self.description,
            mag_path=str(root / self.mag_path),
            graphical_dot_path=(
                None
                if self.graphical_dot_path is None
                else str(root / self.graphical_dot_path)
            ),
            graphical_manifest_path=(
                None
                if self.graphical_manifest_path is None
                else str(root / self.graphical_manifest_path)
            ),
            framework_path=str(root / self.framework_path),
            expected_agents=list(self.expected_agents),
            expected_edges=[dict(edge) for edge in self.expected_edges],
            expected_filters=list(self.expected_filters),
            mandatory_checkpoints=list(self.mandatory_checkpoints),
            metadata={
                **self.metadata,
                "task_type": self.task_type,
                "scenario_type": self.scenario_type,
                "benchmark_task": True,
            },
        )


def load_benchmark_task(path: str | Path) -> BenchmarkTask:
    """Load a benchmark task from YAML."""

    with Path(path).open(encoding="utf-8") as file:
        data = yaml.safe_load(file)
    if not isinstance(data, Mapping):
        raise TypeError("Benchmark task YAML root must be an object.")
    return BenchmarkTask.from_dict(data)


def save_benchmark_task(task: BenchmarkTask, path: str | Path) -> None:
    """Save a benchmark task to YAML."""

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file:
        yaml.safe_dump(task.to_dict(), file, sort_keys=False)


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


def _dict_value(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise TypeError(f"Expected {label} to be a dict.")
    return dict(value)


def _optional_str(value: Any) -> str | None:
    if value is None:
        return None
    return str(value)
