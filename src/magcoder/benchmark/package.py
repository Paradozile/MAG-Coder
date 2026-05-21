"""Mini MAG-Bench package creation and loading."""

from __future__ import annotations

import shutil
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml

from magcoder.benchmark.task import (
    BenchmarkTask,
    load_benchmark_task,
    save_benchmark_task,
)
from magcoder.evaluation import ScenarioSpec, load_scenario_spec


@dataclass(slots=True)
class BenchmarkPackage:
    """A reusable Mini MAG-Bench package."""

    benchmark_id: str
    name: str
    version: str
    tasks: list[BenchmarkTask]
    root_dir: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Render package metadata as plain Python containers."""

        data = asdict(self)
        data["tasks"] = [task.to_dict() for task in self.tasks]
        return data

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> BenchmarkPackage:
        """Build a benchmark package from parsed YAML data."""

        tasks_data = data.get("tasks", [])
        if not isinstance(tasks_data, list):
            raise TypeError("Benchmark field 'tasks' must be a list.")
        metadata = data.get("metadata", {})
        if not isinstance(metadata, dict):
            raise TypeError("Benchmark field 'metadata' must be a dict.")
        return cls(
            benchmark_id=str(data["benchmark_id"]),
            name=str(data["name"]),
            version=str(data["version"]),
            tasks=[BenchmarkTask.from_dict(task) for task in tasks_data],
            root_dir=str(data["root_dir"]),
            metadata=dict(metadata),
        )


DEFAULT_SCENARIO_PATHS = [
    Path("examples/scenarios/robot_rescue_scenario.yaml"),
    Path("examples/scenarios/aircraft_collision_scenario.yaml"),
]


def build_mini_mag_bench(
    output_dir: str | Path,
    *,
    scenario_paths: list[str | Path] | None = None,
) -> BenchmarkPackage:
    """Build the Mini MAG-Bench directory package."""

    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    source_paths = [Path(path) for path in (scenario_paths or DEFAULT_SCENARIO_PATHS)]
    scenarios = [load_scenario_spec(path) for path in source_paths]
    tasks = [
        _package_scenario(root, scenario, source_path=source_path)
        for scenario, source_path in zip(scenarios, source_paths, strict=True)
    ]
    package = BenchmarkPackage(
        benchmark_id="mini_mag_bench",
        name="Mini MAG-Bench",
        version="0.1.0",
        tasks=tasks,
        root_dir=".",
        metadata={
            "generator": "fake-generator",
            "supports_real_llm": False,
            "supports_pass_at_k": False,
            "description": "Small deterministic MAG-Coder benchmark package.",
        },
    )
    save_benchmark_package(package, root / "benchmark.yaml")
    _write_readme(root, package)
    package.root_dir = str(root)
    return package


def load_benchmark_package(path: str | Path) -> BenchmarkPackage:
    """Load a benchmark package from a directory or benchmark.yaml path."""

    input_path = Path(path)
    benchmark_path = (
        input_path / "benchmark.yaml" if input_path.is_dir() else input_path
    )
    with benchmark_path.open(encoding="utf-8") as file:
        data = yaml.safe_load(file)
    if not isinstance(data, Mapping):
        raise TypeError("Benchmark YAML root must be an object.")
    package = BenchmarkPackage.from_dict(data)
    package.root_dir = str(benchmark_path.parent)
    return package


def save_benchmark_package(package: BenchmarkPackage, path: str | Path) -> None:
    """Save benchmark package metadata to YAML."""

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file:
        yaml.safe_dump(package.to_dict(), file, sort_keys=False)


def _package_scenario(
    root: Path,
    scenario: ScenarioSpec,
    *,
    source_path: Path,
) -> BenchmarkTask:
    task_dir = root / "tasks" / scenario.scenario_id
    task_dir.mkdir(parents=True, exist_ok=True)

    mag_path = _copy_to_task(task_dir, scenario.mag_path)
    framework_path = _copy_to_task(task_dir, scenario.framework_path)
    scenario_path = _copy_to_task(task_dir, source_path)
    graphical_dot_path = (
        None
        if scenario.graphical_dot_path is None
        else _copy_to_task(task_dir, scenario.graphical_dot_path)
    )
    graphical_manifest_path = (
        None
        if scenario.graphical_manifest_path is None
        else _copy_to_task(task_dir, scenario.graphical_manifest_path)
    )
    supported_input_modes = ["structured"]
    if graphical_dot_path is not None and graphical_manifest_path is not None:
        supported_input_modes.append("graphical")

    task = BenchmarkTask(
        task_id=scenario.scenario_id,
        name=scenario.name,
        description=scenario.description,
        task_type="abms_interaction_generation",
        scenario_type=str(scenario.metadata.get("scenario_type", "unknown")),
        mag_path=_relative(task_dir, mag_path),
        graphical_dot_path=(
            None
            if graphical_dot_path is None
            else _relative(task_dir, graphical_dot_path)
        ),
        graphical_manifest_path=(
            None
            if graphical_manifest_path is None
            else _relative(task_dir, graphical_manifest_path)
        ),
        framework_path=_relative(task_dir, framework_path),
        scenario_path=_relative(task_dir, scenario_path),
        expected_agents=scenario.expected_agents,
        expected_edges=scenario.expected_edges,
        expected_filters=scenario.expected_filters,
        mandatory_checkpoints=scenario.mandatory_checkpoints,
        supported_input_modes=supported_input_modes,
        metadata={
            **scenario.metadata,
            "source_scenario_id": scenario.scenario_id,
        },
    )
    save_benchmark_task(task, task_dir / "task.yaml")
    return task


def _copy_to_task(task_dir: Path, source: str | Path) -> Path:
    source_path = Path(source)
    destination = task_dir / source_path.name
    shutil.copy2(source_path, destination)
    return destination


def _relative(task_dir: Path, path: Path) -> str:
    return str(path.relative_to(task_dir))


def _write_readme(root: Path, package: BenchmarkPackage) -> None:
    task_lines = "\n".join(
        f"- `{task.task_id}`: {task.name}" for task in package.tasks
    )
    content = f"""# {package.name}

Version: `{package.version}`

This is a small deterministic Mini MAG-Bench package. It uses MAG-Coder's fake
generator and rule-based refinement, so results do not represent real LLM
performance.

## Tasks

{task_lines}

## Run

```bash
python scripts/run_mini_mag_bench.py \\
  --benchmark {root} \\
  --input-mode both \\
  --output {root}/results
```
"""
    (root / "README.md").write_text(content, encoding="utf-8")


def load_task_file(path: str | Path) -> BenchmarkTask:
    """Load a task YAML file."""

    return load_benchmark_task(path)
