"""Manifest objects for generated Full MAG-Bench packages."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(slots=True)
class FullBenchmarkCase:
    """One generated Full MAG-Bench case."""

    case_id: str
    source_task_id: str
    source: str
    data_type: str
    task_type: str
    case_dir: str
    mag_path: str
    mag_json_path: str
    dot_path: str
    manifest_path: str
    tests_json_path: str
    tests_py_path: str
    gold_solution_path: str
    broken_solution_path: str
    framework_path: str
    metadata_path: str
    supported_input_modes: list[str]
    metadata: dict[str, Any] = field(default_factory=dict)
    abms_oracle_path: str = "abms_oracle.json"
    abms_tests_py_path: str = "abms_tests.py"
    gold_abms_solution_path: str = "gold_abms_solution.py"
    broken_abms_solution_path: str = "broken_abms_solution.py"

    def to_dict(self) -> dict[str, Any]:
        """Render this case as JSON/YAML-compatible containers."""

        return asdict(self)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> FullBenchmarkCase:
        """Build a case from parsed package metadata."""

        metadata = data.get("metadata", {})
        if not isinstance(metadata, dict):
            raise TypeError("Full benchmark case metadata must be an object.")
        return cls(
            case_id=str(data["case_id"]),
            source_task_id=str(data["source_task_id"]),
            source=str(data["source"]),
            data_type=str(data["data_type"]),
            task_type=str(data["task_type"]),
            case_dir=str(data["case_dir"]),
            mag_path=str(data["mag_path"]),
            mag_json_path=str(data["mag_json_path"]),
            dot_path=str(data["dot_path"]),
            manifest_path=str(data["manifest_path"]),
            tests_json_path=str(data["tests_json_path"]),
            tests_py_path=str(data["tests_py_path"]),
            gold_solution_path=str(data.get("gold_solution_path", "gold_solution.py")),
            broken_solution_path=str(
                data.get("broken_solution_path", "broken_solution.py")
            ),
            framework_path=str(data["framework_path"]),
            metadata_path=str(data["metadata_path"]),
            supported_input_modes=[str(item) for item in data["supported_input_modes"]],
            metadata=dict(metadata),
            abms_oracle_path=str(data.get("abms_oracle_path", "abms_oracle.json")),
            abms_tests_py_path=str(data.get("abms_tests_py_path", "abms_tests.py")),
            gold_abms_solution_path=str(
                data.get("gold_abms_solution_path", "gold_abms_solution.py")
            ),
            broken_abms_solution_path=str(
                data.get("broken_abms_solution_path", "broken_abms_solution.py")
            ),
        )


@dataclass(slots=True)
class FullBenchmarkPackage:
    """Generated Full MAG-Bench package metadata."""

    benchmark_id: str
    name: str
    version: str
    root_dir: str
    cases: list[FullBenchmarkCase]
    source_files: list[str]
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Render this package as JSON/YAML-compatible containers."""

        data = asdict(self)
        data["cases"] = [case.to_dict() for case in self.cases]
        return data

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> FullBenchmarkPackage:
        """Build package metadata from parsed YAML."""

        cases_data = data.get("cases", [])
        metadata = data.get("metadata", {})
        if not isinstance(cases_data, list):
            raise TypeError("Full benchmark field 'cases' must be a list.")
        if not isinstance(metadata, dict):
            raise TypeError("Full benchmark metadata must be an object.")
        return cls(
            benchmark_id=str(data["benchmark_id"]),
            name=str(data["name"]),
            version=str(data["version"]),
            root_dir=str(data["root_dir"]),
            cases=[FullBenchmarkCase.from_dict(case) for case in cases_data],
            source_files=[str(item) for item in data.get("source_files", [])],
            metadata=dict(metadata),
        )
