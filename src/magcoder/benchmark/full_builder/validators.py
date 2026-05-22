"""Validation for generated Full MAG-Bench packages."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from magcoder.benchmark.full_builder.benchmark_package import (
    load_full_benchmark_package,
)
from magcoder.graphics.dot_parser import load_mag_from_graphical_artifact
from magcoder.models import load_mag_from_yaml


@dataclass(slots=True)
class FullBenchmarkValidationReport:
    """Validation result for a Full MAG-Bench package."""

    benchmark_id: str
    bench_dir: str
    case_count: int
    passed: bool
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Render this report as JSON-compatible containers."""

        return asdict(self)


def validate_full_mag_bench(bench_dir: str | Path) -> FullBenchmarkValidationReport:
    """Validate a generated Full MAG-Bench package."""

    root = Path(bench_dir)
    package = load_full_benchmark_package(root)
    errors: list[str] = []
    warnings: list[str] = []
    seen_case_ids: set[str] = set()
    for case in package.cases:
        if case.case_id in seen_case_ids:
            errors.append(f"Duplicate case id: {case.case_id}")
        seen_case_ids.add(case.case_id)
        case_dir = root / case.case_dir
        _validate_case(root, case_dir, case.to_dict(), errors, warnings)
    return FullBenchmarkValidationReport(
        benchmark_id=package.benchmark_id,
        bench_dir=str(root),
        case_count=len(package.cases),
        passed=not errors,
        errors=errors,
        warnings=warnings,
        metadata={"validator_version": 1},
    )


def save_validation_report(
    report: FullBenchmarkValidationReport,
    bench_dir: str | Path,
) -> tuple[Path, Path]:
    """Save validation report as JSON and Markdown."""

    root = Path(bench_dir)
    json_path = root / "validation_report.json"
    md_path = root / "validation_report.md"
    with json_path.open("w", encoding="utf-8") as file:
        json.dump(report.to_dict(), file, indent=2, sort_keys=True)
        file.write("\n")
    lines = [
        "# Full MAG-Bench v1 Validation Report",
        "",
        f"- Benchmark: `{report.benchmark_id}`",
        f"- Cases: {report.case_count}",
        f"- Passed: {report.passed}",
        "",
    ]
    if report.errors:
        lines.extend(["## Errors", "", *[f"- {error}" for error in report.errors], ""])
    if report.warnings:
        lines.extend(
            ["## Warnings", "", *[f"- {warning}" for warning in report.warnings], ""]
        )
    if not report.errors and not report.warnings:
        lines.append("No validation issues were found.")
    md_path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
    return json_path, md_path


def _validate_case(
    root: Path,
    case_dir: Path,
    case_data: dict[str, Any],
    errors: list[str],
    warnings: list[str],
) -> None:
    if not case_dir.exists():
        errors.append(f"Missing case directory: {case_dir}")
        return
    required_files = [
        "case.yaml",
        case_data["mag_path"],
        case_data["mag_json_path"],
        case_data["dot_path"],
        case_data["manifest_path"],
        case_data["tests_json_path"],
        case_data["tests_py_path"],
        case_data.get("gold_solution_path", "gold_solution.py"),
        case_data.get("broken_solution_path", "broken_solution.py"),
        case_data.get("abms_oracle_path", "abms_oracle.json"),
        case_data.get("abms_tests_py_path", "abms_tests.py"),
        case_data.get("gold_abms_solution_path", "gold_abms_solution.py"),
        case_data.get("broken_abms_solution_path", "broken_abms_solution.py"),
        case_data["framework_path"],
        case_data["metadata_path"],
        "source_task.json",
        "objectified_task.json",
    ]
    for filename in required_files:
        if not (case_dir / str(filename)).exists():
            errors.append(f"Missing required file for {case_dir.name}: {filename}")

    mag_path = case_dir / str(case_data["mag_path"])
    dot_path = case_dir / str(case_data["dot_path"])
    manifest_path = case_dir / str(case_data["manifest_path"])
    metadata_path = case_dir / str(case_data["metadata_path"])
    tests_path = case_dir / str(case_data["tests_json_path"])
    abms_oracle_path = case_dir / str(
        case_data.get("abms_oracle_path", "abms_oracle.json")
    )
    if mag_path.exists():
        try:
            mag = load_mag_from_yaml(mag_path)
            mag.validate()
            if not all(agent.nl_description for agent in mag.agents):
                errors.append(f"MAG agents missing nl_description: {mag_path}")
            if not all(edge.nl_description for edge in mag.edges):
                errors.append(f"MAG edges missing nl_description: {mag_path}")
        except Exception as error:  # noqa: BLE001 - report validation errors.
            errors.append(f"MAG validation failed for {mag_path}: {error}")
    if dot_path.exists() and manifest_path.exists():
        try:
            load_mag_from_graphical_artifact(dot_path, manifest_path)
        except Exception as error:  # noqa: BLE001 - report validation errors.
            errors.append(
                f"DOT+manifest round-trip failed for {case_dir.name}: {error}"
            )
    if tests_path.exists():
        _validate_json_file(tests_path, "tests", errors)
    if abms_oracle_path.exists():
        _validate_json_file(abms_oracle_path, "abms_oracle", errors)
    if metadata_path.exists():
        metadata = _validate_json_file(metadata_path, "metadata", errors)
        if isinstance(metadata, dict):
            for key in ["supports_graph_only", "supports_graph_nl"]:
                if key not in metadata:
                    errors.append(f"Metadata missing {key}: {metadata_path}")
    if not (root / "benchmark.yaml").exists():
        warnings.append("Package benchmark.yaml is missing.")


def _validate_json_file(
    path: Path,
    label: str,
    errors: list[str],
) -> dict[str, Any] | None:
    try:
        with path.open(encoding="utf-8") as file:
            data = json.load(file)
    except Exception as error:  # noqa: BLE001 - report validation errors.
        errors.append(f"Invalid {label} JSON {path}: {error}")
        return None
    if not isinstance(data, dict):
        errors.append(f"{label} JSON root must be an object: {path}")
        return None
    return data
