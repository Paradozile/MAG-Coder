"""Benchmark reporting utilities."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

from magcoder.benchmark.task import BenchmarkTask
from magcoder.evaluation import ScenarioRunResult


def write_benchmark_summary_reports(
    rows: list[dict[str, Any]],
    *,
    csv_path: str | Path,
    markdown_path: str | Path,
) -> None:
    """Write benchmark summary CSV and Markdown reports."""

    _write_csv(rows, Path(csv_path))
    _write_markdown(rows, Path(markdown_path))


def benchmark_summary_row(
    *,
    benchmark_id: str,
    task: BenchmarkTask,
    result: ScenarioRunResult,
) -> dict[str, Any]:
    """Build one summary row for a task/input-mode result."""

    initial_report = result.initial_result.initial_report
    refined_report = result.refined_result.refined_report
    return {
        "benchmark_id": benchmark_id,
        "task_id": task.task_id,
        "scenario_type": task.scenario_type,
        "input_mode": result.input_mode,
        "checkpoint_accuracy_initial": _format_float(
            result.initial_result.checkpoint_accuracy,
        ),
        "checkpoint_accuracy_refined": _format_float(
            result.refined_result.checkpoint_accuracy,
        ),
        "framework_violations_initial": (
            0 if initial_report is None else len(initial_report.framework_violations)
        ),
        "framework_violations_refined": (
            0 if refined_report is None else len(refined_report.framework_violations)
        ),
        "framework_passed_initial": result.initial_result.framework_passed,
        "framework_passed_refined": result.refined_result.framework_passed,
        "mandatory_success_refined": result.refined_result.mandatory_success,
    }


def _write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=_fieldnames())
        writer.writeheader()
        writer.writerows(rows)


def _write_markdown(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    headers = _fieldnames()
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(row[header]) for header in headers) + " |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _fieldnames() -> list[str]:
    return [
        "benchmark_id",
        "task_id",
        "scenario_type",
        "input_mode",
        "checkpoint_accuracy_initial",
        "checkpoint_accuracy_refined",
        "framework_violations_initial",
        "framework_violations_refined",
        "framework_passed_initial",
        "framework_passed_refined",
        "mandatory_success_refined",
    ]


def _format_float(value: float) -> str:
    return f"{value:.3f}"
