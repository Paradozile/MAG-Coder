"""Reporting utilities for scenario evaluation."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from magcoder.evaluation.runner import ScenarioRunResult


def save_scenario_run_result(result: ScenarioRunResult, path: str | Path) -> None:
    """Save a scenario run result as JSON."""

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file:
        json.dump(result.to_dict(), file, indent=2, sort_keys=True)
        file.write("\n")


def write_scenario_summary_reports(
    results: list[ScenarioRunResult],
    *,
    csv_path: str | Path,
    markdown_path: str | Path,
) -> None:
    """Write CSV and Markdown summary reports for scenario runs."""

    rows = [_summary_row(result) for result in results]
    _write_csv(rows, Path(csv_path))
    _write_markdown(rows, Path(markdown_path))


def _summary_row(result: ScenarioRunResult) -> dict[str, Any]:
    initial_report = result.initial_result.initial_report
    refined_report = result.refined_result.refined_report
    initial_violations = (
        0 if initial_report is None else len(initial_report.framework_violations)
    )
    refined_violations = (
        0 if refined_report is None else len(refined_report.framework_violations)
    )
    return {
        "scenario_id": result.scenario.scenario_id,
        "input_mode": result.input_mode,
        "checkpoint_accuracy_initial": _format_float(
            result.initial_result.checkpoint_accuracy,
        ),
        "checkpoint_accuracy_refined": _format_float(
            result.refined_result.checkpoint_accuracy,
        ),
        "framework_passed_initial": result.initial_result.framework_passed,
        "framework_passed_refined": result.refined_result.framework_passed,
        "framework_violations_initial": initial_violations,
        "framework_violations_refined": refined_violations,
        "mandatory_success_refined": result.refined_result.mandatory_success,
    }


def _write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "scenario_id",
        "input_mode",
        "checkpoint_accuracy_initial",
        "checkpoint_accuracy_refined",
        "framework_passed_initial",
        "framework_passed_refined",
        "framework_violations_initial",
        "framework_violations_refined",
        "mandatory_success_refined",
    ]
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_markdown(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    headers = [
        "scenario_id",
        "input_mode",
        "checkpoint_accuracy_initial",
        "checkpoint_accuracy_refined",
        "framework_passed_initial",
        "framework_passed_refined",
        "framework_violations_initial",
        "framework_violations_refined",
        "mandatory_success_refined",
    ]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(row[header]) for header in headers) + " |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _format_float(value: float) -> str:
    return f"{value:.3f}"
