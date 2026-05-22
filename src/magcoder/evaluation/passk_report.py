"""Reporting helpers for multi-sample pass@k evaluation."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any


def write_passk_summary_reports(
    rows: list[dict[str, Any]],
    *,
    csv_path: str | Path,
    markdown_path: str | Path,
) -> None:
    """Write pass@k CSV and Markdown summary files."""

    _write_csv(rows, Path(csv_path), _summary_fieldnames())
    _write_markdown(rows, Path(markdown_path), _summary_fieldnames())


def write_failure_summary(
    failures: list[dict[str, Any]],
    path: str | Path,
) -> None:
    """Write a Markdown summary of failed generation/evaluation samples."""

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    lines = ["# Failure Summary", ""]
    if not failures:
        lines.append("No failed samples were recorded.")
        output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return

    headers = [
        "task_id",
        "input_mode",
        "ablation_id",
        "sample_index",
        "generation_error",
        "success",
    ]
    lines.extend(
        [
            "| " + " | ".join(headers) + " |",
            "| " + " | ".join("---" for _ in headers) + " |",
        ]
    )
    for failure in failures:
        row = " | ".join(str(failure.get(header, "")) for header in headers)
        lines.append(f"| {row} |")
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_csv(
    rows: list[dict[str, Any]],
    path: Path,
    fieldnames: list[str],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_markdown(
    rows: list[dict[str, Any]],
    path: Path,
    fieldnames: list[str],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# pass@k Evaluation Summary",
        "",
        "This table compares pass@k across graph+NL, graph-only, refinement "
        "on/off, structured input, graphical input, and error rates.",
        "",
        "| " + " | ".join(fieldnames) + " |",
        "| " + " | ".join("---" for _ in fieldnames) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(row[field]) for field in fieldnames) + " |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _summary_fieldnames() -> list[str]:
    return [
        "run_id",
        "provider",
        "model_name",
        "task_id",
        "input_mode",
        "ablation_id",
        "num_samples",
        "num_correct",
        "k",
        "estimated_pass_at_k",
        "observed_pass_at_k",
        "success_criterion",
        "avg_checkpoint_accuracy",
        "framework_pass_rate",
        "parse_success_rate",
        "generation_error_rate",
        "cache_hit_rate",
    ]
