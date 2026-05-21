"""Tests for MAG-Bench-Func functional-only export."""

from __future__ import annotations

import csv
from pathlib import Path

from scripts.export_mag_bench_func_views import export_mag_bench_func_views


def test_mag_bench_func_export_uses_functional_pass(tmp_path: Path) -> None:
    """MAG-Bench-Func summaries are derived from source-functional results."""

    run = tmp_path / "run"
    run.mkdir()
    _write_sample_summary(run / "sample_summary.csv")

    report = export_mag_bench_func_views(
        bench=tmp_path / "bench",
        run=run,
        output_dir=tmp_path / "func",
    )

    overall = next(
        row for row in report["summary_rows"]
        if row["group"] == "overall"
    )
    assert overall["functional_pass_count"] == 1
    assert overall["functional_pass_rate"] == "0.500"
    rows = list(
        csv.DictReader(
            (tmp_path / "func" / "mag_bench_vs_func_comparison.csv").open(
                newline="",
                encoding="utf-8",
            )
        )
    )
    assert rows[0]["source_functional_pass_rate"] == "0.500"


def _write_sample_summary(path: Path) -> None:
    fields = [
        "case_id",
        "method",
        "mode",
        "source",
        "task_type",
        "data_type",
        "template_family",
        "contract_pass",
        "functional_pass",
        "abms_pass",
        "functional_error_type",
    ]
    rows = [
        {
            "case_id": "case_a",
            "method": "full_with_refinement",
            "mode": "structured_graph_nl",
            "source": "humaneval",
            "task_type": "strings",
            "data_type": "sequence",
            "template_family": "delegated",
            "contract_pass": True,
            "functional_pass": True,
            "abms_pass": True,
            "functional_error_type": "",
        },
        {
            "case_id": "case_b",
            "method": "no_refinement",
            "mode": "graphical_graph_nl",
            "source": "mbpp",
            "task_type": "math",
            "data_type": "basic",
            "template_family": "delegated",
            "contract_pass": True,
            "functional_pass": False,
            "abms_pass": True,
            "functional_error_type": "AssertionError",
        },
    ]
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
