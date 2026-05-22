"""Tests for paper-scale SourceTask loading variants."""

from __future__ import annotations

import json
from pathlib import Path

from magcoder.benchmark.full_builder.task_loader import load_source_tasks_with_report


def test_source_loader_renames_duplicate_task_ids(tmp_path: Path) -> None:
    """Duplicate task ids can be deterministically renamed with a report."""

    source = tmp_path / "tasks.jsonl"
    row = {
        "task_id": "dup",
        "source": "fixture",
        "prompt": "Return x.",
        "canonical_solution": "def solve(x):\n    return x\n",
        "tests": ["assert solve(1) == 1"],
        "entry_point": "solve",
    }
    source.write_text(
        "\n".join(json.dumps(row) for _ in range(2)) + "\n",
        encoding="utf-8",
    )
    tasks, report = load_source_tasks_with_report([source], duplicate_policy="rename")
    assert [task.task_id for task in tasks] == ["dup", "dup_2"]
    assert report.duplicate_task_ids == 1


def test_source_loader_skips_malformed_rows(tmp_path: Path) -> None:
    """Malformed rows are skipped and recorded in the loader report."""

    source = tmp_path / "tasks.jsonl"
    source.write_text(
        "\n".join(
            [
                '{"task_id": "bad"',
                json.dumps(
                    {
                        "task_id": "ok",
                        "source": "fixture",
                        "prompt": "Return x.",
                        "canonical_solution": "def solve(x):\n    return x\n",
                        "tests": ["assert solve(1) == 1"],
                        "entry_point": "solve",
                    }
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    tasks, report = load_source_tasks_with_report([source])
    assert [task.task_id for task in tasks] == ["ok"]
    assert report.malformed_rows == 1
    assert report.skipped_rows == 1
