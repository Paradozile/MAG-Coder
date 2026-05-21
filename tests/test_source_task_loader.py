"""Tests for Full MAG-Bench source task loading."""

from __future__ import annotations

from pathlib import Path

from magcoder.benchmark.full_builder import (
    classify_source_task,
    load_source_tasks,
    load_source_tasks_from_jsonl,
)

ROOT = Path(__file__).resolve().parents[1]


def test_seed_jsonl_files_can_load() -> None:
    """HumanEval/MBPP-style seed tasks load without network access."""

    humaneval_tasks = load_source_tasks_from_jsonl(
        ROOT / "examples/source_tasks/humaneval_style_seed.jsonl",
    )
    mbpp_tasks = load_source_tasks_from_jsonl(
        ROOT / "examples/source_tasks/mbpp_style_seed.jsonl",
    )
    assert len(humaneval_tasks) == 5
    assert len(mbpp_tasks) == 5
    assert humaneval_tasks[0].entry_point == "has_close_elements"
    assert mbpp_tasks[0].task_id == "mbpp_101"


def test_loading_multiple_sources_preserves_unique_ids() -> None:
    """The loader combines multiple local source files."""

    tasks = load_source_tasks(
        [
            ROOT / "examples/source_tasks/humaneval_style_seed.jsonl",
            ROOT / "examples/source_tasks/mbpp_style_seed.jsonl",
        ]
    )
    assert len(tasks) == 10
    assert len({task.task_id for task in tasks}) == 10


def test_classification_can_be_generated() -> None:
    """Each loaded source task receives deterministic taxonomy labels."""

    task = load_source_tasks_from_jsonl(
        ROOT / "examples/source_tasks/mbpp_style_seed.jsonl",
    )[0]
    classification = classify_source_task(task)
    assert classification.task_id == task.task_id
    assert classification.data_type in {
        "basic",
        "sequence",
        "mapping",
        "nested",
        "tree",
        "matrix",
    }
    assert classification.task_type in {
        "string_processing",
        "sorting_searching",
        "data_structure_operation",
        "numeric_computation",
        "transformation",
    }
