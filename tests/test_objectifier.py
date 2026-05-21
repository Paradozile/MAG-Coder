"""Tests for source-task objectification."""

from __future__ import annotations

from pathlib import Path

from magcoder.benchmark.full_builder import (
    classify_source_task,
    load_source_tasks_from_jsonl,
    objectify_source_task,
)

ROOT = Path(__file__).resolve().parents[1]


def test_function_task_can_be_objectified() -> None:
    """Objectification creates agents, interactions, filters, and tests."""

    task = load_source_tasks_from_jsonl(
        ROOT / "examples/source_tasks/humaneval_style_seed.jsonl",
    )[0]
    classification = classify_source_task(task)
    objectified = objectify_source_task(task, classification)
    assert objectified.source_task_id == task.task_id
    assert len(objectified.agent_classes) == 3
    assert len(objectified.interaction_edges) == 2
    assert objectified.filters
    assert objectified.transformed_tests
    assert objectified.class_interface_spec["required_methods"]


def test_objectifier_generates_varied_agent_names() -> None:
    """Different task categories produce different agent interface names."""

    tasks = load_source_tasks_from_jsonl(
        ROOT / "examples/source_tasks/mbpp_style_seed.jsonl",
    )
    names_by_task = []
    for task in tasks:
        classification = classify_source_task(task)
        objectified = objectify_source_task(task, classification)
        names_by_task.append(tuple(agent["id"] for agent in objectified.agent_classes))
    assert len(set(names_by_task)) > 1
