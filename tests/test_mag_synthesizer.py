"""Tests for MAG synthesis from objectified tasks."""

from __future__ import annotations

from pathlib import Path

from magcoder.benchmark.full_builder import (
    classify_source_task,
    load_source_tasks_from_jsonl,
    objectify_source_task,
    synthesize_mag_from_objectified_task,
)
from magcoder.graphics.dot_parser import load_mag_from_graphical_artifact
from magcoder.graphics.dot_renderer import save_mag_dot
from magcoder.graphics.manifest import (
    build_graphical_mag_manifest,
    save_graphical_mag_manifest,
)
from magcoder.models import save_mag_to_yaml

ROOT = Path(__file__).resolve().parents[1]


def test_objectified_task_can_convert_to_valid_mag() -> None:
    """MAG synthesis preserves agents, edges, filters, and metadata."""

    task = load_source_tasks_from_jsonl(
        ROOT / "examples/source_tasks/humaneval_style_seed.jsonl",
    )[0]
    objectified = objectify_source_task(task, classify_source_task(task))
    mag = synthesize_mag_from_objectified_task(objectified)
    mag.validate()
    assert len(mag.agents) == 3
    assert len(mag.edges) == 2
    assert mag.filters
    assert mag.metadata["required_method_names"]


def test_synthesized_mag_can_round_trip_through_dot_manifest(tmp_path: Path) -> None:
    """Generated MAGs support standardized graphical artifacts."""

    task = load_source_tasks_from_jsonl(
        ROOT / "examples/source_tasks/mbpp_style_seed.jsonl",
    )[2]
    mag = synthesize_mag_from_objectified_task(
        objectify_source_task(task, classify_source_task(task))
    )
    mag_path = tmp_path / "mag.yaml"
    dot_path = tmp_path / "mag.dot"
    manifest_path = tmp_path / "mag_manifest.json"
    save_mag_to_yaml(mag, mag_path)
    save_mag_dot(mag, dot_path)
    save_graphical_mag_manifest(build_graphical_mag_manifest(mag), manifest_path)
    recovered = load_mag_from_graphical_artifact(dot_path, manifest_path)
    assert recovered.id == mag.id
    recovered.validate()
