"""Tests for deterministic template variant generation."""

from __future__ import annotations

from magcoder.benchmark.full_builder.source_task import SourceTask
from magcoder.benchmark.full_builder.task_taxonomy import classify_source_task
from magcoder.benchmark.full_builder.template_variants import (
    VariantGenerationConfig,
    generate_template_variant_specs,
)


def test_fixture_source_task_can_produce_multiple_variants() -> None:
    """Variant ids and template families are deterministic."""

    task = SourceTask(
        task_id="HumanEval/fixture",
        source="humaneval",
        prompt="Return x.",
        canonical_solution="def solve(x):\n    return x\n",
        tests=["assert solve(1) == 1"],
        entry_point="solve",
    )
    classification = classify_source_task(task)
    config = VariantGenerationConfig(
        variants_per_task=5,
        variant_policy="paper2500",
        seed=2026,
    )
    specs = generate_template_variant_specs(task, classification, config)
    repeat = generate_template_variant_specs(task, classification, config)

    assert len(specs) == 5
    assert [spec.variant_id for spec in specs] == [
        spec.variant_id for spec in repeat
    ]
    assert len({spec.template_family for spec in specs}) > 1
    assert specs[0].expected_behavior_preservation
