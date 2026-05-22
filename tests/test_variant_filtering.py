"""Tests for deterministic template variant filtering."""

from __future__ import annotations

from magcoder.benchmark.full_builder.objectifier import objectify_source_task
from magcoder.benchmark.full_builder.source_task import SourceTask
from magcoder.benchmark.full_builder.task_taxonomy import classify_source_task
from magcoder.benchmark.full_builder.template_variants import (
    VariantGenerationConfig,
    generate_template_variant_specs,
)
from magcoder.benchmark.full_builder.variant_filters import (
    VariantFilterState,
    evaluate_variant,
)


def test_duplicate_variants_are_filtered() -> None:
    """A repeated spec for the same source/interface is rejected."""

    task = _task()
    classification = classify_source_task(task)
    spec = generate_template_variant_specs(
        task,
        classification,
        VariantGenerationConfig(variants_per_task=1, variant_policy="paper2500"),
    )[0]
    objectified = objectify_source_task(
        task,
        classification,
        variant_index=0,
        variant_spec=spec,
    )
    state = VariantFilterState()

    first = evaluate_variant(
        source_task=task,
        classification=classification,
        objectified=objectified,
        variant_spec=spec,
        state=state,
        deduplicate=True,
        drop_illogical=True,
    )
    second = evaluate_variant(
        source_task=task,
        classification=classification,
        objectified=objectified,
        variant_spec=spec,
        state=state,
        deduplicate=True,
        drop_illogical=True,
    )

    assert first.accepted
    assert not second.accepted
    assert second.reason.startswith("duplicate_")


def test_illogical_variant_is_skipped_with_reason() -> None:
    """No-edge objectifications are treated as illogical."""

    task = _task()
    classification = classify_source_task(task)
    spec = generate_template_variant_specs(
        task,
        classification,
        VariantGenerationConfig(variants_per_task=1),
    )[0]
    objectified = objectify_source_task(task, classification, variant_spec=spec)
    objectified.interaction_edges = []

    decision = evaluate_variant(
        source_task=task,
        classification=classification,
        objectified=objectified,
        variant_spec=spec,
        state=VariantFilterState(),
        deduplicate=True,
        drop_illogical=True,
    )

    assert not decision.accepted
    assert decision.reason == "no_interaction_edge"


def _task() -> SourceTask:
    return SourceTask(
        task_id="fixture_task",
        source="fixture",
        prompt="Return x.",
        canonical_solution="def solve(x):\n    return x\n",
        tests=["assert solve(1) == 1"],
        entry_point="solve",
    )
