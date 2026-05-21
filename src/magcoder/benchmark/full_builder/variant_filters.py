"""Deterministic filters for template variant acceptance."""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass, field
from typing import Any

from magcoder.benchmark.full_builder.objectifier import ObjectifiedTask
from magcoder.benchmark.full_builder.source_task import SourceTask
from magcoder.benchmark.full_builder.task_taxonomy import TaskClassification
from magcoder.benchmark.full_builder.template_variants import TemplateVariantSpec


@dataclass(slots=True)
class VariantFilterDecision:
    """Acceptance decision for one template variant."""

    accepted: bool
    reason: str = "accepted"
    duplicate_key: str = ""
    illogical: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Render this decision as JSON-compatible containers."""

        return asdict(self)


@dataclass(slots=True)
class VariantFilterReport:
    """Aggregate deterministic filtering report."""

    candidate_variants: int = 0
    accepted_variants: int = 0
    skipped_variants: int = 0
    duplicate_variants_removed: int = 0
    illogical_variants_removed: int = 0
    unsupported_oracle_variants: int = 0
    unsupported_abms_oracle_variants: int = 0
    skip_reason_distribution: dict[str, int] = field(default_factory=dict)
    rows: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Render this report as JSON-compatible containers."""

        return asdict(self)


class VariantFilterState:
    """Mutable state for duplicate filtering across variants."""

    def __init__(self) -> None:
        self.semantic_keys: set[str] = set()
        self.interface_keys: set[str] = set()


def evaluate_variant(
    *,
    source_task: SourceTask,
    classification: TaskClassification,
    objectified: ObjectifiedTask,
    variant_spec: TemplateVariantSpec,
    state: VariantFilterState,
    deduplicate: bool,
    drop_illogical: bool,
) -> VariantFilterDecision:
    """Return an acceptance decision for one generated variant."""

    semantic_key = _semantic_duplicate_key(source_task, classification, variant_spec)
    interface_key = _interface_signature_key(objectified, variant_spec)
    if deduplicate and semantic_key in state.semantic_keys:
        return VariantFilterDecision(
            accepted=False,
            reason="duplicate_semantic_variant",
            duplicate_key=semantic_key,
            metadata={"variant_id": variant_spec.variant_id},
        )
    if deduplicate and interface_key in state.interface_keys:
        return VariantFilterDecision(
            accepted=False,
            reason="duplicate_interface_signature",
            duplicate_key=interface_key,
            metadata={"variant_id": variant_spec.variant_id},
        )
    illogical_reason = _illogical_reason(source_task, objectified)
    if drop_illogical and illogical_reason is not None:
        return VariantFilterDecision(
            accepted=False,
            reason=illogical_reason,
            illogical=True,
            metadata={"variant_id": variant_spec.variant_id},
        )
    oracle_reason = _oracle_reason(source_task)
    if oracle_reason is not None:
        return VariantFilterDecision(
            accepted=False,
            reason=oracle_reason,
            metadata={"variant_id": variant_spec.variant_id},
        )
    state.semantic_keys.add(semantic_key)
    state.interface_keys.add(interface_key)
    return VariantFilterDecision(
        accepted=True,
        reason="accepted",
        duplicate_key=semantic_key,
        metadata={
            "variant_id": variant_spec.variant_id,
            "semantic_key": semantic_key,
            "interface_key": interface_key,
        },
    )


def build_variant_filter_report(
    decisions: list[dict[str, Any]],
) -> VariantFilterReport:
    """Aggregate row-level filter decisions."""

    accepted = [row for row in decisions if row.get("accepted")]
    skipped = [row for row in decisions if not row.get("accepted")]
    reasons = Counter(str(row.get("reason", "unknown")) for row in skipped)
    duplicate_count = sum(
        count
        for reason, count in reasons.items()
        if reason.startswith("duplicate_")
    )
    illogical_count = sum(
        count
        for reason, count in reasons.items()
        if reason.startswith("illogical_")
        or reason
        in {
            "no_agent_class",
            "no_interaction_edge",
            "method_signature_cannot_be_mapped",
            "expected_behavior_cannot_be_observed",
            "source_tests_cannot_be_adapted",
        }
    )
    return VariantFilterReport(
        candidate_variants=len(decisions),
        accepted_variants=len(accepted),
        skipped_variants=len(skipped),
        duplicate_variants_removed=duplicate_count,
        illogical_variants_removed=illogical_count,
        unsupported_oracle_variants=reasons.get("unsupported_source_oracle", 0),
        unsupported_abms_oracle_variants=reasons.get("unsupported_abms_oracle", 0),
        skip_reason_distribution=dict(reasons),
        rows=decisions,
    )


def _semantic_duplicate_key(
    source_task: SourceTask,
    classification: TaskClassification,
    variant_spec: TemplateVariantSpec,
) -> str:
    return "|".join(
        [
            source_task.task_id,
            classification.data_type,
            classification.task_type,
            variant_spec.template_family,
            variant_spec.agent_naming_scheme,
            variant_spec.interaction_scheme,
            variant_spec.framework_api_scheme,
        ]
    )


def _interface_signature_key(
    objectified: ObjectifiedTask,
    variant_spec: TemplateVariantSpec,
) -> str:
    agents = ",".join(agent.get("id", "") for agent in objectified.agent_classes)
    methods = ",".join(
        edge.get("required_method_name", "")
        for edge in objectified.interaction_edges
    )
    return "|".join(
        [
            objectified.source_task_id,
            variant_spec.template_family,
            agents,
            methods,
        ]
    )


def _illogical_reason(
    source_task: SourceTask,
    objectified: ObjectifiedTask,
) -> str | None:
    if not objectified.agent_classes:
        return "no_agent_class"
    if not objectified.interaction_edges:
        return "no_interaction_edge"
    required_methods = [
        edge.get("required_method_name") for edge in objectified.interaction_edges
    ]
    if not all(required_methods):
        return "method_signature_cannot_be_mapped"
    if not source_task.entry_point:
        return "expected_behavior_cannot_be_observed"
    if not source_task.tests:
        return "source_tests_cannot_be_adapted"
    return None


def _oracle_reason(source_task: SourceTask) -> str | None:
    if not source_task.canonical_solution:
        return "unsupported_source_oracle"
    if not source_task.tests:
        return "unsupported_source_oracle"
    if not source_task.entry_point:
        return "unsupported_abms_oracle"
    return None
