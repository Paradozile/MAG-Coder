"""Deterministic template variants for paper-scale MAG-Bench reproduction."""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass, field
from typing import Any

from magcoder.benchmark.full_builder.objectifier import sanitize_identifier
from magcoder.benchmark.full_builder.source_task import SourceTask
from magcoder.benchmark.full_builder.task_taxonomy import TaskClassification


@dataclass(slots=True)
class TemplateVariantSpec:
    """One deterministic objectification/template variant."""

    variant_id: str
    source_task_id: str
    source_dataset: str
    data_type: str
    task_type: str
    template_family: str
    agent_naming_scheme: str
    attribute_scheme: str
    interaction_scheme: str
    filter_scheme: str
    nl_description_scheme: str
    framework_api_scheme: str
    expected_behavior_preservation: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Render this spec as JSON-compatible containers."""

        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TemplateVariantSpec:
        """Load a variant spec from JSON-compatible containers."""

        metadata = data.get("metadata", {})
        if not isinstance(metadata, dict):
            metadata = {}
        return cls(
            variant_id=str(data["variant_id"]),
            source_task_id=str(data["source_task_id"]),
            source_dataset=str(data["source_dataset"]),
            data_type=str(data["data_type"]),
            task_type=str(data["task_type"]),
            template_family=str(data["template_family"]),
            agent_naming_scheme=str(data["agent_naming_scheme"]),
            attribute_scheme=str(data["attribute_scheme"]),
            interaction_scheme=str(data["interaction_scheme"]),
            filter_scheme=str(data["filter_scheme"]),
            nl_description_scheme=str(data["nl_description_scheme"]),
            framework_api_scheme=str(data["framework_api_scheme"]),
            expected_behavior_preservation=str(
                data["expected_behavior_preservation"],
            ),
            metadata=metadata,
        )


@dataclass(slots=True)
class VariantGenerationConfig:
    """Configuration for deterministic template variant generation."""

    variants_per_task: int = 1
    variant_policy: str = "single"
    target_cases: int | None = None
    seed: int | None = None
    deduplicate_variants: bool = False
    drop_illogical: bool = False

    def to_dict(self) -> dict[str, Any]:
        """Render this config as JSON-compatible containers."""

        return asdict(self)


@dataclass(slots=True)
class VariantGenerationReport:
    """Aggregate report for generated template variants."""

    policy: str
    variants_per_task: int
    target_cases: int | None
    candidate_variants_generated: int = 0
    source_task_count: int = 0
    source_distribution: dict[str, int] = field(default_factory=dict)
    task_type_distribution: dict[str, int] = field(default_factory=dict)
    data_type_distribution: dict[str, int] = field(default_factory=dict)
    template_family_distribution: dict[str, int] = field(default_factory=dict)
    rows: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Render this report as JSON-compatible containers."""

        return asdict(self)


_TEMPLATE_FAMILIES = [
    "pipeline",
    "brokered_exchange",
    "guarded_handoff",
    "state_monitor",
    "result_auditor",
]
_AGENT_NAMING_SCHEMES = [
    "domain_compact",
    "role_explicit",
    "data_type_prefixed",
    "task_type_prefixed",
    "coordinator_prefixed",
]
_ATTRIBUTE_SCHEMES = [
    "input_result_state",
    "payload_cache_state",
    "source_target_state",
    "observed_value_state",
    "contract_state",
]
_INTERACTION_SCHEMES = [
    "two_edge_pipeline",
    "request_response",
    "guarded_dispatch",
    "monitor_then_emit",
    "audit_then_publish",
]
_FILTER_SCHEMES = [
    "input_ready",
    "payload_present",
    "guard_satisfied",
    "state_available",
    "contract_enabled",
]
_NL_DESCRIPTION_SCHEMES = [
    "concise_contract",
    "role_descriptive",
    "data_flow_descriptive",
    "condition_focused",
    "audit_focused",
]
_FRAMEWORK_API_SCHEMES = [
    "send_message",
    "send_message_with_target",
    "recorded_message_bus",
    "stateful_handoff",
    "observable_return",
]


def generate_template_variant_specs(
    task: SourceTask,
    classification: TaskClassification,
    config: VariantGenerationConfig,
) -> list[TemplateVariantSpec]:
    """Generate deterministic variant specs for one source task."""

    count = max(1, config.variants_per_task)
    specs: list[TemplateVariantSpec] = []
    source_stem = sanitize_identifier(task.task_id)
    offset = 0 if config.seed is None else config.seed % len(_TEMPLATE_FAMILIES)
    for variant_index in range(count):
        family_index = (variant_index + offset) % len(_TEMPLATE_FAMILIES)
        variant_number = variant_index + 1
        specs.append(
            TemplateVariantSpec(
                variant_id=f"{source_stem}__v{variant_number:02d}",
                source_task_id=task.task_id,
                source_dataset=task.source,
                data_type=classification.data_type,
                task_type=classification.task_type,
                template_family=_TEMPLATE_FAMILIES[family_index],
                agent_naming_scheme=_AGENT_NAMING_SCHEMES[
                    (variant_index + classification_index(classification.task_type))
                    % len(_AGENT_NAMING_SCHEMES)
                ],
                attribute_scheme=_ATTRIBUTE_SCHEMES[
                    (variant_index + classification_index(classification.data_type))
                    % len(_ATTRIBUTE_SCHEMES)
                ],
                interaction_scheme=_INTERACTION_SCHEMES[family_index],
                filter_scheme=_FILTER_SCHEMES[family_index],
                nl_description_scheme=_NL_DESCRIPTION_SCHEMES[family_index],
                framework_api_scheme=_FRAMEWORK_API_SCHEMES[family_index],
                expected_behavior_preservation="source_task_tests_and_entry_point",
                metadata={
                    "variant_index": variant_index,
                    "variant_policy": config.variant_policy,
                    "seed": config.seed,
                    "source_entry_point": task.entry_point,
                },
            )
        )
    return specs


def summarize_variant_specs(
    specs: list[TemplateVariantSpec],
    *,
    source_task_count: int,
    config: VariantGenerationConfig,
) -> VariantGenerationReport:
    """Create an aggregate report for generated variant specs."""

    return VariantGenerationReport(
        policy=config.variant_policy,
        variants_per_task=config.variants_per_task,
        target_cases=config.target_cases,
        candidate_variants_generated=len(specs),
        source_task_count=source_task_count,
        source_distribution=dict(Counter(spec.source_dataset for spec in specs)),
        task_type_distribution=dict(Counter(spec.task_type for spec in specs)),
        data_type_distribution=dict(Counter(spec.data_type for spec in specs)),
        template_family_distribution=dict(
            Counter(spec.template_family for spec in specs)
        ),
        rows=[spec.to_dict() for spec in specs],
    )


def classification_index(value: str) -> int:
    """Map known classification labels to stable small integers."""

    labels = [
        "basic",
        "sequence",
        "mapping",
        "nested",
        "tree",
        "matrix",
        "string_processing",
        "sorting_searching",
        "data_structure_operation",
        "numeric_computation",
        "transformation",
    ]
    try:
        return labels.index(value)
    except ValueError:
        return 0
