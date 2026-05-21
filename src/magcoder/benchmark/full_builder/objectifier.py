"""Convert function-level source tasks into agent-interaction tasks."""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from typing import TYPE_CHECKING, Any

from magcoder.benchmark.full_builder.source_task import SourceTask
from magcoder.benchmark.full_builder.task_taxonomy import TaskClassification

if TYPE_CHECKING:
    from magcoder.benchmark.full_builder.template_variants import TemplateVariantSpec


@dataclass(slots=True)
class ObjectifiedTask:
    """An object-oriented, interaction-oriented version of a source task."""

    task_id: str
    source_task_id: str
    objectified_prompt: str
    agent_classes: list[dict[str, Any]]
    interaction_edges: list[dict[str, Any]]
    filters: list[dict[str, Any]]
    class_interface_spec: dict[str, Any]
    expected_behavior: str
    transformed_tests: list[dict[str, Any]]
    classification: dict[str, Any]
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Render this task as JSON-compatible containers."""

        return asdict(self)


def objectify_source_task(
    task: SourceTask,
    classification: TaskClassification,
    *,
    variant_index: int = 0,
    variant_spec: TemplateVariantSpec | None = None,
) -> ObjectifiedTask:
    """Create a deterministic agent-interaction task from a function task."""

    case_stem = sanitize_identifier(task.task_id)
    role = _role_names(classification, case_stem, variant_index)
    input_agent = role["input_agent"]
    processor_agent = role["processor_agent"]
    output_agent = role["output_agent"]
    edge_ingest = f"{_short_agent(input_agent)}_to_{_short_agent(processor_agent)}"
    edge_emit = f"{_short_agent(processor_agent)}_to_{_short_agent(output_agent)}"
    filter_token = (
        sanitize_identifier(variant_spec.filter_scheme)
        if variant_spec is not None
        else "input_ready"
    )
    filter_id = f"{case_stem}_{filter_token}"
    entry_point = task.entry_point or "solve"
    method_name = f"interact_{edge_ingest}"
    final_method_name = f"interact_{edge_emit}"
    template_note = (
        f" Use the `{variant_spec.template_family}` template family with "
        f"`{variant_spec.framework_api_scheme}` framework-facing API."
        if variant_spec is not None
        else ""
    )
    objectified_prompt = (
        f"Transform source task {task.task_id} into cooperating classes. "
        f"{input_agent} provides the original input, {processor_agent} performs "
        f"the behavior of `{entry_point}`, and {output_agent} receives the "
        f"computed result. Preserve the source tests as contract checks.{template_note}"
    )
    agent_classes: list[dict[str, Any]] = [
        {
            "id": input_agent,
            "attributes": [
                {
                    "name": "input_data",
                    "type": task.input_signature or "Any",
                    "description": "Input payload for the original function task.",
                }
            ],
            "ports": {
                "input": [],
                "output": ["task_input"],
            },
            "nl_description": f"Provides inputs for source task {task.task_id}.",
        },
        {
            "id": processor_agent,
            "attributes": [
                {
                    "name": "last_result",
                    "type": task.output_type or "Any",
                    "description": "Most recent computed result.",
                }
            ],
            "ports": {
                "input": ["task_input"],
                "output": ["task_output"],
            },
            "nl_description": (
                f"Implements the behavior of `{entry_point}` as an agent method."
            ),
        },
        {
            "id": output_agent,
            "attributes": [
                {
                    "name": "received_result",
                    "type": task.output_type or "Any",
                    "description": "Result received from the processor agent.",
                }
            ],
            "ports": {
                "input": ["task_output"],
                "output": [],
            },
            "nl_description": f"Receives outputs for source task {task.task_id}.",
        },
    ]
    interaction_edges: list[dict[str, Any]] = [
        {
            "id": edge_ingest,
            "src_agent": input_agent,
            "src_port": "task_input",
            "dst_agent": processor_agent,
            "dst_port": "task_input",
            "filter_id": filter_id,
            "required_method_name": method_name,
            "nl_description": (
                f"Send source-task input from {input_agent} to {processor_agent}."
            ),
        },
        {
            "id": edge_emit,
            "src_agent": processor_agent,
            "src_port": "task_output",
            "dst_agent": output_agent,
            "dst_port": "task_output",
            "filter_id": None,
            "required_method_name": final_method_name,
            "nl_description": (
                f"Send computed result from {processor_agent} to {output_agent}."
            ),
        },
    ]
    filters: list[dict[str, Any]] = [
        {
            "id": filter_id,
            "expressions": ["input_data is not None"],
            "nl_description": "Interaction is enabled when the input is available.",
        }
    ]
    transformed_tests: list[dict[str, Any]] = [
        {
            "test_id": f"{case_stem}_test_{index:03d}",
            "source_assertion": test,
            "entry_point": entry_point,
            "expected_contract": {
                "agents": [input_agent, processor_agent, output_agent],
                "required_methods": [method_name, final_method_name],
            },
        }
        for index, test in enumerate(task.tests)
    ]
    variant_suffix = "" if variant_index == 0 else f"_variant_{variant_index + 1:02d}"
    variant_metadata: dict[str, Any] = {}
    if variant_spec is not None:
        variant_metadata = {
            "variant_id": variant_spec.variant_id,
            "template_family": variant_spec.template_family,
            "agent_naming_scheme": variant_spec.agent_naming_scheme,
            "attribute_scheme": variant_spec.attribute_scheme,
            "interaction_scheme": variant_spec.interaction_scheme,
            "filter_scheme": variant_spec.filter_scheme,
            "nl_description_scheme": variant_spec.nl_description_scheme,
            "framework_api_scheme": variant_spec.framework_api_scheme,
        }
    return ObjectifiedTask(
        task_id=f"{case_stem}{variant_suffix}_mag",
        source_task_id=task.task_id,
        objectified_prompt=objectified_prompt,
        agent_classes=agent_classes,
        interaction_edges=interaction_edges,
        filters=filters,
        class_interface_spec={
            "entry_point": entry_point,
            "input_agent": input_agent,
            "processor_agent": processor_agent,
            "output_agent": output_agent,
            "required_methods": [method_name, final_method_name],
            "message_api_name": "send_message",
        },
        expected_behavior=(
            f"The processor agent should implement behavior equivalent to "
            f"`{entry_point}` from the original function task."
        ),
        transformed_tests=transformed_tests,
        classification=classification.to_dict(),
        metadata={
            "source": task.source,
            "objectifier_version": 1,
            "variant_index": variant_index,
            **variant_metadata,
            "canonical_solution_available": task.canonical_solution is not None,
        },
    )


def sanitize_identifier(value: str) -> str:
    """Convert arbitrary source ids into stable snake_case ids."""

    cleaned = re.sub(r"[^0-9A-Za-z]+", "_", value).strip("_").lower()
    if not cleaned:
        return "task"
    if cleaned[0].isdigit():
        return f"task_{cleaned}"
    return cleaned


def _role_names(
    classification: TaskClassification,
    case_stem: str,
    variant_index: int = 0,
) -> dict[str, str]:
    prefix = {
        "string_processing": "Text",
        "sorting_searching": "Search",
        "data_structure_operation": "Structure",
        "numeric_computation": "Numeric",
        "transformation": "Transform",
    }[classification.task_type]
    data_prefix = {
        "basic": "Value",
        "sequence": "Sequence",
        "mapping": "Mapping",
        "nested": "Nested",
        "tree": "Tree",
        "matrix": "Matrix",
    }[classification.data_type]
    case_prefix = _compact_pascal_case(case_stem)
    if variant_index > 0:
        variant_token = ["Flow", "Relay", "Coordinator", "Mediator"][
            (variant_index - 1) % 4
        ]
        case_prefix = f"{case_prefix}{variant_token}{variant_index + 1}"
    return {
        "input_agent": f"{case_prefix}{data_prefix}InputAgent",
        "processor_agent": f"{case_prefix}{prefix}ProcessorAgent",
        "output_agent": f"{case_prefix}{prefix}ResultAgent",
    }


def _short_agent(agent_id: str) -> str:
    value = re.sub(r"Agent$", "", agent_id)
    words = re.findall(r"[A-Z]?[a-z]+|[A-Z]+(?=[A-Z]|$)|[0-9]+", value)
    return "_".join(word.lower() for word in words) or sanitize_identifier(agent_id)


def _compact_pascal_case(value: str) -> str:
    words = [word for word in re.split(r"[^0-9A-Za-z]+|_", value) if word]
    selected = words[-3:] if len(words) > 3 else words
    pascal = "".join(word[:1].upper() + word[1:] for word in selected)
    return pascal or "Task"
