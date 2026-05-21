"""Entry-point adapter specification for Full MAG-Bench cases."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from typing import Any

from magcoder.benchmark.full_builder.objectifier import ObjectifiedTask
from magcoder.benchmark.full_builder.source_task import SourceTask


@dataclass(slots=True)
class EntryPointAdapterSpec:
    """How source-task tests can be aligned with generated MAG code."""

    source_entry_point: str
    source_input_signature: str | None
    source_output_type: str | None
    objectified_entry_point: str
    candidate_agent_class: str
    candidate_interaction_method: str
    adapter_strategy: str
    input_mapping: dict[str, Any]
    output_mapping: dict[str, Any]
    required_candidate_symbols: list[str]
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Render this spec as JSON/YAML-compatible containers."""

        return asdict(self)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> EntryPointAdapterSpec:
        """Build a spec from parsed metadata."""

        metadata = data.get("metadata", {})
        input_mapping = data.get("input_mapping", {})
        output_mapping = data.get("output_mapping", {})
        required_symbols = data.get("required_candidate_symbols", [])
        if not isinstance(metadata, dict):
            raise TypeError("adapter spec metadata must be an object.")
        if not isinstance(input_mapping, dict):
            raise TypeError("adapter spec input_mapping must be an object.")
        if not isinstance(output_mapping, dict):
            raise TypeError("adapter spec output_mapping must be an object.")
        if not isinstance(required_symbols, list):
            raise TypeError("adapter spec required symbols must be a list.")
        return cls(
            source_entry_point=str(data["source_entry_point"]),
            source_input_signature=_optional_str(data.get("source_input_signature")),
            source_output_type=_optional_str(data.get("source_output_type")),
            objectified_entry_point=str(data["objectified_entry_point"]),
            candidate_agent_class=str(data["candidate_agent_class"]),
            candidate_interaction_method=str(data["candidate_interaction_method"]),
            adapter_strategy=str(data["adapter_strategy"]),
            input_mapping=dict(input_mapping),
            output_mapping=dict(output_mapping),
            required_candidate_symbols=[str(symbol) for symbol in required_symbols],
            metadata=dict(metadata),
        )


def build_entry_point_adapter_spec(
    source_task: SourceTask,
    objectified_task: ObjectifiedTask,
) -> EntryPointAdapterSpec:
    """Derive a deterministic adapter spec from objectification metadata."""

    interface = objectified_task.class_interface_spec
    input_agent = str(interface.get("input_agent", ""))
    processor_agent = str(interface.get("processor_agent", ""))
    required_methods = interface.get("required_methods", [])
    first_method = (
        str(required_methods[0])
        if isinstance(required_methods, list) and required_methods
        else ""
    )
    source_entry_point = source_task.entry_point or "solve"
    adapter_status = (
        "supported"
        if input_agent and processor_agent and first_method
        else "unsupported"
    )
    return EntryPointAdapterSpec(
        source_entry_point=source_entry_point,
        source_input_signature=source_task.input_signature,
        source_output_type=source_task.output_type,
        objectified_entry_point=f"{input_agent}.{first_method}",
        candidate_agent_class=input_agent,
        candidate_interaction_method=first_method,
        adapter_strategy="thin_agent_interaction_wrapper",
        input_mapping={
            "args": "source function positional arguments",
            "kwargs": "source function keyword arguments",
            "source_agent_attribute": "input_data",
            "target_agent_class": processor_agent,
        },
        output_mapping={
            "preferred_target_attribute": "last_result",
            "fallback_return_keys": ["result", "payload", "message", "data", "value"],
            "fallback": "return interaction method result",
        },
        required_candidate_symbols=[
            symbol
            for symbol in [input_agent, processor_agent, first_method]
            if symbol
        ],
        metadata={
            "adapter_status": adapter_status,
            "adapter_spec_version": 1,
            "source_task_id": source_task.task_id,
        },
    )


def _optional_str(value: Any) -> str | None:
    if value is None:
        return None
    return str(value)
