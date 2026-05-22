"""ABMS-level oracle specifications for Full MAG-Bench cases."""

from __future__ import annotations

import ast
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from typing import Any

from magcoder.benchmark.full_builder.objectifier import ObjectifiedTask


@dataclass(slots=True)
class ABMSOracleSpec:
    """Agent-interaction-level oracle for a synthesized MAG case."""

    case_id: str
    agent_instances: list[dict[str, Any]]
    initial_state: dict[str, dict[str, Any]]
    interaction_sequence: list[dict[str, Any]]
    expected_messages: list[dict[str, Any]]
    expected_state_changes: list[dict[str, Any]]
    expected_return: dict[str, Any]
    expected_observations: dict[str, Any]
    tolerance: dict[str, Any]
    oracle_strategy: str
    required_candidate_symbols: list[str]
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Render this spec as JSON-compatible containers."""

        return asdict(self)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> ABMSOracleSpec:
        """Build a spec from parsed JSON metadata."""

        return cls(
            case_id=str(data["case_id"]),
            agent_instances=_dict_list(data.get("agent_instances", [])),
            initial_state=_nested_dict(data.get("initial_state", {})),
            interaction_sequence=_dict_list(data.get("interaction_sequence", [])),
            expected_messages=_dict_list(data.get("expected_messages", [])),
            expected_state_changes=_dict_list(
                data.get("expected_state_changes", []),
            ),
            expected_return=_dict(data.get("expected_return", {})),
            expected_observations=_dict(data.get("expected_observations", {})),
            tolerance=_dict(data.get("tolerance", {})),
            oracle_strategy=str(data.get("oracle_strategy", "")),
            required_candidate_symbols=[
                str(symbol)
                for symbol in data.get("required_candidate_symbols", [])
            ],
            metadata=_dict(data.get("metadata", {})),
        )


def build_abms_oracle_spec(objectified_task: ObjectifiedTask) -> ABMSOracleSpec:
    """Derive a conservative ABMS-level oracle from objectification metadata."""

    interface = objectified_task.class_interface_spec
    input_agent = str(interface.get("input_agent", ""))
    processor_agent = str(interface.get("processor_agent", ""))
    required_methods = interface.get("required_methods", [])
    interaction_method = (
        str(required_methods[0])
        if isinstance(required_methods, list) and required_methods
        else ""
    )
    message_api_name = str(interface.get("message_api_name", "send_message"))
    status = (
        "supported"
        if input_agent and processor_agent and interaction_method
        else "unsupported"
    )
    unsupported_reason = "" if status == "supported" else "missing_interface_fields"
    probe_payload = _json_safe(_probe_payload_from_tests(objectified_task))
    return ABMSOracleSpec(
        case_id=objectified_task.task_id,
        agent_instances=[
            {"id": "source", "class_name": input_agent, "role": "source"},
            {"id": "target", "class_name": processor_agent, "role": "target"},
        ],
        initial_state={
            "source": {
                "input_data": probe_payload,
            },
            "target": {"last_result": None},
        },
        interaction_sequence=[
            {
                "call_id": "primary_interaction",
                "caller": "source",
                "method": interaction_method,
                "args": ["target"],
            }
        ],
        expected_messages=[
            {
                "message_api_name": message_api_name,
                "sender": "source",
                "receiver": "target",
                "min_count": 1,
            }
        ],
        expected_state_changes=[
            {
                "agent": "target",
                "attribute": "last_result",
                "predicate": "not_none",
            }
        ],
        expected_return={"allow_any": True},
        expected_observations={
            "min_messages": 1,
            "target_last_result_not_none": True,
        },
        tolerance={},
        oracle_strategy="message_api_and_state_probe",
        required_candidate_symbols=[
            symbol
            for symbol in [input_agent, processor_agent, interaction_method]
            if symbol
        ],
        metadata={
            "abms_oracle_status": status,
            "unsupported_reason": unsupported_reason,
            "abms_oracle_spec_version": 1,
            "source_task_id": objectified_task.source_task_id,
            "probe_payload_kind": type(probe_payload).__name__,
        },
    )


def _probe_payload_from_tests(objectified_task: ObjectifiedTask) -> Any:
    if not objectified_task.transformed_tests:
        return "magbench_probe"
    assertion = str(objectified_task.transformed_tests[0].get("source_assertion", ""))
    entry_point = str(objectified_task.class_interface_spec.get("entry_point", ""))
    try:
        tree = ast.parse(assertion)
    except SyntaxError:
        return "magbench_probe"
    call = next(
        (
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == entry_point
        ),
        None,
    )
    if call is None:
        return "magbench_probe"
    try:
        args = [ast.literal_eval(arg) for arg in call.args]
        kwargs = {
            keyword.arg: ast.literal_eval(keyword.value)
            for keyword in call.keywords
            if keyword.arg is not None
        }
    except (ValueError, TypeError):
        return "magbench_probe"
    if kwargs:
        return {"args": args, "kwargs": kwargs}
    if len(args) == 1:
        return args[0]
    return args


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, complex):
        return {
            "__type__": "complex",
            "real": value.real,
            "imag": value.imag,
        }
    if isinstance(value, set):
        return {
            "__type__": "set",
            "items": [_json_safe(item) for item in sorted(value, key=repr)],
        }
    if isinstance(value, tuple):
        return [_json_safe(item) for item in value]
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    return repr(value)


def _dict(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        return {}
    return {str(key): item for key, item in value.items()}


def _nested_dict(value: Any) -> dict[str, dict[str, Any]]:
    if not isinstance(value, Mapping):
        return {}
    output: dict[str, dict[str, Any]] = {}
    for key, item in value.items():
        output[str(key)] = _dict(item)
    return output


def _dict_list(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    return [_dict(item) for item in value]
