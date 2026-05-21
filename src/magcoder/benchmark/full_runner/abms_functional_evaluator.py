"""ABMS-level functional evaluation for generated Full MAG-Bench code."""

from __future__ import annotations

import inspect
import json
import multiprocessing
import re
import traceback
from dataclasses import asdict, dataclass, field
from pathlib import Path
from types import MethodType, ModuleType
from typing import Any

from magcoder.benchmark.full_builder.abms_oracle_spec import ABMSOracleSpec
from magcoder.benchmark.full_runner.functional_evaluator import (
    load_python_module_from_path,
)


@dataclass(slots=True)
class ABMSFunctionalResult:
    """ABMS-level behavior report for one candidate."""

    case_id: str
    solution_path: str
    abms_supported: bool
    abms_pass: bool
    abms_error_type: str | None = None
    abms_error_message: str | None = None
    instantiated_agents: list[str] = field(default_factory=list)
    interaction_calls_executed: int = 0
    observed_messages: list[dict[str, Any]] = field(default_factory=list)
    observed_state_changes: dict[str, Any] = field(default_factory=dict)
    observed_return: Any = None
    expected_observations: dict[str, Any] = field(default_factory=dict)
    required_candidate_symbols_present: bool = False
    failing_stage: str | None = None
    failing_agent_class: str | None = None
    failing_interaction_method: str | None = None
    missing_symbol: str | None = None
    observed_signature: str | None = None
    expected_signature: str | None = None
    expected_message: dict[str, Any] | None = None
    traceback_excerpt: str | None = None
    root_cause_category: str | None = None
    failing_symbol: str | None = None
    failing_call_repr: str | None = None
    expected_call_signature: str | None = None
    observed_call_signature: str | None = None
    message_capture_mode: str | None = None
    candidate_send_message_signature: str | None = None
    candidate_defined_symbols: list[str] = field(default_factory=list)
    candidate_class_methods: dict[str, list[str]] = field(default_factory=dict)
    candidate_init_attributes: dict[str, list[str]] = field(default_factory=dict)
    safe_fix_hint: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Render this report as JSON-compatible containers."""

        return asdict(self)


def run_abms_case_tests(
    case_dir: str | Path,
    solution_path: str | Path,
    timeout_seconds: int = 5,
) -> ABMSFunctionalResult:
    """Run a case's ABMS oracle against a candidate solution in a child process."""

    case_path = Path(case_dir)
    solution = Path(solution_path)
    if not solution.exists():
        return _failure_result(
            case_path=case_path,
            solution_path=solution,
            error_type="CandidateArtifactMissing",
            error_message=f"Candidate file not found: {solution}",
        )
    ctx = multiprocessing.get_context("spawn")
    queue = ctx.Queue()
    process = ctx.Process(
        target=_run_abms_worker,
        args=(str(case_path), str(solution), queue),
    )
    process.start()
    process.join(timeout_seconds)
    if process.is_alive():
        process.terminate()
        process.join(1)
        return _failure_result(
            case_path=case_path,
            solution_path=solution,
            error_type="Timeout",
            error_message=f"ABMS tests exceeded {timeout_seconds} seconds.",
        )
    if queue.empty():
        return _failure_result(
            case_path=case_path,
            solution_path=solution,
            error_type="WorkerNoResult",
            error_message=f"ABMS worker exited with code {process.exitcode}.",
        )
    payload = queue.get()
    if not isinstance(payload, dict):
        return _failure_result(
            case_path=case_path,
            solution_path=solution,
            error_type="InvalidWorkerPayload",
            error_message="ABMS worker returned a non-object payload.",
        )
    return _result_from_dict(payload)


def run_abms_case_tests_from_module(
    case_dir: str | Path,
    module: ModuleType,
) -> ABMSFunctionalResult:
    """Run a case's ABMS oracle against an already-loaded candidate module."""

    case_path = Path(case_dir)
    spec = _load_abms_oracle_spec(case_path)
    return _evaluate_module(
        case_path=case_path,
        solution_path=Path(getattr(module, "__file__", "<module>")),
        module=module,
        spec=spec,
    )


def _run_abms_worker(case_dir: str, solution_path: str, queue: Any) -> None:
    try:
        case_path = Path(case_dir)
        solution = Path(solution_path)
        spec = _load_abms_oracle_spec(case_path)
        module = load_python_module_from_path(solution)
        queue.put(
            _evaluate_module(
                case_path=case_path,
                solution_path=solution,
                module=module,
                spec=spec,
            ).to_dict()
        )
    except BaseException as error:  # noqa: BLE001 - isolate candidate failures.
        queue.put(
            _failure_result(
                case_path=Path(case_dir),
                solution_path=Path(solution_path),
                error_type=type(error).__name__,
                error_message=str(error),
            ).to_dict()
        )


def _evaluate_module(
    *,
    case_path: Path,
    solution_path: Path,
    module: ModuleType,
    spec: ABMSOracleSpec | None,
) -> ABMSFunctionalResult:
    if spec is None:
        return _failure_result(
            case_path=case_path,
            solution_path=solution_path,
            error_type="ABMSOracleMissing",
            error_message="abms_oracle.json is missing or invalid.",
            supported=False,
        )
    status = str(spec.metadata.get("abms_oracle_status", "supported"))
    if status == "unsupported":
        return _failure_result(
            case_path=case_path,
            solution_path=solution_path,
            error_type="ABMSOracleUnsupported",
            error_message=str(spec.metadata.get("unsupported_reason", "")),
            supported=False,
        )
    symbol_report = _required_symbols_present(module, spec)
    if not symbol_report["present"]:
        root_cause = "undefined_symbol"
        return ABMSFunctionalResult(
            case_id=case_path.name,
            solution_path=str(solution_path),
            abms_supported=True,
            abms_pass=False,
            abms_error_type="MissingRequiredCandidateSymbols",
            abms_error_message=", ".join(symbol_report["missing"]),
            expected_observations=dict(spec.expected_observations),
            required_candidate_symbols_present=False,
            failing_stage="symbol_check",
            missing_symbol=", ".join(symbol_report["missing"]),
            root_cause_category=root_cause,
            failing_symbol=", ".join(symbol_report["missing"]),
            expected_call_signature=_expected_signature(spec),
            expected_message=_expected_message(spec),
            candidate_defined_symbols=_candidate_defined_symbols(module),
            candidate_class_methods=_candidate_class_methods(module),
            candidate_init_attributes=_candidate_init_attributes(module),
            candidate_send_message_signature=_candidate_send_message_signature(module),
            safe_fix_hint=_safe_fix_hint(root_cause),
            metadata={"required_symbols": spec.required_candidate_symbols},
        )

    failing_stage = "instantiate_agents"
    failing_agent_class: str | None = None
    failing_interaction_method: str | None = None
    observed_messages: list[dict[str, Any]] = []
    try:
        failing_agent_class = ";".join(
            str(agent.get("class_name", "")) for agent in spec.agent_instances
        )
        agents = _instantiate_agents(module, spec)
        failing_stage = "install_message_api"
        _install_framework_message_api(agents, spec, observed_messages)
        before_state = _snapshot_agent_state(agents)
        observed_return: Any = None
        executed = 0
        failing_stage = "run_interaction"
        for step in spec.interaction_sequence:
            caller_id = str(step.get("caller", ""))
            method_name = str(step.get("method", ""))
            failing_interaction_method = method_name
            caller = agents[caller_id]
            args = [agents[str(arg)] for arg in step.get("args", [])]
            method = getattr(caller, method_name)
            observed_return = method(*args)
            executed += 1
        after_state = _snapshot_agent_state(agents)
        state_changes = _state_changes(before_state, after_state)
        pass_checks = _check_abms_observations(
            spec=spec,
            observed_messages=observed_messages,
            state_changes=state_changes,
            executed=executed,
        )
        return ABMSFunctionalResult(
            case_id=case_path.name,
            solution_path=str(solution_path),
            abms_supported=True,
            abms_pass=pass_checks["passed"],
            abms_error_type=pass_checks["error_type"],
            abms_error_message=pass_checks["error_message"],
            instantiated_agents=sorted(agents),
            interaction_calls_executed=executed,
            observed_messages=observed_messages,
            observed_state_changes=state_changes,
            observed_return=_json_safe(observed_return),
            expected_observations=dict(spec.expected_observations),
            required_candidate_symbols_present=True,
            failing_stage=None if pass_checks["passed"] else "check_observations",
            failing_interaction_method=failing_interaction_method,
            expected_message=_expected_message(spec),
            message_capture_mode="framework_api_wrapper",
            candidate_send_message_signature=_candidate_send_message_signature(module),
            root_cause_category=(
                None
                if pass_checks["passed"]
                else _root_cause_category(
                    pass_checks["error_type"],
                    pass_checks["error_message"],
                )
            ),
            metadata={"oracle_strategy": spec.oracle_strategy},
        )
    except Exception as error:  # noqa: BLE001 - structured ABMS diagnostics.
        error_type = type(error).__name__
        error_message = str(error)
        root_cause = _root_cause_category(error_type, error_message)
        expected_signature = _expected_signature(spec)
        observed_signature = _observed_signature(error_message)
        return ABMSFunctionalResult(
            case_id=case_path.name,
            solution_path=str(solution_path),
            abms_supported=True,
            abms_pass=False,
            abms_error_type=error_type,
            abms_error_message=error_message,
            expected_observations=dict(spec.expected_observations),
            required_candidate_symbols_present=True,
            failing_stage=failing_stage,
            failing_agent_class=failing_agent_class,
            failing_interaction_method=failing_interaction_method,
            observed_messages=observed_messages,
            missing_symbol=_missing_symbol(error_type, error_message),
            observed_signature=observed_signature,
            expected_signature=expected_signature,
            expected_message=_expected_message(spec),
            traceback_excerpt=_traceback_excerpt(),
            root_cause_category=root_cause,
            failing_symbol=_missing_symbol(error_type, error_message),
            failing_call_repr=_failing_call_repr(failing_interaction_method),
            expected_call_signature=expected_signature,
            observed_call_signature=observed_signature,
            message_capture_mode="framework_api_wrapper",
            candidate_send_message_signature=_candidate_send_message_signature(module),
            candidate_defined_symbols=_candidate_defined_symbols(module),
            candidate_class_methods=_candidate_class_methods(module),
            candidate_init_attributes=_candidate_init_attributes(module),
            safe_fix_hint=_safe_fix_hint(root_cause),
            metadata={"oracle_strategy": spec.oracle_strategy},
        )


def _load_abms_oracle_spec(case_path: Path) -> ABMSOracleSpec | None:
    oracle_path = case_path / "abms_oracle.json"
    if not oracle_path.exists():
        return None
    try:
        with oracle_path.open(encoding="utf-8") as file:
            data = json.load(file)
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(data, dict):
        return None
    return ABMSOracleSpec.from_dict(data)


def _required_symbols_present(
    module: ModuleType,
    spec: ABMSOracleSpec,
) -> dict[str, Any]:
    missing: list[str] = []
    available = set(vars(module))
    for symbol in spec.required_candidate_symbols:
        if symbol in available:
            continue
        matching_step = next(
            (
                step
                for step in spec.interaction_sequence
                if symbol == str(step.get("method", ""))
            ),
            None,
        )
        if matching_step is not None:
            caller_class = _caller_class_name(
                spec,
                str(matching_step.get("caller", "")),
            )
            candidate_class = getattr(module, caller_class, None)
            if candidate_class is not None and hasattr(candidate_class, symbol):
                continue
        missing.append(symbol)
    return {"present": not missing, "missing": missing}


def _caller_class_name(spec: ABMSOracleSpec, caller_id: str) -> str:
    for agent in spec.agent_instances:
        if str(agent.get("id", "")) == caller_id:
            return str(agent.get("class_name", ""))
    return ""


def _instantiate_agents(
    module: ModuleType,
    spec: ABMSOracleSpec,
) -> dict[str, Any]:
    agents: dict[str, Any] = {}
    for agent in spec.agent_instances:
        agent_id = str(agent.get("id", ""))
        class_name = str(agent.get("class_name", ""))
        state = dict(spec.initial_state.get(agent_id, {}))
        class_object = getattr(module, class_name)
        try:
            instance = class_object(**state)
        except TypeError:
            instance = class_object()
            for key, value in state.items():
                setattr(instance, key, value)
        agents[agent_id] = instance
    return agents


def _install_framework_message_api(
    agents: dict[str, Any],
    spec: ABMSOracleSpec,
    observed_messages: list[dict[str, Any]],
) -> None:
    api_name = "send_message"
    if spec.expected_messages:
        api_name = str(spec.expected_messages[0].get("message_api_name", api_name))

    def _record_message(self: Any, target: Any, message: Any) -> dict[str, Any]:
        payload = {
            "sender": self.__class__.__name__,
            "receiver": target.__class__.__name__,
            "message": _json_safe(message),
        }
        observed_messages.append(payload)
        if not hasattr(self, "sent_messages"):
            self.sent_messages = []
        self.sent_messages.append(payload)
        if hasattr(target, "messages"):
            target.messages.append(message)
        target.last_result = message
        target.received_result = message
        return {"delivered": True, "message": message}

    def _record_local_message(self: Any, message: Any) -> dict[str, Any]:
        payload = {
            "sender": self.__class__.__name__,
            "receiver": self.__class__.__name__,
            "message": _json_safe(message),
            "capture_mode": "receiver_local_message_api",
        }
        observed_messages.append(payload)
        if not hasattr(self, "sent_messages"):
            self.sent_messages = []
        self.sent_messages.append(payload)
        if hasattr(self, "messages"):
            self.messages.append(message)
        self.last_result = message
        self.received_result = message
        return {"delivered": True, "message": message}

    for instance in agents.values():
        existing = getattr(instance, api_name, None)
        if callable(existing):
            wrapped = _wrap_message_api(
                existing,
                _record_message,
                _record_local_message,
                instance,
            )
            setattr(
                instance,
                api_name,
                wrapped,
            )
        else:
            compatible_api = _compatible_message_api(
                _record_message,
                _record_local_message,
            )
            setattr(
                instance,
                api_name,
                MethodType(compatible_api, instance),
            )


def _compatible_message_api(recorder: Any, local_recorder: Any) -> Any:
    def _send(
        self: Any,
        target: Any,
        message: Any = None,
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        if message is None:
            return local_recorder(self, target)
        return recorder(self, target, message)

    return _send


def _wrap_message_api(
    existing: Any,
    recorder: Any,
    local_recorder: Any,
    instance: Any,
) -> Any:
    def _wrapped(*call_args: Any, **kwargs: Any) -> Any:
        if len(call_args) >= 2:
            target, message = call_args[0], call_args[1]
            recorded = recorder(instance, target, message)
            try:
                return existing(*call_args, **kwargs)
            except TypeError:
                try:
                    return existing(message)
                except TypeError:
                    return recorded
        if len(call_args) == 1:
            message = call_args[0]
            recorded = local_recorder(instance, message)
            try:
                return existing(message, **kwargs)
            except TypeError:
                return recorded
        message = kwargs.get("message")
        recorded = local_recorder(instance, message)
        try:
            return existing(**kwargs)
        except TypeError:
            return recorded

    return _wrapped


def _snapshot_agent_state(agents: dict[str, Any]) -> dict[str, dict[str, Any]]:
    output: dict[str, dict[str, Any]] = {}
    for agent_id, instance in agents.items():
        output[agent_id] = {
            key: _json_safe(value)
            for key, value in vars(instance).items()
            if not key.startswith("_")
        }
    return output


def _state_changes(
    before: dict[str, dict[str, Any]],
    after: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    changes: dict[str, Any] = {}
    for agent_id, after_state in after.items():
        before_state = before.get(agent_id, {})
        for key, value in after_state.items():
            if before_state.get(key) != value:
                changes[f"{agent_id}.{key}"] = {
                    "before": before_state.get(key),
                    "after": value,
                }
    return changes


def _check_abms_observations(
    *,
    spec: ABMSOracleSpec,
    observed_messages: list[dict[str, Any]],
    state_changes: dict[str, Any],
    executed: int,
) -> dict[str, Any]:
    if executed < len(spec.interaction_sequence):
        return {
            "passed": False,
            "error_type": "InteractionSequenceIncomplete",
            "error_message": "Not all configured interactions executed.",
        }
    expected_min_messages = int(spec.expected_observations.get("min_messages", 0))
    if len(observed_messages) < expected_min_messages:
        return {
            "passed": False,
            "error_type": "ExpectedMessageMissing",
            "error_message": (
                f"Expected at least {expected_min_messages} message(s), "
                f"observed {len(observed_messages)}."
            ),
        }
    for expected_change in spec.expected_state_changes:
        if expected_change.get("predicate") != "not_none":
            continue
        key = f"{expected_change.get('agent')}.{expected_change.get('attribute')}"
        change = state_changes.get(key)
        if not change or change.get("after") is None:
            return {
                "passed": False,
                "error_type": "ExpectedStateChangeMissing",
                "error_message": f"Expected non-null state change for {key}.",
            }
    return {"passed": True, "error_type": None, "error_message": None}


def _failure_result(
    *,
    case_path: Path,
    solution_path: Path,
    error_type: str,
    error_message: str,
    supported: bool = True,
) -> ABMSFunctionalResult:
    root_cause = _root_cause_category(error_type, error_message)
    return ABMSFunctionalResult(
        case_id=case_path.name,
        solution_path=str(solution_path),
        abms_supported=supported,
        abms_pass=False,
        abms_error_type=error_type,
        abms_error_message=error_message,
        failing_stage="preflight",
        missing_symbol=_missing_symbol(error_type, error_message),
        root_cause_category=root_cause,
        failing_symbol=_missing_symbol(error_type, error_message),
        safe_fix_hint=_safe_fix_hint(root_cause),
    )


def _result_from_dict(data: dict[str, Any]) -> ABMSFunctionalResult:
    return ABMSFunctionalResult(
        case_id=str(data["case_id"]),
        solution_path=str(data["solution_path"]),
        abms_supported=bool(data["abms_supported"]),
        abms_pass=bool(data["abms_pass"]),
        abms_error_type=_optional_str(data.get("abms_error_type")),
        abms_error_message=_optional_str(data.get("abms_error_message")),
        instantiated_agents=[str(item) for item in data.get("instantiated_agents", [])],
        interaction_calls_executed=int(data.get("interaction_calls_executed", 0)),
        observed_messages=[
            dict(item)
            for item in data.get("observed_messages", [])
            if isinstance(item, dict)
        ],
        observed_state_changes=dict(data.get("observed_state_changes", {})),
        observed_return=data.get("observed_return"),
        expected_observations=dict(data.get("expected_observations", {})),
        required_candidate_symbols_present=bool(
            data.get("required_candidate_symbols_present", False)
        ),
        failing_stage=_optional_str(data.get("failing_stage")),
        failing_agent_class=_optional_str(data.get("failing_agent_class")),
        failing_interaction_method=_optional_str(
            data.get("failing_interaction_method")
        ),
        missing_symbol=_optional_str(data.get("missing_symbol")),
        observed_signature=_optional_str(data.get("observed_signature")),
        expected_signature=_optional_str(data.get("expected_signature")),
        expected_message=(
            dict(data["expected_message"])
            if isinstance(data.get("expected_message"), dict)
            else None
        ),
        traceback_excerpt=_optional_str(data.get("traceback_excerpt")),
        root_cause_category=_optional_str(data.get("root_cause_category")),
        failing_symbol=_optional_str(data.get("failing_symbol")),
        failing_call_repr=_optional_str(data.get("failing_call_repr")),
        expected_call_signature=_optional_str(data.get("expected_call_signature")),
        observed_call_signature=_optional_str(data.get("observed_call_signature")),
        message_capture_mode=_optional_str(data.get("message_capture_mode")),
        candidate_send_message_signature=_optional_str(
            data.get("candidate_send_message_signature")
        ),
        candidate_defined_symbols=[
            str(item) for item in data.get("candidate_defined_symbols", [])
        ],
        candidate_class_methods={
            str(key): [str(item) for item in value]
            for key, value in dict(data.get("candidate_class_methods", {})).items()
            if isinstance(value, list)
        },
        candidate_init_attributes={
            str(key): [str(item) for item in value]
            for key, value in dict(data.get("candidate_init_attributes", {})).items()
            if isinstance(value, list)
        },
        safe_fix_hint=_optional_str(data.get("safe_fix_hint")),
        metadata=dict(data.get("metadata", {})),
    )


def _json_safe(value: Any) -> Any:
    try:
        json.dumps(value)
    except TypeError:
        return repr(value)
    return value


def _optional_str(value: Any) -> str | None:
    if value is None:
        return None
    return str(value)


def _traceback_excerpt() -> str:
    return "\n".join(traceback.format_exc().splitlines()[-6:])


def _observed_signature(message: str) -> str | None:
    if "positional argument" in message or "required positional" in message:
        return message
    return None


def _missing_symbol(error_type: str | None, message: str | None) -> str | None:
    text = message or ""
    if error_type == "AttributeError":
        match = re.search(r"has no attribute '([^']+)'", text)
        if match:
            return match.group(1)
    if error_type == "NameError":
        match = re.search(r"name '([^']+)' is not defined", text)
        if match:
            return match.group(1)
    if error_type == "KeyError":
        stripped = text.strip("'\"")
        return stripped or None
    return None


def _failing_call_repr(method_name: str | None) -> str | None:
    if not method_name:
        return None
    return f"{method_name}(...)"


def _expected_message(spec: ABMSOracleSpec) -> dict[str, Any] | None:
    if not spec.expected_messages:
        return None
    return dict(spec.expected_messages[0])


def _expected_signature(spec: ABMSOracleSpec) -> str | None:
    if not spec.interaction_sequence:
        return None
    step = spec.interaction_sequence[0]
    args = ", ".join(["self", *[str(arg) for arg in step.get("args", [])]])
    return f"{step.get('method', '<method>')}({args})"


def _root_cause_category(error_type: str | None, message: str | None) -> str:
    text = f"{error_type or ''}: {message or ''}".lower()
    if "missingrequiredcandidatesymbols" in text:
        return "undefined_symbol"
    if "attributeerror" in text:
        if "'dict' object has no attribute" in text:
            return "missing_peer_attribute"
        if "send_message" in text:
            return "wrong_message_api_usage"
        if "self." in text:
            return "missing_self_attribute"
        if "has no attribute" in text:
            return "missing_attribute_initialization"
        return "missing_attribute_initialization"
    if "typeerror" in text:
        if "record_message" in text or "send_message" in text:
            return "wrong_send_message_signature"
        if "nonetype" in text:
            return "missing_attribute_initialization"
        if "missing" in text and "positional argument" in text:
            return "wrong_positional_argument_count"
        if "unexpected keyword" in text:
            return "wrong_keyword_argument_name"
        return "wrong_method_signature"
    if "keyerror" in text:
        return "missing_state_key"
    if "nameerror" in text:
        if "begin_python" in text or "end_python" in text:
            return "undefined_helper_function"
        return "undefined_symbol"
    if "recursionerror" in text:
        return "recursive_self_call"
    if "expectedmessagemissing" in text:
        return "expected_message_missing"
    if "expectedstatechangemissing" in text:
        return "missing_attribute_initialization"
    if "assertionerror" in text:
        return "behavior_assertion_failure"
    return "unknown"


def _candidate_defined_symbols(module: ModuleType) -> list[str]:
    return sorted(name for name in vars(module) if not name.startswith("__"))


def _candidate_class_methods(module: ModuleType) -> dict[str, list[str]]:
    output: dict[str, list[str]] = {}
    for name, value in vars(module).items():
        if isinstance(value, type):
            output[name] = sorted(
                attr
                for attr, member in vars(value).items()
                if callable(member) and not attr.startswith("__")
            )
    return output


def _candidate_init_attributes(module: ModuleType) -> dict[str, list[str]]:
    output: dict[str, list[str]] = {}
    for name, value in vars(module).items():
        if not isinstance(value, type):
            continue
        try:
            instance = value()
        except Exception:  # noqa: BLE001 - diagnostics must never fail evaluation.
            output[name] = []
            continue
        output[name] = sorted(
            attr for attr in vars(instance) if not attr.startswith("_")
        )
    return output


def _candidate_send_message_signature(module: ModuleType) -> str | None:
    signatures: list[str] = []
    for class_name, value in vars(module).items():
        if not isinstance(value, type):
            continue
        member = vars(value).get("send_message")
        if member is None:
            continue
        try:
            signature = str(inspect.signature(member))
        except (TypeError, ValueError):
            signature = "(unknown)"
        signatures.append(f"{class_name}.send_message{signature}")
    if not signatures:
        return None
    return "; ".join(sorted(signatures))


def _safe_fix_hint(root_cause: str) -> str:
    hints = {
        "wrong_send_message_signature": (
            "normalize send_message(target, message) compatibility"
        ),
        "expected_message_missing": "insert or bridge framework message API call",
        "missing_peer_attribute": "normalize message API target/message ordering",
        "missing_self_attribute": "initialize declared MAG attribute in __init__",
        "missing_attribute_initialization": "initialize declared MAG attributes",
        "wrong_positional_argument_count": (
            "normalize interaction/message method signature"
        ),
        "wrong_keyword_argument_name": "avoid unsupported keyword arguments",
        "wrong_method_signature": "normalize interaction method signature",
        "missing_state_key": "initialize state key before interaction",
        "undefined_helper_function": (
            "remove output-format markers or unsupported helper call"
        ),
        "undefined_symbol": (
            "resolve symbol from MAG class/attribute names or mark unsafe"
        ),
        "recursive_self_call": (
            "replace recursive wrapper with direct interaction wrapper"
        ),
    }
    return hints.get(root_cause, "manual triage required")
