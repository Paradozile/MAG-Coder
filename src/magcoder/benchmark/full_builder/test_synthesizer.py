"""Synthesize tests and contract metadata for full MAG-Bench cases."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from magcoder.benchmark.full_builder.abms_oracle_spec import (
    ABMSOracleSpec,
    build_abms_oracle_spec,
)
from magcoder.benchmark.full_builder.adapter_spec import (
    EntryPointAdapterSpec,
    build_entry_point_adapter_spec,
)
from magcoder.benchmark.full_builder.objectifier import ObjectifiedTask
from magcoder.benchmark.full_builder.source_task import SourceTask
from magcoder.models import MAGModel


@dataclass(slots=True)
class SynthesizedTestBundle:
    """Tests and contract metadata for one benchmark case."""

    case_id: str
    tests: list[dict[str, Any]]
    expected_contract: dict[str, Any]
    framework_ref: str
    entry_point_adapter_spec: EntryPointAdapterSpec
    abms_oracle_spec: ABMSOracleSpec
    gold_solution: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Render this bundle as JSON-compatible containers."""

        return asdict(self)


def synthesize_tests(
    *,
    source_task: SourceTask,
    objectified_task: ObjectifiedTask,
    mag: MAGModel,
    framework_ref: str,
) -> SynthesizedTestBundle:
    """Build contract-oriented tests for one synthesized MAG case."""

    required_methods = [
        str(edge.get("required_method_name"))
        for edge in objectified_task.interaction_edges
        if edge.get("required_method_name") is not None
    ]
    adapter_spec = build_entry_point_adapter_spec(source_task, objectified_task)
    abms_oracle_spec = build_abms_oracle_spec(objectified_task)
    return SynthesizedTestBundle(
        case_id=objectified_task.task_id,
        tests=list(objectified_task.transformed_tests),
        expected_contract={
            "agents": [agent.id for agent in mag.agents],
            "edges": [edge.id for edge in mag.edges],
            "filters": [attribute_filter.id for attribute_filter in mag.filters],
            "required_methods": required_methods,
            "message_api_name": "send_message",
        },
        framework_ref=framework_ref,
        entry_point_adapter_spec=adapter_spec,
        abms_oracle_spec=abms_oracle_spec,
        gold_solution=source_task.canonical_solution,
        metadata={
            "source_task_id": source_task.task_id,
            "entry_point": source_task.entry_point,
            "test_synthesizer_version": 1,
        },
    )


def save_tests_json(bundle: SynthesizedTestBundle, path: str | Path) -> None:
    """Save synthesized tests as JSON."""

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file:
        json.dump(bundle.to_dict(), file, indent=2, sort_keys=True)
        file.write("\n")


def save_tests_py(bundle: SynthesizedTestBundle, path: str | Path) -> None:
    """Save an executable pytest contract stub for benchmark consumers."""

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    entry_point = str(bundle.metadata.get("entry_point") or "solve")
    content = f'''"""Functional oracle tests for MAG-Bench case {bundle.case_id}."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from types import ModuleType
from typing import Any

ENTRY_POINT = {entry_point!r}
EXPECTED_CONTRACT = {bundle.expected_contract!r}
SOURCE_TESTS = {bundle.tests!r}


@dataclass(slots=True)
class TestRunResult:
    """Functional test result returned to the MAG-Bench runner."""

    passed: bool
    total: int
    passed_count: int
    failed_count: int
    results: list[dict[str, Any]]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def run_tests(module: ModuleType) -> TestRunResult:
    """Run source assertions against a candidate module."""

    namespace = dict(vars(module))
    results: list[dict[str, Any]] = []
    if ENTRY_POINT not in namespace:
        return TestRunResult(
            passed=False,
            total=len(SOURCE_TESTS),
            passed_count=0,
            failed_count=len(SOURCE_TESTS),
            results=[
                {{
                    "test_id": item["test_id"],
                    "passed": False,
                    "error_type": "MissingEntryPoint",
                    "error_message": f"Missing entry point {{ENTRY_POINT}}",
                    "source_assertion": item["source_assertion"],
                }}
                for item in SOURCE_TESTS
            ],
        )
    for item in SOURCE_TESTS:
        assertion = item["source_assertion"]
        try:
            namespace.setdefault("__builtins__", __builtins__)
            exec(assertion, namespace, namespace)
        except AssertionError as error:
            results.append(
                {{
                    "test_id": item["test_id"],
                    "passed": False,
                    "error_type": "AssertionError",
                    "error_message": str(error),
                    "source_assertion": assertion,
                }}
            )
        except Exception as error:  # noqa: BLE001 - report oracle failures.
            results.append(
                {{
                    "test_id": item["test_id"],
                    "passed": False,
                    "error_type": type(error).__name__,
                    "error_message": str(error),
                    "source_assertion": assertion,
                }}
            )
        else:
            results.append(
                {{
                    "test_id": item["test_id"],
                    "passed": True,
                    "error_type": "",
                    "error_message": "",
                    "source_assertion": assertion,
                }}
            )
    passed_count = sum(1 for item in results if item["passed"])
    failed_count = len(results) - passed_count
    return TestRunResult(
        passed=failed_count == 0,
        total=len(results),
        passed_count=passed_count,
        failed_count=failed_count,
        results=results,
    )


def test_gold_or_candidate_module_contract_placeholder():
    assert EXPECTED_CONTRACT["agents"]
    assert EXPECTED_CONTRACT["required_methods"]
'''
    output_path.write_text(content, encoding="utf-8")


def save_abms_oracle_json(bundle: SynthesizedTestBundle, path: str | Path) -> None:
    """Save ABMS-level oracle metadata as JSON."""

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file:
        json.dump(bundle.abms_oracle_spec.to_dict(), file, indent=2, sort_keys=True)
        file.write("\n")


def save_abms_tests_py(bundle: SynthesizedTestBundle, path: str | Path) -> None:
    """Save a lightweight executable ABMS oracle fixture."""

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    content = f'''"""ABMS-level oracle fixture for MAG-Bench case {bundle.case_id}."""

from __future__ import annotations

from pathlib import Path
from types import ModuleType

from magcoder.benchmark.full_runner.abms_functional_evaluator import (
    run_abms_case_tests_from_module,
)


def run_abms_tests(module: ModuleType):
    """Run the ABMS-level oracle against an already-loaded module."""

    return run_abms_case_tests_from_module(Path(__file__).parent, module)
'''
    output_path.write_text(content, encoding="utf-8")


def save_gold_solution_py(bundle: SynthesizedTestBundle, path: str | Path) -> None:
    """Save the canonical solution as the executable gold oracle."""

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    entry_point = str(bundle.metadata.get("entry_point") or "solve")
    if bundle.gold_solution is None or not bundle.gold_solution.strip():
        content = (
            f"def {entry_point}(*args, **kwargs):\n"
            "    raise NotImplementedError\n"
        )
    else:
        content = bundle.gold_solution.rstrip() + "\n"
    output_path.write_text(content, encoding="utf-8")


def save_broken_solution_py(bundle: SynthesizedTestBundle, path: str | Path) -> None:
    """Save a deterministic negative oracle solution."""

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    entry_point = str(bundle.metadata.get("entry_point") or "solve")
    content = (
        f"def {entry_point}(*args, **kwargs):\n"
        "    # Deterministically wrong baseline for oracle validation.\n"
        "    return None\n"
    )
    output_path.write_text(content, encoding="utf-8")


def save_gold_abms_solution_py(
    bundle: SynthesizedTestBundle,
    path: str | Path,
) -> None:
    """Save a positive ABMS-level agent-interaction oracle solution."""

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    spec = bundle.abms_oracle_spec
    source_class = _agent_class(spec, "source")
    target_class = _agent_class(spec, "target")
    method_name = _interaction_method(spec)
    message_api_name = _message_api_name(spec)
    content = f'''class {target_class}:
    def __init__(self, last_result=None, **kwargs):
        self.last_result = last_result
        self.messages = []
        for key, value in kwargs.items():
            setattr(self, key, value)


class {source_class}:
    def __init__(self, input_data=None, **kwargs):
        self.input_data = input_data
        self.messages = []
        for key, value in kwargs.items():
            setattr(self, key, value)

    def {message_api_name}(self, target, message):
        target.last_result = message
        if hasattr(target, "messages"):
            target.messages.append(message)
        self.messages.append(message)
        return {{"delivered": True, "message": message}}

    def {method_name}(self, target):
        payload = {{"message": "interaction_triggered", "input_data": self.input_data}}
        return self.{message_api_name}(target, payload)
'''
    output_path.write_text(content, encoding="utf-8")


def save_broken_abms_solution_py(
    bundle: SynthesizedTestBundle,
    path: str | Path,
) -> None:
    """Save a deterministic ABMS-level negative oracle solution."""

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    spec = bundle.abms_oracle_spec
    source_class = _agent_class(spec, "source")
    target_class = _agent_class(spec, "target")
    method_name = _interaction_method(spec)
    content = f'''class {target_class}:
    def __init__(self, last_result=None, **kwargs):
        self.last_result = last_result
        self.messages = []
        for key, value in kwargs.items():
            setattr(self, key, value)


class {source_class}:
    def __init__(self, input_data=None, **kwargs):
        self.input_data = input_data
        for key, value in kwargs.items():
            setattr(self, key, value)

    def {method_name}(self, target):
        # Deterministically wrong ABMS baseline: no message API call.
        return None
'''
    output_path.write_text(content, encoding="utf-8")


def _agent_class(spec: ABMSOracleSpec, role: str) -> str:
    for item in spec.agent_instances:
        if item.get("role") == role or item.get("id") == role:
            return str(item["class_name"])
    return "UnsupportedAgent"


def _interaction_method(spec: ABMSOracleSpec) -> str:
    if not spec.interaction_sequence:
        return "interact_unsupported"
    return str(spec.interaction_sequence[0].get("method", "interact_unsupported"))


def _message_api_name(spec: ABMSOracleSpec) -> str:
    if spec.expected_messages:
        return str(spec.expected_messages[0].get("message_api_name", "send_message"))
    return "send_message"
