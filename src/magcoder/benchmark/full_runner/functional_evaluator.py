"""Functional oracle execution for generated Full MAG-Bench cases."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import multiprocessing
import sys
from dataclasses import asdict, dataclass, field, is_dataclass
from pathlib import Path
from types import ModuleType
from typing import Any

from magcoder.benchmark.full_builder.adapter_spec import EntryPointAdapterSpec
from magcoder.benchmark.full_runner.candidate_adapter import (
    AdaptedCandidate,
    build_candidate_with_entrypoint_adapter,
)


@dataclass(slots=True)
class FunctionalTestResult:
    """One functional assertion result."""

    test_id: str
    passed: bool
    error_type: str = ""
    error_message: str = ""
    source_assertion: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Render this result as JSON-compatible containers."""

        return asdict(self)


@dataclass(slots=True)
class FunctionalEvaluationReport:
    """Functional test report for one candidate solution."""

    case_id: str
    solution_path: str
    passed: bool
    timed_out: bool
    total: int
    passed_count: int
    failed_count: int
    test_results: list[FunctionalTestResult]
    error_type: str | None = None
    error_message: str | None = None
    entrypoint_mode: str = "missing"
    adapter_used: bool = False
    adapter_success: bool = False
    adapter_error_type: str | None = None
    adapter_error_message: str | None = None
    direct_entrypoint_present: bool = False
    required_candidate_symbols_present: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Render this report as JSON-compatible containers."""

        return {
            "case_id": self.case_id,
            "solution_path": self.solution_path,
            "passed": self.passed,
            "timed_out": self.timed_out,
            "total": self.total,
            "passed_count": self.passed_count,
            "failed_count": self.failed_count,
            "test_results": [result.to_dict() for result in self.test_results],
            "error_type": self.error_type,
            "error_message": self.error_message,
            "entrypoint_mode": self.entrypoint_mode,
            "adapter_used": self.adapter_used,
            "adapter_success": self.adapter_success,
            "adapter_error_type": self.adapter_error_type,
            "adapter_error_message": self.adapter_error_message,
            "direct_entrypoint_present": self.direct_entrypoint_present,
            "required_candidate_symbols_present": (
                self.required_candidate_symbols_present
            ),
            "metadata": self.metadata,
        }


def load_python_module_from_path(path: str | Path) -> ModuleType:
    """Load a Python module from a file path using an isolated module name."""

    module_path = Path(path)
    digest = hashlib.sha256(str(module_path.resolve()).encode("utf-8")).hexdigest()
    module_name = f"magcoder_dynamic_{module_path.stem}_{digest[:12]}"
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load Python module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def load_python_module_from_source(source: str, module_name_hint: str) -> ModuleType:
    """Load a Python module from source text using an isolated module name."""

    digest = hashlib.sha256(source.encode("utf-8")).hexdigest()
    safe_hint = "".join(
        character if character.isalnum() or character == "_" else "_"
        for character in module_name_hint
    )
    module_name = f"magcoder_dynamic_{safe_hint}_{digest[:12]}"
    module = ModuleType(module_name)
    module.__file__ = f"<{module_name}>"
    sys.modules[module_name] = module
    exec(compile(source, module.__file__ or module_name, "exec"), module.__dict__)
    return module


def run_case_tests(
    case_dir: str | Path,
    solution_path: str | Path,
    timeout_seconds: int = 5,
) -> FunctionalEvaluationReport:
    """Run a case's tests.py against a candidate solution in a child process."""

    case_path = Path(case_dir)
    solution = Path(solution_path)
    candidate_source, adapter_result = _prepare_candidate_source(case_path, solution)
    if adapter_result is not None and not adapter_result.adapter_success:
        return _adapter_failure_report(case_path, solution, adapter_result)
    ctx = multiprocessing.get_context("spawn")
    queue = ctx.Queue()
    process = ctx.Process(
        target=_run_case_tests_worker,
        args=(str(case_path), str(solution), candidate_source, queue),
    )
    process.start()
    process.join(timeout_seconds)
    if process.is_alive():
        process.terminate()
        process.join(1)
        return FunctionalEvaluationReport(
            case_id=case_path.name,
            solution_path=str(solution),
            passed=False,
            timed_out=True,
            total=0,
            passed_count=0,
            failed_count=0,
            test_results=[],
            error_type="Timeout",
            error_message=f"Functional tests exceeded {timeout_seconds} seconds.",
            metadata=_functional_metadata(case_path, adapter_result),
            **_functional_adapter_fields(adapter_result),
        )
    if queue.empty():
        return FunctionalEvaluationReport(
            case_id=case_path.name,
            solution_path=str(solution),
            passed=False,
            timed_out=False,
            total=0,
            passed_count=0,
            failed_count=0,
            test_results=[],
            error_type="WorkerNoResult",
            error_message=(
                f"Functional test worker exited with code {process.exitcode}."
            ),
            metadata=_functional_metadata(case_path, adapter_result),
            **_functional_adapter_fields(adapter_result),
        )
    payload = queue.get()
    if not isinstance(payload, dict):
        return FunctionalEvaluationReport(
            case_id=case_path.name,
            solution_path=str(solution),
            passed=False,
            timed_out=False,
            total=0,
            passed_count=0,
            failed_count=0,
            test_results=[],
            error_type="InvalidWorkerPayload",
            error_message="Functional test worker returned a non-object payload.",
            metadata=_functional_metadata(case_path, adapter_result),
            **_functional_adapter_fields(adapter_result),
        )
    return _report_from_worker_payload(case_path, solution, payload, adapter_result)


def _run_case_tests_worker(
    case_dir: str,
    solution_path: str,
    candidate_source: str,
    queue: Any,
) -> None:
    try:
        tests_module = load_python_module_from_path(Path(case_dir) / "tests.py")
        solution_module = load_python_module_from_source(
            candidate_source,
            Path(solution_path).stem,
        )
        run_tests = getattr(tests_module, "run_tests", None)
        if not callable(run_tests):
            queue.put(
                {
                    "ok": False,
                    "error_type": "MissingRunTests",
                    "error_message": "tests.py does not define callable run_tests.",
                }
            )
            return
        raw_result = run_tests(solution_module)
        queue.put({"ok": True, "result": _normalize_raw_test_result(raw_result)})
    except BaseException as error:  # noqa: BLE001 - isolate candidate failures.
        queue.put(
            {
                "ok": False,
                "error_type": type(error).__name__,
                "error_message": str(error),
            }
        )


def _normalize_raw_test_result(raw_result: Any) -> dict[str, Any]:
    if hasattr(raw_result, "to_dict") and callable(raw_result.to_dict):
        data = raw_result.to_dict()
    elif is_dataclass(raw_result) and not isinstance(raw_result, type):
        data = asdict(raw_result)
    elif isinstance(raw_result, dict):
        data = raw_result
    else:
        raise TypeError(f"Unsupported test result type: {type(raw_result).__name__}")
    if not isinstance(data, dict):
        raise TypeError("Functional test result must normalize to a dict.")
    return data


def _report_from_worker_payload(
    case_path: Path,
    solution: Path,
    payload: dict[str, Any],
    adapter_result: AdaptedCandidate | None,
) -> FunctionalEvaluationReport:
    if not bool(payload.get("ok", False)):
        return FunctionalEvaluationReport(
            case_id=case_path.name,
            solution_path=str(solution),
            passed=False,
            timed_out=False,
            total=0,
            passed_count=0,
            failed_count=0,
            test_results=[],
            error_type=str(payload.get("error_type", "FunctionalError")),
            error_message=str(payload.get("error_message", "")),
            metadata=_functional_metadata(case_path, adapter_result),
            **_functional_adapter_fields(adapter_result),
        )
    result_data = payload.get("result", {})
    if not isinstance(result_data, dict):
        raise TypeError("Functional worker result must be an object.")
    raw_results = result_data.get("results", [])
    if not isinstance(raw_results, list):
        raise TypeError("Functional test results field must be a list.")
    test_results = [_functional_result_from_dict(item) for item in raw_results]
    total = int(result_data.get("total", len(test_results)))
    passed_count = int(
        result_data.get(
            "passed_count",
            sum(1 for result in test_results if result.passed),
        )
    )
    failed_count = int(result_data.get("failed_count", total - passed_count))
    passed = bool(result_data.get("passed", failed_count == 0))
    error_type = None
    error_message = None
    if not passed and test_results:
        first_failure = next((item for item in test_results if not item.passed), None)
        if first_failure is not None:
            error_type = first_failure.error_type or "FunctionalFailure"
            error_message = first_failure.error_message
            if (
                adapter_result is not None
                and adapter_result.adapter_used
                and error_type not in {"AssertionError", "MissingEntryPoint"}
            ):
                error_type = "AdapterRuntimeFailure"
    return FunctionalEvaluationReport(
        case_id=case_path.name,
        solution_path=str(solution),
        passed=passed,
        timed_out=False,
        total=total,
        passed_count=passed_count,
        failed_count=failed_count,
        test_results=test_results,
        error_type=error_type,
        error_message=error_message,
        metadata=_functional_metadata(case_path, adapter_result),
        adapter_error_type=(
            error_type
            if adapter_result is not None
            and adapter_result.adapter_used
            and error_type == "AdapterRuntimeFailure"
            else None
        ),
        adapter_error_message=(
            error_message
            if adapter_result is not None
            and adapter_result.adapter_used
            and error_type == "AdapterRuntimeFailure"
            else None
        ),
        **_functional_adapter_fields(adapter_result),
    )


def _functional_result_from_dict(data: Any) -> FunctionalTestResult:
    if not isinstance(data, dict):
        raise TypeError("Each functional test result must be an object.")
    metadata = data.get("metadata", {})
    if not isinstance(metadata, dict):
        metadata = {}
    return FunctionalTestResult(
        test_id=str(data.get("test_id", "")),
        passed=bool(data.get("passed", False)),
        error_type=str(data.get("error_type", "")),
        error_message=str(data.get("error_message", "")),
        source_assertion=str(data.get("source_assertion", "")),
        metadata=dict(metadata),
    )


def _prepare_candidate_source(
    case_path: Path,
    solution: Path,
) -> tuple[str, AdaptedCandidate | None]:
    candidate_source = solution.read_text(encoding="utf-8")
    adapter_spec = _load_adapter_spec(case_path)
    if adapter_spec is None:
        return candidate_source, None
    status = str(adapter_spec.metadata.get("adapter_status", "supported"))
    if status == "unsupported":
        return candidate_source, None
    adapter_result = build_candidate_with_entrypoint_adapter(
        candidate_source,
        adapter_spec,
    )
    return adapter_result.code, adapter_result


def _load_adapter_spec(case_path: Path) -> EntryPointAdapterSpec | None:
    tests_json = case_path / "tests.json"
    if not tests_json.exists():
        return None
    try:
        with tests_json.open(encoding="utf-8") as file:
            data = json.load(file)
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(data, dict):
        return None
    raw_spec = data.get("entry_point_adapter_spec")
    if not isinstance(raw_spec, dict):
        return None
    return EntryPointAdapterSpec.from_dict(raw_spec)


def _adapter_failure_report(
    case_path: Path,
    solution: Path,
    adapter_result: AdaptedCandidate,
) -> FunctionalEvaluationReport:
    error_type = adapter_result.error_type or "AdapterConstructionFailure"
    return FunctionalEvaluationReport(
        case_id=case_path.name,
        solution_path=str(solution),
        passed=False,
        timed_out=False,
        total=0,
        passed_count=0,
        failed_count=0,
        test_results=[],
        error_type=error_type,
        error_message=adapter_result.error_message,
        metadata=_functional_metadata(case_path, adapter_result),
        adapter_error_type=error_type,
        adapter_error_message=adapter_result.error_message,
        **_functional_adapter_fields(adapter_result),
    )


def _functional_metadata(
    case_path: Path,
    adapter_result: AdaptedCandidate | None,
) -> dict[str, Any]:
    metadata: dict[str, Any] = {"tests_path": str(case_path / "tests.py")}
    if adapter_result is not None:
        metadata["adapter"] = adapter_result.to_dict()
    return metadata


def _functional_adapter_fields(
    adapter_result: AdaptedCandidate | None,
) -> dict[str, Any]:
    if adapter_result is None:
        return {
            "entrypoint_mode": "missing",
            "adapter_used": False,
            "adapter_success": False,
            "direct_entrypoint_present": False,
            "required_candidate_symbols_present": False,
        }
    entrypoint_mode = str(adapter_result.metadata.get("entrypoint_mode", "missing"))
    return {
        "entrypoint_mode": entrypoint_mode,
        "adapter_used": adapter_result.adapter_used,
        "adapter_success": adapter_result.adapter_success,
        "direct_entrypoint_present": adapter_result.direct_entrypoint,
        "required_candidate_symbols_present": (
            adapter_result.required_candidate_symbols_present
        ),
    }
