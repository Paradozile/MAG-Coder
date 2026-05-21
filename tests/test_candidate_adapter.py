"""Tests for candidate source-entry adapter construction."""

from __future__ import annotations

import json
from pathlib import Path

from magcoder.benchmark.full_builder import EntryPointAdapterSpec
from magcoder.benchmark.full_runner.candidate_adapter import (
    build_candidate_with_entrypoint_adapter,
)
from magcoder.benchmark.full_runner.functional_evaluator import run_case_tests


def test_candidate_with_direct_entrypoint_does_not_get_wrapper() -> None:
    """Direct source-compatible functions are left untouched."""

    spec = _spec()
    candidate = "def solve(value):\n    return value + 1\n"
    adapted = build_candidate_with_entrypoint_adapter(candidate, spec)
    assert adapted.direct_entrypoint
    assert not adapted.adapter_used
    assert adapted.code == candidate


def test_candidate_missing_entrypoint_gets_thin_wrapper() -> None:
    """Candidates with required class/method symbols get a source wrapper."""

    spec = _spec()
    candidate = _candidate_with_symbols()
    adapted = build_candidate_with_entrypoint_adapter(candidate, spec)
    assert adapted.adapter_used
    assert adapted.adapter_success
    assert "def solve" in adapted.code
    assert "gold_solution" not in adapted.code


def test_missing_required_symbols_produces_construction_failure() -> None:
    """The adapter reports missing class/method symbols without guessing logic."""

    adapted = build_candidate_with_entrypoint_adapter(
        "class Other:\n    pass\n",
        _spec(),
    )
    assert not adapted.adapter_success
    assert adapted.error_type == "AdapterConstructionFailure"


def test_functional_evaluator_reports_adapter_mode(tmp_path: Path) -> None:
    """Functional evaluation can run through an adapter-generated entry point."""

    case_dir = _write_adapter_case(tmp_path)
    candidate = tmp_path / "candidate.py"
    candidate.write_text(_candidate_with_symbols(), encoding="utf-8")
    report = run_case_tests(case_dir, candidate)
    assert report.passed
    assert report.entrypoint_mode == "adapter"
    assert report.adapter_used
    assert report.adapter_success
    assert not report.direct_entrypoint_present
    assert report.required_candidate_symbols_present


def test_functional_evaluator_reports_adapter_construction_failure(
    tmp_path: Path,
) -> None:
    """Missing candidate symbols are classified before tests.py runs."""

    case_dir = _write_adapter_case(tmp_path)
    candidate = tmp_path / "candidate.py"
    candidate.write_text("class Other:\n    pass\n", encoding="utf-8")
    report = run_case_tests(case_dir, candidate)
    assert not report.passed
    assert report.error_type == "AdapterConstructionFailure"
    assert report.entrypoint_mode == "missing"
    assert not report.required_candidate_symbols_present


def _spec() -> EntryPointAdapterSpec:
    return EntryPointAdapterSpec(
        source_entry_point="solve",
        source_input_signature="value: int",
        source_output_type="int",
        objectified_entry_point="SourceAgent.interact_source_to_processor",
        candidate_agent_class="SourceAgent",
        candidate_interaction_method="interact_source_to_processor",
        adapter_strategy="thin_agent_interaction_wrapper",
        input_mapping={
            "source_agent_attribute": "input_data",
            "target_agent_class": "ProcessorAgent",
        },
        output_mapping={
            "preferred_target_attribute": "last_result",
            "fallback_return_keys": ["result"],
        },
        required_candidate_symbols=[
            "SourceAgent",
            "ProcessorAgent",
            "interact_source_to_processor",
        ],
        metadata={"adapter_status": "supported"},
    )


def _candidate_with_symbols() -> str:
    return """
class SourceAgent:
    def __init__(self, input_data=None):
        self.input_data = input_data

    def interact_source_to_processor(self, target):
        target.last_result = self.input_data["args"][0] + 1
        return {"result": target.last_result}


class ProcessorAgent:
    def __init__(self):
        self.last_result = None
""".lstrip()


def _write_adapter_case(tmp_path: Path) -> Path:
    case_dir = tmp_path / "case"
    case_dir.mkdir()
    (case_dir / "tests.json").write_text(
        json.dumps({"entry_point_adapter_spec": _spec().to_dict()}),
        encoding="utf-8",
    )
    (case_dir / "tests.py").write_text(
        """
ENTRY_POINT = "solve"


def run_tests(module):
    try:
        assert module.solve(1) == 2
    except Exception as error:
        return {
            "passed": False,
            "total": 1,
            "passed_count": 0,
            "failed_count": 1,
            "results": [{
                "test_id": "adapter",
                "passed": False,
                "error_type": type(error).__name__,
                "error_message": str(error),
                "source_assertion": "assert solve(1) == 2",
            }],
        }
    return {
        "passed": True,
        "total": 1,
        "passed_count": 1,
        "failed_count": 0,
        "results": [{
            "test_id": "adapter",
            "passed": True,
            "error_type": "",
            "error_message": "",
            "source_assertion": "assert solve(1) == 2",
        }],
    }
""".lstrip(),
        encoding="utf-8",
    )
    return case_dir
