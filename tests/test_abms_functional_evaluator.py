"""Tests for ABMS-level functional evaluation."""

from __future__ import annotations

from pathlib import Path

from magcoder.benchmark.full_builder import build_full_mag_bench
from magcoder.benchmark.full_runner import (
    load_full_bench_run_cases,
    run_abms_case_tests,
)

ROOT = Path(__file__).resolve().parents[1]
SOURCES = [
    ROOT / "examples/source_tasks/medium_humaneval_style_seed.jsonl",
    ROOT / "examples/source_tasks/medium_mbpp_style_seed.jsonl",
]


def test_gold_abms_solution_passes(tmp_path: Path) -> None:
    """Generated gold ABMS solution passes the ABMS oracle."""

    bench = _build_one_case(tmp_path)
    case = load_full_bench_run_cases(bench, max_cases=1)[0]
    report = run_abms_case_tests(case.case_dir, case.gold_abms_solution_path)
    assert report.abms_supported
    assert report.abms_pass
    assert report.interaction_calls_executed == 1
    assert report.observed_messages


def test_broken_abms_solution_fails(tmp_path: Path) -> None:
    """Generated broken ABMS solution fails the ABMS oracle."""

    bench = _build_one_case(tmp_path)
    case = load_full_bench_run_cases(bench, max_cases=1)[0]
    report = run_abms_case_tests(case.case_dir, case.broken_abms_solution_path)
    assert report.abms_supported
    assert not report.abms_pass
    assert report.abms_error_type == "ExpectedMessageMissing"


def test_abms_evaluator_reports_missing_agent_class(tmp_path: Path) -> None:
    """Missing candidate classes are structured ABMS failures."""

    bench = _build_one_case(tmp_path)
    case = load_full_bench_run_cases(bench, max_cases=1)[0]
    candidate = tmp_path / "missing_class.py"
    candidate.write_text("class OtherAgent:\n    pass\n", encoding="utf-8")
    report = run_abms_case_tests(case.case_dir, candidate)
    assert not report.abms_pass
    assert report.abms_error_type == "MissingRequiredCandidateSymbols"


def test_abms_evaluator_reports_wrong_state_or_message(tmp_path: Path) -> None:
    """A candidate with classes but no message API call fails ABMS checks."""

    bench = _build_one_case(tmp_path)
    case = load_full_bench_run_cases(bench, max_cases=1)[0]
    candidate = tmp_path / "wrong_message.py"
    candidate.write_text(
        case.broken_abms_solution_path.read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    report = run_abms_case_tests(case.case_dir, candidate)
    assert not report.abms_pass
    assert report.abms_error_type == "ExpectedMessageMissing"


def _build_one_case(tmp_path: Path) -> Path:
    bench = tmp_path / "bench"
    build_full_mag_bench(
        source_paths=SOURCES,
        output_dir=bench,
        max_cases=1,
        formats=["structured", "graphical"],
    )
    return bench
