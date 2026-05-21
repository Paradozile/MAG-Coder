"""Tests for Full MAG-Bench functional oracle execution."""

from __future__ import annotations

from pathlib import Path

from magcoder.benchmark.full_runner import load_full_bench_run_cases, run_case_tests

ROOT = Path(__file__).resolve().parents[1]
BENCH = ROOT / "benchmarks/full_mag_bench_v1"


def test_seed_gold_solution_passes_case_tests() -> None:
    """A seed case's gold solution passes its generated functional tests."""

    case = load_full_bench_run_cases(BENCH, max_cases=1)[0]
    report = run_case_tests(case.case_dir, case.gold_solution_path)
    assert report.passed
    assert report.total > 0


def test_seed_broken_solution_fails_case_tests() -> None:
    """A seed case's broken solution fails at least one functional test."""

    case = load_full_bench_run_cases(BENCH, max_cases=1)[0]
    report = run_case_tests(case.case_dir, case.broken_solution_path)
    assert not report.passed
    assert report.failed_count > 0


def test_functional_evaluator_captures_syntax_error(tmp_path: Path) -> None:
    """Candidate syntax errors are reported as functional failures."""

    case_dir = _write_simple_case(tmp_path)
    bad_solution = tmp_path / "bad_syntax.py"
    bad_solution.write_text("def solve(:\n    return 2\n", encoding="utf-8")
    report = run_case_tests(case_dir, bad_solution)
    assert not report.passed
    assert report.error_type == "SyntaxError"


def test_functional_evaluator_captures_runtime_error(tmp_path: Path) -> None:
    """Runtime errors raised during test execution are captured."""

    case_dir = _write_simple_case(tmp_path)
    bad_solution = tmp_path / "bad_runtime.py"
    bad_solution.write_text(
        "def solve(value):\n    raise RuntimeError('boom')\n",
        encoding="utf-8",
    )
    report = run_case_tests(case_dir, bad_solution)
    assert not report.passed
    assert report.error_type == "RuntimeError"


def _write_simple_case(tmp_path: Path) -> Path:
    case_dir = tmp_path / "case"
    case_dir.mkdir()
    (case_dir / "tests.py").write_text(
        '''
def run_tests(module):
    results = []
    try:
        assert module.solve(1) == 2
    except Exception as error:
        results.append({
            "test_id": "simple",
            "passed": False,
            "error_type": type(error).__name__,
            "error_message": str(error),
            "source_assertion": "assert solve(1) == 2",
        })
    else:
        results.append({
            "test_id": "simple",
            "passed": True,
            "error_type": "",
            "error_message": "",
            "source_assertion": "assert solve(1) == 2",
        })
    return {
        "passed": all(item["passed"] for item in results),
        "total": len(results),
        "passed_count": sum(1 for item in results if item["passed"]),
        "failed_count": sum(1 for item in results if not item["passed"]),
        "results": results,
    }
'''.lstrip(),
        encoding="utf-8",
    )
    return case_dir
