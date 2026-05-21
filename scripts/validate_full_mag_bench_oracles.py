"""Validate Full MAG-Bench functional gold/broken oracles."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from magcoder.benchmark.full_runner import load_full_bench_run_cases, run_case_tests


def main() -> None:
    """CLI entry point."""

    parser = argparse.ArgumentParser(
        description="Validate gold_solution.py and broken_solution.py for each case.",
    )
    parser.add_argument("--bench", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--timeout-seconds", type=int, default=5)
    args = parser.parse_args()
    report = validate_oracles(args.bench, args.timeout_seconds)
    _save_report(report, args.output)
    print(f"Oracle validation passed: {report['passed']}")
    print(f"Oracle validation JSON: {args.output}")
    print(f"Oracle validation Markdown: {args.output.with_suffix('.md')}")
    if not report["passed"]:
        raise SystemExit(1)


def validate_oracles(
    bench_path: str | Path,
    timeout_seconds: int = 5,
) -> dict[str, Any]:
    """Validate that gold passes and broken fails for all Full MAG-Bench cases."""

    cases = load_full_bench_run_cases(bench_path)
    rows: list[dict[str, Any]] = []
    for case in cases:
        gold = run_case_tests(
            case.case_dir,
            case.gold_solution_path,
            timeout_seconds=timeout_seconds,
        )
        broken = run_case_tests(
            case.case_dir,
            case.broken_solution_path,
            timeout_seconds=timeout_seconds,
        )
        rows.append(
            {
                "case_id": case.case_id,
                "gold_passed": gold.passed,
                "broken_failed": not broken.passed,
                "gold_error_type": gold.error_type,
                "gold_error_message": gold.error_message,
                "broken_error_type": broken.error_type,
                "broken_error_message": broken.error_message,
                "gold_report": gold.to_dict(),
                "broken_report": broken.to_dict(),
            }
        )
    passed = all(row["gold_passed"] and row["broken_failed"] for row in rows)
    return {
        "bench_path": str(bench_path),
        "case_count": len(rows),
        "passed": passed,
        "gold_passed_count": sum(1 for row in rows if row["gold_passed"]),
        "broken_failed_count": sum(1 for row in rows if row["broken_failed"]),
        "cases": rows,
    }


def _save_report(report: dict[str, Any], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file:
        json.dump(report, file, indent=2, sort_keys=True)
        file.write("\n")
    lines = [
        "# Full MAG-Bench Oracle Validation Report",
        "",
        f"- Cases: {report['case_count']}",
        f"- Passed: {report['passed']}",
        f"- Gold passed: {report['gold_passed_count']}",
        f"- Broken failed: {report['broken_failed_count']}",
        "",
        "| case_id | gold_passed | broken_failed | gold_error | broken_error |",
        "| --- | --- | --- | --- | --- |",
    ]
    for row in report["cases"]:
        lines.append(
            "| {case_id} | {gold_passed} | {broken_failed} | {gold_error_type} | "
            "{broken_error_type} |".format(**row)
        )
    output_path.with_suffix(".md").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
