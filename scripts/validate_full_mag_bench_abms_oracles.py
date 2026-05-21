"""Validate Full MAG-Bench ABMS-level gold/broken oracles."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from magcoder.benchmark.full_runner import (
    load_full_bench_run_cases,
    run_abms_case_tests,
)


def main() -> None:
    """CLI entry point."""

    parser = argparse.ArgumentParser(
        description=(
            "Validate gold_abms_solution.py and broken_abms_solution.py for each "
            "Full MAG-Bench case."
        ),
    )
    parser.add_argument("--bench", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--timeout-seconds", type=int, default=5)
    args = parser.parse_args()
    report = validate_abms_oracles(args.bench, args.timeout_seconds)
    _save_report(report, args.output)
    print(f"ABMS oracle validation passed: {report['passed']}")
    print(f"ABMS oracle validation JSON: {args.output}")
    print(f"ABMS oracle validation Markdown: {args.output.with_suffix('.md')}")
    if not report["passed"]:
        raise SystemExit(1)


def validate_abms_oracles(
    bench_path: str | Path,
    timeout_seconds: int = 5,
) -> dict[str, Any]:
    """Validate ABMS gold passes and ABMS broken fails for all supported cases."""

    cases = load_full_bench_run_cases(bench_path)
    rows: list[dict[str, Any]] = []
    for case in cases:
        gold = run_abms_case_tests(
            case.case_dir,
            case.gold_abms_solution_path,
            timeout_seconds=timeout_seconds,
        )
        broken = run_abms_case_tests(
            case.case_dir,
            case.broken_abms_solution_path,
            timeout_seconds=timeout_seconds,
        )
        supported = bool(gold.abms_supported and broken.abms_supported)
        rows.append(
            {
                "case_id": case.case_id,
                "abms_supported": supported,
                "unsupported_reason": _unsupported_reason(gold, broken),
                "gold_abms_passed": gold.abms_pass,
                "broken_abms_failed": not broken.abms_pass,
                "gold_error_type": gold.abms_error_type,
                "gold_error_message": gold.abms_error_message,
                "broken_error_type": broken.abms_error_type,
                "broken_error_message": broken.abms_error_message,
                "gold_report": gold.to_dict(),
                "broken_report": broken.to_dict(),
            }
        )
    supported_rows = [row for row in rows if row["abms_supported"]]
    unsupported_rows = [row for row in rows if not row["abms_supported"]]
    passed = bool(supported_rows) and all(
        row["gold_abms_passed"] and row["broken_abms_failed"]
        for row in supported_rows
    )
    return {
        "bench_path": str(bench_path),
        "case_count": len(rows),
        "passed": passed,
        "supported_cases": len(supported_rows),
        "unsupported_cases": len(unsupported_rows),
        "unsupported_reasons": _reason_counts(unsupported_rows),
        "gold_abms_pass_count": sum(
            1 for row in supported_rows if row["gold_abms_passed"]
        ),
        "broken_abms_fail_count": sum(
            1 for row in supported_rows if row["broken_abms_failed"]
        ),
        "cases": rows,
    }


def _unsupported_reason(gold: Any, broken: Any) -> str:
    if gold.abms_supported and broken.abms_supported:
        return ""
    return gold.abms_error_type or broken.abms_error_type or "unsupported"


def _reason_counts(rows: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        reason = str(row.get("unsupported_reason") or "unknown")
        counts[reason] = counts.get(reason, 0) + 1
    return counts


def _save_report(report: dict[str, Any], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file:
        json.dump(report, file, indent=2, sort_keys=True)
        file.write("\n")
    lines = [
        "# Full MAG-Bench ABMS Oracle Validation Report",
        "",
        f"- Cases: {report['case_count']}",
        f"- Passed: {report['passed']}",
        f"- Supported cases: {report['supported_cases']}",
        f"- Unsupported cases: {report['unsupported_cases']}",
        f"- Gold ABMS passed: {report['gold_abms_pass_count']}",
        f"- Broken ABMS failed: {report['broken_abms_fail_count']}",
        "",
        "| case_id | supported | gold_abms_passed | broken_abms_failed | errors |",
        "| --- | --- | --- | --- | --- |",
    ]
    for row in report["cases"]:
        errors = "; ".join(
            item
            for item in [
                row.get("unsupported_reason", ""),
                row.get("gold_error_type", ""),
                row.get("broken_error_type", ""),
            ]
            if item
        )
        lines.append(
            "| {case_id} | {abms_supported} | {gold_abms_passed} | "
            "{broken_abms_failed} | {errors} |".format(
                case_id=row["case_id"],
                abms_supported=row["abms_supported"],
                gold_abms_passed=row["gold_abms_passed"],
                broken_abms_failed=row["broken_abms_failed"],
                errors=errors,
            )
        )
    output_path.with_suffix(".md").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
