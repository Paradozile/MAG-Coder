"""Validate Full MAG-Bench source-entry adapter specifications."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from magcoder.benchmark.full_runner import (
    load_full_bench_run_cases,
    load_python_module_from_path,
)


def main() -> None:
    """CLI entry point."""

    parser = argparse.ArgumentParser(
        description="Validate entry_point_adapter_spec metadata for Full MAG-Bench.",
    )
    parser.add_argument("--bench", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = validate_adapters(args.bench)
    _save_report(report, args.output)
    print(f"Adapter validation passed: {report['passed']}")
    print(f"Adapter validation JSON: {args.output}")
    print(f"Adapter validation Markdown: {args.output.with_suffix('.md')}")
    if not report["passed"]:
        raise SystemExit(1)


def validate_adapters(bench_path: str | Path) -> dict[str, Any]:
    """Validate that each accepted case exposes a usable adapter spec."""

    rows: list[dict[str, Any]] = []
    for case in load_full_bench_run_cases(bench_path):
        row = _validate_case_adapter(case)
        rows.append(row)
    passed = all(row["passed"] for row in rows)
    return {
        "bench_path": str(bench_path),
        "case_count": len(rows),
        "passed": passed,
        "supported_count": sum(1 for row in rows if row["adapter_supported"]),
        "unsupported_count": sum(
            1 for row in rows if not row["adapter_supported"]
        ),
        "cases": rows,
    }


def _validate_case_adapter(case: Any) -> dict[str, Any]:
    errors: list[str] = []
    spec = case.entry_point_adapter_spec
    if spec is None:
        errors.append("missing_entry_point_adapter_spec")
        return {
            "case_id": case.case_id,
            "passed": False,
            "adapter_supported": False,
            "source_entry_point": "",
            "candidate_agent_class": "",
            "candidate_interaction_method": "",
            "tests_entry_point": "",
            "errors": errors,
        }
    tests_entry_point = _load_tests_entry_point(case.tests_py_path)
    adapter_status = str(spec.metadata.get("adapter_status", "supported"))
    adapter_supported = adapter_status == "supported"
    if not spec.source_entry_point:
        errors.append("missing_source_entry_point")
    if tests_entry_point != spec.source_entry_point:
        errors.append("tests_entry_point_mismatch")
    if adapter_supported and not spec.candidate_agent_class:
        errors.append("missing_candidate_agent_class")
    if adapter_supported and not spec.candidate_interaction_method:
        errors.append("missing_candidate_interaction_method")
    if adapter_supported and not spec.required_candidate_symbols:
        errors.append("missing_required_candidate_symbols")
    return {
        "case_id": case.case_id,
        "passed": not errors,
        "adapter_supported": adapter_supported,
        "adapter_status": adapter_status,
        "source_entry_point": spec.source_entry_point,
        "candidate_agent_class": spec.candidate_agent_class,
        "candidate_interaction_method": spec.candidate_interaction_method,
        "objectified_entry_point": spec.objectified_entry_point,
        "required_candidate_symbols": spec.required_candidate_symbols,
        "tests_entry_point": tests_entry_point,
        "errors": errors,
    }


def _load_tests_entry_point(tests_py_path: Path) -> str:
    try:
        module = load_python_module_from_path(tests_py_path)
    except Exception:  # noqa: BLE001 - validation will report mismatch.
        return ""
    return str(getattr(module, "ENTRY_POINT", ""))


def _save_report(report: dict[str, Any], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file:
        json.dump(report, file, indent=2, sort_keys=True)
        file.write("\n")
    lines = [
        "# Full MAG-Bench Adapter Validation Report",
        "",
        f"- Cases: {report['case_count']}",
        f"- Passed: {report['passed']}",
        f"- Supported adapters: {report['supported_count']}",
        f"- Unsupported adapters: {report['unsupported_count']}",
        "",
        "| case_id | supported | source_entry_point | candidate_agent_class | "
        "candidate_interaction_method | errors |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for row in report["cases"]:
        row_errors = ", ".join(row["errors"])
        lines.append(
            "| {case_id} | {supported} | {source_entry_point} | "
            "{agent_class} | {method_name} | {errors} |".format(
                case_id=row["case_id"],
                supported=row["adapter_supported"],
                source_entry_point=row["source_entry_point"],
                agent_class=row["candidate_agent_class"],
                method_name=row["candidate_interaction_method"],
                errors=row_errors,
            )
        )
    output_path.with_suffix(".md").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
