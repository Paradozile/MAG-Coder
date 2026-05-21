"""Export MAG-Bench-Func functional-only views from existing run outputs."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

GROUP_FIELDS = [
    "overall",
    "method",
    "mode",
    "source",
    "task_type",
    "data_type",
    "template_family",
]


def main() -> None:
    """CLI entry point."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bench", type=Path, required=True)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = export_mag_bench_func_views(
        bench=args.bench,
        run=args.run,
        output_dir=args.output,
    )
    print(f"MAG-Bench-Func output: {args.output}")
    print(f"Functional-only samples: {report['total_samples']}")


def export_mag_bench_func_views(
    *,
    bench: str | Path,
    run: str | Path,
    output_dir: str | Path,
) -> dict[str, Any]:
    """Build functional-only summary files from an existing run."""

    bench_path = Path(bench)
    run_path = Path(run)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    rows = _read_csv(run_path / "sample_summary.csv")
    summary = _summary_rows(rows)
    comparison = _mag_bench_vs_func(rows)
    report = {
        "schema_version": 1,
        "bench": str(bench_path),
        "run": str(run_path),
        "view": "MAG-Bench-Func",
        "definition": "functional-only view using source-entry functional_pass",
        "total_samples": len(rows),
        "summary_rows": summary,
        "comparison_rows": comparison,
    }
    _write_csv(
        output / "mag_bench_func_summary.csv",
        [
            "group",
            "value",
            "samples",
            "functional_pass_count",
            "functional_pass_rate",
            "functional_error_distribution",
        ],
        summary,
    )
    _write_csv(
        output / "mag_bench_vs_func_comparison.csv",
        [
            "group",
            "value",
            "samples",
            "contract_pass_rate",
            "source_functional_pass_rate",
            "abms_pass_rate",
        ],
        comparison,
    )
    (output / "mag_bench_func_report.md").write_text(
        _render_report(report),
        encoding="utf-8",
    )
    (output / "README_MAG_BENCH_FUNC.md").write_text(
        _render_readme(),
        encoding="utf-8",
    )
    _write_json(report, output / "mag_bench_func_report.json")
    return report


def _summary_rows(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for group in GROUP_FIELDS:
        for value, bucket in _group(rows, group).items():
            output.append(
                {
                    "group": group,
                    "value": value,
                    "samples": len(bucket),
                    "functional_pass_count": _count_true(bucket, "functional_pass"),
                    "functional_pass_rate": _rate(
                        _count_true(bucket, "functional_pass"),
                        len(bucket),
                    ),
                    "functional_error_distribution": json.dumps(
                        _distribution(bucket, "functional_error_type"),
                        sort_keys=True,
                    ),
                }
            )
    return output


def _mag_bench_vs_func(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for group in GROUP_FIELDS:
        for value, bucket in _group(rows, group).items():
            output.append(
                {
                    "group": group,
                    "value": value,
                    "samples": len(bucket),
                    "contract_pass_rate": _rate(
                        _count_true(bucket, "contract_pass"),
                        len(bucket),
                    ),
                    "source_functional_pass_rate": _rate(
                        _count_true(bucket, "functional_pass"),
                        len(bucket),
                    ),
                    "abms_pass_rate": _rate(
                        _count_true(bucket, "abms_pass"),
                        len(bucket),
                    ),
                }
            )
    return output


def _group(rows: list[dict[str, str]], field: str) -> dict[str, list[dict[str, str]]]:
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    if field == "overall":
        grouped["overall"] = rows
        return dict(grouped)
    for row in rows:
        value = row.get(field, "")
        if field == "mode" and not value:
            value = row.get("input_mode", "")
        grouped[value or "unknown"].append(row)
    return dict(sorted(grouped.items()))


def _render_report(report: dict[str, Any]) -> str:
    overall = next(
        row
        for row in report["summary_rows"]
        if row["group"] == "overall" and row["value"] == "overall"
    )
    lines = [
        "# MAG-Bench-Func Functional-Only View",
        "",
        f"- Source run: `{report['run']}`",
        f"- Samples: {report['total_samples']}",
        "- Metric: `functional_pass` from source-entry functional evaluation",
        f"- Overall functional pass rate: {overall['functional_pass_rate']}",
        "",
        "MAG-Bench-Func is reported separately from contract and ABMS metrics.",
    ]
    return "\n".join(lines) + "\n"


def _render_readme() -> str:
    return (
        "# MAG-Bench-Func\n\n"
        "MAG-Bench-Func is the functional-only view of MAG-Bench outputs. "
        "It reads `functional_pass` / source-entry evaluation results from an "
        "existing run and does not rerun models. Contract and ABMS conformance "
        "remain separate MAG-Bench metrics.\n"
    )


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as file:
        return [dict(row) for row in csv.DictReader(file)]


def _write_csv(path: Path, fields: list[str], rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _write_json(data: dict[str, Any], path: Path) -> None:
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _count_true(rows: list[dict[str, str]], field: str) -> int:
    return sum(1 for row in rows if _truthy(row.get(field, "")))


def _truthy(value: str | bool | None) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def _rate(count: int, total: int) -> str:
    if total <= 0:
        return "N/A"
    return f"{count / total:.3f}"


def _distribution(rows: list[dict[str, str]], field: str) -> dict[str, int]:
    output: dict[str, int] = {}
    for row in rows:
        value = row.get(field, "") or "none"
        output[value] = output.get(value, 0) + 1
    return output


if __name__ == "__main__":
    main()
