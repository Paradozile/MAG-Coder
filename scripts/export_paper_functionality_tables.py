"""Export paper-facing functionality tables from frozen/read-only artifacts."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any


def main() -> None:
    """CLI entry point."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--paper2500-result", type=Path, required=True)
    parser.add_argument("--passk-result", type=Path, required=True)
    parser.add_argument("--rq2-analysis", type=Path, required=True)
    parser.add_argument("--mag-bench-func", type=Path, required=True)
    parser.add_argument("--functionality-audit", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    export_paper_functionality_tables(
        paper2500_result=args.paper2500_result,
        passk_result=args.passk_result,
        rq2_analysis=args.rq2_analysis,
        mag_bench_func=args.mag_bench_func,
        functionality_audit=args.functionality_audit,
        output_dir=args.output,
    )
    print(f"Paper functionality tables: {args.output}")


def export_paper_functionality_tables(
    *,
    paper2500_result: str | Path,
    passk_result: str | Path,
    rq2_analysis: str | Path,
    mag_bench_func: str | Path,
    functionality_audit: str | Path,
    output_dir: str | Path,
) -> None:
    """Write paper table CSVs from existing artifacts."""

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    result_root = Path(paper2500_result)
    run_rows = _read_csv(_first_existing(result_root, [
        "run/ollama_native_2500_stratified100_s1_8g/sample_summary.csv",
        "sample_summary.csv",
    ]))
    passk_root = Path(passk_result)
    passk_manifest = _read_json_if_exists(passk_root / "manifest.json")
    rq2_report = _read_json_if_exists(Path(rq2_analysis) / "rq2_ablation_report.json")
    func_rows = _read_csv(Path(mag_bench_func) / "mag_bench_func_summary.csv")
    audit = _read_json_if_exists(Path(functionality_audit))

    _write_csv(
        output / "table_rq1_framework_conformance.csv",
        ["metric", "samples", "pass_count", "pass_rate", "evidence_type"],
        _overall_metrics(run_rows),
    )
    _write_csv(
        output / "table_rq2_hybrid_ablation.csv",
        ["comparison", "metric", "left", "right", "delta", "winner", "evidence_type"],
        _rq2_rows(rq2_report),
    )
    _write_csv(
        output / "table_rq3_ast_refinement.csv",
        [
            "method",
            "samples",
            "contract_rate",
            "source_functional_rate",
            "abms_rate",
            "evidence_type",
        ],
        _group_metric_rows(run_rows, "method"),
    )
    _write_csv(
        output / "table_passk_subset.csv",
        [
            "metric",
            "pass@1",
            "pass@3",
            "pass@5",
            "pass@10",
            "pass@50",
            "pass@100",
            "evidence_type",
        ],
        _passk_rows(passk_manifest),
    )
    _write_csv(
        output / "table_mag_bench_func.csv",
        [
            "group",
            "value",
            "samples",
            "functional_pass_count",
            "functional_pass_rate",
            "evidence_type",
        ],
        _func_rows(func_rows),
    )
    _write_csv(
        output / "table_generation_time.csv",
        ["group", "value", "samples", "generation_time_seconds", "evidence_type"],
        _generation_time_rows(run_rows),
    )
    _write_csv(
        output / "table_functionality_parity.csv",
        ["feature_id", "paper_feature", "status", "real_run_evidence", "caveat"],
        _functionality_rows(audit),
    )
    (output / "paper_table_notes.md").write_text(
        _notes(),
        encoding="utf-8",
    )


def _overall_metrics(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    samples = len(rows)
    return [
        {
            "metric": "contract_pass",
            "samples": samples,
            "pass_count": _count_true(rows, "contract_pass"),
            "pass_rate": _rate(_count_true(rows, "contract_pass"), samples),
            "evidence_type": "real-run",
        },
        {
            "metric": "source_functional_pass",
            "samples": samples,
            "pass_count": _count_true(rows, "functional_pass"),
            "pass_rate": _rate(_count_true(rows, "functional_pass"), samples),
            "evidence_type": "real-run",
        },
        {
            "metric": "abms_pass",
            "samples": samples,
            "pass_count": _count_true(rows, "abms_pass"),
            "pass_rate": _rate(_count_true(rows, "abms_pass"), samples),
            "evidence_type": "real-run",
        },
    ]


def _rq2_rows(report: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for comparison, key in [
        ("graph_nl_vs_graph_only", "graph_nl_vs_graph_only"),
        ("structured_vs_graphical", "structured_vs_graphical"),
    ]:
        data = report.get(key, {})
        if not isinstance(data, dict):
            continue
        for metric, values in data.items():
            if not isinstance(values, dict):
                continue
            rows.append(
                {
                    "comparison": comparison,
                    "metric": metric,
                    "left": _first_value(values, ["graph_nl", "structured"]),
                    "right": _first_value(values, ["graph_only", "graphical"]),
                    "delta": values.get("delta", "N/A"),
                    "winner": values.get("winner", "N/A"),
                    "evidence_type": "real-run",
                }
            )
    return rows or [
        {
            "comparison": "rq2",
            "metric": "N/A",
            "left": "N/A",
            "right": "N/A",
            "delta": "N/A",
            "winner": "N/A",
            "evidence_type": "not-run",
        }
    ]


def _group_metric_rows(rows: list[dict[str, str]], field: str) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for value, bucket in _group(rows, field).items():
        output.append(
                {
                    field: value,
                    "samples": len(bucket),
                    "contract_rate": _rate(
                        _count_true(bucket, "contract_pass"),
                        len(bucket),
                    ),
                "source_functional_rate": _rate(
                    _count_true(bucket, "functional_pass"),
                    len(bucket),
                ),
                "abms_rate": _rate(_count_true(bucket, "abms_pass"), len(bucket)),
                "evidence_type": "real-run",
            }
        )
    return output


def _passk_rows(manifest: dict[str, Any]) -> list[dict[str, Any]]:
    metrics = manifest.get("passk_metrics", {})
    rows: list[dict[str, Any]] = []
    if not isinstance(metrics, dict):
        metrics = {}
    metric_keys = [
        ("contract", "contract"),
        ("source-functional", "source_functional"),
        ("ABMS", "abms"),
    ]
    for label, key in metric_keys:
        values = metrics.get(key, {})
        if not isinstance(values, dict):
            values = {}
        rows.append(
            {
                "metric": label,
                "pass@1": values.get("pass@1", "N/A"),
                "pass@3": values.get("pass@3", "N/A"),
                "pass@5": values.get("pass@5", "N/A"),
                "pass@10": values.get("pass@10", "N/A"),
                "pass@50": values.get("pass@50", "N/A"),
                "pass@100": values.get("pass@100", "N/A"),
                "evidence_type": "real-run-bounded",
            }
        )
    return rows


def _func_rows(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for row in rows:
        if row.get("group") in {"overall", "method", "mode", "source"}:
            output.append(
                {
                    "group": row.get("group", ""),
                    "value": row.get("value", ""),
                    "samples": row.get("samples", ""),
                    "functional_pass_count": row.get("functional_pass_count", ""),
                    "functional_pass_rate": row.get("functional_pass_rate", ""),
                    "evidence_type": "read-only-view",
                }
            )
    return output


def _generation_time_rows(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for field in ["overall", "method", "mode"]:
        for value, bucket in _group(rows, field).items():
            times = [
                _as_float(
                    row.get("latency_seconds")
                    or row.get("generation_time_seconds")
                )
                for row in bucket
            ]
            known = [value for value in times if value is not None]
            output.append(
                {
                    "group": field,
                    "value": value,
                    "samples": len(bucket),
                    "generation_time_seconds": (
                        f"{sum(known) / len(known):.3f}" if known else "N/A"
                    ),
                    "evidence_type": "real-run" if known else "N/A",
                }
            )
    return output


def _functionality_rows(report: dict[str, Any]) -> list[dict[str, Any]]:
    rows = report.get("features", [])
    if not isinstance(rows, list):
        return []
    output: list[dict[str, Any]] = []
    for row in rows:
        if isinstance(row, dict):
            output.append(
                {
                    "feature_id": row.get("feature_id", ""),
                    "paper_feature": row.get("paper_feature", ""),
                    "status": row.get("status", ""),
                    "real_run_evidence": row.get("real_run_evidence", ""),
                    "caveat": row.get("caveat", ""),
                }
            )
    return output


def _first_existing(root: Path, candidates: list[str]) -> Path:
    for candidate in candidates:
        path = root / candidate
        if path.exists():
            return path
    raise FileNotFoundError(
        f"None of the candidate files exist under {root}: {candidates}"
    )


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as file:
        return [dict(row) for row in csv.DictReader(file)]


def _read_json_if_exists(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    return data if isinstance(data, dict) else {}


def _write_csv(path: Path, fields: list[str], rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _group(rows: list[dict[str, str]], field: str) -> dict[str, list[dict[str, str]]]:
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    if field == "overall":
        grouped["overall"] = rows
        return grouped
    for row in rows:
        value = row.get(field, "")
        if field == "mode" and not value:
            value = row.get("input_mode", "")
        grouped[value or "unknown"].append(row)
    return dict(sorted(grouped.items()))


def _count_true(rows: list[dict[str, str]], field: str) -> int:
    return sum(1 for row in rows if str(row.get(field, "")).lower() == "true")


def _rate(count: int, total: int) -> str:
    if total <= 0:
        return "N/A"
    return f"{count / total:.3f}"


def _as_float(value: Any) -> float | None:
    if value in {None, "", "N/A"}:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _first_value(data: dict[str, Any], keys: list[str]) -> Any:
    for key in keys:
        if key in data:
            return data[key]
    return "N/A"


def _notes() -> str:
    return (
        "# Paper Functionality Table Notes\n\n"
        "These tables are assembled from frozen or read-only artifacts. Missing "
        "or interface-only functionality is reported as `N/A`; no values are "
        "hand-entered as substitute metrics.\n"
    )


if __name__ == "__main__":
    main()
