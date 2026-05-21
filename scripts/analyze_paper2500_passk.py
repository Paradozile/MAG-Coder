"""Analyze Phase 8J paper2500 pass@k pilot outputs."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from magcoder.evaluation.passk import estimate_pass_at_k, observed_pass_at_k

METRICS = ["contract_pass", "functional_pass", "abms_pass"]
K_VALUES = [1, 3, 5]
CLAIM_FILES = [
    "UPDATED_CLAIM_COMPLETION_MATRIX.md",
    "UPDATED_SAFE_PAPER_CLAIMS.md",
    "UPDATED_UNSUPPORTED_CLAIMS.md",
    "C9_PASSK_EVIDENCE_SUMMARY.md",
]


def main() -> None:
    """CLI entry point."""

    parser = argparse.ArgumentParser(description="Analyze paper2500 pass@k pilot.")
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--claim-output",
        type=Path,
        default=Path("claim_artifacts_8j"),
    )
    args = parser.parse_args()
    report = analyze_paper2500_passk(
        run_dir=args.run,
        output_dir=args.output,
        claim_output_dir=args.claim_output,
    )
    print(f"Pass@k analysis output: {args.output}")
    print(f"C9 claim status: {report['claim_status']['C9_PASSK_MULTISAMPLE']}")


def analyze_paper2500_passk(
    *,
    run_dir: str | Path,
    output_dir: str | Path,
    claim_output_dir: str | Path = "claim_artifacts_8j",
) -> dict[str, Any]:
    """Analyze sample-level pass@k data and write reports/tables."""

    run = Path(run_dir)
    output = Path(output_dir)
    claim_output = Path(claim_output_dir)
    output.mkdir(parents=True, exist_ok=True)
    claim_output.mkdir(parents=True, exist_ok=True)
    rows = [_normalize_row(row) for row in _read_csv(run / "sample_summary.csv")]
    run_config = _read_json_if_exists(run / "run_config.json")
    report = {
        "run_dir": str(run),
        "total_samples": len(rows),
        "selected_cases": len({row["case_id"] for row in rows if row["case_id"]}),
        "samples_per_cell": _samples_per_cell_distribution(rows),
        "modes_present": sorted({row["mode"] for row in rows}),
        "methods_present": sorted({row["method"] for row in rows}),
        "k_values": K_VALUES,
        "overall_passk": _passk_rows(rows, "overall"),
        "method_passk": _passk_rows(rows, "method"),
        "mode_passk": _passk_rows(rows, "mode"),
        "source_passk": _passk_rows(rows, "source"),
        "task_type_passk": _passk_rows(rows, "task_type"),
        "data_type_passk": _passk_rows(rows, "data_type"),
        "template_family_passk": _passk_rows(rows, "template_family"),
        "error_taxonomy": _error_taxonomy(rows),
        "infrastructure_errors": _infrastructure_errors(rows),
        "missing_entrypoint_count": sum(
            1 for row in rows if row.get("functional_error_type") == "MissingEntryPoint"
        ),
        "run_config": run_config,
    }
    report["method_comparison"] = _compare_named_groups(
        report["method_passk"],
        "full_with_refinement",
        "no_refinement",
    )
    report["mode_comparison"] = _compare_named_groups(
        report["mode_passk"],
        "structured_graph_nl",
        "graphical_graph_nl",
    )
    report["claim_status"] = _claim_status(report)
    _write_json(report, output / "passk_analysis_report.json")
    (output / "passk_analysis_report.md").write_text(
        _render_report(report),
        encoding="utf-8",
    )
    _write_csv(
        output / "table_overall_passk.csv",
        _passk_table_fields("overall"),
        report["overall_passk"],
    )
    _write_csv(
        output / "table_method_passk.csv",
        _passk_table_fields("method"),
        report["method_passk"],
    )
    _write_csv(
        output / "table_mode_passk.csv",
        _passk_table_fields("mode"),
        report["mode_passk"],
    )
    _write_csv(
        output / "table_source_passk.csv",
        _passk_table_fields("source"),
        report["source_passk"],
    )
    _write_csv(
        output / "table_task_type_passk.csv",
        _passk_table_fields("task_type"),
        report["task_type_passk"],
    )
    _write_csv(
        output / "table_error_taxonomy.csv",
        ["level", "error_type", "count"],
        report["error_taxonomy"],
    )
    _write_claim_artifacts(report, claim_output)
    return report


def _passk_rows(rows: list[dict[str, str]], group_field: str) -> list[dict[str, Any]]:
    buckets: dict[str, list[list[dict[str, str]]]] = defaultdict(list)
    for cell in _sample_cells(rows):
        if group_field == "overall":
            label = "overall"
        else:
            label = cell[0].get(group_field, "") or "unknown"
        buckets[label].append(cell)
    output: list[dict[str, Any]] = []
    for label, cells in sorted(buckets.items()):
        for metric in METRICS:
            item: dict[str, Any] = {
                group_field: label,
                "metric": metric,
                "groups": len(cells),
                "samples": sum(len(cell) for cell in cells),
                "num_correct": sum(
                    1 for cell in cells for row in cell if _truthy(row.get(metric))
                ),
            }
            for k in K_VALUES:
                estimates = [_cell_pass_at_k(cell, metric, k) for cell in cells]
                known = [value for value in estimates if value is not None]
                item[f"pass_at_{k}"] = _round(sum(known) / len(known)) if known else ""
                observed = [_cell_observed_at_k(cell, metric, k) for cell in cells]
                observed_known = [value for value in observed if value is not None]
                item[f"observed_pass_at_{k}"] = (
                    _round(
                        sum(1 for value in observed_known if value)
                        / len(observed_known)
                    )
                    if observed_known
                    else ""
                )
            output.append(item)
    return output


def _sample_cells(rows: list[dict[str, str]]) -> list[list[dict[str, str]]]:
    cells: dict[tuple[str, str, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        cells[(row["case_id"], row["mode"], row["method"])].append(row)
    return [
        sorted(group, key=lambda row: int(row.get("sample_index") or 0))
        for group in cells.values()
    ]


def _cell_pass_at_k(
    cell: list[dict[str, str]],
    metric: str,
    k: int,
) -> float | None:
    if k > len(cell):
        return None
    successes = [_truthy(row.get(metric)) for row in cell]
    return estimate_pass_at_k(len(successes), sum(successes), k)


def _cell_observed_at_k(
    cell: list[dict[str, str]],
    metric: str,
    k: int,
) -> bool | None:
    if k > len(cell):
        return None
    return observed_pass_at_k([_truthy(row.get(metric)) for row in cell], k)


def _compare_named_groups(
    rows: list[dict[str, Any]],
    left: str,
    right: str,
) -> dict[str, dict[str, Any]]:
    by_label_metric = {
        (str(row.get(next(iter(row.keys())), "")), str(row["metric"])): row
        for row in rows
    }
    output: dict[str, dict[str, Any]] = {}
    for metric in METRICS:
        left_row = by_label_metric.get((left, metric), {})
        right_row = by_label_metric.get((right, metric), {})
        output[metric] = {}
        for k in K_VALUES:
            left_value = _as_float(left_row.get(f"pass_at_{k}", ""))
            right_value = _as_float(right_row.get(f"pass_at_{k}", ""))
            delta = round(left_value - right_value, 6)
            winner = "tie"
            if delta > 0:
                winner = left
            elif delta < 0:
                winner = right
            output[metric][f"pass_at_{k}"] = {
                left: left_value,
                right: right_value,
                "delta": delta,
                "winner": winner,
            }
    return output


def _error_taxonomy(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for level, field in [
        ("functional", "functional_error_type"),
        ("abms", "abms_error_type"),
    ]:
        counts = Counter(row.get(field) or "none" for row in rows)
        for error_type, count in sorted(counts.items()):
            output.append({"level": level, "error_type": error_type, "count": count})
    return output


def _infrastructure_errors(rows: list[dict[str, str]]) -> dict[str, int]:
    return {
        field: sum(1 for row in rows if _truthy(row.get(field)))
        for field in [
            "backend_error",
            "extraction_failure",
            "syntax_parse_failure",
            "refinement_error",
            "final_content_empty",
        ]
    }


def _samples_per_cell_distribution(rows: list[dict[str, str]]) -> dict[str, int]:
    counts = Counter(str(len(cell)) for cell in _sample_cells(rows))
    return dict(sorted(counts.items()))


def _claim_status(report: dict[str, Any]) -> dict[str, str]:
    infra = report["infrastructure_errors"]
    has_pass5 = all(
        row.get("pass_at_5") != ""
        for row in report["overall_passk"]
        if row.get("metric") in METRICS
    )
    if report["total_samples"] and has_pass5 and not any(infra.values()):
        status = "bounded"
    else:
        status = "needs_additional_experiment"
    return {"C9_PASSK_MULTISAMPLE": status}


def _render_report(report: dict[str, Any]) -> str:
    lines = [
        "# Paper2500 Pass@k Pilot Analysis",
        "",
        f"- Total samples: {report['total_samples']}",
        f"- Selected cases: {report['selected_cases']}",
        f"- Modes present: `{report['modes_present']}`",
        f"- Methods present: `{report['methods_present']}`",
        f"- Samples per cell: `{report['samples_per_cell']}`",
        f"- Infrastructure errors: `{report['infrastructure_errors']}`",
        f"- MissingEntryPoint: {report['missing_entrypoint_count']}",
        f"- C9 claim status: `{report['claim_status']['C9_PASSK_MULTISAMPLE']}`",
        "",
        "## Overall pass@k",
        "",
        _markdown_table(_passk_table_fields("overall"), report["overall_passk"]),
        "",
        "## Method pass@k",
        "",
        _markdown_table(_passk_table_fields("method"), report["method_passk"]),
        "",
        "## Mode pass@k",
        "",
        _markdown_table(_passk_table_fields("mode"), report["mode_passk"]),
        "",
        "## Source pass@k",
        "",
        _markdown_table(_passk_table_fields("source"), report["source_passk"]),
        "",
        "## Error Taxonomy",
        "",
        _markdown_table(["level", "error_type", "count"], report["error_taxonomy"]),
        "",
        "## Claim Boundary",
        "",
        "- This is a 20-case stratified pass@k pilot, not a full 2,500-case run.",
        "- It reports pass@1/pass@3/pass@5 only, not pass@10.",
        "- It is a single local qwen3.5:9b result.",
    ]
    return "\n".join(lines) + "\n"


def _write_claim_artifacts(report: dict[str, Any], output: Path) -> None:
    c9_status = report["claim_status"]["C9_PASSK_MULTISAMPLE"]
    summary = _render_c9_summary(report)
    (output / "C9_PASSK_EVIDENCE_SUMMARY.md").write_text(summary, encoding="utf-8")
    (output / "UPDATED_CLAIM_COMPLETION_MATRIX.md").write_text(
        "\n".join(
            [
                "# Updated Claim Completion Matrix",
                "",
                "| claim_id | status | evidence | caveat |",
                "| --- | --- | --- | --- |",
                "| C9_PASSK_MULTISAMPLE | "
                f"{c9_status} | 20-case paper2500 pass@1/pass@3/pass@5 "
                "pilot with samples-per-case=5 | Not full 2,500-case "
                "evaluation and not pass@10. |",
                "",
            ]
        ),
        encoding="utf-8",
    )
    (output / "UPDATED_SAFE_PAPER_CLAIMS.md").write_text(
        "\n".join(
            [
                "# Updated Safe Paper Claims",
                "",
                "- On a 20-case stratified paper2500 pilot with five samples per "
                "case/mode/method, we report contract, source-functional, "
                "and ABMS pass@1/pass@3/pass@5.",
                "- These pass@k results provide bounded multi-sample evidence for "
                "the reproduced paper2500 benchmark.",
                "- The results should not be described as full 2,500-case "
                "exhaustive pass@k, pass@10, or multi-model generality.",
                "",
            ]
        ),
        encoding="utf-8",
    )
    (output / "UPDATED_UNSUPPORTED_CLAIMS.md").write_text(
        "\n".join(
            [
                "# Updated Unsupported Claims",
                "",
                "- Full 2,500-case exhaustive real-model pass@k validation.",
                "- pass@10 or pass@100 on the paper2500 benchmark.",
                "- Multi-model generality.",
                "- Arbitrary PNG/VLM graphical recognition.",
                "",
            ]
        ),
        encoding="utf-8",
    )


def _render_c9_summary(report: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# C9 Pass@k Evidence Summary",
            "",
            f"- Total samples: {report['total_samples']}",
            f"- Selected cases: {report['selected_cases']}",
            f"- C9 status: `{report['claim_status']['C9_PASSK_MULTISAMPLE']}`",
            f"- Samples per cell: `{report['samples_per_cell']}`",
            "",
            "## Overall",
            "",
            _markdown_table(_passk_table_fields("overall"), report["overall_passk"]),
            "",
            "## Method Breakdown",
            "",
            _markdown_table(_passk_table_fields("method"), report["method_passk"]),
            "",
            "## Mode Breakdown",
            "",
            _markdown_table(_passk_table_fields("mode"), report["mode_passk"]),
            "",
            "This is bounded paper2500 pass@k evidence on a 20-case stratified "
            "subset. It is not a full 2,500-case or pass@10 result.",
            "",
        ]
    )


def _passk_table_fields(group_field: str) -> list[str]:
    return [
        group_field,
        "metric",
        "groups",
        "samples",
        "num_correct",
        "pass_at_1",
        "pass_at_3",
        "pass_at_5",
        "observed_pass_at_1",
        "observed_pass_at_3",
        "observed_pass_at_5",
    ]


def _metric_value(row: dict[str, Any], field: str) -> str:
    value = row.get(field, "")
    return str(value)


def _markdown_table(fields: list[str], rows: list[dict[str, Any]]) -> str:
    lines = [
        "| " + " | ".join(fields) + " |",
        "| " + " | ".join("---" for _ in fields) + " |",
    ]
    for row in rows:
        values = [_metric_value(row, field) for field in fields]
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def _normalize_row(row: dict[str, str]) -> dict[str, str]:
    output = dict(row)
    output["mode"] = row.get("mode") or row.get("input_mode", "")
    return output


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def _read_json_if_exists(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open(encoding="utf-8") as file:
        data = json.load(file)
    return dict(data) if isinstance(data, dict) else {}


def _write_csv(path: Path, fields: list[str], rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _write_json(data: dict[str, Any], path: Path) -> None:
    with path.open("w", encoding="utf-8") as file:
        json.dump(data, file, indent=2, sort_keys=True)
        file.write("\n")


def _truthy(value: object) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "pass", "passed"}


def _round(value: float) -> float:
    return round(value, 6)


def _as_float(value: object) -> float:
    if value == "":
        return 0.0
    return float(value)


if __name__ == "__main__":
    main()
