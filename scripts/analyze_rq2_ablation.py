"""Analyze Phase 8I.5 RQ2 input-mode ablation results."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

METRICS = ["contract_pass", "functional_pass", "abms_pass"]
EXPECTED_MODES = [
    "structured_graph_nl",
    "structured_graph_only",
    "graphical_graph_nl",
    "graphical_graph_only",
]


def main() -> None:
    """CLI entry point."""

    parser = argparse.ArgumentParser(description="Analyze RQ2 ablation outputs.")
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--baseline-run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--claim-output",
        type=Path,
        default=Path("claim_artifacts_8i5"),
    )
    args = parser.parse_args()
    report = analyze_rq2_ablation(
        run_dir=args.run,
        baseline_run_dir=args.baseline_run,
        output_dir=args.output,
        claim_output_dir=args.claim_output,
    )
    print(f"RQ2 analysis output: {args.output}")
    print(f"RQ2 claim status: {report['rq2_claim_status']}")


def analyze_rq2_ablation(
    *,
    run_dir: str | Path,
    baseline_run_dir: str | Path,
    output_dir: str | Path,
    claim_output_dir: str | Path = "claim_artifacts_8i5",
) -> dict[str, Any]:
    """Analyze one RQ2 ablation run and write reports."""

    run = Path(run_dir)
    baseline = Path(baseline_run_dir)
    output = Path(output_dir)
    claim_output = Path(claim_output_dir)
    output.mkdir(parents=True, exist_ok=True)
    claim_output.mkdir(parents=True, exist_ok=True)
    rows = _read_csv(run / "sample_summary.csv")
    baseline_rows = _read_csv(baseline / "sample_summary.csv")
    rows = [_normalize_row(row) for row in rows]
    baseline_rows = [_normalize_row(row) for row in baseline_rows]
    mode_metrics = _group_metrics(rows, "mode")
    condition_metrics = _group_metrics(rows, "nl_condition")
    family_metrics = _group_metrics(rows, "mode_family")
    graph_compare = _condition_comparison(condition_metrics)
    family_compare = _family_comparison(family_metrics)
    graph_pairs = _paired_graph_nl_vs_graph_only(rows)
    family_pairs = _paired_structured_vs_graphical(rows)
    report = {
        "run_dir": str(run),
        "baseline_run_dir": str(baseline),
        "total_samples": len(rows),
        "selected_cases": len({row.get("case_id", "") for row in rows}),
        "modes_present": sorted({row.get("mode", "") for row in rows}),
        "methods_present": sorted({row.get("method", "") for row in rows}),
        "mode_condition_metrics": mode_metrics,
        "nl_condition_metrics": condition_metrics,
        "mode_family_metrics": family_metrics,
        "graph_nl_vs_graph_only": graph_compare,
        "structured_vs_graphical": family_compare,
        "paired_graph_nl_vs_graph_only": graph_pairs,
        "paired_structured_vs_graphical": family_pairs,
        "baseline_metrics": {
            "total_samples": len(baseline_rows),
            "mode_metrics": _group_metrics(baseline_rows, "mode"),
        },
        "error_taxonomy_by_mode": _error_taxonomy_by_group(rows, "mode"),
        "rq2_claim_status": _rq2_claim_status(graph_compare, family_compare),
        "warnings": _warnings(rows),
    }
    _write_json(report, output / "rq2_ablation_report.json")
    (output / "rq2_ablation_report.md").write_text(
        _render_report(report),
        encoding="utf-8",
    )
    _write_csv(
        output / "table_mode_condition_metrics.csv",
        _metric_table_fields("mode"),
        _metric_table_rows(mode_metrics, "mode"),
    )
    _write_csv(
        output / "table_graph_nl_vs_graph_only.csv",
        ["metric", "graph_nl", "graph_only", "delta", "winner"],
        _comparison_rows(graph_compare),
    )
    _write_csv(
        output / "table_structured_vs_graphical.csv",
        ["metric", "structured", "graphical", "delta", "winner"],
        _comparison_rows(family_compare),
    )
    _write_csv(
        output / "table_per_case_mode_win_loss.csv",
        [
            "comparison",
            "metric",
            "pairs",
            "left_wins",
            "right_wins",
            "ties",
            "left_label",
            "right_label",
        ],
        _paired_rows(graph_pairs, family_pairs),
    )
    _write_updated_claim_artifacts(report, claim_output)
    return report


def _normalize_row(row: dict[str, str]) -> dict[str, str]:
    mode = row.get("mode") or row.get("input_mode", "")
    normalized = dict(row)
    normalized["mode"] = mode
    normalized["mode_family"] = row.get("mode_family") or _mode_family(mode)
    normalized["nl_condition"] = row.get("nl_condition") or _nl_condition(mode)
    return normalized


def _group_metrics(rows: list[dict[str, str]], key: str) -> dict[str, dict[str, Any]]:
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[row.get(key, "") or "unknown"].append(row)
    return {
        value: _metrics_for_rows(group)
        for value, group in sorted(grouped.items())
    }


def _metrics_for_rows(rows: list[dict[str, str]]) -> dict[str, Any]:
    total = len(rows)
    output: dict[str, Any] = {"samples": total}
    for metric in METRICS:
        count = _count_true(rows, metric)
        output[f"{metric}_count"] = count
        output[f"{metric}_rate"] = _rate(count, total)
    output["backend_errors"] = _count_true(rows, "backend_error")
    output["extraction_failures"] = _count_true(rows, "extraction_failure")
    output["syntax_parse_failures"] = _count_true(rows, "syntax_parse_failure")
    output["refinement_errors"] = _count_true(rows, "refinement_error")
    output["functional_error_type_distribution"] = _distribution(
        rows,
        "functional_error_type",
    )
    output["abms_error_type_distribution"] = _distribution(rows, "abms_error_type")
    return output


def _condition_comparison(
    condition_metrics: dict[str, dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    return _compare_groups(condition_metrics, "graph_nl", "graph_only")


def _family_comparison(
    family_metrics: dict[str, dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    return _compare_groups(family_metrics, "structured", "graphical")


def _compare_groups(
    metrics: dict[str, dict[str, Any]],
    left: str,
    right: str,
) -> dict[str, dict[str, Any]]:
    left_metrics = metrics.get(left, {})
    right_metrics = metrics.get(right, {})
    output: dict[str, dict[str, Any]] = {}
    for metric in METRICS:
        left_rate = float(left_metrics.get(f"{metric}_rate", 0.0))
        right_rate = float(right_metrics.get(f"{metric}_rate", 0.0))
        delta = round(left_rate - right_rate, 6)
        winner = "tie"
        if delta > 0:
            winner = left
        elif delta < 0:
            winner = right
        output[metric] = {
            left: left_rate,
            right: right_rate,
            "delta": delta,
            "winner": winner,
        }
    return output


def _paired_graph_nl_vs_graph_only(rows: list[dict[str, str]]) -> dict[str, Any]:
    pairs: dict[tuple[str, str, str, str], dict[str, dict[str, str]]] = defaultdict(
        dict
    )
    for row in rows:
        key = (
            row.get("case_id", ""),
            row.get("mode_family", ""),
            row.get("method", ""),
            row.get("sample_id", row.get("sample_index", "")),
        )
        pairs[key][row.get("nl_condition", "")] = row
    return _paired_counts(
        pairs,
        left_label="graph_nl",
        right_label="graph_only",
        comparison="graph_nl_vs_graph_only",
    )


def _paired_structured_vs_graphical(rows: list[dict[str, str]]) -> dict[str, Any]:
    pairs: dict[tuple[str, str, str, str], dict[str, dict[str, str]]] = defaultdict(
        dict
    )
    for row in rows:
        key = (
            row.get("case_id", ""),
            row.get("nl_condition", ""),
            row.get("method", ""),
            row.get("sample_id", row.get("sample_index", "")),
        )
        pairs[key][row.get("mode_family", "")] = row
    return _paired_counts(
        pairs,
        left_label="structured",
        right_label="graphical",
        comparison="structured_vs_graphical",
    )


def _paired_counts(
    pairs: dict[tuple[str, str, str, str], dict[str, dict[str, str]]],
    *,
    left_label: str,
    right_label: str,
    comparison: str,
) -> dict[str, Any]:
    output: dict[str, Any] = {
        "comparison": comparison,
        "left_label": left_label,
        "right_label": right_label,
        "metrics": {},
    }
    for metric in METRICS:
        left_wins = 0
        right_wins = 0
        ties = 0
        complete = 0
        for pair in pairs.values():
            if left_label not in pair or right_label not in pair:
                continue
            complete += 1
            left_value = _truthy(pair[left_label].get(metric, ""))
            right_value = _truthy(pair[right_label].get(metric, ""))
            if left_value == right_value:
                ties += 1
            elif left_value:
                left_wins += 1
            else:
                right_wins += 1
        output["metrics"][metric] = {
            "pairs": complete,
            "left_wins": left_wins,
            "right_wins": right_wins,
            "ties": ties,
        }
    return output


def _rq2_claim_status(
    graph_compare: dict[str, dict[str, Any]],
    family_compare: dict[str, dict[str, Any]],
) -> dict[str, str]:
    graph_nl_contract_delta = float(graph_compare["contract_pass"]["delta"])
    graph_nl_abms_delta = float(graph_compare["abms_pass"]["delta"])
    if graph_nl_contract_delta > 0.02 and graph_nl_abms_delta > 0.02:
        c7 = "bounded_supported"
    elif graph_nl_contract_delta >= 0 and graph_nl_abms_delta >= 0:
        c7 = "bounded_mixed"
    else:
        c7 = "not_supported"
    structured_contract_delta = float(family_compare["contract_pass"]["delta"])
    structured_abms_delta = float(family_compare["abms_pass"]["delta"])
    if abs(structured_contract_delta) <= 0.02 and abs(structured_abms_delta) <= 0.02:
        c8 = "bounded_no_material_difference"
    elif structured_contract_delta > 0 and structured_abms_delta > 0:
        c8 = "bounded_structured_higher"
    elif structured_contract_delta < 0 and structured_abms_delta < 0:
        c8 = "bounded_graphical_higher"
    else:
        c8 = "bounded_mixed"
    return {"C7_RQ2_HYBRID_INPUT_CONTRIBUTION": c7, "C8_GRAPH_ONLY_VS_GRAPH_NL": c8}


def _write_updated_claim_artifacts(report: dict[str, Any], output: Path) -> None:
    status = report["rq2_claim_status"]
    c7 = status["C7_RQ2_HYBRID_INPUT_CONTRIBUTION"]
    c8 = status["C8_GRAPH_ONLY_VS_GRAPH_NL"]
    evidence = _rq2_evidence_text(report)
    (output / "RQ2_EVIDENCE_SUMMARY.md").write_text(evidence, encoding="utf-8")
    (output / "UPDATED_CLAIM_COMPLETION_MATRIX.md").write_text(
        "# Updated Claim Completion Matrix\n\n"
        "This update supplements Phase 8I with the Phase 8I.5 RQ2 ablation. "
        "It does not change any frozen bundle.\n\n"
        "| claim id | previous status | updated status | evidence |\n"
        "| --- | --- | --- | --- |\n"
        f"| C7_RQ2_HYBRID_INPUT_CONTRIBUTION | needs_additional_experiment | "
        f"{c7} | See RQ2_EVIDENCE_SUMMARY.md |\n"
        f"| C8_GRAPH_ONLY_VS_GRAPH_NL | needs_additional_experiment | {c8} | "
        "See RQ2_EVIDENCE_SUMMARY.md |\n",
        encoding="utf-8",
    )
    (output / "UPDATED_SAFE_PAPER_CLAIMS.md").write_text(
        "# Updated Safe Paper Claims\n\n"
        "- Phase 8I.5 adds a matched 100-case RQ2 ablation over the same "
        "selected paper2500 subset used by Phase 8G.\n"
        "- Treat RQ2 as bounded evidence from one local qwen3.5:9b run, not "
        "multi-model generality.\n"
        f"- C7 updated status: {c7}.\n"
        f"- C8 updated status: {c8}.\n"
        "- Graphical input remains standardized DOT+manifest MAG artifacts, "
        "not arbitrary image recognition.\n",
        encoding="utf-8",
    )
    (output / "UPDATED_UNSUPPORTED_CLAIMS.md").write_text(
        "# Updated Unsupported Claims\n\n"
        "- Full 2,500-case real-model exhaustive validation remains unsupported.\n"
        "- Paper2500 pass@10/pass@100 remains unsupported.\n"
        "- Multi-model generality remains unsupported.\n"
        "- Arbitrary PNG/VLM recognition remains unsupported.\n"
        "- Strong source-functional correctness remains unsupported.\n"
        "- Exact reproduction of original authors' hidden template list remains "
        "unsupported.\n",
        encoding="utf-8",
    )


def _rq2_evidence_text(report: dict[str, Any]) -> str:
    return (
        "# RQ2 Evidence Summary\n\n"
        f"- Total samples: {report['total_samples']}\n"
        f"- Selected cases: {report['selected_cases']}\n"
        f"- Modes present: `{report['modes_present']}`\n"
        f"- Methods present: `{report['methods_present']}`\n\n"
        "## Graph+NL vs Graph-only\n\n"
        f"`{report['graph_nl_vs_graph_only']}`\n\n"
        "## Structured vs Graphical\n\n"
        f"`{report['structured_vs_graphical']}`\n\n"
        "## Paired Win/Loss\n\n"
        f"- graph_nl_vs_graph_only: `{report['paired_graph_nl_vs_graph_only']}`\n"
        f"- structured_vs_graphical: `{report['paired_structured_vs_graphical']}`\n\n"
        "## Claim Status\n\n"
        f"`{report['rq2_claim_status']}`\n"
    )


def _render_report(report: dict[str, Any]) -> str:
    lines = [
        "# RQ2 Ablation Report",
        "",
        "This report analyzes graph+NL versus graph-only and structured versus "
        "graphical input modes on a matched paper2500 subset.",
        "",
        f"- Total samples: {report['total_samples']}",
        f"- Selected cases: {report['selected_cases']}",
        f"- Modes present: `{report['modes_present']}`",
        f"- Methods present: `{report['methods_present']}`",
        f"- Warnings: `{report['warnings']}`",
        "",
        "## Mode Metrics",
        "",
    ]
    for mode, metrics in report["mode_condition_metrics"].items():
        lines.append(f"- {mode}: `{metrics}`")
    lines.extend(
        [
            "",
            "## Graph+NL vs Graph-only",
            "",
            f"`{report['graph_nl_vs_graph_only']}`",
            "",
            "## Structured vs Graphical",
            "",
            f"`{report['structured_vs_graphical']}`",
            "",
            "## Paired Analysis",
            "",
            f"- graph_nl_vs_graph_only: `{report['paired_graph_nl_vs_graph_only']}`",
            f"- structured_vs_graphical: `{report['paired_structured_vs_graphical']}`",
            "",
            "## RQ2 Claim Status",
            "",
            f"`{report['rq2_claim_status']}`",
            "",
            "## Limitations",
            "",
            "- This is a 100-case stratified subset, not a full 2,500-case run.",
            "- This is a single local model run.",
            "- Graphical means DOT+manifest standardized artifacts.",
        ]
    )
    return "\n".join(lines) + "\n"


def _comparison_rows(comparison: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for metric, values in comparison.items():
        labels = [label for label in values if label not in {"delta", "winner"}]
        left = labels[0] if labels else "left"
        right = labels[1] if len(labels) > 1 else "right"
        rows.append(
            {
                "metric": metric,
                left: values.get(left, ""),
                right: values.get(right, ""),
                "delta": values.get("delta", ""),
                "winner": values.get("winner", ""),
            }
        )
    return rows


def _paired_rows(
    graph_pairs: dict[str, Any],
    family_pairs: dict[str, Any],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for paired in [graph_pairs, family_pairs]:
        for metric, values in paired["metrics"].items():
            rows.append(
                {
                    "comparison": paired["comparison"],
                    "metric": metric,
                    "pairs": values["pairs"],
                    "left_wins": values["left_wins"],
                    "right_wins": values["right_wins"],
                    "ties": values["ties"],
                    "left_label": paired["left_label"],
                    "right_label": paired["right_label"],
                }
            )
    return rows


def _metric_table_rows(
    grouped: dict[str, dict[str, Any]],
    group_key: str,
) -> list[dict[str, Any]]:
    return [{group_key: key, **metrics} for key, metrics in grouped.items()]


def _metric_table_fields(group_key: str) -> list[str]:
    return [
        group_key,
        "samples",
        "contract_pass_count",
        "contract_pass_rate",
        "functional_pass_count",
        "functional_pass_rate",
        "abms_pass_count",
        "abms_pass_rate",
        "backend_errors",
        "extraction_failures",
        "syntax_parse_failures",
        "refinement_errors",
        "functional_error_type_distribution",
        "abms_error_type_distribution",
    ]


def _error_taxonomy_by_group(
    rows: list[dict[str, str]],
    key: str,
) -> dict[str, dict[str, dict[str, int]]]:
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[row.get(key, "") or "unknown"].append(row)
    return {
        value: {
            "functional_error_type": _distribution(group, "functional_error_type"),
            "abms_error_type": _distribution(group, "abms_error_type"),
        }
        for value, group in sorted(grouped.items())
    }


def _warnings(rows: list[dict[str, str]]) -> list[str]:
    warnings: list[str] = []
    modes = {row.get("mode", "") for row in rows}
    missing = [mode for mode in EXPECTED_MODES if mode not in modes]
    if missing:
        warnings.append(f"missing expected modes: {missing}")
    methods = {row.get("method", "") for row in rows}
    if "full_with_refinement" not in methods:
        warnings.append("full_with_refinement samples are missing")
    return warnings


def _mode_family(mode: str) -> str:
    if mode.startswith("structured_"):
        return "structured"
    if mode.startswith("graphical_"):
        return "graphical"
    return "unknown"


def _nl_condition(mode: str) -> str:
    if mode.endswith("_graph_nl"):
        return "graph_nl"
    if mode.endswith("_graph_only"):
        return "graph_only"
    return "unknown"


def _count_true(rows: list[dict[str, str]], key: str) -> int:
    return sum(1 for row in rows if _truthy(row.get(key, "")))


def _truthy(value: str | object) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "pass", "passed"}


def _rate(count: int, total: int) -> float:
    return round(count / total, 6) if total else 0.0


def _distribution(rows: list[dict[str, str]], key: str) -> dict[str, int]:
    return dict(Counter(row.get(key, "") or "none" for row in rows))


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def _write_csv(path: Path, fields: list[str], rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _write_json(payload: dict[str, Any], path: Path) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", "utf-8")


if __name__ == "__main__":
    main()
