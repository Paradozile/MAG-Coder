"""Tests for Phase 8K paper functionality table exporter."""

from __future__ import annotations

import csv
import json
from pathlib import Path

from scripts.export_paper_functionality_tables import export_paper_functionality_tables


def test_paper_functionality_tables_distinguish_evidence_types(tmp_path: Path) -> None:
    """Table exporter combines real-run, read-only, and interface-only evidence."""

    paper_result = tmp_path / "paper_result"
    run_dir = paper_result / "run" / "ollama_native_2500_stratified100_s1_8g"
    run_dir.mkdir(parents=True)
    _write_sample_summary(run_dir / "sample_summary.csv")
    passk = tmp_path / "passk"
    passk.mkdir()
    (passk / "manifest.json").write_text(
        json.dumps(
            {
                "passk_metrics": {
                    "contract": {"pass@1": 0.8, "pass@3": 0.9, "pass@5": 1.0},
                    "source_functional": {"pass@1": 0.2, "pass@3": 0.2, "pass@5": 0.2},
                    "abms": {"pass@1": 0.7, "pass@3": 0.8, "pass@5": 0.9},
                }
            }
        )
        + "\n",
        encoding="utf-8",
    )
    rq2 = tmp_path / "rq2"
    rq2.mkdir()
    (rq2 / "rq2_ablation_report.json").write_text(
        json.dumps(
            {
                "graph_nl_vs_graph_only": {
                    "contract_pass": {
                        "graph_nl": 0.8,
                        "graph_only": 0.7,
                        "delta": 0.1,
                        "winner": "graph_nl",
                    }
                },
                "structured_vs_graphical": {},
            }
        )
        + "\n",
        encoding="utf-8",
    )
    func = tmp_path / "func"
    func.mkdir()
    _write_func_summary(func / "mag_bench_func_summary.csv")
    audit = tmp_path / "audit.json"
    audit.write_text(
        json.dumps(
            {
                "features": [
                    {
                        "feature_id": "F9_PROVIDER_REGISTRY",
                        "paper_feature": "provider registry",
                        "status": "interface_only",
                        "real_run_evidence": "not_required",
                        "caveat": "template only",
                    }
                ]
            }
        )
        + "\n",
        encoding="utf-8",
    )

    export_paper_functionality_tables(
        paper2500_result=paper_result,
        passk_result=passk,
        rq2_analysis=rq2,
        mag_bench_func=func,
        functionality_audit=audit,
        output_dir=tmp_path / "tables",
    )

    rows = list(
        csv.DictReader(
            (tmp_path / "tables" / "table_functionality_parity.csv").open(
                newline="",
                encoding="utf-8",
            )
        )
    )
    assert rows[0]["status"] == "interface_only"
    passk_rows = list(
        csv.DictReader(
            (tmp_path / "tables" / "table_passk_subset.csv").open(
                newline="",
                encoding="utf-8",
            )
        )
    )
    assert passk_rows[0]["pass@10"] == "N/A"


def _write_sample_summary(path: Path) -> None:
    fields = [
        "case_id",
        "method",
        "mode",
        "source",
        "task_type",
        "data_type",
        "template_family",
        "contract_pass",
        "functional_pass",
        "abms_pass",
        "latency_seconds",
    ]
    rows = [
        {
            "case_id": "case_a",
            "method": "full_with_refinement",
            "mode": "structured_graph_nl",
            "source": "humaneval",
            "task_type": "strings",
            "data_type": "sequence",
            "template_family": "delegated",
            "contract_pass": True,
            "functional_pass": False,
            "abms_pass": True,
            "latency_seconds": "2.0",
        }
    ]
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _write_func_summary(path: Path) -> None:
    fields = [
        "group",
        "value",
        "samples",
        "functional_pass_count",
        "functional_pass_rate",
        "functional_error_distribution",
    ]
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerow(
            {
                "group": "overall",
                "value": "overall",
                "samples": "1",
                "functional_pass_count": "0",
                "functional_pass_rate": "0.000",
                "functional_error_distribution": "{}",
            }
        )
