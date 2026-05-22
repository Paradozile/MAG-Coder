"""Tests for Phase 8J paper2500 pass@k analysis."""

from __future__ import annotations

import csv
import json
from pathlib import Path

from scripts.analyze_paper2500_passk import analyze_paper2500_passk


def test_passk_analysis_computes_contract_functional_abms(tmp_path: Path) -> None:
    """The analysis computes pass@1/pass@3/pass@5 for all success metrics."""

    run = tmp_path / "run"
    run.mkdir()
    _write_sample_summary(run / "sample_summary.csv")
    (run / "selected_cases.json").write_text('{"case_ids": ["case_a"]}\n')

    report = analyze_paper2500_passk(
        run_dir=run,
        output_dir=tmp_path / "analysis",
        claim_output_dir=tmp_path / "claims",
    )

    overall = {
        row["metric"]: row for row in report["overall_passk"]
    }
    assert overall["contract_pass"]["pass_at_1"] == 0.4
    assert overall["contract_pass"]["pass_at_3"] == 0.9
    assert overall["contract_pass"]["pass_at_5"] == 1.0
    assert overall["functional_pass"]["pass_at_5"] == 1.0
    assert overall["abms_pass"]["pass_at_5"] == 1.0
    assert report["claim_status"]["C9_PASSK_MULTISAMPLE"] == "bounded"
    assert (tmp_path / "analysis" / "table_method_passk.csv").exists()
    assert (tmp_path / "claims" / "C9_PASSK_EVIDENCE_SUMMARY.md").exists()


def test_passk_analysis_method_breakdown(tmp_path: Path) -> None:
    """Method tables preserve full-with-refinement and no-refinement groups."""

    run = tmp_path / "run"
    run.mkdir()
    _write_sample_summary(run / "sample_summary.csv", include_no_refinement=True)
    (run / "selected_cases.json").write_text('{"case_ids": ["case_a"]}\n')

    analyze_paper2500_passk(
        run_dir=run,
        output_dir=tmp_path / "analysis",
        claim_output_dir=tmp_path / "claims",
    )

    rows = list(
        csv.DictReader(
            (tmp_path / "analysis" / "table_method_passk.csv").open(
                newline="",
                encoding="utf-8",
            )
        )
    )
    methods = {row["method"] for row in rows}
    assert methods == {"full_with_refinement", "no_refinement"}


def _write_sample_summary(
    path: Path,
    *,
    include_no_refinement: bool = False,
) -> None:
    fields = [
        "case_id",
        "source",
        "task_type",
        "data_type",
        "template_family",
        "mode",
        "method",
        "sample_id",
        "sample_index",
        "contract_pass",
        "functional_pass",
        "abms_pass",
        "backend_error",
        "extraction_failure",
        "syntax_parse_failure",
        "refinement_error",
        "final_content_empty",
        "functional_error_type",
        "abms_error_type",
    ]
    rows = _rows_for_method("full_with_refinement", [True, True, False, False, False])
    if include_no_refinement:
        rows.extend(
            _rows_for_method("no_refinement", [True, False, False, False, False])
        )
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _rows_for_method(method: str, successes: list[bool]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for index, success in enumerate(successes):
        rows.append(
            {
                "case_id": "case_a",
                "source": "humaneval",
                "task_type": "sorting_searching",
                "data_type": "sequence",
                "template_family": "brokered_exchange",
                "mode": "structured_graph_nl",
                "method": method,
                "sample_id": f"sample_{index:03d}",
                "sample_index": index,
                "contract_pass": success,
                "functional_pass": success,
                "abms_pass": success,
                "backend_error": False,
                "extraction_failure": False,
                "syntax_parse_failure": False,
                "refinement_error": False,
                "final_content_empty": False,
                "functional_error_type": "" if success else "AssertionError",
                "abms_error_type": "" if success else "ExpectedMessageMissing",
            }
        )
    return rows


def test_analysis_json_can_be_loaded(tmp_path: Path) -> None:
    """The JSON report is stable enough for downstream validators."""

    run = tmp_path / "run"
    run.mkdir()
    _write_sample_summary(run / "sample_summary.csv")
    (run / "selected_cases.json").write_text('{"case_ids": ["case_a"]}\n')

    analyze_paper2500_passk(
        run_dir=run,
        output_dir=tmp_path / "analysis",
        claim_output_dir=tmp_path / "claims",
    )

    data = json.loads(
        (tmp_path / "analysis" / "passk_analysis_report.json").read_text(
            encoding="utf-8"
        )
    )
    assert data["overall_passk"]
    assert data["claim_status"]["C9_PASSK_MULTISAMPLE"] == "bounded"
