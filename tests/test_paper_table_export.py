from __future__ import annotations

import csv
from pathlib import Path

from scripts.export_paper_tables import export_paper_tables

FIELDS = [
    "case_id",
    "task_id",
    "input_mode",
    "ablation_id",
    "provider",
    "think",
    "native_endpoint",
    "use_nl",
    "use_refinement",
    "num_samples",
    "num_correct",
    "pass_at_1_estimated",
    "pass_at_3_estimated",
    "pass_at_5_estimated",
    "pass_at_10_estimated",
    "pass_at_1_observed",
    "pass_at_3_observed",
    "pass_at_5_observed",
    "pass_at_10_observed",
    "avg_checkpoint_accuracy",
    "mandatory_success_rate",
    "framework_pass_rate",
    "parse_success_rate",
    "generation_error_rate",
    "final_content_empty_rate",
    "syntax_parse_failure_rate",
    "extraction_failure_rate",
    "contract_mismatch_rate",
    "cache_hit_rate",
    "expected_refinement_gap_rate",
    "secret_findings",
    "failure_reason_summary",
]


def _mock_run_dir(tmp_path: Path) -> Path:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    rows = [
        {
            "case_id": "robot_structured_full",
            "task_id": "robot_rescue",
            "input_mode": "structured",
            "ablation_id": "graph_nl_with_refinement",
            "provider": "fake",
            "think": "False",
            "native_endpoint": "/api/chat",
            "use_nl": "True",
            "use_refinement": "True",
            "num_samples": "3",
            "num_correct": "3",
            "pass_at_1_estimated": "1.000",
            "pass_at_3_estimated": "1.000",
            "pass_at_5_estimated": "1.000",
            "pass_at_10_estimated": "1.000",
            "pass_at_1_observed": "True",
            "pass_at_3_observed": "True",
            "pass_at_5_observed": "True",
            "pass_at_10_observed": "True",
            "avg_checkpoint_accuracy": "1.000",
            "mandatory_success_rate": "1.000",
            "framework_pass_rate": "1.000",
            "parse_success_rate": "1.000",
            "generation_error_rate": "0.000",
            "final_content_empty_rate": "0.000",
            "syntax_parse_failure_rate": "0.000",
            "extraction_failure_rate": "0.000",
            "contract_mismatch_rate": "0.000",
            "cache_hit_rate": "0.000",
            "expected_refinement_gap_rate": "0.000",
            "secret_findings": "0",
            "failure_reason_summary": "",
        },
        {
            "case_id": "aircraft_graphical_graph_only",
            "task_id": "aircraft_collision",
            "input_mode": "graphical",
            "ablation_id": "graph_only_with_refinement",
            "provider": "fake",
            "think": "False",
            "native_endpoint": "/api/chat",
            "use_nl": "False",
            "use_refinement": "True",
            "num_samples": "3",
            "num_correct": "3",
            "pass_at_1_estimated": "1.000",
            "pass_at_3_estimated": "1.000",
            "pass_at_5_estimated": "1.000",
            "pass_at_10_estimated": "1.000",
            "pass_at_1_observed": "True",
            "pass_at_3_observed": "True",
            "pass_at_5_observed": "True",
            "pass_at_10_observed": "True",
            "avg_checkpoint_accuracy": "1.000",
            "mandatory_success_rate": "1.000",
            "framework_pass_rate": "1.000",
            "parse_success_rate": "1.000",
            "generation_error_rate": "0.000",
            "final_content_empty_rate": "0.000",
            "syntax_parse_failure_rate": "0.000",
            "extraction_failure_rate": "0.000",
            "contract_mismatch_rate": "0.000",
            "cache_hit_rate": "0.000",
            "expected_refinement_gap_rate": "0.000",
            "secret_findings": "0",
            "failure_reason_summary": "",
        },
        {
            "case_id": "robot_no_refinement",
            "task_id": "robot_rescue",
            "input_mode": "structured",
            "ablation_id": "graph_nl_without_refinement",
            "provider": "fake",
            "think": "False",
            "native_endpoint": "/api/chat",
            "use_nl": "True",
            "use_refinement": "False",
            "num_samples": "3",
            "num_correct": "1",
            "pass_at_1_estimated": "0.333",
            "pass_at_3_estimated": "1.000",
            "pass_at_5_estimated": "1.000",
            "pass_at_10_estimated": "1.000",
            "pass_at_1_observed": "True",
            "pass_at_3_observed": "True",
            "pass_at_5_observed": "True",
            "pass_at_10_observed": "True",
            "avg_checkpoint_accuracy": "0.800",
            "mandatory_success_rate": "0.333",
            "framework_pass_rate": "0.333",
            "parse_success_rate": "1.000",
            "generation_error_rate": "0.000",
            "final_content_empty_rate": "0.000",
            "syntax_parse_failure_rate": "0.000",
            "extraction_failure_rate": "0.000",
            "contract_mismatch_rate": "0.000",
            "cache_hit_rate": "0.000",
            "expected_refinement_gap_rate": "0.667",
            "secret_findings": "0",
            "failure_reason_summary": "expected_refinement_gap:2",
        },
    ]
    with (run_dir / "mini_pass3_summary.csv").open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.DictWriter(file, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    with (run_dir / "no_refinement_analysis.csv").open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "task_id",
                "input_mode",
                "num_samples",
                "natural_success_count",
                "expected_refinement_gap_count",
                "parse_success_rate",
                "framework_pass_rate",
                "missing_contract_rate",
                "missing_message_api_rate",
                "notes",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "task_id": "robot_rescue",
                "input_mode": "structured",
                "num_samples": "3",
                "natural_success_count": "1",
                "expected_refinement_gap_count": "2",
                "parse_success_rate": "1.000",
                "framework_pass_rate": "0.333",
                "missing_contract_rate": "0.000",
                "missing_message_api_rate": "0.667",
                "notes": "expected refinement gap",
            }
        )
    return run_dir


def _read_table(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def test_export_paper_tables_generates_csv_md_tex(tmp_path: Path) -> None:
    output_dir = tmp_path / "paper_tables"

    written = export_paper_tables(_mock_run_dir(tmp_path), output_dir)

    assert len(written) == 18
    assert (output_dir / "table_overall_passk.csv").exists()
    assert (output_dir / "table_overall_passk.md").exists()
    assert (output_dir / "table_overall_passk.tex").exists()


def test_overall_table_groups_refinement_correctly(tmp_path: Path) -> None:
    output_dir = tmp_path / "paper_tables"

    export_paper_tables(_mock_run_dir(tmp_path), output_dir)
    rows = _read_table(output_dir / "table_overall_passk.csv")

    full = next(row for row in rows if row["method"] == "full_with_refinement")
    no_refinement = next(row for row in rows if row["method"] == "no_refinement")
    assert full["num_cases"] == "2"
    assert full["pass@3"] == "1.000"
    assert full["pass@10"] == "1.000"
    assert no_refinement["num_cases"] == "1"
    assert no_refinement["mandatory_success_rate"] == "0.333"


def test_ablation_input_scenario_and_error_tables_generate(
    tmp_path: Path,
) -> None:
    output_dir = tmp_path / "paper_tables"

    export_paper_tables(_mock_run_dir(tmp_path), output_dir)

    assert _read_table(output_dir / "table_ablation.csv")
    assert _read_table(output_dir / "table_input_modes.csv")
    assert _read_table(output_dir / "table_scenarios.csv")
    assert _read_table(output_dir / "table_no_refinement_analysis.csv")
    error_rows = _read_table(output_dir / "table_error_analysis.csv")
    assert error_rows[0]["final_content_empty_rate"] == "0.000"
    assert error_rows[0]["contract_mismatch_rate"] == "0.000"
