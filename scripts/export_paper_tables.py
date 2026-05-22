"""Export paper-ready tables from a Mini MAG-Bench pass@k run."""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(slots=True)
class TableSpec:
    """A table ready to be written in CSV, Markdown, and LaTeX."""

    name: str
    fields: list[str]
    rows: list[dict[str, Any]]


def export_paper_tables(run_dir: str | Path, output_dir: str | Path) -> list[Path]:
    """Export all paper tables from a Mini MAG-Bench pass@k run directory."""

    run_path = Path(run_dir)
    output_path = Path(output_dir)
    summary_rows = _load_summary_rows(_summary_path(run_path))
    tables = [
        _overall_passk_table(summary_rows),
        _ablation_table(summary_rows),
        _input_modes_table(summary_rows),
        _scenarios_table(summary_rows),
        _error_analysis_table(summary_rows),
    ]
    no_refinement_path = run_path / "no_refinement_analysis.csv"
    if no_refinement_path.exists():
        tables.append(_raw_table("table_no_refinement_analysis", no_refinement_path))
    written: list[Path] = []
    for table in tables:
        written.extend(_write_table(table, output_path))
    return written


def main() -> None:
    """CLI entry point."""

    parser = argparse.ArgumentParser(
        description="Export paper-ready tables from mini_pass{k}_summary.csv.",
    )
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    written = export_paper_tables(args.run_dir, args.output_dir)
    print(f"Exported {len(written)} table files to {args.output_dir}")


def _load_summary_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(f"Missing summary file: {path}")
    with path.open(newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))
    if not rows:
        raise ValueError(f"Summary file has no rows: {path}")
    return rows


def _summary_path(run_path: Path) -> Path:
    for name in [
        "mini_pass10_summary.csv",
        "mini_pass5_summary.csv",
        "mini_pass3_summary.csv",
    ]:
        path = run_path / name
        if path.exists():
            return path
    raise FileNotFoundError(f"No mini_pass summary CSV found in {run_path}")


def _overall_passk_table(rows: list[dict[str, str]]) -> TableSpec:
    return TableSpec(
        name="table_overall_passk",
        fields=[
            "method",
            "num_cases",
            "num_samples",
            "pass@1",
            "pass@3",
            "pass@5",
            "pass@10",
            "mandatory_success_rate",
            "framework_pass_rate",
            "parse_success_rate",
            "generation_error_rate",
        ],
        rows=[
            _aggregate_row(
                "method",
                "full_with_refinement",
                [row for row in rows if _bool(row.get("use_refinement"))],
            ),
            _aggregate_row(
                "method",
                "no_refinement",
                [row for row in rows if not _bool(row.get("use_refinement"))],
            ),
        ],
    )


def _ablation_table(rows: list[dict[str, str]]) -> TableSpec:
    labels = {
        "graph_nl_with_refinement": "graph+NL with refinement",
        "graph_only_with_refinement": "graph-only with refinement",
        "graph_nl_without_refinement": "graph+NL without refinement",
    }
    return TableSpec(
        name="table_ablation",
        fields=[
            "ablation",
            "num_cases",
            "num_samples",
            "pass@1",
            "pass@3",
            "pass@5",
            "pass@10",
            "mandatory_success_rate",
            "framework_pass_rate",
            "avg_checkpoint_accuracy",
        ],
        rows=[
            _aggregate_row(
                "ablation",
                label,
                [row for row in rows if row.get("ablation_id") == ablation_id],
            )
            for ablation_id, label in labels.items()
        ],
    )


def _input_modes_table(rows: list[dict[str, str]]) -> TableSpec:
    return TableSpec(
        name="table_input_modes",
        fields=[
            "input_mode",
            "num_cases",
            "num_samples",
            "pass@1",
            "pass@3",
            "pass@5",
            "pass@10",
            "mandatory_success_rate",
            "parse_success_rate",
            "framework_pass_rate",
        ],
        rows=[
            _aggregate_row(
                "input_mode",
                input_mode,
                [row for row in rows if row.get("input_mode") == input_mode],
            )
            for input_mode in ["structured", "graphical"]
        ],
    )


def _scenarios_table(rows: list[dict[str, str]]) -> TableSpec:
    return TableSpec(
        name="table_scenarios",
        fields=[
            "scenario",
            "num_cases",
            "num_samples",
            "pass@1",
            "pass@3",
            "pass@5",
            "pass@10",
            "mandatory_success_rate",
            "framework_pass_rate",
            "avg_checkpoint_accuracy",
        ],
        rows=[
            _aggregate_row(
                "scenario",
                scenario,
                [row for row in rows if row.get("task_id") == scenario],
            )
            for scenario in ["robot_rescue", "aircraft_collision"]
        ],
    )


def _error_analysis_table(rows: list[dict[str, str]]) -> TableSpec:
    return TableSpec(
        name="table_error_analysis",
        fields=[
            "scope",
            "num_cases",
            "num_samples",
            "final_content_empty_rate",
            "syntax_parse_failure_rate",
            "backend_error_rate",
            "extraction_failure_rate",
            "contract_mismatch_rate",
            "secret_findings",
        ],
        rows=[
            {
                "scope": "overall",
                "num_cases": len(rows),
                "num_samples": _num_samples(rows),
                "final_content_empty_rate": _weighted_rate(
                    rows,
                    "final_content_empty_rate",
                ),
                "syntax_parse_failure_rate": _weighted_rate(
                    rows,
                    "syntax_parse_failure_rate",
                ),
                "backend_error_rate": _weighted_rate(rows, "generation_error_rate"),
                "extraction_failure_rate": _weighted_rate(
                    rows,
                    "extraction_failure_rate",
                ),
                "contract_mismatch_rate": _weighted_rate(
                    rows,
                    "contract_mismatch_rate",
                ),
                "secret_findings": sum(
                    _int(row.get("secret_findings")) for row in rows
                ),
            }
        ],
    )


def _raw_table(name: str, path: Path) -> TableSpec:
    with path.open(newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))
        fields = list(rows[0]) if rows else []
    return TableSpec(name=name, fields=fields, rows=[dict(row) for row in rows])


def _aggregate_row(
    label_field: str,
    label: str,
    rows: list[dict[str, str]],
) -> dict[str, Any]:
    return {
        label_field: label,
        "num_cases": len(rows),
        "num_samples": _num_samples(rows),
        "pass@1": _macro_average(rows, "pass_at_1_estimated"),
        "pass@3": _macro_average(rows, "pass_at_3_estimated"),
        "pass@5": _macro_average(rows, "pass_at_5_estimated"),
        "pass@10": _macro_average(rows, "pass_at_10_estimated"),
        "mandatory_success_rate": _weighted_rate(rows, "mandatory_success_rate"),
        "framework_pass_rate": _weighted_rate(rows, "framework_pass_rate"),
        "parse_success_rate": _weighted_rate(rows, "parse_success_rate"),
        "generation_error_rate": _weighted_rate(rows, "generation_error_rate"),
        "avg_checkpoint_accuracy": _weighted_rate(rows, "avg_checkpoint_accuracy"),
    }


def _write_table(table: TableSpec, output_dir: Path) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / f"{table.name}.csv"
    md_path = output_dir / f"{table.name}.md"
    tex_path = output_dir / f"{table.name}.tex"
    _write_csv(table, csv_path)
    _write_markdown(table, md_path)
    _write_latex(table, tex_path)
    return [csv_path, md_path, tex_path]


def _write_csv(table: TableSpec, path: Path) -> None:
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=table.fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(table.rows)


def _write_markdown(table: TableSpec, path: Path) -> None:
    lines = [
        f"# {table.name}",
        "",
        "| " + " | ".join(table.fields) + " |",
        "| " + " | ".join("---" for _field in table.fields) + " |",
    ]
    for row in table.rows:
        values = [str(row.get(field, "")) for field in table.fields]
        lines.append("| " + " | ".join(values) + " |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_latex(table: TableSpec, path: Path) -> None:
    alignment = "l" + "r" * max(len(table.fields) - 1, 0)
    lines = [
        "\\begin{table}[t]",
        "\\centering",
        f"\\caption{{{_latex_escape(table.name.replace('_', ' ').title())}}}",
        f"\\begin{{tabular}}{{{alignment}}}",
        "\\toprule",
        " & ".join(_latex_escape(field) for field in table.fields) + " \\\\",
        "\\midrule",
    ]
    for row in table.rows:
        values = [_latex_escape(str(row.get(field, ""))) for field in table.fields]
        lines.append(" & ".join(values) + " \\\\")
    lines.extend(["\\bottomrule", "\\end{tabular}", "\\end{table}"])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _num_samples(rows: list[dict[str, str]]) -> int:
    return sum(_int(row.get("num_samples")) for row in rows)


def _macro_average(rows: list[dict[str, str]], field: str) -> str:
    values = [
        _float(row.get(field))
        for row in rows
        if row.get(field) not in {None, ""}
    ]
    if not values:
        return "0.000"
    return f"{sum(values) / len(values):.3f}"


def _weighted_rate(rows: list[dict[str, str]], field: str) -> str:
    total_weight = _num_samples(rows)
    if total_weight <= 0:
        return "0.000"
    numerator = sum(
        _float(row.get(field)) * _int(row.get("num_samples")) for row in rows
    )
    return f"{numerator / total_weight:.3f}"


def _int(value: str | None) -> int:
    try:
        return int(str(value))
    except (TypeError, ValueError):
        return 0


def _float(value: str | None) -> float:
    try:
        return float(str(value))
    except (TypeError, ValueError):
        return 0.0


def _bool(value: str | None) -> bool:
    return str(value).lower() in {"true", "1", "1.0", "yes"}


def _latex_escape(value: str) -> str:
    replacements = {
        "\\": "\\textbackslash{}",
        "&": "\\&",
        "%": "\\%",
        "$": "\\$",
        "#": "\\#",
        "_": "\\_",
        "{": "\\{",
        "}": "\\}",
    }
    return "".join(replacements.get(char, char) for char in value)


if __name__ == "__main__":
    main()
