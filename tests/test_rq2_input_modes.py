"""Tests for Phase 8I.5 RQ2 input modes."""

from __future__ import annotations

import csv
from pathlib import Path

from magcoder.benchmark.full_runner import (
    FullBenchPassKConfig,
    FullMagBenchPassKRunner,
    build_full_bench_prompt_input,
    load_full_bench_run_cases,
)
from magcoder.benchmark.full_runner.input_modes import mode_family, mode_nl_condition

ROOT = Path(__file__).resolve().parents[1]
PAPER2500_BENCH = ROOT / "benchmarks/full_mag_bench_2500_v1"


def test_graph_only_prompt_omits_long_objectified_nl() -> None:
    """Graph-only keeps contracts while omitting objectified NL prompt text."""

    case = load_full_bench_run_cases(PAPER2500_BENCH, max_cases=1)[0]
    graph_nl = build_full_bench_prompt_input(case, "structured_graph_nl")
    graph_only = build_full_bench_prompt_input(case, "structured_graph_only")
    assert graph_nl.hybrid_input.mode == "graph+nl"
    assert graph_only.hybrid_input.mode == "graph-only"
    assert "Objectified prompt" in graph_nl.prompt_text
    assert "Expected behavior" in graph_nl.prompt_text
    assert "Objectified prompt" not in graph_only.prompt_text
    assert "Expected behavior" not in graph_only.prompt_text
    assert "Functional interface requirement" in graph_only.prompt_text
    assert "ABMS oracle interface requirement" in graph_only.prompt_text


def test_rq2_mode_helpers_classify_modes() -> None:
    """Mode helper fields support RQ2 grouping."""

    assert mode_family("structured_graph_only") == "structured"
    assert mode_family("graphical_graph_nl") == "graphical"
    assert mode_nl_condition("structured_graph_only") == "graph_only"
    assert mode_nl_condition("graphical_graph_nl") == "graph_nl"


def test_runner_writes_rq2_mode_fields_for_fake_run(tmp_path: Path) -> None:
    """Run summaries include mode_family and nl_condition fields."""

    output = tmp_path / "rq2_fake"
    result = FullMagBenchPassKRunner().run(
        FullBenchPassKConfig(
            bench_path=str(PAPER2500_BENCH),
            output_dir=str(output),
            provider="fake",
            model_name="fake-generator",
            samples_per_case=1,
            k_values=[1],
            modes=["structured_graph_nl", "structured_graph_only"],
            methods=["full_with_refinement"],
            max_cases=1,
            overwrite=True,
            evaluation="all",
            refinement_strategy="balanced",
        )
    )
    assert len(result.sample_results) == 2
    with (output / "sample_summary.csv").open(newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))
    assert {row["mode_family"] for row in rows} == {"structured"}
    assert {row["nl_condition"] for row in rows} == {"graph_nl", "graph_only"}
    assert {row["mode"] for row in rows} == {
        "structured_graph_nl",
        "structured_graph_only",
    }
