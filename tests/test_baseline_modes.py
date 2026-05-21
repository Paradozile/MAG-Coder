"""Tests for paper-facing baseline input/method modes."""

from __future__ import annotations

from pathlib import Path

from magcoder.benchmark.full_runner import (
    FullBenchPassKConfig,
    FullMagBenchPassKRunner,
    build_full_bench_prompt_input,
    load_full_bench_run_cases,
)
from magcoder.benchmark.full_runner.input_modes import (
    baseline_family,
    mode_nl_condition,
)

ROOT = Path(__file__).resolve().parents[1]
PAPER2500_BENCH = ROOT / "benchmarks/full_mag_bench_2500_v1"


def test_zero_shot_and_few_shot_direct_prompts_render() -> None:
    """Direct baselines expose source prompts without MAG contract prose."""

    case = load_full_bench_run_cases(PAPER2500_BENCH, max_cases=1)[0]
    zero = build_full_bench_prompt_input(case, "zero_shot_direct")
    few = build_full_bench_prompt_input(case, "few_shot_direct")
    assert zero.source_kind == "direct"
    assert zero.hybrid_mode == "direct"
    assert "Functional interface requirement" not in zero.prompt_text
    assert "Example:" not in zero.prompt_text
    assert "Example:" in few.prompt_text


def test_without_nl_and_baseline_family_helpers() -> None:
    """Paper baseline labels are stable for table export."""

    assert mode_nl_condition("without_nl") == "graph_only"
    assert baseline_family("zero_shot_direct", "no_refinement") == "zero_shot_direct"
    assert baseline_family("structured_graph_nl", "without_ast") == "without_ast"
    assert baseline_family("structured_graph_nl", "mag_coder_full") == "mag_coder_full"


def test_without_ast_skips_refinement_in_fake_smoke(tmp_path: Path) -> None:
    """The without_ast method keeps initial code as the active sample artifact."""

    result = FullMagBenchPassKRunner().run(
        FullBenchPassKConfig(
            bench_path=str(PAPER2500_BENCH),
            output_dir=str(tmp_path / "baseline_smoke"),
            provider="fake",
            model_name="fake-generator",
            samples_per_case=1,
            k_values=[1],
            modes=["structured_graph_nl"],
            methods=["without_ast"],
            max_cases=1,
            overwrite=True,
            evaluation="all",
            refinement_strategy="balanced",
        )
    )
    sample = result.sample_results[0]
    assert sample.method == "without_ast"
    assert sample.code_refined_path is None
    assert sample.metadata["baseline_family"] == "without_ast"
