"""Tests for Full MAG-Bench runner integration."""

from __future__ import annotations

import csv
from pathlib import Path

from magcoder.benchmark.full_runner import (
    FullBenchPassKConfig,
    FullMagBenchPassKRunner,
    build_full_bench_prompt_input,
    load_full_bench_run_cases,
)
from magcoder.generation import FakeCodeGenerator, GenerationConfig, GenerationRequest

ROOT = Path(__file__).resolve().parents[1]
BENCH = ROOT / "benchmarks/full_mag_bench_v1"


def test_full_bench_case_loader_loads_seed_cases() -> None:
    """Seed Full MAG-Bench cases can be loaded for experiments."""

    cases = load_full_bench_run_cases(BENCH, max_cases=3)
    assert len(cases) == 3
    assert cases[0].mag_yaml_path.exists()
    assert cases[0].dot_path.exists()
    assert cases[0].manifest_path.exists()
    assert cases[0].expected_contract["required_methods"]


def test_structured_graph_nl_mode_generates_hybrid_input() -> None:
    """Structured graph+NL mode builds a promptable HybridInput."""

    case = load_full_bench_run_cases(BENCH, max_cases=1)[0]
    prompt_input = build_full_bench_prompt_input(case, "structured_graph_nl")
    assert prompt_input.hybrid_input.agents
    assert prompt_input.hybrid_input.edges
    assert "Agents" in prompt_input.prompt_text


def test_graphical_graph_nl_mode_round_trips_dot_manifest() -> None:
    """Graphical mode recovers MAG through DOT + manifest before prompting."""

    case = load_full_bench_run_cases(BENCH, max_cases=1)[0]
    prompt_input = build_full_bench_prompt_input(case, "graphical_graph_nl")
    assert prompt_input.mag.id == case.case_id
    prompt_input.mag.validate()


def test_fake_provider_can_generate_code_initial() -> None:
    """Full benchmark HybridInput can be consumed by the fake generator."""

    case = load_full_bench_run_cases(BENCH, max_cases=1)[0]
    prompt_input = build_full_bench_prompt_input(case, "structured_graph_nl")
    result = FakeCodeGenerator().generate(
        GenerationRequest(
            hybrid_input=prompt_input.hybrid_input,
            config=GenerationConfig(),
            prompt_text=prompt_input.prompt_text,
        )
    )
    assert "class" in result.code_initial
    assert result.code_initial.strip()


def test_full_bench_passk_runner_fake_smoke(tmp_path: Path) -> None:
    """Fake pass@k run writes sample outputs and contract-level summaries."""

    output = tmp_path / "fake_smoke"
    config = FullBenchPassKConfig(
        bench_path=str(BENCH),
        output_dir=str(output),
        provider="fake",
        model_name="fake-generator",
        samples_per_case=1,
        k_values=[1],
        modes=["structured_graph_nl", "graphical_graph_nl"],
        methods=["full_with_refinement", "no_refinement"],
        max_cases=1,
        overwrite=True,
    )
    result = FullMagBenchPassKRunner().run(config)
    assert len(result.sample_results) == 4
    assert (output / "summary.csv").exists()
    assert (output / "report.md").exists()
    assert any(sample.code_refined_path for sample in result.sample_results)
    assert any(sample.code_refined_path is None for sample in result.sample_results)

    with (output / "summary.csv").open(newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))
    assert rows
    assert {row["evaluation_level"] for row in rows} == {"contract"}
