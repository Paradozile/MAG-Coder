"""Tests for paper2500-style benchmark construction."""

from __future__ import annotations

import json
from pathlib import Path

from scripts.validate_full_mag_bench_abms_oracles import validate_abms_oracles
from scripts.validate_full_mag_bench_oracles import validate_oracles

from magcoder.benchmark.full_builder import (
    build_full_mag_bench,
    validate_full_mag_bench,
)
from magcoder.benchmark.full_runner import FullBenchPassKConfig, FullMagBenchPassKRunner


def test_target_cases_build_emits_variant_specs(tmp_path: Path) -> None:
    """Target-cases builds accept a deterministic subset of variants."""

    source = _write_tasks(tmp_path / "tasks.jsonl", count=4)
    bench = tmp_path / "bench2500_fixture"
    package, report = build_full_mag_bench(
        source_paths=[source],
        output_dir=bench,
        target_cases=7,
        variants_per_task=3,
        variant_policy="paper2500",
        split="auto",
        skip_unsupported=True,
        overwrite=True,
    )

    assert report.accepted_cases == 7
    assert report.candidate_variants_generated == 12
    assert report.target_reached
    assert (bench / "variant_generation_report.json").exists()
    assert (bench / "variant_filter_report.json").exists()
    assert (bench / "paper2500_reproduction_notes.md").exists()
    assert (bench / package.cases[0].case_dir / "variant_spec.json").exists()
    assert validate_full_mag_bench(bench).passed
    assert validate_oracles(bench)["passed"]
    assert validate_abms_oracles(bench)["passed"]


def test_paper2500_fake_smoke_does_not_invoke_real_provider(tmp_path: Path) -> None:
    """A small paper2500 fixture can run fake smoke only."""

    source = _write_tasks(tmp_path / "tasks.jsonl", count=2)
    bench = tmp_path / "bench"
    build_full_mag_bench(
        source_paths=[source],
        output_dir=bench,
        target_cases=3,
        variants_per_task=2,
        variant_policy="paper2500",
        overwrite=True,
    )
    output = tmp_path / "fake"
    result = FullMagBenchPassKRunner().run(
        FullBenchPassKConfig(
            bench_path=str(bench),
            output_dir=str(output),
            provider="fake",
            model_name="fake-generator",
            samples_per_case=1,
            k_values=[1],
            modes=["structured_graph_nl"],
            methods=["full_with_refinement"],
            evaluation="all",
            refinement_strategy="balanced",
            overwrite=True,
        )
    )

    assert result.sample_results
    assert (output / "summary.csv").exists()
    assert (output / "selected_cases.json").exists()


def _write_tasks(path: Path, *, count: int) -> Path:
    rows = [
        {
            "task_id": f"fixture_{index}",
            "source": "fixture",
            "prompt": "Return x.",
            "canonical_solution": "def solve(x):\n    return x\n",
            "tests": ["assert solve(1) == 1"],
            "entry_point": "solve",
        }
        for index in range(count)
    ]
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )
    return path
