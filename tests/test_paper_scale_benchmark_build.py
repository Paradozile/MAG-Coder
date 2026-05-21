"""Tests for paper-scale Full MAG-Bench build plumbing."""

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


def test_paper_scale_build_fixture_generates_full_case(tmp_path: Path) -> None:
    """A small official-style fixture builds with paper-scale metadata files."""

    source = _write_source_fixture(tmp_path / "official.jsonl")
    output = tmp_path / "paper_bench"
    package, report = build_full_mag_bench(
        source_paths=[source],
        output_dir=output,
        max_cases=None,
        formats=["structured", "graphical"],
        split="auto",
        seed=2026,
        skip_unsupported=True,
        write_source_loader_report=True,
        variants_per_task=1,
    )
    assert report.accepted_cases == 2
    assert len(package.cases) == 2
    assert (output / "source_loader_report.json").exists()
    assert (output / "case_index.json").exists()
    assert (output / "split_manifest.json").exists()

    validation = validate_full_mag_bench(output)
    assert validation.passed
    oracle = validate_oracles(output)
    assert oracle["passed"]
    abms = validate_abms_oracles(output)
    assert abms["passed"]


def test_paper_scale_fake_smoke_uses_fake_provider(tmp_path: Path) -> None:
    """The paper-scale fixture can run fake smoke without a real provider."""

    source = _write_source_fixture(tmp_path / "official.jsonl")
    bench = tmp_path / "paper_bench"
    build_full_mag_bench(source_paths=[source], output_dir=bench, max_cases=1)
    output = tmp_path / "fake_smoke"
    result = FullMagBenchPassKRunner().run(
        FullBenchPassKConfig(
            bench_path=str(bench),
            output_dir=str(output),
            provider="fake",
            model_name="fake-generator",
            samples_per_case=1,
            k_values=[1],
            modes=["structured_graph_nl", "graphical_graph_nl"],
            methods=["full_with_refinement", "no_refinement"],
            evaluation="all",
            overwrite=True,
        )
    )
    assert len(result.sample_results) == 4
    assert (output / "summary.csv").exists()


def _write_source_fixture(path: Path) -> Path:
    rows = [
        {
            "task_id": "HumanEval/fixture_add",
            "source": "humaneval",
            "prompt": "def add_one(x):\n    return x + 1\n",
            "canonical_solution": "def add_one(x):\n    return x + 1\n",
            "tests": ["assert add_one(1) == 2"],
            "entry_point": "add_one",
        },
        {
            "task_id": "mbpp_fixture_square",
            "source": "mbpp",
            "prompt": "Square a number.",
            "canonical_solution": "def square(x):\n    return x * x\n",
            "tests": ["assert square(3) == 9"],
            "entry_point": "square",
        },
    ]
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )
    return path
