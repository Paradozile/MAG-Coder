"""Tests for ABMS-oriented deterministic refinement and prompt contracts."""

from __future__ import annotations

from pathlib import Path

from magcoder.benchmark.full_builder import build_full_mag_bench
from magcoder.benchmark.full_runner import (
    build_full_bench_prompt_input,
    load_full_bench_run_cases,
)
from magcoder.models import load_mag_from_yaml
from magcoder.refinement import load_framework_spec_from_yaml, refine_code

ROOT = Path(__file__).resolve().parents[1]
SOURCES = [ROOT / "examples/source_tasks/medium_humaneval_style_seed.jsonl"]


def test_refiner_adds_abms_safe_init_and_message_api(tmp_path: Path) -> None:
    """Refinement adds structural ABMS helpers without copying gold logic."""

    bench = _build_one_case(tmp_path)
    case = load_full_bench_run_cases(bench, max_cases=1)[0]
    mag = load_mag_from_yaml(case.mag_yaml_path)
    framework = load_framework_spec_from_yaml(case.framework_path)
    source_class = case.abms_oracle_spec.required_candidate_symbols[0]  # type: ignore[union-attr]
    target_class = case.abms_oracle_spec.required_candidate_symbols[1]  # type: ignore[union-attr]
    method = case.abms_oracle_spec.required_candidate_symbols[2]  # type: ignore[union-attr]
    candidate = f"""
class {source_class}:
    def {method}(self):
        return None

class {target_class}:
    pass
"""
    result = refine_code(candidate, mag, framework)
    assert "def __init__(self, *args, **kwargs):" in result.code_refined
    assert "def send_message(self, target, message=None):" in result.code_refined
    assert "self.sent_messages" in result.code_refined
    assert f"def {method}(self, target):" in result.code_refined
    assert any(
        action.startswith("ensure_abms_attribute_initialization")
        for action in result.rewrite_actions
    )
    assert "reverse_words" not in result.code_refined


def test_full_prompt_includes_abms_interface_constraints(tmp_path: Path) -> None:
    """Full runner prompts include machine-readable ABMS interface requirements."""

    bench = _build_one_case(tmp_path)
    case = load_full_bench_run_cases(bench, max_cases=1)[0]
    prompt = build_full_bench_prompt_input(case, "structured_graph_only").prompt_text
    assert "ABMS oracle interface requirement" in prompt
    assert "Initialize every MAG attribute" in prompt
    assert "Interaction methods MUST accept self" in prompt
    assert "Required candidate symbols" in prompt


def _build_one_case(tmp_path: Path) -> Path:
    bench = tmp_path / "bench"
    build_full_mag_bench(
        source_paths=SOURCES,
        output_dir=bench,
        max_cases=1,
        formats=["structured", "graphical"],
    )
    return bench
