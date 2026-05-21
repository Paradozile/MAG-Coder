"""Regression tests for residual ABMS runtime repair rules."""

from __future__ import annotations

from pathlib import Path

from magcoder.benchmark.full_builder import build_full_mag_bench
from magcoder.benchmark.full_runner import load_full_bench_run_cases
from magcoder.models import load_mag_from_yaml
from magcoder.refinement import load_framework_spec_from_yaml, refine_code

ROOT = Path(__file__).resolve().parents[1]
SOURCES = [ROOT / "examples/source_tasks/medium_humaneval_style_seed.jsonl"]


def test_refiner_removes_begin_end_python_markers(tmp_path: Path) -> None:
    """Output-format markers are removed instead of becoming NameError at import."""

    case = _build_case(tmp_path)
    mag = load_mag_from_yaml(case.mag_yaml_path)
    framework = load_framework_spec_from_yaml(case.framework_path)
    spec = case.abms_oracle_spec
    assert spec is not None
    source_class, target_class, method = spec.required_candidate_symbols[:3]
    candidate = f"""
BEGIN_PYTHON
class {source_class}:
    def {method}(self, target):
        self.send_message(target, "ok")

class {target_class}:
    pass
END_PYTHON
"""
    result = refine_code(candidate, mag, framework)
    assert "BEGIN_PYTHON" not in result.code_refined
    assert "END_PYTHON" not in result.code_refined
    assert "remove_output_format_markers" in result.rewrite_actions


def test_refiner_normalizes_message_api_argument_order(tmp_path: Path) -> None:
    """Reversed message/target arguments trigger a contract-compliant override."""

    case = _build_case(tmp_path)
    mag = load_mag_from_yaml(case.mag_yaml_path)
    framework = load_framework_spec_from_yaml(case.framework_path)
    spec = case.abms_oracle_spec
    assert spec is not None
    source_class, target_class, method = spec.required_candidate_symbols[:3]
    candidate = f"""
class {source_class}:
    def {method}(self, target):
        message = {{"value": 1}}
        self.send_message(message, target)

class {target_class}:
    pass
"""
    result = refine_code(candidate, mag, framework)
    assert f"def {method}(self, target):" in result.code_refined
    assert any(
        action.startswith("normalize_message_api_argument_order")
        for action in result.rewrite_actions
    )
    assert "gold_solution" not in result.code_refined


def test_refiner_normalizes_missing_message_api_argument(tmp_path: Path) -> None:
    """A one-argument send_message call is repaired by overriding the method."""

    case = _build_case(tmp_path)
    mag = load_mag_from_yaml(case.mag_yaml_path)
    framework = load_framework_spec_from_yaml(case.framework_path)
    spec = case.abms_oracle_spec
    assert spec is not None
    source_class, target_class, method = spec.required_candidate_symbols[:3]
    candidate = f"""
class {source_class}:
    def {method}(self, target):
        self.send_message("payload")

class {target_class}:
    pass
"""
    result = refine_code(candidate, mag, framework)
    assert any(
        action.startswith("normalize_message_api_signature")
        for action in result.rewrite_actions
    )
    assert "KeyError" not in result.code_refined
    assert "RecursionError" not in result.code_refined


def _build_case(tmp_path: Path):
    bench = tmp_path / "bench"
    build_full_mag_bench(
        source_paths=SOURCES,
        output_dir=bench,
        max_cases=1,
        formats=["structured", "graphical"],
    )
    return load_full_bench_run_cases(bench, max_cases=1)[0]
