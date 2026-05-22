"""Tests for the clean GitHub code-release surface."""

from __future__ import annotations

import json
from pathlib import Path

from scripts.audit_release_tree import (
    REQUIRED_FEATURES,
    REQUIRED_IGNORE_PATTERNS,
    audit_release_tree,
)
from scripts.run_code_submission_smoke import run_code_submission_smoke

ROOT = Path(__file__).resolve().parents[1]


def test_release_manifest_has_required_paper_features() -> None:
    """The JSON release manifest replaces process Markdown matrices."""

    manifest = json.loads((ROOT / "release_manifest.json").read_text(encoding="utf-8"))
    features = " ".join(manifest["implemented_paper_features"])
    for item in [
        "structured MAG input",
        "graphical DOT+manifest input",
        "graph+NL and graph-only modes",
        "zero-shot and few-shot baselines",
        "AST refinement",
        "MAG-Bench-Func export",
        "ABMS evaluation",
        "pass@1/3/5/10/50/100 aggregation",
    ]:
        assert item in features
    bounded = json.dumps(manifest["bounded_or_interface_only_features"])
    assert "interface_only" in bounded
    assert "arbitrary PNG/VLM recognition is not claimed" in bounded


def test_release_audit_detects_missing_required_script(tmp_path: Path) -> None:
    """Release audit reports missing required paths in a fixture repository."""

    repo = tmp_path / "repo"
    _make_minimal_repo(repo, omit="scripts/build_full_mag_bench.py")
    report = audit_release_tree(repo_root=repo, output_dir=tmp_path / "audit")
    assert report["ready"] is False
    assert any("scripts/build_full_mag_bench.py" in error for error in report["errors"])


def test_gitignore_includes_generated_artifact_dirs() -> None:
    """Generated and process artifacts are ignored for code-only submission."""

    text = (ROOT / ".gitignore").read_text(encoding="utf-8")
    for pattern in REQUIRED_IGNORE_PATTERNS:
        assert pattern in text
    assert "benchmarks/full_mag_bench_2500_v1/" in text
    assert ".release_audit/" in text
    assert "*.md" in text
    assert "!/README.md" in text


def test_code_submission_smoke_does_not_call_real_provider(tmp_path: Path) -> None:
    """The final smoke uses only fake/mock local functionality."""

    report = run_code_submission_smoke(tmp_path / "smoke")
    assert report["passed"] is True
    assert report["fake_provider_smoke"]["real_provider_invoked"] is False
    assert report["vlm_mock"]["real_vlm_invoked"] is False
    assert report["provider_registry"]["external_calls_made"] is False


def test_release_docs_do_not_contain_local_absolute_paths() -> None:
    """Release package-level docs avoid machine-local paths."""

    for path in [ROOT / "README.md", ROOT / "release_manifest.json"]:
        text = path.read_text(encoding="utf-8")
        assert "/Users/" not in text
        assert "/home/" not in text
        assert "C:\\Users\\" not in text
        assert "paradozile" not in text.lower()


def test_interface_only_and_overclaim_boundaries_are_preserved() -> None:
    """Release wording preserves interface-only status and avoids overclaims."""

    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    manifest = (ROOT / "release_manifest.json").read_text(encoding="utf-8")
    combined = readme + "\n" + manifest
    assert "interface-only" in combined or "interface_only" in combined
    assert "not claimed as completed" in combined
    assert "full 2500 exhaustive real-model result" not in combined
    assert "pass@10 real result" not in combined
    assert "proves multi-model generality" not in combined
    assert "arbitrary PNG/VLM recognition is implemented" not in combined


def _make_minimal_repo(repo: Path, *, omit: str) -> None:
    repo.mkdir(parents=True)
    (repo / ".gitignore").write_text(
        "\n".join(REQUIRED_IGNORE_PATTERNS) + "\n",
        encoding="utf-8",
    )
    (repo / "pyproject.toml").write_text(
        """
[project]
name = "mag-coder"

[tool.ruff]
exclude = ["benchmarks/full_mag_bench_2500_v1", ".release_audit"]

[tool.mypy]
exclude = "^(.release_audit|benchmarks)/"
""".lstrip(),
        encoding="utf-8",
    )
    (repo / "README.md").write_text(_fixture_readme(), encoding="utf-8")
    (repo / "release_manifest.json").write_text(
        json.dumps({"implemented_paper_features": []}) + "\n",
        encoding="utf-8",
    )
    for feature in REQUIRED_FEATURES:
        for raw_path in [
            feature.module_path,
            feature.cli_entrypoint,
            feature.test_file,
        ]:
            if raw_path == omit:
                continue
            _write_fixture_path(repo / raw_path)
    for raw_path in [
        "src/magcoder/__init__.py",
        "configs/providers/openai_compatible_template.yaml",
        "examples/source_tasks/humaneval_style_seed.jsonl",
        "scripts/download_official_source_tasks.py",
        "scripts/build_full_mag_bench.py",
        "scripts/validate_full_mag_bench.py",
        "scripts/validate_full_mag_bench_oracles.py",
        "scripts/validate_full_mag_bench_adapters.py",
        "scripts/validate_full_mag_bench_abms_oracles.py",
        "scripts/run_full_mag_bench_passk.py",
        "scripts/check_full_mag_bench_real_run_readiness.py",
        "scripts/select_2500_stratified_cases.py",
        "scripts/analyze_paper2500_passk.py",
        "scripts/analyze_rq2_ablation.py",
        "scripts/export_mag_bench_func_views.py",
        "scripts/export_paper_functionality_tables.py",
        "scripts/render_mag_graphics.py",
        "scripts/extract_mag_from_rendered_graph.py",
        "scripts/scan_experiment_secrets.py",
        "scripts/scan_submission_paths.py",
        "scripts/audit_release_tree.py",
        "scripts/run_code_submission_smoke.py",
    ]:
        if raw_path != omit:
            _write_fixture_path(repo / raw_path)


def _write_fixture_path(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("# fixture\n", encoding="utf-8")


def _fixture_readme() -> str:
    return (
        """
# MAG-Coder

Generated large benchmark outputs and frozen experiment results are excluded.
Real-model runs require local model availability and are never auto-pulled.
Multi-model provider templates are included, but multi-model real validation
is not included.
arbitrary PNG/VLM recognition is exposed as an optional/mock interface.
Full 2500-case exhaustive real-model evaluation is not included.
Source-functional correctness, contract conformance, and ABMS conformance are
separate metrics.
"""
    ).lstrip()
