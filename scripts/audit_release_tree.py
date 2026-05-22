"""Audit MAG-Coder's code-only GitHub release tree.

The audit is intentionally conservative: it checks the source tree and ignore
rules, reports local non-submission artifacts, and verifies claim-boundary
wording. It never removes files and never calls providers.
"""

from __future__ import annotations

import argparse
import fnmatch
import json
import re
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ReleaseFeature:
    """One paper-facing functionality surface expected in the release."""

    feature: str
    module_path: str
    cli_entrypoint: str
    test_file: str
    status: str
    note: str


REQUIRED_FEATURES: tuple[ReleaseFeature, ...] = (
    ReleaseFeature(
        "MAG-Coder method pipeline",
        "src/magcoder/benchmark/full_runner/passk_runner.py",
        "scripts/run_full_mag_bench_passk.py",
        "tests/test_full_benchmark_runner.py",
        "implemented",
        "Structured/graphical generation, evaluation, and reporting entry point.",
    ),
    ReleaseFeature(
        "official HumanEval/MBPP downloader",
        "src/magcoder/benchmark/full_builder/task_loader.py",
        "scripts/download_official_source_tasks.py",
        "tests/test_official_source_task_download.py",
        "implemented",
        "Allowlisted source-task download and normalization.",
    ),
    ReleaseFeature(
        "template variant generation",
        "src/magcoder/benchmark/full_builder/template_variants.py",
        "scripts/build_full_mag_bench.py",
        "tests/test_template_variant_generation.py",
        "implemented",
        "Deterministic paper-scale variant generation.",
    ),
    ReleaseFeature(
        "variant filtering",
        "src/magcoder/benchmark/full_builder/variant_filters.py",
        "scripts/build_full_mag_bench.py",
        "tests/test_variant_filtering.py",
        "implemented",
        "Duplicate and illogical variant filtering.",
    ),
    ReleaseFeature(
        "benchmark validation gates",
        "src/magcoder/benchmark/full_builder/validators.py",
        "scripts/validate_full_mag_bench.py",
        "tests/test_paper_scale_benchmark_build.py",
        "implemented",
        "Package/source/adapter/ABMS validation surfaces.",
    ),
    ReleaseFeature(
        "structured and graphical input modes",
        "src/magcoder/benchmark/full_runner/input_modes.py",
        "scripts/run_full_mag_bench_passk.py",
        "tests/test_rq2_input_modes.py",
        "implemented",
        "graph+NL and graph-only RQ2 modes.",
    ),
    ReleaseFeature(
        "zero-shot and few-shot baselines",
        "src/magcoder/benchmark/full_runner/input_modes.py",
        "scripts/run_full_mag_bench_passk.py",
        "tests/test_baseline_modes.py",
        "implemented",
        "MAG-free direct baseline prompt surfaces.",
    ),
    ReleaseFeature(
        "AST refinement",
        "src/magcoder/refinement/refiner.py",
        "scripts/run_full_mag_bench_passk.py",
        "tests/test_refiner.py",
        "implemented",
        "Deterministic AST/framework/MAG compatibility rules.",
    ),
    ReleaseFeature(
        "MAG-Bench-Func view",
        "src/magcoder/benchmark/full_runner/functional_evaluator.py",
        "scripts/export_mag_bench_func_views.py",
        "tests/test_mag_bench_func_export.py",
        "implemented",
        "Functional-only view separate from contract and ABMS.",
    ),
    ReleaseFeature(
        "ABMS oracle",
        "src/magcoder/benchmark/full_runner/abms_functional_evaluator.py",
        "scripts/validate_full_mag_bench_abms_oracles.py",
        "tests/test_abms_functional_evaluator.py",
        "implemented",
        "Agent-interaction-level oracle, not full simulation validation.",
    ),
    ReleaseFeature(
        "pass@k aggregation",
        "src/magcoder/benchmark/full_runner/passk_runner.py",
        "scripts/analyze_paper2500_passk.py",
        "tests/test_passk_extended_metrics.py",
        "implemented",
        "pass@1/3/5/10/50/100 with N/A for k greater than n.",
    ),
    ReleaseFeature(
        "provider registry",
        "src/magcoder/providers/provider_registry.py",
        "configs/providers/openai_compatible_template.yaml",
        "tests/test_provider_registry.py",
        "interface-only",
        "Config templates only; no credentials and no external calls.",
    ),
    ReleaseFeature(
        "VLM/PNG mock interface",
        "src/magcoder/graphics/vlm_extractor.py",
        "scripts/extract_mag_from_rendered_graph.py",
        "tests/test_vlm_graphics_interface.py",
        "interface-only",
        "Mock/manifest extraction only; arbitrary PNG recognition is future work.",
    ),
    ReleaseFeature(
        "paper table export",
        "scripts/export_paper_functionality_tables.py",
        "scripts/export_paper_functionality_tables.py",
        "tests/test_paper_functionality_tables.py",
        "implemented",
        "Exports table-ready CSVs from user-supplied run artifacts.",
    ),
    ReleaseFeature(
        "secret and path scans",
        "scripts/scan_experiment_secrets.py",
        "scripts/scan_submission_paths.py",
        "tests/test_final_code_submission.py",
        "implemented",
        "Release hygiene utilities.",
    ),
)


REQUIRED_CLI_SCRIPTS = (
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
)


REQUIRED_IGNORE_PATTERNS = (
    "benchmarks/**/passk_runs/",
    "benchmarks/**/frozen_results/",
    "benchmarks/full_mag_bench_2500_v1/",
    "benchmarks/full_mag_bench_paper_v1/",
    "benchmarks/full_mag_bench_medium_v1/",
    "submission_artifacts/",
    "paper_artifacts/",
    "paper_artifacts_8i/",
    "paper_integration_artifacts/",
    "claim_artifacts/",
    "claim_artifacts_8i5/",
    "claim_artifacts_8j/",
    "paper_functionality_artifacts/",
    "final_submission_audit/",
    ".release_audit/",
    "audits/",
    "data/source_tasks/official_raw/",
    "data/source_tasks/*_official.jsonl",
    "data/source_tasks/official_*report*.json",
    "data/source_tasks/official_*manifest*.json",
    "*.log",
    ".pytest_cache/",
    ".mypy_cache/",
    ".ruff_cache/",
    "__pycache__/",
    "*.pyc",
    "*.md",
    "!/README.md",
)


FORBIDDEN_RELEASE_PATTERNS = (
    "benchmarks/**/passk_runs/*",
    "benchmarks/**/frozen_results/*",
    "benchmarks/full_mag_bench_2500_v1/*",
    "benchmarks/full_mag_bench_paper_v1/*",
    "benchmarks/full_mag_bench_medium_v1/*",
    "submission_artifacts/*",
    "paper_artifacts/*",
    "paper_artifacts_8i/*",
    "paper_integration_artifacts/*",
    "claim_artifacts/*",
    "claim_artifacts_8i5/*",
    "claim_artifacts_8j/*",
    "paper_functionality_artifacts/*",
    "final_submission_audit/*",
    ".release_audit/*",
    "data/source_tasks/official_raw/*",
    "data/source_tasks/*_official.jsonl",
    "data/source_tasks/official_*report*.json",
    "data/source_tasks/official_*manifest*.json",
)


LOCAL_PATH_PATTERNS = (
    re.compile(r"/Users/"),
    re.compile(r"/home/"),
    re.compile(r"C:\\\\Users\\\\"),
    re.compile(r"/private/var/"),
    re.compile(r"\bparadozile\b", re.IGNORECASE),
)


SECRET_PATTERNS = (
    re.compile(r"sk-[A-Za-z0-9_-]{20,}"),
    re.compile(r"AIza[0-9A-Za-z_-]{20,}"),
    re.compile(r"(?i)(api[_-]?key|secret|token)\s*[:=]\s*['\"][^'\"]{8,}['\"]"),
)


SCAN_SUFFIXES = {".py", ".yaml", ".yml", ".json", ".toml", ".txt", ".md"}


def main() -> None:
    """CLI entry point."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = audit_release_tree(repo_root=args.repo_root, output_dir=args.output)
    print(f"Release tree audit: {args.output}")
    print(f"Ready: {report['ready']}")


def audit_release_tree(
    *, repo_root: str | Path, output_dir: str | Path
) -> dict[str, Any]:
    """Audit the release tree and write JSON/Markdown reports."""

    root = Path(repo_root).resolve()
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    feature_rows = [_feature_row(root, feature) for feature in REQUIRED_FEATURES]
    required_files = _required_file_report(root)
    ignore_report = _ignore_report(root)
    markdown_report = _markdown_report(root)
    artifact_report = _artifact_report(root)
    tracked_report = _tracked_file_report(root)
    path_report = _path_report(root)
    secret_report = _secret_report(root)
    provider_report = _provider_template_report(root)
    wording_report = _claim_wording_report(root)
    pyproject_report = _pyproject_report(root)

    errors: list[str] = []
    for section in (
        required_files,
        ignore_report,
        markdown_report,
        tracked_report,
        path_report,
        secret_report,
        provider_report,
        wording_report,
        pyproject_report,
    ):
        errors.extend(section["errors"])
    errors.extend(row["error"] for row in feature_rows if row["error"])

    warnings: list[str] = []
    warnings.extend(artifact_report["warnings"])
    warnings.extend(tracked_report["warnings"])

    report = {
        "schema_version": 1,
        "ready": not errors,
        "errors": errors,
        "warnings": warnings,
        "features": feature_rows,
        "required_files": required_files,
        "ignore_report": ignore_report,
        "markdown_report": markdown_report,
        "artifact_report": artifact_report,
        "tracked_file_report": tracked_report,
        "path_report": path_report,
        "secret_report": secret_report,
        "provider_template_report": provider_report,
        "claim_wording_report": wording_report,
        "pyproject_report": pyproject_report,
    }
    _write_json(report, output / "release_tree_report.json")
    (output / "RELEASE_READINESS.md").write_text(
        _render_readiness(report), encoding="utf-8"
    )
    (output / "non_submission_artifacts.md").write_text(
        _render_non_submission_artifacts(artifact_report),
        encoding="utf-8",
    )
    return report


def _feature_row(root: Path, feature: ReleaseFeature) -> dict[str, Any]:
    paths = [feature.module_path, feature.cli_entrypoint, feature.test_file]
    missing = [path for path in paths if not (root / path).exists()]
    row = asdict(feature)
    row.update(
        {
            "present": not missing,
            "missing_paths": missing,
            "error": (
                "" if not missing else f"missing release feature: {feature.feature}"
            ),
        }
    )
    return row


def _required_file_report(root: Path) -> dict[str, Any]:
    required = [
        "README.md",
        ".gitignore",
        "pyproject.toml",
        "release_manifest.json",
        "src/magcoder/__init__.py",
        "configs/providers/openai_compatible_template.yaml",
        "examples/source_tasks/humaneval_style_seed.jsonl",
        *REQUIRED_CLI_SCRIPTS,
    ]
    missing = [path for path in required if not (root / path).exists()]
    return {
        "required_files": required,
        "missing_files": missing,
        "errors": [f"missing required release file: {path}" for path in missing],
    }


def _ignore_report(root: Path) -> dict[str, Any]:
    path = root / ".gitignore"
    text = path.read_text(encoding="utf-8") if path.exists() else ""
    missing = [pattern for pattern in REQUIRED_IGNORE_PATTERNS if pattern not in text]
    return {
        "gitignore_exists": path.exists(),
        "required_patterns": list(REQUIRED_IGNORE_PATTERNS),
        "missing_patterns": missing,
        "errors": [f".gitignore missing pattern: {pattern}" for pattern in missing],
    }


def _markdown_report(root: Path) -> dict[str, Any]:
    md_files = sorted(
        _rel(path, root)
        for path in root.rglob("*.md")
        if _should_scan_local_file(path, root)
    )
    non_readme = [path for path in md_files if path != "README.md"]
    candidates = [
        path for path in non_readme if not _is_ignored_by_release(root / path, root)
    ]
    return {
        "markdown_files": md_files,
        "non_readme_markdown_files": non_readme,
        "non_readme_markdown_release_candidates": candidates,
        "errors": [
            f"non-README markdown would be included in release: {path}"
            for path in candidates
        ],
    }


def _artifact_report(root: Path) -> dict[str, Any]:
    all_files = [
        _rel(path, root)
        for path in root.rglob("*")
        if path.is_file() and _should_scan_local_file(path, root)
    ]
    non_submission = [
        path
        for path in all_files
        if _matches_any(path, FORBIDDEN_RELEASE_PATTERNS)
        or (path.endswith(".md") and path != "README.md")
        or path.endswith(".pyc")
    ]
    ignored = [
        path
        for path in non_submission
        if _is_ignored_by_release(root / path, root)
    ]
    not_ignored = [path for path in non_submission if path not in ignored]
    warnings = [
        f"local non-submission artifact is not ignored: {path}" for path in not_ignored
    ]
    return {
        "non_submission_artifacts": non_submission,
        "ignored_non_submission_artifacts": ignored,
        "not_ignored_non_submission_artifacts": not_ignored,
        "warnings": warnings,
    }


def _tracked_file_report(root: Path) -> dict[str, Any]:
    git_root = _git_root(root)
    tracked = _git_ls_files(root)
    forbidden = [
        path
        for path in tracked
        if _matches_any(path, FORBIDDEN_RELEASE_PATTERNS)
        or (path.endswith(".md") and path != "README.md")
    ]
    warnings: list[str] = []
    if git_root and git_root != root:
        warnings.append(f"git root differs from repo root: {git_root}")
    if not tracked:
        warnings.append("no tracked files reported for repo root")
    return {
        "git_root": str(git_root) if git_root else "",
        "repo_root": str(root),
        "tracked_file_count": len(tracked),
        "forbidden_tracked_files": forbidden,
        "errors": [f"forbidden tracked release file: {path}" for path in forbidden],
        "warnings": warnings,
    }


def _path_report(root: Path) -> dict[str, Any]:
    scanned = [
        root / "README.md",
        root / "release_manifest.json",
        root / "src",
        root / "configs",
    ]
    findings: list[str] = []
    for path in _iter_files(scanned, root):
        text = path.read_text(encoding="utf-8", errors="ignore")
        for pattern in LOCAL_PATH_PATTERNS:
            if pattern.search(text):
                findings.append(
                    f"local absolute path or username in {_rel(path, root)}"
                )
                break
    return {"findings": findings, "errors": findings}


def _secret_report(root: Path) -> dict[str, Any]:
    findings: list[str] = []
    scan_roots = [
        root / "README.md",
        root / "release_manifest.json",
        root / "src",
        root / "scripts",
        root / "configs",
        root / "examples/source_tasks",
        root / "tests",
    ]
    for path in _iter_files(scan_roots, root):
        text = path.read_text(encoding="utf-8", errors="ignore")
        for pattern in SECRET_PATTERNS:
            match = pattern.search(text)
            if match and not _placeholder_secret_match(match.group(0)):
                findings.append(f"possible secret in {_rel(path, root)}")
                break
    return {"findings": findings, "errors": findings}


def _provider_template_report(root: Path) -> dict[str, Any]:
    errors: list[str] = []
    template_dir = root / "configs/providers"
    for path in sorted(template_dir.glob("*.yaml")):
        text = path.read_text(encoding="utf-8", errors="ignore").lower()
        rel = _rel(path, root)
        if "template" in path.name and "template" not in text:
            errors.append(f"provider template missing template wording: {rel}")
        if (
            "api key" not in text
            and "credential" not in text
            and "local_only: true" not in text
        ):
            errors.append(f"provider template lacks credential boundary: {rel}")
    return {"errors": errors}


def _claim_wording_report(root: Path) -> dict[str, Any]:
    readme = (root / "README.md").read_text(encoding="utf-8")
    normalized = " ".join(readme.split())
    required = [
        "Generated large benchmark outputs and frozen experiment results are excluded",
        "Real-model runs require local model availability and are never auto-pulled",
        "Multi-model provider templates are included, "
        "but multi-model real validation is not included",
        "arbitrary PNG/VLM recognition is exposed as an optional/mock interface",
        "Full 2500-case exhaustive real-model evaluation is not included",
        "Source-functional correctness, contract conformance, "
        "and ABMS conformance are separate metrics",
    ]
    unsupported = [
        "full 2500 exhaustive real-model result",
        "proves multi-model generality",
        "arbitrary PNG/VLM recognition is implemented",
        "pass@10 real result",
    ]
    missing = [item for item in required if item not in normalized]
    overclaims = [item for item in unsupported if item in normalized]
    errors = [f"README missing claim boundary: {item}" for item in missing]
    errors.extend(f"README contains unsupported claim: {item}" for item in overclaims)
    return {
        "missing_required_boundaries": missing,
        "overclaims": overclaims,
        "errors": errors,
    }


def _pyproject_report(root: Path) -> dict[str, Any]:
    path = root / "pyproject.toml"
    if not path.exists():
        return {"errors": ["pyproject.toml missing"]}
    text = path.read_text(encoding="utf-8")
    errors: list[str] = []
    for required in ["name = \"mag-coder\"", "[tool.ruff]", "[tool.mypy]"]:
        if required not in text:
            errors.append(f"pyproject missing {required}")
    for excluded in ["benchmarks/full_mag_bench_2500_v1", ".release_audit"]:
        if excluded not in text:
            errors.append(f"pyproject excludes missing generated artifact: {excluded}")
    return {"errors": errors}


def _iter_files(paths: list[Path], root: Path) -> list[Path]:
    files: list[Path] = []
    for path in paths:
        if path.is_file() and path.suffix in SCAN_SUFFIXES:
            files.append(path)
        elif path.is_dir():
            files.extend(
                child
                for child in path.rglob("*")
                if child.is_file()
                and child.suffix in SCAN_SUFFIXES
                and _should_scan_local_file(child, root)
            )
    return files


def _should_scan_local_file(path: Path, root: Path) -> bool:
    if path.is_symlink():
        return False
    rel = _rel(path, root)
    ignored_dirs = (
        ".git/",
        ".venv/",
        ".venv_",
        "__pycache__/",
        ".pytest_cache/",
        ".mypy_cache/",
        ".ruff_cache/",
    )
    return not any(rel.startswith(prefix) for prefix in ignored_dirs)


def _git_root(root: Path) -> Path | None:
    result = subprocess.run(
        ["git", "-C", str(root), "rev-parse", "--show-toplevel"],
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        return None
    return Path(result.stdout.strip()).resolve()


def _git_ls_files(root: Path) -> list[str]:
    result = subprocess.run(
        ["git", "-C", str(root), "ls-files"],
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        return []
    git_root = _git_root(root)
    raw = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    if not git_root or git_root == root:
        return raw
    prefix = root.relative_to(git_root).as_posix() + "/"
    return [line.removeprefix(prefix) for line in raw if line.startswith(prefix)]


def _is_ignored_by_release(path: Path, root: Path) -> bool:
    rel = _rel(path, root)
    ignored = False
    for pattern in _gitignore_patterns(root):
        negated = pattern.startswith("!")
        raw_pattern = pattern[1:] if negated else pattern
        if _matches_pattern(rel, raw_pattern):
            ignored = not negated
    return ignored


def _gitignore_patterns(root: Path) -> list[str]:
    path = root / ".gitignore"
    if not path.exists():
        return []
    patterns: list[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if stripped and not stripped.startswith("#"):
            patterns.append(stripped)
    return patterns


def _matches_any(path: str, patterns: tuple[str, ...]) -> bool:
    return any(_matches_pattern(path, pattern) for pattern in patterns)


def _matches_pattern(path: str, pattern: str) -> bool:
    if pattern.startswith("/"):
        return fnmatch.fnmatch(path, pattern.lstrip("/"))
    if pattern.endswith("/"):
        directory = pattern.rstrip("/")
        return (
            path == directory
            or path.startswith(directory + "/")
            or fnmatch.fnmatch(path, directory)
            or fnmatch.fnmatch(path, directory + "/*")
        )
    if "/" not in pattern:
        return fnmatch.fnmatch(Path(path).name, pattern) or fnmatch.fnmatch(
            path, pattern
        )
    return fnmatch.fnmatch(path, pattern)


def _placeholder_secret_match(value: str) -> bool:
    lowered = value.lower()
    return (
        any(
            token in lowered
            for token in (
                "dummy",
                "example",
                "template",
                "placeholder",
                "test",
                "secret",
                "smoke",
                "benchmark",
            )
        )
        or "api_key_env" in lowered
        or "env" in lowered
    )


def _rel(path: Path, root: Path) -> str:
    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        try:
            return path.resolve().relative_to(root).as_posix()
        except ValueError:
            return path.as_posix()


def _render_readiness(report: dict[str, Any]) -> str:
    lines = [
        "# Release Readiness",
        "",
        f"- Ready: {report['ready']}",
        f"- Errors: {len(report['errors'])}",
        f"- Warnings: {len(report['warnings'])}",
        "",
    ]
    if report["errors"]:
        lines.extend(["## Errors", ""])
        lines.extend(f"- {error}" for error in report["errors"])
        lines.append("")
    if report["warnings"]:
        lines.extend(["## Warnings", ""])
        lines.extend(f"- {warning}" for warning in report["warnings"])
        lines.append("")
    lines.extend(
        [
            "## Summary",
            "",
            "The audit checks code entry points, tests, ignore rules, provider "
            "template boundaries, README claim boundaries, path privacy, and "
            "obvious secret patterns. It does not remove or stage files.",
        ]
    )
    return "\n".join(lines) + "\n"


def _render_non_submission_artifacts(report: dict[str, Any]) -> str:
    lines = [
        "# Non-submission Artifacts",
        "",
        "These files exist locally but are not intended for the clean code-only "
        "GitHub release. The audit only lists them; it does not delete files.",
        "",
        f"- Total listed: {len(report['non_submission_artifacts'])}",
        f"- Not ignored: {len(report['not_ignored_non_submission_artifacts'])}",
        "",
    ]
    if report["not_ignored_non_submission_artifacts"]:
        lines.extend(["## Not Ignored", ""])
        lines.extend(
            f"- `{path}`"
            for path in report["not_ignored_non_submission_artifacts"][:200]
        )
        lines.append("")
    lines.extend(["## Local Non-submission Files", ""])
    lines.extend(f"- `{path}`" for path in report["non_submission_artifacts"][:500])
    if len(report["non_submission_artifacts"]) > 500:
        lines.append("- ...")
    return "\n".join(lines) + "\n"


def _write_json(data: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
