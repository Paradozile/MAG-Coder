"""Path-safety helpers for local execution workflows."""

from __future__ import annotations

from pathlib import Path


def is_path_within_repo(path: str | Path, repo_root: str | Path) -> bool:
    """Return whether a path resolves inside the repository root."""

    root = Path(repo_root).resolve()
    candidate = _resolve_against_repo(path, root)
    return candidate == root or root in candidate.parents


def ensure_output_path_within_repo(
    path: str | Path,
    repo_root: str | Path,
) -> Path:
    """Resolve a writable output path and fail if it escapes the repository."""

    root = Path(repo_root).resolve()
    candidate = _resolve_against_repo(path, root)
    if not (candidate == root or root in candidate.parents):
        raise ValueError(
            f"Output path must be inside repository {root}: {candidate}"
        )
    return candidate


def _resolve_against_repo(path: str | Path, repo_root: Path) -> Path:
    candidate = Path(path)
    if not candidate.is_absolute():
        candidate = repo_root / candidate
    return candidate.resolve(strict=False)
