"""File-based cache for generation results."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from magcoder.generation.core import (
    GenerationRequest,
    GenerationResult,
    load_generation_result,
    request_hash,
    save_generation_result,
)

DEFAULT_GENERATION_CACHE_DIR = Path("runs/generation_cache")


def generation_cache_key(request: GenerationRequest) -> str:
    """Build the stable cache key for a generation request."""

    return request_hash(request)


def generation_cache_path(
    request: GenerationRequest,
    *,
    cache_dir: str | Path = DEFAULT_GENERATION_CACHE_DIR,
) -> Path:
    """Return the cache file path for a request."""

    return Path(cache_dir) / f"{generation_cache_key(request)}.json"


def load_cached_generation_result(
    request: GenerationRequest,
    *,
    cache_dir: str | Path = DEFAULT_GENERATION_CACHE_DIR,
) -> GenerationResult | None:
    """Load a cached generation result if one exists."""

    path = generation_cache_path(request, cache_dir=cache_dir)
    if not path.exists():
        return None
    result = load_generation_result(path)
    result.metadata["cache_hit"] = True
    result.metadata["cache_path"] = str(path)
    return result


def save_cached_generation_result(
    result: GenerationResult,
    request: GenerationRequest,
    *,
    cache_dir: str | Path = DEFAULT_GENERATION_CACHE_DIR,
) -> Path:
    """Persist a generation result in the request cache."""

    path = generation_cache_path(request, cache_dir=cache_dir)
    result.metadata["cache_path"] = str(path)
    result.metadata.setdefault("cache_hit", False)
    save_generation_result(result, path)
    return path


def cache_index_record(
    request: GenerationRequest,
    result: GenerationResult,
    *,
    cache_dir: str | Path = DEFAULT_GENERATION_CACHE_DIR,
) -> dict[str, Any]:
    """Return a small inspectable cache index record."""

    return {
        "cache_key": generation_cache_key(request),
        "cache_path": str(generation_cache_path(request, cache_dir=cache_dir)),
        "provider": result.metadata.get("provider"),
        "model_name": result.metadata.get("model_name"),
        "prompt_hash": result.metadata.get("prompt_hash"),
        "request_hash": result.metadata.get("request_hash"),
    }


def save_cache_index_record(
    request: GenerationRequest,
    result: GenerationResult,
    *,
    cache_dir: str | Path = DEFAULT_GENERATION_CACHE_DIR,
) -> Path:
    """Save a small cache index record next to cached results."""

    directory = Path(cache_dir)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{generation_cache_key(request)}.index.json"
    with path.open("w", encoding="utf-8") as file:
        json.dump(
            cache_index_record(request, result, cache_dir=cache_dir),
            file,
            indent=2,
            sort_keys=True,
        )
        file.write("\n")
    return path
