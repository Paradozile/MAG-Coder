"""Tests for official HumanEval/MBPP source-task downloader normalization."""

from __future__ import annotations

import gzip
import json

import pytest
from scripts.download_official_source_tasks import (
    OFFICIAL_DATASETS,
    normalize_humaneval_bytes,
    normalize_mbpp_bytes,
    validate_official_url,
)


def test_downloader_allowlist_accepts_official_urls() -> None:
    """Official URLs are accepted by the downloader allowlist."""

    for dataset, spec in OFFICIAL_DATASETS.items():
        validate_official_url(dataset, spec["url"])


def test_downloader_refuses_non_official_url() -> None:
    """Non-official source URLs are rejected before any download."""

    with pytest.raises(ValueError):
        validate_official_url("humaneval", "https://example.com/HumanEval.jsonl.gz")


def test_humaneval_style_fixture_normalizes_to_source_task() -> None:
    """HumanEval official rows normalize into executable SourceTask JSON rows."""

    raw = {
        "task_id": "HumanEval/0",
        "prompt": "def add_one(x):\n    \"\"\"Return x + 1.\"\"\"\n",
        "canonical_solution": "    return x + 1\n",
        "test": "def check(candidate):\n    assert candidate(1) == 2\n",
        "entry_point": "add_one",
    }
    rows, report = normalize_humaneval_bytes(
        gzip.compress((json.dumps(raw) + "\n").encode("utf-8"))
    )
    assert report["loaded_task_count"] == 1
    assert rows[0]["entry_point"] == "add_one"
    assert "check(add_one)" in rows[0]["tests"][0]
    assert rows[0]["canonical_solution"].startswith("def add_one")


def test_mbpp_style_fixture_normalizes_to_source_task() -> None:
    """Sanitized MBPP official rows normalize into SourceTask JSON rows."""

    raw = [
        {
            "task_id": 1,
            "prompt": "Write a function to square a number.",
            "code": "def square(x):\n    return x * x\n",
            "test_list": ["assert square(3) == 9"],
        }
    ]
    rows, report = normalize_mbpp_bytes(json.dumps(raw).encode("utf-8"))
    assert report["loaded_task_count"] == 1
    assert rows[0]["task_id"] == "mbpp_1"
    assert rows[0]["entry_point"] == "square"
