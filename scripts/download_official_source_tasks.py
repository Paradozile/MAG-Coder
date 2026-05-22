"""Download and normalize official HumanEval/MBPP source-task data."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import re
import urllib.error
import urllib.request
from collections.abc import Iterable, Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

OFFICIAL_DATASETS: dict[str, dict[str, str]] = {
    "humaneval": {
        "url": (
            "https://raw.githubusercontent.com/openai/human-eval/master/"
            "data/HumanEval.jsonl.gz"
        ),
        "raw_filename": "HumanEval.jsonl.gz",
        "normalized_filename": "humaneval_official.jsonl",
        "repository": "openai/human-eval",
        "license_note": (
            "Official HumanEval source data from openai/human-eval. "
            "Consult the upstream repository for the current license/README."
        ),
    },
    "mbpp": {
        "url": (
            "https://raw.githubusercontent.com/google-research/google-research/"
            "master/mbpp/sanitized-mbpp.json"
        ),
        "raw_filename": "sanitized-mbpp.json",
        "normalized_filename": "mbpp_official.jsonl",
        "repository": "google-research/google-research/mbpp",
        "license_note": (
            "Official sanitized MBPP source data from google-research. "
            "Consult the upstream repository for the current license/README."
        ),
    },
}


def main() -> None:
    """CLI entry point."""

    parser = argparse.ArgumentParser(
        description=(
            "Download official HumanEval/MBPP data and normalize to "
            "SourceTask JSONL."
        ),
    )
    parser.add_argument(
        "--dataset",
        choices=["humaneval", "mbpp", "all"],
        required=True,
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--normalized-output", type=Path, default=None)
    parser.add_argument("--normalized-output-dir", type=Path, default=None)
    parser.add_argument(
        "--source-url",
        default=None,
        help="Optional official URL override. Must exactly match the allowlist.",
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    datasets = ["humaneval", "mbpp"] if args.dataset == "all" else [args.dataset]
    if len(datasets) > 1 and args.normalized_output is not None:
        raise SystemExit("--normalized-output is only valid for one dataset.")
    manifest_dir = _manifest_dir(args)
    manifest_dir.mkdir(parents=True, exist_ok=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    downloads: list[dict[str, Any]] = []
    normalizations: list[dict[str, Any]] = []
    for dataset in datasets:
        spec = OFFICIAL_DATASETS[dataset]
        source_url = args.source_url or spec["url"]
        validate_official_url(dataset, source_url)
        raw_path = args.output_dir / spec["raw_filename"]
        raw_bytes = _download(source_url, raw_path, overwrite=args.overwrite)
        normalized_path = _normalized_path(args, dataset)
        if dataset == "humaneval":
            rows, normalization = normalize_humaneval_bytes(raw_bytes)
        else:
            rows, normalization = normalize_mbpp_bytes(raw_bytes)
        _write_jsonl(rows, normalized_path)
        raw_sha = _sha256_bytes(raw_bytes)
        normalized_sha = _sha256_file(normalized_path)
        downloads.append(
            {
                "dataset": dataset,
                "repository": spec["repository"],
                "source_url": source_url,
                "raw_path": str(raw_path),
                "download_time": datetime.now(UTC).isoformat(),
                "file_size": len(raw_bytes),
                "sha256": raw_sha,
                "raw_file_sha256": raw_sha,
                "license_note": spec["license_note"],
                "status": "downloaded",
            }
        )
        normalizations.append(
            {
                "dataset": dataset,
                "normalized_path": str(normalized_path),
                "normalized_file_sha256": normalized_sha,
                **normalization,
            }
        )
        print(
            f"{dataset}: loaded={normalization['loaded_task_count']} "
            f"skipped={normalization['skipped_count']} -> {normalized_path}"
        )
    _write_json(
        {
            "created_at": datetime.now(UTC).isoformat(),
            "allowed_sources": {
                key: value["url"] for key, value in OFFICIAL_DATASETS.items()
            },
            "downloads": downloads,
        },
        manifest_dir / "official_download_manifest.json",
    )
    _write_json(
        {
            "created_at": datetime.now(UTC).isoformat(),
            "normalizations": normalizations,
            "total_loaded_task_count": sum(
                int(item["loaded_task_count"]) for item in normalizations
            ),
            "total_skipped_count": sum(
                int(item["skipped_count"]) for item in normalizations
            ),
        },
        manifest_dir / "official_normalization_report.json",
    )
    print(f"Download manifest: {manifest_dir / 'official_download_manifest.json'}")
    print(
        "Normalization report: "
        f"{manifest_dir / 'official_normalization_report.json'}"
    )


def validate_official_url(dataset: str, url: str) -> None:
    """Reject non-allowlisted source URLs."""

    expected = OFFICIAL_DATASETS[dataset]["url"]
    if url != expected:
        raise ValueError(
            f"Refusing non-official URL for {dataset}: {url!r}; expected {expected!r}."
        )


def normalize_humaneval_bytes(
    raw_bytes: bytes,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Normalize official HumanEval JSONL.GZ bytes to SourceTask rows."""

    rows: list[dict[str, Any]] = []
    skipped: list[dict[str, str]] = []
    text = gzip.decompress(raw_bytes).decode("utf-8")
    for line_number, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            continue
        try:
            raw = json.loads(line)
            row = _normalize_humaneval_row(raw)
        except Exception as error:  # noqa: BLE001 - report malformed rows.
            skipped.append(
                {
                    "line_number": str(line_number),
                    "reason": f"{type(error).__name__}: {error}",
                }
            )
            continue
        rows.append(row)
    return rows, _normalization_summary(
        "humaneval",
        len(text.splitlines()),
        rows,
        skipped,
    )


def normalize_mbpp_bytes(
    raw_bytes: bytes,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Normalize official sanitized MBPP JSON bytes to SourceTask rows."""

    parsed = json.loads(raw_bytes.decode("utf-8"))
    if not isinstance(parsed, list):
        raise TypeError("Official sanitized MBPP JSON root must be a list.")
    rows: list[dict[str, Any]] = []
    skipped: list[dict[str, str]] = []
    for index, raw in enumerate(parsed):
        try:
            if not isinstance(raw, Mapping):
                raise TypeError("row must be a JSON object")
            row = _normalize_mbpp_row(raw)
        except Exception as error:  # noqa: BLE001 - report malformed rows.
            skipped.append(
                {
                    "index": str(index),
                    "reason": f"{type(error).__name__}: {error}",
                }
            )
            continue
        rows.append(row)
    return rows, _normalization_summary("mbpp", len(parsed), rows, skipped)


def _normalize_humaneval_row(raw: Mapping[str, Any]) -> dict[str, Any]:
    task_id = str(raw["task_id"])
    prompt = str(raw["prompt"])
    canonical_body = str(raw["canonical_solution"])
    entry_point = str(raw["entry_point"])
    test_code = str(raw["test"])
    canonical_solution = prompt.rstrip() + "\n" + canonical_body.rstrip() + "\n"
    tests = [test_code.rstrip() + f"\ncheck({entry_point})"]
    return {
        "task_id": task_id,
        "source": "humaneval",
        "prompt": prompt,
        "canonical_solution": canonical_solution,
        "tests": tests,
        "entry_point": entry_point,
        "metadata": {
            "official_dataset": "humaneval",
            "original_task_id": task_id,
            "normalization": "prompt_plus_canonical_solution_with_check_call",
        },
    }


def _normalize_mbpp_row(raw: Mapping[str, Any]) -> dict[str, Any]:
    raw_id = raw.get("task_id", raw.get("id"))
    task_id = f"mbpp_{raw_id}"
    raw_tests = raw.get("test_list", raw.get("tests", []))
    tests = [_prefix_required_imports(str(item)) for item in _list_value(raw_tests)]
    code = str(raw.get("code", raw.get("canonical_solution", "")))
    entry_point = str(raw.get("entry_point") or _infer_entry_point(tests, code) or "")
    if not entry_point:
        raise ValueError(f"Cannot infer MBPP entry point for {task_id}")
    return {
        "task_id": task_id,
        "source": "mbpp",
        "prompt": str(raw.get("prompt", raw.get("text", ""))),
        "canonical_solution": code,
        "tests": tests,
        "entry_point": entry_point,
        "metadata": {
            "official_dataset": "mbpp",
            "original_task_id": raw_id,
            "normalization": "sanitized_mbpp_assertions",
        },
    }


def _infer_entry_point(tests: Iterable[str], code: str) -> str | None:
    match = re.search(r"^\s*def\s+([A-Za-z_][A-Za-z0-9_]*)\s*\(", code, re.M)
    if match:
        return match.group(1)
    for test in tests:
        match = re.search(r"\bassert\s+([A-Za-z_][A-Za-z0-9_]*)\s*\(", test)
        if match:
            return match.group(1)
    return None


def _prefix_required_imports(test: str) -> str:
    imports: list[str] = []
    for module_name in ["math", "cmath", "re", "heapq", "itertools", "collections"]:
        if re.search(rf"\b{module_name}\.", test):
            imports.append(f"import {module_name}")
    if re.search(r"\bstring\.", test):
        imports.append("import string")
    if not imports:
        return test
    return "\n".join(imports + [test])


def _list_value(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]


def _normalization_summary(
    dataset: str,
    input_row_count: int,
    rows: list[dict[str, Any]],
    skipped: list[dict[str, str]],
) -> dict[str, Any]:
    return {
        "dataset": dataset,
        "input_row_count": input_row_count,
        "loaded_task_count": len(rows),
        "skipped_count": len(skipped),
        "malformed_rows": skipped,
        "skipped_or_malformed_rows": skipped,
    }


def _download(url: str, output_path: Path, *, overwrite: bool) -> bytes:
    if output_path.exists() and not overwrite:
        return output_path.read_bytes()
    try:
        with urllib.request.urlopen(url, timeout=60) as response:  # nosec B310
            data = response.read()
    except urllib.error.URLError as error:
        raise RuntimeError(
            f"Failed to download official benchmark source data from {url}: {error}"
        ) from error
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(data)
    return data


def _normalized_path(args: argparse.Namespace, dataset: str) -> Path:
    if args.normalized_output is not None:
        return args.normalized_output
    output_dir = args.normalized_output_dir or args.output_dir.parent
    return output_dir / OFFICIAL_DATASETS[dataset]["normalized_filename"]


def _manifest_dir(args: argparse.Namespace) -> Path:
    if args.normalized_output_dir is not None:
        return args.normalized_output_dir
    if args.normalized_output is not None:
        return args.normalized_output.parent
    return args.output_dir.parent


def _write_jsonl(rows: list[dict[str, Any]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file:
        for row in rows:
            json.dump(row, file, sort_keys=True)
            file.write("\n")


def _write_json(data: dict[str, Any], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file:
        json.dump(data, file, indent=2, sort_keys=True)
        file.write("\n")


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_file(path: Path) -> str:
    return _sha256_bytes(path.read_bytes())


if __name__ == "__main__":
    main()
