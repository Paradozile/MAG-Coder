"""Render one Full MAG-Bench case's MAG as DOT/SVG/PNG smoke artifacts."""

from __future__ import annotations

import argparse
import base64
import json
import shutil
from pathlib import Path
from typing import Any

from magcoder.benchmark.full_runner.case_loader import load_full_bench_run_cases
from magcoder.graphics import render_mag_artifact

_PNG_1X1 = (
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+/p9sAAAAASUVORK5CYII="
)


def main() -> None:
    """CLI entry point."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bench", type=Path, required=True)
    parser.add_argument("--case-id", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest_path = render_case_graphics(
        bench=args.bench,
        case_id=args.case_id,
        output_dir=args.output,
    )
    print(f"Rendered graph manifest: {manifest_path}")


def render_case_graphics(
    *,
    bench: str | Path,
    case_id: str,
    output_dir: str | Path,
) -> Path:
    """Render one benchmark case and write a smoke manifest."""

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    cases = load_full_bench_run_cases(bench)
    case = next((item for item in cases if item.case_id == case_id), None)
    if case is None:
        raise ValueError(f"Case not found: {case_id}")

    artifact = render_mag_artifact(
        case.mag_json_path,
        output,
        formats=("dot", "svg", "png"),
    )
    mag_copy = output / f"{case_id}_mag.json"
    dot_copy = output / f"{case_id}.dot"
    manifest_copy = output / f"{case_id}_graphical_manifest.json"
    shutil.copy2(case.mag_json_path, mag_copy)
    shutil.copy2(artifact.dot_path, dot_copy)
    shutil.copy2(artifact.manifest_path, manifest_copy)
    svg_copy = _copy_optional(artifact.svg_path, output / f"{case_id}.svg")
    png_copy = _copy_optional(artifact.png_path, output / f"{case_id}.png")
    if png_copy is None:
        png_copy = output / f"{case_id}.png"
        png_copy.write_bytes(base64.b64decode(_PNG_1X1))

    wrapper = {
        "schema_version": 1,
        "case_id": case_id,
        "image_path": png_copy.name,
        "svg_path": None if svg_copy is None else svg_copy.name,
        "dot_path": dot_copy.name,
        "mag_json_path": mag_copy.name,
        "graphical_manifest_path": manifest_copy.name,
        "extractor_note": (
            "This smoke artifact supports mock/manifest recovery and does not "
            "perform arbitrary VLM recognition."
        ),
        "render_metadata": artifact.metadata,
    }
    wrapper_path = output / f"{case_id}_manifest.json"
    _write_json(wrapper, wrapper_path)
    return wrapper_path


def _copy_optional(source: str | None, target: Path) -> Path | None:
    if source is None:
        return None
    source_path = Path(source)
    if not source_path.exists():
        return None
    shutil.copy2(source_path, target)
    return target


def _write_json(data: dict[str, Any], path: Path) -> None:
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
