"""Tests for optional rendered MAG graphical extraction interface."""

from __future__ import annotations

import json
from pathlib import Path

from scripts.extract_mag_from_rendered_graph import extract_mag_from_rendered_graph
from scripts.render_mag_graphics import render_case_graphics

ROOT = Path(__file__).resolve().parents[1]
PAPER2500_BENCH = ROOT / "benchmarks/full_mag_bench_2500_v1"


def test_mock_vlm_extractor_recovers_mag_from_manifest(tmp_path: Path) -> None:
    """Mock extractor recovers MAG from sidecar manifest without VLM calls."""

    case_id = _first_case_id()
    manifest = render_case_graphics(
        bench=PAPER2500_BENCH,
        case_id=case_id,
        output_dir=tmp_path / "graphics",
    )
    wrapper = json.loads(manifest.read_text(encoding="utf-8"))
    image = manifest.parent / wrapper["image_path"]
    output = tmp_path / "recovered_mag.json"

    result = extract_mag_from_rendered_graph(
        image_path=image,
        manifest_path=manifest,
        extractor_name="mock",
        output_path=output,
    )

    assert result["extractor"] == "mock"
    assert output.exists()
    recovered = json.loads(output.read_text(encoding="utf-8"))
    assert recovered["agents"]


def _first_case_id() -> str:
    index = json.loads(
        (PAPER2500_BENCH / "case_index.json").read_text(encoding="utf-8")
    )
    cases = index.get("cases", index)
    if isinstance(cases, list) and cases:
        first = cases[0]
        if isinstance(first, dict):
            return str(first["case_id"])
        return str(first)
    raise AssertionError("fixture benchmark has no cases")
