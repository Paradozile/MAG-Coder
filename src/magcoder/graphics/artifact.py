"""Graphical MAG artifact rendering utilities."""

from __future__ import annotations

import shutil
import subprocess
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from magcoder.graphics.dot_renderer import save_mag_dot
from magcoder.graphics.manifest import (
    build_graphical_mag_manifest,
    save_graphical_mag_manifest,
)
from magcoder.models import MAGModel, load_mag_from_json, load_mag_from_yaml


@dataclass(slots=True)
class GraphicalMAGArtifact:
    """Paths and metadata for rendered graphical MAG artifacts."""

    mag_id: str
    dot_path: str
    manifest_path: str
    svg_path: str | None = None
    png_path: str | None = None
    source_mag_path: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Render the artifact as plain Python containers."""

        return asdict(self)


def render_mag_artifact(
    mag_path: str | Path,
    output_dir: str | Path,
    formats: Sequence[str] = ("dot", "svg"),
) -> GraphicalMAGArtifact:
    """Render DOT plus optional Graphviz image artifacts for a MAG file."""

    mag_input_path = Path(mag_path)
    mag = _load_mag(mag_input_path)
    normalized_formats = _normalize_formats(formats)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    artifact_stem = _artifact_stem(mag_input_path)

    dot_path = output_path / f"{artifact_stem}.dot"
    manifest_path = output_path / f"{artifact_stem}_manifest.json"
    save_mag_dot(mag, dot_path)
    save_graphical_mag_manifest(
        build_graphical_mag_manifest(
            mag,
            metadata={"source_mag_path": str(mag_input_path)},
        ),
        manifest_path,
    )

    warnings: list[str] = []
    svg_path: Path | None = None
    png_path: Path | None = None
    image_formats = [item for item in normalized_formats if item in {"svg", "png"}]
    dot_executable = shutil.which("dot")
    if image_formats and dot_executable is None:
        warnings.append(
            "Graphviz executable 'dot' is unavailable; image rendering skipped.",
        )
    elif dot_executable is not None:
        for image_format in image_formats:
            target_path = output_path / f"{artifact_stem}.{image_format}"
            if _run_graphviz(
                dot_executable=dot_executable,
                dot_path=dot_path,
                output_path=target_path,
                image_format=image_format,
                warnings=warnings,
            ):
                if image_format == "svg":
                    svg_path = target_path
                elif image_format == "png":
                    png_path = target_path

    return GraphicalMAGArtifact(
        mag_id=mag.id,
        dot_path=str(dot_path),
        svg_path=None if svg_path is None else str(svg_path),
        png_path=None if png_path is None else str(png_path),
        manifest_path=str(manifest_path),
        source_mag_path=str(mag_input_path),
        metadata={
            "formats_requested": normalized_formats,
            "warnings": warnings,
        },
    )


def _run_graphviz(
    *,
    dot_executable: str,
    dot_path: Path,
    output_path: Path,
    image_format: str,
    warnings: list[str],
) -> bool:
    try:
        subprocess.run(
            [
                dot_executable,
                f"-T{image_format}",
                str(dot_path),
                "-o",
                str(output_path),
            ],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError) as error:
        warnings.append(f"Graphviz failed for {image_format}: {error}")
        return False
    return True


def _load_mag(path: Path) -> MAGModel:
    if path.suffix.lower() == ".json":
        return load_mag_from_json(path)
    if path.suffix.lower() in {".yaml", ".yml"}:
        return load_mag_from_yaml(path)
    msg = f"Unsupported MAG file extension: {path.suffix!r}."
    raise ValueError(msg)


def _normalize_formats(formats: Sequence[str]) -> list[str]:
    normalized = []
    for item in formats:
        value = item.strip().lower()
        if not value:
            continue
        if value not in {"dot", "svg", "png"}:
            msg = f"Unsupported graphical MAG format: {item!r}."
            raise ValueError(msg)
        if value not in normalized:
            normalized.append(value)
    if "dot" not in normalized:
        normalized.insert(0, "dot")
    return normalized


def _artifact_stem(path: Path) -> str:
    stem = path.stem
    if stem.endswith("_mag"):
        stem = stem[: -len("_mag")]
    return stem
