"""Optional rendered-graph extraction interfaces for MAG artifacts.

The concrete implementation here is intentionally non-networked. It supports
manifest-backed recovery for smoke tests and defines the interface that a real
VLM extractor would implement when explicitly configured by a user.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Protocol

from magcoder.graphics.dot_parser import load_mag_from_graphical_artifact
from magcoder.models import MAGModel, load_mag_from_json


@dataclass(slots=True)
class RenderedGraphArtifact:
    """A rendered graph plus optional manifest references."""

    image_path: str
    manifest_path: str | None = None
    dot_path: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class VLMExtractionConfig:
    """Configuration for a rendered-graph extractor."""

    extractor: str = "mock"
    provider: str | None = None
    model: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class VLMExtractionResult:
    """Result of extracting a MAGModel from a rendered graph."""

    mag: MAGModel
    extractor: str
    used_manifest: bool
    warnings: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Render result metadata without duplicating full model internals."""

        data = asdict(self)
        data["mag"] = self.mag.to_dict()
        return data


class VLMExtractor(Protocol):
    """Interface for rendered-graph MAG extraction."""

    def extract(
        self,
        artifact: RenderedGraphArtifact,
        config: VLMExtractionConfig,
    ) -> VLMExtractionResult:
        """Extract a MAG model from a rendered graph artifact."""


class MockVLMExtractor:
    """Recover MAG from a sidecar manifest without calling a VLM."""

    def extract(
        self,
        artifact: RenderedGraphArtifact,
        config: VLMExtractionConfig,
    ) -> VLMExtractionResult:
        manifest = _load_sidecar_manifest(artifact)
        base_dir = Path(artifact.manifest_path or artifact.image_path).parent
        warnings: list[str] = []
        mag_json = _resolve_optional(base_dir, manifest.get("mag_json_path"))
        if mag_json is None:
            mag_json = _resolve_optional(base_dir, manifest.get("source_mag_path"))
        if mag_json is not None and mag_json.exists():
            mag = load_mag_from_json(mag_json)
            return VLMExtractionResult(
                mag=mag,
                extractor="mock",
                used_manifest=True,
                warnings=warnings,
                metadata={"source": "mag_json", "config": config.metadata},
            )
        dot_path = _resolve_optional(base_dir, manifest.get("dot_path")) or Path(
            artifact.dot_path or "",
        )
        graph_manifest = _resolve_optional(
            base_dir,
            manifest.get("graphical_manifest_path")
            or manifest.get("source_manifest_path"),
        )
        if dot_path.exists() and graph_manifest is not None and graph_manifest.exists():
            mag = load_mag_from_graphical_artifact(dot_path, graph_manifest)
            return VLMExtractionResult(
                mag=mag,
                extractor="mock",
                used_manifest=True,
                warnings=warnings,
                metadata={"source": "dot_manifest", "config": config.metadata},
            )
        raise ValueError("Mock VLM extraction requires MAG JSON or DOT+manifest.")


class DotManifestExtractor:
    """Recover MAG directly from standardized DOT plus manifest."""

    def extract(
        self,
        artifact: RenderedGraphArtifact,
        config: VLMExtractionConfig,
    ) -> VLMExtractionResult:
        manifest = _load_sidecar_manifest(artifact)
        base_dir = Path(artifact.manifest_path or artifact.image_path).parent
        dot_path = _resolve_optional(base_dir, manifest.get("dot_path")) or Path(
            artifact.dot_path or "",
        )
        graph_manifest = _resolve_optional(
            base_dir,
            manifest.get("graphical_manifest_path")
            or manifest.get("source_manifest_path"),
        )
        if graph_manifest is None and artifact.manifest_path is not None:
            graph_manifest = Path(artifact.manifest_path)
        if graph_manifest is None:
            raise ValueError("DOT manifest extraction requires a manifest path.")
        mag = load_mag_from_graphical_artifact(dot_path, graph_manifest)
        return VLMExtractionResult(
            mag=mag,
            extractor="dot_manifest",
            used_manifest=True,
            metadata={"config": config.metadata},
        )


class ProviderVLMExtractor:
    """Interface placeholder for explicitly configured provider-backed VLMs."""

    def extract(
        self,
        artifact: RenderedGraphArtifact,
        config: VLMExtractionConfig,
    ) -> VLMExtractionResult:
        raise RuntimeError(
            "Provider-backed VLM extraction is interface-only in this repository "
            "unless a user explicitly wires a provider. No external API is called."
        )


def extractor_from_name(name: str) -> VLMExtractor:
    """Return a non-networked extractor implementation by name."""

    normalized = name.strip().lower()
    if normalized == "mock":
        return MockVLMExtractor()
    if normalized in {"dot", "dot_manifest"}:
        return DotManifestExtractor()
    if normalized in {"provider", "vlm"}:
        return ProviderVLMExtractor()
    raise ValueError(f"Unsupported rendered-graph extractor: {name}")


def _load_sidecar_manifest(artifact: RenderedGraphArtifact) -> dict[str, Any]:
    if artifact.manifest_path is None:
        return {}
    path = Path(artifact.manifest_path)
    if not path.exists():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise TypeError("Rendered graph manifest root must be an object.")
    return data


def _resolve_optional(base_dir: Path, value: Any) -> Path | None:
    if value is None:
        return None
    path = Path(str(value))
    if path.is_absolute():
        return path
    return base_dir / path
