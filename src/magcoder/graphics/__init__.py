"""Graphical MAG rendering and round-trip utilities."""

from magcoder.graphics.artifact import GraphicalMAGArtifact, render_mag_artifact
from magcoder.graphics.dot_parser import load_mag_from_graphical_artifact
from magcoder.graphics.dot_renderer import render_mag_to_dot, save_mag_dot
from magcoder.graphics.manifest import (
    GraphicalMAGManifest,
    build_graphical_mag_manifest,
    load_graphical_mag_manifest,
    save_graphical_mag_manifest,
)
from magcoder.graphics.pipeline import (
    GraphicalMAGPipelineResult,
    run_graphical_mag_pipeline,
)
from magcoder.graphics.vlm_extractor import (
    DotManifestExtractor,
    MockVLMExtractor,
    ProviderVLMExtractor,
    RenderedGraphArtifact,
    VLMExtractionConfig,
    VLMExtractionResult,
    VLMExtractor,
    extractor_from_name,
)

__all__ = [
    "GraphicalMAGArtifact",
    "GraphicalMAGManifest",
    "GraphicalMAGPipelineResult",
    "DotManifestExtractor",
    "MockVLMExtractor",
    "ProviderVLMExtractor",
    "RenderedGraphArtifact",
    "VLMExtractionConfig",
    "VLMExtractionResult",
    "VLMExtractor",
    "build_graphical_mag_manifest",
    "extractor_from_name",
    "load_graphical_mag_manifest",
    "load_mag_from_graphical_artifact",
    "render_mag_artifact",
    "render_mag_to_dot",
    "run_graphical_mag_pipeline",
    "save_graphical_mag_manifest",
    "save_mag_dot",
]
