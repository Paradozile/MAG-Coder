"""Pipeline entry point from graphical MAG artifacts to analysis reports."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from magcoder.generation import (
    FakeCodeGenerator,
    GenerationConfig,
    GenerationRequest,
    save_generation_result,
)
from magcoder.graphics.dot_parser import load_mag_from_graphical_artifact
from magcoder.models import save_mag_to_yaml
from magcoder.refinement import (
    analyze_python_source,
    check_consistency,
    constraints_with_interaction_contracts,
    extract_model_constraints,
    load_framework_spec_from_yaml,
    save_consistency_report,
)
from magcoder.refinement.contracts import build_interaction_contracts
from magcoder.semantic import build_hybrid_input


@dataclass(slots=True)
class GraphicalMAGPipelineResult:
    """Output paths and summary for a graphical MAG pipeline run."""

    mag_id: str
    prompt_path: str
    generation_path: str
    code_path: str
    analysis_path: str
    analysis_passed: bool
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Render the pipeline result as plain Python containers."""

        return asdict(self)


def run_graphical_mag_pipeline(
    *,
    dot_path: str | Path,
    manifest_path: str | Path,
    framework_path: str | Path,
    output_dir: str | Path,
    mode: str = "graph+nl",
) -> GraphicalMAGPipelineResult:
    """Run DOT+manifest through semantic input, generation, and analysis."""

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    mag = load_mag_from_graphical_artifact(dot_path, manifest_path)
    mag.validate()
    hybrid_input = build_hybrid_input(mag, mode=mode)
    prompt_text = hybrid_input.to_prompt_text()

    prompt_path = output_path / "prompt.txt"
    mag_path = output_path / "mag_recovered.yaml"
    generation_path = output_path / "generation.json"
    code_path = output_path / "code_initial.py"
    analysis_path = output_path / "analysis.json"

    save_mag_to_yaml(mag, mag_path)
    prompt_path.write_text(prompt_text, encoding="utf-8")

    request = GenerationRequest(
        hybrid_input=hybrid_input,
        config=GenerationConfig(
            mode=mode,
            metadata={
                "entrypoint": "graphical_mag",
                "dot_path": str(dot_path),
                "manifest_path": str(manifest_path),
            },
        ),
        prompt_text=prompt_text,
    )
    generation = FakeCodeGenerator().generate(request)
    save_generation_result(generation, generation_path)
    code_path.write_text(generation.code_initial, encoding="utf-8")

    framework = load_framework_spec_from_yaml(framework_path)
    constraints = constraints_with_interaction_contracts(
        extract_model_constraints(mag),
        build_interaction_contracts(mag, framework),
    )
    analysis = analyze_python_source(generation.code_initial)
    report = check_consistency(analysis, constraints, framework)
    save_consistency_report(report, analysis_path)

    return GraphicalMAGPipelineResult(
        mag_id=mag.id,
        prompt_path=str(prompt_path),
        generation_path=str(generation_path),
        code_path=str(code_path),
        analysis_path=str(analysis_path),
        analysis_passed=report.passed,
        metadata={
            "dot_path": str(dot_path),
            "manifest_path": str(manifest_path),
            "framework_path": str(framework_path),
            "mag_path": str(mag_path),
            "mode": mode,
        },
    )
