import ast
from pathlib import Path

import pytest

from magcoder.generation import (
    FakeCodeGenerator,
    GenerationConfig,
    GenerationRequest,
    GenerationResult,
    extract_python_code,
    load_generation_result,
    save_generation_result,
)
from magcoder.models import load_mag_from_yaml
from magcoder.semantic import build_hybrid_input

ROOT = Path(__file__).resolve().parents[1]
EXAMPLE_CASES = [
    (
        ROOT / "examples" / "robot_rescue_mag.yaml",
        ROOT / "examples" / "robot_rescue_generation.json",
        ("SonarDetection", "InfraredThermal"),
    ),
    (
        ROOT / "examples" / "aircraft_collision_mag.yaml",
        ROOT / "examples" / "aircraft_collision_generation.json",
        ("AircraftA", "ConflictDetector"),
    ),
]


def _generation_request(mag_path: Path) -> GenerationRequest:
    mag = load_mag_from_yaml(mag_path)
    hybrid_input = build_hybrid_input(mag)
    return GenerationRequest(
        hybrid_input=hybrid_input,
        config=GenerationConfig(),
        prompt_text=hybrid_input.to_prompt_text(),
    )


def test_extract_python_code_from_fenced_block() -> None:
    raw_response = "Here is code:\n```python\nx = 1\nprint(x)\n```\nDone."

    assert extract_python_code(raw_response) == "x = 1\nprint(x)\n"


def test_extract_python_code_without_fence() -> None:
    raw_response = "x = 1\nprint(x)\n"

    assert extract_python_code(raw_response) == raw_response


def test_fake_generator_returns_generation_result() -> None:
    result = FakeCodeGenerator().generate(_generation_request(EXAMPLE_CASES[0][0]))

    assert isinstance(result, GenerationResult)
    assert result.generator_name == "fake-generator"
    assert "class SonarDetection" in result.code_initial
    assert "class InfraredThermal" in result.code_initial
    assert "def interact_sonar_to_infrared" in result.code_initial
    assert "distance < 10" in result.code_initial


@pytest.mark.parametrize(("mag_path", "generation_path", "agent_ids"), EXAMPLE_CASES)
def test_examples_generate_code_with_required_structure(
    mag_path: Path,
    generation_path: Path,
    agent_ids: tuple[str, str],
) -> None:
    result = FakeCodeGenerator().generate(_generation_request(mag_path))

    ast.parse(result.code_initial)
    for agent_id in agent_ids:
        assert f"class {agent_id}" in result.code_initial
    assert "def interact_" in result.code_initial
    assert "if not _filter_" in result.code_initial

    persisted = load_generation_result(generation_path)
    ast.parse(persisted.code_initial)


def test_save_and_load_generation_result(tmp_path: Path) -> None:
    result = FakeCodeGenerator().generate(_generation_request(EXAMPLE_CASES[0][0]))
    output_path = tmp_path / "generation.json"

    save_generation_result(result, output_path)
    loaded = load_generation_result(output_path)

    assert loaded.to_dict() == result.to_dict()


def test_generation_from_robot_rescue_example() -> None:
    result = FakeCodeGenerator().generate(_generation_request(EXAMPLE_CASES[0][0]))

    assert "nearby_target" in result.prompt_text
    assert "_filter_nearby_target" in result.code_initial
    assert "filter_failed" in result.code_initial
    assert result.warnings == []


def test_fake_generator_outputs_different_code_for_different_mags() -> None:
    robot_result = FakeCodeGenerator().generate(
        _generation_request(EXAMPLE_CASES[0][0]),
    )
    aircraft_result = FakeCodeGenerator().generate(
        _generation_request(EXAMPLE_CASES[1][0]),
    )

    assert robot_result.code_initial != aircraft_result.code_initial
    assert "class SonarDetection" in robot_result.code_initial
    assert "class AircraftA" in aircraft_result.code_initial


def test_generation_result_metadata_contains_phase4_context() -> None:
    result = FakeCodeGenerator().generate(_generation_request(EXAMPLE_CASES[1][0]))

    assert result.metadata["generation_interface_version"] == 1
    assert result.metadata["mag_id"] == "aircraft_collision_demo"
    assert "prompt_text_sha256" in result.metadata
    assert result.metadata["hybrid_input"]["mag_id"] == "aircraft_collision_demo"
