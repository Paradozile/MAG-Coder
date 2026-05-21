import ast
import json
from pathlib import Path
from typing import Any

from magcoder.graphics import run_graphical_mag_pipeline

ROOT = Path(__file__).resolve().parents[1]
FRAMEWORK_PATH = ROOT / "examples" / "simple_abms_framework.yaml"
GRAPHICAL_CASES = {
    "robot_rescue": (
        ROOT / "examples" / "graphics" / "robot_rescue" / "robot_rescue.dot",
        ROOT
        / "examples"
        / "graphics"
        / "robot_rescue"
        / "robot_rescue_manifest.json",
    ),
    "aircraft_collision": (
        ROOT
        / "examples"
        / "graphics"
        / "aircraft_collision"
        / "aircraft_collision.dot",
        ROOT
        / "examples"
        / "graphics"
        / "aircraft_collision"
        / "aircraft_collision_manifest.json",
    ),
}


def test_dot_and_manifest_can_enter_full_pipeline(tmp_path: Path) -> None:
    dot_path, manifest_path = GRAPHICAL_CASES["robot_rescue"]

    result = run_graphical_mag_pipeline(
        dot_path=dot_path,
        manifest_path=manifest_path,
        framework_path=FRAMEWORK_PATH,
        output_dir=tmp_path,
    )

    assert result.mag_id == "robot_rescue_demo"
    assert Path(result.prompt_path).exists()
    assert Path(result.generation_path).exists()
    assert Path(result.code_path).exists()
    assert Path(result.analysis_path).exists()


def test_generated_prompt_text_is_non_empty(tmp_path: Path) -> None:
    dot_path, manifest_path = GRAPHICAL_CASES["robot_rescue"]
    result = run_graphical_mag_pipeline(
        dot_path=dot_path,
        manifest_path=manifest_path,
        framework_path=FRAMEWORK_PATH,
        output_dir=tmp_path,
    )

    prompt_text = Path(result.prompt_path).read_text(encoding="utf-8")

    assert "## Agents" in prompt_text
    assert "SonarDetection" in prompt_text


def test_code_initial_can_be_parsed_by_ast(tmp_path: Path) -> None:
    dot_path, manifest_path = GRAPHICAL_CASES["robot_rescue"]
    result = run_graphical_mag_pipeline(
        dot_path=dot_path,
        manifest_path=manifest_path,
        framework_path=FRAMEWORK_PATH,
        output_dir=tmp_path,
    )

    code_initial = Path(result.code_path).read_text(encoding="utf-8")

    ast.parse(code_initial)


def test_analysis_json_can_be_generated(tmp_path: Path) -> None:
    dot_path, manifest_path = GRAPHICAL_CASES["robot_rescue"]
    result = run_graphical_mag_pipeline(
        dot_path=dot_path,
        manifest_path=manifest_path,
        framework_path=FRAMEWORK_PATH,
        output_dir=tmp_path,
    )

    analysis = _load_json(result.analysis_path)

    assert set(analysis) == {
        "framework_violations",
        "missing_agent_classes",
        "missing_attributes",
        "missing_filter_guards",
        "missing_interaction_methods",
        "passed",
        "warnings",
    }
    assert isinstance(analysis["passed"], bool)


def test_graphical_pipeline_outputs_are_different(tmp_path: Path) -> None:
    robot_result = run_graphical_mag_pipeline(
        dot_path=GRAPHICAL_CASES["robot_rescue"][0],
        manifest_path=GRAPHICAL_CASES["robot_rescue"][1],
        framework_path=FRAMEWORK_PATH,
        output_dir=tmp_path / "robot",
    )
    aircraft_result = run_graphical_mag_pipeline(
        dot_path=GRAPHICAL_CASES["aircraft_collision"][0],
        manifest_path=GRAPHICAL_CASES["aircraft_collision"][1],
        framework_path=FRAMEWORK_PATH,
        output_dir=tmp_path / "aircraft",
    )

    robot_code = Path(robot_result.code_path).read_text(encoding="utf-8")
    aircraft_code = Path(aircraft_result.code_path).read_text(encoding="utf-8")

    assert robot_code != aircraft_code
    assert "SonarDetection" in robot_code
    assert "AircraftA" in aircraft_code


def test_analysis_is_structured_even_when_not_passing(tmp_path: Path) -> None:
    dot_path, manifest_path = GRAPHICAL_CASES["aircraft_collision"]
    result = run_graphical_mag_pipeline(
        dot_path=dot_path,
        manifest_path=manifest_path,
        framework_path=FRAMEWORK_PATH,
        output_dir=tmp_path,
    )

    analysis = _load_json(result.analysis_path)

    assert isinstance(analysis["framework_violations"], list)
    assert isinstance(analysis["warnings"], list)
    assert analysis["passed"] is result.analysis_passed


def _load_json(path: str) -> dict[str, Any]:
    with Path(path).open(encoding="utf-8") as file:
        data = json.load(file)
    if not isinstance(data, dict):
        raise TypeError("Expected JSON object.")
    return data
