import ast
from pathlib import Path

from magcoder.generation import load_generation_result
from magcoder.models import load_mag_from_yaml
from magcoder.refinement import (
    analyze_python_source,
    build_interaction_contracts,
    check_consistency,
    constraints_with_interaction_contracts,
    extract_model_constraints,
    load_framework_spec_from_yaml,
    refine_code,
)

ROOT = Path(__file__).resolve().parents[1]
FRAMEWORK_PATH = ROOT / "examples" / "simple_abms_framework.yaml"
ROBOT_MAG_PATH = ROOT / "examples" / "robot_rescue_mag.yaml"
ROBOT_GENERATION_PATH = ROOT / "examples" / "robot_rescue_generation.json"
AIRCRAFT_MAG_PATH = ROOT / "examples" / "aircraft_collision_mag.yaml"
AIRCRAFT_GENERATION_PATH = ROOT / "examples" / "aircraft_collision_generation.json"


def _refine_example(mag_path: Path, generation_path: Path):
    generation = load_generation_result(generation_path)
    mag = load_mag_from_yaml(mag_path)
    framework = load_framework_spec_from_yaml(FRAMEWORK_PATH)
    return refine_code(generation.code_initial, mag, framework)


def test_refinement_produces_refined_code() -> None:
    result = _refine_example(ROBOT_MAG_PATH, ROBOT_GENERATION_PATH)

    assert result.code_refined
    assert result.code_refined != result.code_initial


def test_refined_code_is_parseable_by_ast() -> None:
    result = _refine_example(ROBOT_MAG_PATH, ROBOT_GENERATION_PATH)

    ast.parse(result.code_refined)


def test_missing_interaction_method_is_added() -> None:
    result = _refine_example(ROBOT_MAG_PATH, ROBOT_GENERATION_PATH)

    assert "def interact_sonar_to_infrared(self, target):" in result.code_refined


def test_send_message_call_is_inserted() -> None:
    result = _refine_example(ROBOT_MAG_PATH, ROBOT_GENERATION_PATH)

    assert 'self.send_message(target, "interaction_triggered")' in result.code_refined


def test_filter_guard_is_preserved_or_inserted() -> None:
    result = _refine_example(ROBOT_MAG_PATH, ROBOT_GENERATION_PATH)

    assert "if distance < 10:" in result.code_refined


def test_refined_report_has_fewer_framework_violations() -> None:
    result = _refine_example(ROBOT_MAG_PATH, ROBOT_GENERATION_PATH)

    assert len(result.refined_report.framework_violations) < len(
        result.initial_report.framework_violations
    )


def test_robot_rescue_refinement() -> None:
    result = _refine_example(ROBOT_MAG_PATH, ROBOT_GENERATION_PATH)

    assert result.refined_report.passed
    assert "add_required_edge_method:sonar_to_infrared" in (
        result.rewrite_actions
    )


def test_aircraft_collision_refinement() -> None:
    result = _refine_example(AIRCRAFT_MAG_PATH, AIRCRAFT_GENERATION_PATH)

    assert result.refined_report.passed
    assert "def interact_aircraft_to_detector(self, target):" in result.code_refined
    assert "if distance < safe_distance:" in result.code_refined


def test_refiner_inserts_send_message_into_existing_required_method() -> None:
    mag = load_mag_from_yaml(ROBOT_MAG_PATH)
    framework = load_framework_spec_from_yaml(FRAMEWORK_PATH)
    code = """
class SonarDetection:
    def __init__(self, distance=None):
        self.distance = distance

    def interact_sonar_to_infrared(self, target):
        if distance < 10:
            return "triggered"
        return None

class InfraredThermal:
    def __init__(self, thermal_confidence=None):
        self.thermal_confidence = thermal_confidence
"""

    result = refine_code(code, mag, framework)

    assert "insert_message_api_call_in_existing_method:sonar_to_infrared" in (
        result.rewrite_actions
    )
    assert 'self.send_message(target, "interaction_triggered")' in result.code_refined
    assert result.refined_report.passed


def test_checker_reports_method_level_message_api_missing() -> None:
    mag = load_mag_from_yaml(ROBOT_MAG_PATH)
    framework = load_framework_spec_from_yaml(FRAMEWORK_PATH)
    constraints = constraints_with_interaction_contracts(
        extract_model_constraints(mag),
        build_interaction_contracts(mag, framework),
    )
    code = """
class SonarDetection:
    def __init__(self, distance=None):
        self.distance = distance

    def interact_sonar_to_infrared(self, target):
        if distance < 10:
            return "triggered"
        return None

class InfraredThermal:
    def __init__(self, thermal_confidence=None):
        self.thermal_confidence = thermal_confidence
"""

    report = check_consistency(analyze_python_source(code), constraints, framework)

    assert any(
        "class 'SonarDetection' method 'interact_sonar_to_infrared'" in violation
        and "missing message API 'send_message'" in violation
        for violation in report.framework_violations
    )
