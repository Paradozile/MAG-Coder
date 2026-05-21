import ast
from pathlib import Path

from magcoder.generation import load_generation_result
from magcoder.models import load_mag_from_yaml
from magcoder.refinement import (
    FrameworkSpec,
    analyze_python_source,
    check_consistency,
    extract_model_constraints,
    load_framework_spec_from_yaml,
)

ROOT = Path(__file__).resolve().parents[1]
ROBOT_MAG_PATH = ROOT / "examples" / "robot_rescue_mag.yaml"
ROBOT_GENERATION_PATH = ROOT / "examples" / "robot_rescue_generation.json"
FRAMEWORK_PATH = ROOT / "examples" / "simple_abms_framework.yaml"


def _robot_code() -> str:
    return load_generation_result(ROBOT_GENERATION_PATH).code_initial


def _fake_compatible_framework() -> FrameworkSpec:
    return FrameworkSpec(
        name="fake_compatible",
        required_agent_methods=["__init__"],
        interaction_method_prefix="interact",
        message_api_name="",
        naming_conventions={"interaction_method_style": "edge_function"},
    )


def test_extract_model_constraints_from_robot_rescue() -> None:
    constraints = extract_model_constraints(load_mag_from_yaml(ROBOT_MAG_PATH))

    assert constraints.expected_agent_class_names == [
        "SonarDetection",
        "InfraredThermal",
    ]
    assert constraints.expected_attributes["SonarDetection"] == ["distance"]
    assert constraints.expected_filter_expressions["nearby_target"] == [
        "distance < 10",
    ]
    assert constraints.expected_source_target_direction["sonar_to_infrared"] == (
        "SonarDetection.detection_signal -> InfraredThermal.sonar_input"
    )


def test_ast_analyzer_detects_classes_methods_and_guards() -> None:
    analysis = analyze_python_source(_robot_code())

    assert analysis.parse_errors == []
    assert "SonarDetection" in analysis.class_names
    assert "__init__" in analysis.methods_by_class["SonarDetection"]
    assert "distance" in analysis.attributes_by_class["SonarDetection"]
    assert "not _filter_nearby_target(source, target)" in analysis.if_conditions
    assert "_filter_nearby_target" in analysis.function_calls
    ast.parse(_robot_code())


def test_checker_passes_on_reasonable_fake_generation() -> None:
    constraints = extract_model_constraints(load_mag_from_yaml(ROBOT_MAG_PATH))
    analysis = analyze_python_source(_robot_code())

    report = check_consistency(
        analysis,
        constraints,
        _fake_compatible_framework(),
    )

    assert report.passed


def test_checker_reports_missing_class() -> None:
    code = _robot_code().replace("class InfraredThermal:", "class MissingTarget:")
    constraints = extract_model_constraints(load_mag_from_yaml(ROBOT_MAG_PATH))

    report = check_consistency(
        analyze_python_source(code),
        constraints,
        _fake_compatible_framework(),
    )

    assert report.missing_agent_classes == ["InfraredThermal"]
    assert not report.passed


def test_checker_reports_missing_guard() -> None:
    code = """
class SonarDetection:
    def __init__(self) -> None:
        self.distance = None

class InfraredThermal:
    def __init__(self) -> None:
        self.thermal_confidence = None

def interact_sonar_to_infrared(source, target):
    return {"delivered": True}
"""
    constraints = extract_model_constraints(load_mag_from_yaml(ROBOT_MAG_PATH))

    report = check_consistency(
        analyze_python_source(code),
        constraints,
        _fake_compatible_framework(),
    )

    assert report.missing_filter_guards == ["sonar_to_infrared"]
    assert not report.passed


def test_framework_spec_loads_from_yaml() -> None:
    framework = load_framework_spec_from_yaml(FRAMEWORK_PATH)

    assert framework.name == "simple_abms"
    assert framework.required_agent_methods == ["__init__"]
    assert framework.interaction_method_prefix == "interact_with"
    assert framework.message_api_name == "send_message"
