from pathlib import Path

from magcoder.models import load_mag_from_yaml
from magcoder.semantic import build_hybrid_input

ROOT = Path(__file__).resolve().parents[1]
EXAMPLE_MAG_PATH = ROOT / "examples" / "robot_rescue_mag.yaml"
EXAMPLE_PROMPT_PATH = ROOT / "examples" / "robot_rescue_prompt.txt"
AIRCRAFT_MAG_PATH = ROOT / "examples" / "aircraft_collision_mag.yaml"
AIRCRAFT_PROMPT_PATH = ROOT / "examples" / "aircraft_collision_prompt.txt"


def test_build_hybrid_input_from_robot_rescue_example() -> None:
    mag = load_mag_from_yaml(EXAMPLE_MAG_PATH)

    hybrid = build_hybrid_input(mag)

    assert hybrid.mag_id == "robot_rescue_demo"
    assert hybrid.mode == "graph+nl"


def test_hybrid_input_preserves_agent_edge_filter_counts() -> None:
    mag = load_mag_from_yaml(EXAMPLE_MAG_PATH)

    hybrid = build_hybrid_input(mag)

    assert len(hybrid.agents) == len(mag.agents)
    assert len(hybrid.edges) == len(mag.edges)
    assert len(hybrid.filters) == len(mag.filters)


def test_prompt_text_contains_agent_edge_filter_and_condition_information() -> None:
    mag = load_mag_from_yaml(EXAMPLE_MAG_PATH)
    prompt_text = build_hybrid_input(mag).to_prompt_text()

    assert "## Agents" in prompt_text
    assert "## Interactions" in prompt_text
    assert "## Filters / Conditions" in prompt_text
    assert "## Code generation requirements" in prompt_text
    assert "## Constraints to preserve" in prompt_text
    assert "SonarDetection" in prompt_text
    assert "InfraredThermal" in prompt_text
    assert "sonar_to_infrared" in prompt_text
    assert "nearby_target" in prompt_text
    assert "distance < 10" in prompt_text


def test_graph_only_and_graph_plus_nl_modes_produce_different_outputs() -> None:
    mag = load_mag_from_yaml(EXAMPLE_MAG_PATH)

    graph_only = build_hybrid_input(mag, mode="graph-only")
    graph_plus_nl = build_hybrid_input(mag, mode="graph+nl")

    assert graph_only.to_dict() != graph_plus_nl.to_dict()
    assert graph_only.to_prompt_text() != graph_plus_nl.to_prompt_text()


def test_robot_rescue_prompt_output_matches_renderer() -> None:
    mag = load_mag_from_yaml(EXAMPLE_MAG_PATH)

    assert build_hybrid_input(mag).to_prompt_text() == EXAMPLE_PROMPT_PATH.read_text(
        encoding="utf-8",
    )


def test_aircraft_collision_prompt_output_matches_renderer() -> None:
    mag = load_mag_from_yaml(AIRCRAFT_MAG_PATH)

    assert build_hybrid_input(mag).to_prompt_text() == AIRCRAFT_PROMPT_PATH.read_text(
        encoding="utf-8",
    )
