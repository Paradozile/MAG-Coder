from pathlib import Path

from magcoder.graphics import (
    build_graphical_mag_manifest,
    load_graphical_mag_manifest,
    load_mag_from_graphical_artifact,
    render_mag_artifact,
    render_mag_to_dot,
)
from magcoder.models import load_mag_from_yaml

ROOT = Path(__file__).resolve().parents[1]
ROBOT_MAG_PATH = ROOT / "examples" / "robot_rescue_mag.yaml"


def test_mag_model_can_be_rendered_to_dot() -> None:
    mag = load_mag_from_yaml(ROBOT_MAG_PATH)

    dot_text = render_mag_to_dot(mag)

    assert dot_text.startswith("digraph")
    assert "robot_rescue_demo" in dot_text


def test_dot_contains_agent_nodes() -> None:
    mag = load_mag_from_yaml(ROBOT_MAG_PATH)

    dot_text = render_mag_to_dot(mag)

    assert '"SonarDetection"' in dot_text
    assert '"InfraredThermal"' in dot_text


def test_dot_contains_directed_edges() -> None:
    mag = load_mag_from_yaml(ROBOT_MAG_PATH)

    dot_text = render_mag_to_dot(mag)

    assert '"SonarDetection" -> "InfraredThermal"' in dot_text
    assert "nearby_target" in dot_text


def test_manifest_contains_nodes_edges_and_filters() -> None:
    mag = load_mag_from_yaml(ROBOT_MAG_PATH)

    manifest = build_graphical_mag_manifest(mag)

    assert [node["id"] for node in manifest.nodes] == [
        "SonarDetection",
        "InfraredThermal",
    ]
    assert manifest.edges[0]["id"] == "sonar_to_infrared"
    assert manifest.filters[0]["id"] == "nearby_target"


def test_dot_and_manifest_round_trip_back_to_valid_mag(tmp_path: Path) -> None:
    artifact = render_mag_artifact(
        mag_path=ROBOT_MAG_PATH,
        output_dir=tmp_path,
        formats=("dot", "svg"),
    )

    round_tripped = load_mag_from_graphical_artifact(
        artifact.dot_path,
        artifact.manifest_path,
    )

    round_tripped.validate()
    original = load_mag_from_yaml(ROBOT_MAG_PATH)
    assert round_tripped.id == original.id
    assert round_tripped.to_dict()["agents"] == original.to_dict()["agents"]
    assert round_tripped.to_dict()["edges"] == original.to_dict()["edges"]
    assert round_tripped.to_dict()["filters"] == original.to_dict()["filters"]


def test_render_mag_artifact_does_not_require_graphviz(tmp_path: Path) -> None:
    artifact = render_mag_artifact(
        mag_path=ROBOT_MAG_PATH,
        output_dir=tmp_path,
        formats=("dot", "svg", "png"),
    )

    assert Path(artifact.dot_path).exists()
    assert Path(artifact.manifest_path).exists()
    manifest = load_graphical_mag_manifest(artifact.manifest_path)
    assert manifest.mag_id == "robot_rescue_demo"
    if artifact.svg_path is not None:
        assert Path(artifact.svg_path).exists()
    if artifact.png_path is not None:
        assert Path(artifact.png_path).exists()
