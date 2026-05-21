from pathlib import Path

import pytest

from magcoder.models import (
    AgentAttribute,
    AgentClass,
    AttributeFilter,
    InteractionEdge,
    MAGModel,
    MAGValidationError,
    Port,
    load_mag_from_json,
    load_mag_from_yaml,
    save_mag_to_json,
    save_mag_to_yaml,
)


def _valid_mag() -> MAGModel:
    return MAGModel(
        id="valid_mag",
        agents=[
            AgentClass(
                id="Sender",
                ports=[
                    Port(
                        name="out",
                        direction="output",
                        description="Outbound signal.",
                    ),
                    Port(name="in", direction="input"),
                ],
                attributes=[
                    AgentAttribute(
                        name="energy",
                        type="float",
                        description="Available energy.",
                    ),
                ],
                nl_description="Sends interaction signals.",
            ),
            AgentClass(
                id="Receiver",
                ports=[
                    Port(name="in", direction="input"),
                    Port(name="out", direction="output"),
                ],
                nl_description="Receives interaction signals.",
            ),
        ],
        edges=[
            InteractionEdge(
                id="sender_to_receiver",
                src_agent="Sender",
                src_port="out",
                dst_agent="Receiver",
                dst_port="in",
                filter_id="close_enough",
                nl_description="Sender passes a signal to Receiver.",
            ),
        ],
        filters=[
            AttributeFilter(
                id="close_enough",
                expressions=["distance < 10"],
                nl_description="Only interact when agents are nearby.",
            ),
        ],
        metadata={"source": "test"},
    )


def test_valid_mag_passes_validation() -> None:
    _valid_mag().validate()


def test_edge_referencing_missing_agent_fails() -> None:
    mag = _valid_mag()
    mag.edges[0].dst_agent = "MissingAgent"

    with pytest.raises(MAGValidationError, match="unknown dst_agent"):
        mag.validate()


def test_output_to_input_direction_is_valid() -> None:
    mag = _valid_mag()
    mag.edges[0].src_port = "out"
    mag.edges[0].dst_port = "in"

    mag.validate()


@pytest.mark.parametrize(
    ("src_port", "dst_port", "message"),
    [
        ("in", "in", "expected 'output'"),
        ("out", "out", "expected 'input'"),
    ],
)
def test_invalid_edge_directions_fail(
    src_port: str,
    dst_port: str,
    message: str,
) -> None:
    mag = _valid_mag()
    mag.edges[0].src_port = src_port
    mag.edges[0].dst_port = dst_port

    with pytest.raises(MAGValidationError, match=message):
        mag.validate()


def test_json_yaml_round_trip_preserves_content(tmp_path: Path) -> None:
    mag = _valid_mag()
    json_path = tmp_path / "mag.json"
    yaml_path = tmp_path / "mag.yaml"

    save_mag_to_json(mag, json_path)
    from_json = load_mag_from_json(json_path)
    save_mag_to_yaml(from_json, yaml_path)
    from_yaml = load_mag_from_yaml(yaml_path)

    assert from_json.to_dict() == mag.to_dict()
    assert from_yaml.to_dict() == mag.to_dict()
