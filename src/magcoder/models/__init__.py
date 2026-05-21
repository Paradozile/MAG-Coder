"""Data model definitions for MAG structures."""

from magcoder.models.mag import (
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

__all__ = [
    "AgentAttribute",
    "AgentClass",
    "AttributeFilter",
    "InteractionEdge",
    "MAGModel",
    "MAGValidationError",
    "Port",
    "load_mag_from_json",
    "load_mag_from_yaml",
    "save_mag_to_json",
    "save_mag_to_yaml",
]
