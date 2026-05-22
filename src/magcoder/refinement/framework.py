"""Framework constraints for refinement analysis."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass(slots=True)
class FrameworkSpec:
    """Minimal target-framework constraints for generated ABMS code."""

    name: str
    required_base_class: str | None = None
    required_agent_methods: list[str] = field(default_factory=list)
    interaction_method_prefix: str = "interact_with"
    message_api_name: str = "send_message"
    state_access_style: str = "attribute"
    naming_conventions: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Render the framework spec as plain Python containers."""

        return asdict(self)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> FrameworkSpec:
        """Build a framework spec from parsed data."""

        required_agent_methods = data.get("required_agent_methods", [])
        naming_conventions = data.get("naming_conventions", {})
        metadata = data.get("metadata", {})

        if not isinstance(required_agent_methods, list):
            raise TypeError("Framework field 'required_agent_methods' must be a list.")
        if not isinstance(naming_conventions, dict):
            raise TypeError("Framework field 'naming_conventions' must be a dict.")
        if not isinstance(metadata, dict):
            raise TypeError("Framework field 'metadata' must be a dict.")

        required_base_class = data.get("required_base_class")
        return cls(
            name=str(data["name"]),
            required_base_class=(
                None if required_base_class is None else str(required_base_class)
            ),
            required_agent_methods=[
                str(method_name) for method_name in required_agent_methods
            ],
            interaction_method_prefix=str(
                data.get("interaction_method_prefix", "interact_with"),
            ),
            message_api_name=str(data.get("message_api_name", "send_message")),
            state_access_style=str(data.get("state_access_style", "attribute")),
            naming_conventions=dict(naming_conventions),
            metadata=dict(metadata),
        )


def load_framework_spec_from_yaml(path: str | Path) -> FrameworkSpec:
    """Load a framework spec from YAML."""

    with Path(path).open(encoding="utf-8") as file:
        data = yaml.safe_load(file)
    if not isinstance(data, Mapping):
        raise TypeError("Framework YAML root must be an object.")
    return FrameworkSpec.from_dict(data)


def save_framework_spec_to_yaml(spec: FrameworkSpec, path: str | Path) -> None:
    """Save a framework spec to YAML."""

    with Path(path).open("w", encoding="utf-8") as file:
        yaml.safe_dump(spec.to_dict(), file, sort_keys=False)
