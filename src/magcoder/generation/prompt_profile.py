"""Prompt profile data model and loading utilities."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass(slots=True)
class PromptProfile:
    """A reusable prompt-stabilization profile for generator requests."""

    profile_id: str
    description: str
    system_message: str
    user_prefix: str
    user_suffix: str
    output_format: str
    code_fence_required: bool
    allow_markdown: bool
    require_final_content: bool
    extraction_strategy: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Render this profile as plain Python containers."""

        return asdict(self)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> PromptProfile:
        """Build a prompt profile from parsed YAML data."""

        metadata = data.get("metadata", {})
        if not isinstance(metadata, dict):
            raise TypeError("Prompt profile field 'metadata' must be a dict.")
        return cls(
            profile_id=str(data["profile_id"]),
            description=str(data.get("description", "")),
            system_message=str(data.get("system_message", "")),
            user_prefix=str(data.get("user_prefix", "")),
            user_suffix=str(data.get("user_suffix", "")),
            output_format=str(data.get("output_format", "python_code")),
            code_fence_required=bool(data.get("code_fence_required", False)),
            allow_markdown=bool(data.get("allow_markdown", False)),
            require_final_content=bool(data.get("require_final_content", True)),
            extraction_strategy=str(data.get("extraction_strategy", "python_code")),
            metadata=dict(metadata),
        )


def load_prompt_profile(path: str | Path) -> PromptProfile:
    """Load a prompt profile from YAML."""

    with Path(path).open(encoding="utf-8") as file:
        data = yaml.safe_load(file)
    if not isinstance(data, Mapping):
        raise TypeError("Prompt profile YAML root must be an object.")
    return PromptProfile.from_dict(data)


def find_prompt_profile_path(profile_id: str, *, root: str | Path = ".") -> Path:
    """Return the conventional config path for a profile id."""

    return Path(root) / "configs" / "prompt_profiles" / f"{profile_id}.yaml"


def load_prompt_profile_by_id(
    profile_id: str,
    *,
    root: str | Path = ".",
) -> PromptProfile:
    """Load a prompt profile by id from configs/prompt_profiles."""

    return load_prompt_profile(find_prompt_profile_path(profile_id, root=root))
