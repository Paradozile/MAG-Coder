"""Provider registry and dry-run readiness surface for paper parity."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass(slots=True)
class ProviderTemplate:
    """Configuration template for one provider family."""

    provider_id: str
    family: str
    model_name: str
    config_path: str
    credential_env_vars: list[str] = field(default_factory=list)
    local_only: bool = False
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Render as JSON-compatible data."""

        return asdict(self)


DEFAULT_PROVIDER_TEMPLATES: tuple[ProviderTemplate, ...] = (
    ProviderTemplate(
        provider_id="ollama_native_qwen35_9b",
        family="ollama_native",
        model_name="qwen3.5:9b",
        config_path="configs/providers/ollama_native_qwen35_9b.yaml",
        local_only=True,
        notes=["Local Ollama model must already exist; no auto-pull is performed."],
    ),
    ProviderTemplate(
        provider_id="openai_compatible_template",
        family="openai_compatible",
        model_name="gpt-4o-compatible-template",
        config_path="configs/providers/openai_compatible_template.yaml",
        credential_env_vars=["OPENAI_API_KEY"],
        notes=["Template only; readiness reports missing credentials without calls."],
    ),
    ProviderTemplate(
        provider_id="gemini_template",
        family="gemini",
        model_name="gemini-template",
        config_path="configs/providers/gemini_template.yaml",
        credential_env_vars=["GEMINI_API_KEY"],
        notes=["Template only; no external API call is made by registry readiness."],
    ),
    ProviderTemplate(
        provider_id="anthropic_template",
        family="anthropic",
        model_name="claude-template",
        config_path="configs/providers/anthropic_template.yaml",
        credential_env_vars=["ANTHROPIC_API_KEY"],
        notes=["Template only; no external API call is made by registry readiness."],
    ),
    ProviderTemplate(
        provider_id="local_vllm_template",
        family="local_vllm",
        model_name="local-vllm-template",
        config_path="configs/providers/local_vllm_template.yaml",
        local_only=True,
        notes=["Local endpoint template; no model download is performed."],
    ),
)


def list_provider_templates() -> list[ProviderTemplate]:
    """Return built-in provider templates."""

    return list(DEFAULT_PROVIDER_TEMPLATES)


def load_provider_template(path: str | Path) -> ProviderTemplate:
    """Load a provider template YAML without reading credentials."""

    data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise TypeError(f"Provider template must be a mapping: {path}")
    return ProviderTemplate(
        provider_id=str(data["provider_id"]),
        family=str(data["family"]),
        model_name=str(data.get("model_name", "")),
        config_path=str(path),
        credential_env_vars=[str(item) for item in data.get("credential_env_vars", [])],
        local_only=bool(data.get("local_only", False)),
        notes=[str(item) for item in data.get("notes", [])],
    )


def readiness_without_secrets(
    template: ProviderTemplate | str,
    *,
    available_env_vars: set[str] | None = None,
) -> dict[str, Any]:
    """Return structured readiness without reading or emitting secret values."""

    template_obj = _resolve_template(template)
    available = available_env_vars or set()
    missing = [
        name for name in template_obj.credential_env_vars if name not in available
    ]
    status = "ready" if not missing else "missing_credentials"
    if (
        template_obj.local_only
        and template_obj.family in {"ollama_native", "local_vllm"}
    ):
        status = "interface_available"
    warnings = [
        f"Missing credential environment variable: {name}" for name in missing
    ]
    return {
        "provider_id": template_obj.provider_id,
        "family": template_obj.family,
        "model_name": template_obj.model_name,
        "status": status,
        "readiness_status": status,
        "missing_credential_env_vars": missing,
        "warnings": warnings,
        "external_api_called": False,
        "secrets_read": False,
        "notes": template_obj.notes,
    }


def registry_report() -> dict[str, Any]:
    """Return a registry report suitable for parity audit artifacts."""

    templates = [template.to_dict() for template in list_provider_templates()]
    return {
        "provider_families": sorted({item["family"] for item in templates}),
        "templates": templates,
        "external_calls_made": False,
        "readiness": [
            readiness_without_secrets(template)
            for template in list_provider_templates()
        ],
    }


def save_registry_report(path: str | Path) -> None:
    """Save provider registry report JSON."""

    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(registry_report(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _resolve_template(template: ProviderTemplate | str) -> ProviderTemplate:
    if isinstance(template, ProviderTemplate):
        return template
    for candidate in DEFAULT_PROVIDER_TEMPLATES:
        if candidate.provider_id == template or candidate.family == template:
            return candidate
    raise ValueError(f"Unknown provider template: {template}")
