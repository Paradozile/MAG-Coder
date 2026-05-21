"""Tests for provider registry/config templates."""

from __future__ import annotations

from magcoder.providers.provider_registry import (
    list_provider_templates,
    readiness_without_secrets,
    registry_report,
)


def test_provider_registry_lists_templates_without_api_calls() -> None:
    """Registry exposes provider families without touching external services."""

    templates = list_provider_templates()
    families = {template.family for template in templates}
    assert "ollama_native" in families
    assert "openai_compatible" in families
    assert "gemini" in families
    assert "anthropic" in families
    assert "local_vllm" in families


def test_provider_readiness_reports_missing_credentials() -> None:
    """Credential-backed templates are marked unavailable by schema only."""

    report = readiness_without_secrets("gemini_template")
    assert report["readiness_status"] == "missing_credentials"
    assert "GEMINI_API_KEY" in report["warnings"][0]
    assert "api_key" not in report


def test_registry_report_has_no_external_call_marker() -> None:
    """The registry report is explicitly dry-run/interface-only."""

    report = registry_report()
    assert report["external_calls_made"] is False
    assert report["templates"]
