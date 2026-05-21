"""Utilities for real generator pilot checks and audits."""

from __future__ import annotations

import csv
import json
import os
import re
import urllib.error
import urllib.request
from collections.abc import Iterable, Iterator, Mapping
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import yaml


@dataclass(slots=True)
class BackendCheckConfig:
    """Configuration for a generator backend preflight check."""

    provider: str
    model_name: str
    api_base_url: str | None = None
    api_key_env: str | None = None
    do_request: bool = False
    timeout_seconds: int = 10
    output_path: str = "backend_check.json"


@dataclass(slots=True)
class BackendCheckResult:
    """Result of a generator backend preflight check."""

    provider: str
    model_name: str
    status: str
    ok: bool
    message: str
    api_base_url: str | None = None
    api_key_env: str | None = None
    env_var_present: bool = False
    request_attempted: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Render this check result as plain Python containers."""

        return asdict(self)


@dataclass(slots=True)
class PilotAuditResult:
    """Summary of a pass@k pilot run audit."""

    run_dir: str
    passk_summary_exists: bool
    sample_results_exists: bool
    total_samples: int
    generation_error_rate: float
    parse_success_rate: float
    framework_pass_rate: float
    mandatory_success_rate: float
    cache_hit_rate: float
    failed_samples: list[dict[str, Any]]
    secret_findings: list[dict[str, Any]]
    audit_path: str

    def to_dict(self) -> dict[str, Any]:
        """Render this audit result as plain Python containers."""

        return asdict(self)


@dataclass(slots=True)
class PilotRunConfig:
    """Config-driven real generator pilot settings."""

    benchmark_path: str
    provider: str
    model_name: str
    api_base_url: str | None
    api_key_env: str | None
    temperature: float
    top_p: float | None
    timeout_seconds: int
    num_samples: int
    k_values: list[int]
    success_criterion: str
    output_dir: str
    notes: str = ""
    dry_run: bool = False
    do_backend_request_check: bool = False
    local_dummy_api_key_allowed: bool = False
    input_modes: list[str] = field(
        default_factory=lambda: ["structured", "graphical"],
    )
    task_ids: list[str] = field(default_factory=list)
    ablation_ids: list[str] = field(
        default_factory=lambda: [
            "graph_nl_with_refinement",
            "graph_only_with_refinement",
            "graph_nl_without_refinement",
        ],
    )
    max_tokens: int | None = None
    stop: list[str] | None = None
    think: bool | None = None
    native_endpoint: str | None = None
    num_predict: int | None = None
    prompt_profile_id: str | None = None
    prompt_profile_path: str | None = None
    require_non_empty_code: bool = False
    smoke_required: bool = False
    benchmark_smoke_required: bool = False
    base_seed: int | None = None
    cache_enabled: bool = True
    resume: bool = True
    run_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def effective_provider(self) -> str:
        """Return the provider that will actually be executed."""

        if self.dry_run:
            return "dry_run"
        return self.provider

    @property
    def resolved_run_id(self) -> str:
        """Return a stable run id for this pilot."""

        if self.run_id:
            return self.run_id
        provider = self.effective_provider.replace("/", "_")
        model = self.model_name.replace("/", "_").replace(" ", "_")
        return f"{provider}_{model}_pilot"

    def to_dict(self) -> dict[str, Any]:
        """Render this config as plain Python containers."""

        data = asdict(self)
        data["effective_provider"] = self.effective_provider
        data["resolved_run_id"] = self.resolved_run_id
        return data


@dataclass(slots=True)
class PilotRunBundle:
    """Output locations for a config-driven pilot run."""

    output_dir: str
    pilot_config_path: str
    backend_check_path: str
    passk_summary_csv_path: str
    passk_summary_md_path: str
    sample_results_path: str
    pilot_audit_path: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Render this bundle as plain Python containers."""

        return asdict(self)


@dataclass(slots=True)
class PilotComparisonResult:
    """Summary output for comparing multiple pilot run directories."""

    rows: list[dict[str, Any]]
    csv_path: str
    markdown_path: str

    def to_dict(self) -> dict[str, Any]:
        """Render this comparison result as plain Python containers."""

        return asdict(self)


SECRET_PATTERNS = [
    re.compile(r"sk-[A-Za-z0-9_\-]{16,}"),
    re.compile(r"Bearer\s+[A-Za-z0-9._\-]{16,}", re.IGNORECASE),
    re.compile(r'"api_key"\s*:\s*"[^"]{8,}"', re.IGNORECASE),
]


def load_pilot_run_config(path: str | Path) -> PilotRunConfig:
    """Load a config-driven pilot YAML file."""

    with Path(path).open(encoding="utf-8") as file:
        data = yaml.safe_load(file)
    if not isinstance(data, Mapping):
        raise TypeError("Pilot config YAML root must be an object.")
    return pilot_run_config_from_mapping(data)


def pilot_run_config_from_mapping(data: Mapping[str, Any]) -> PilotRunConfig:
    """Build a pilot config from parsed YAML data."""

    k_values = data.get("k_values", [1, 3])
    if not isinstance(k_values, list):
        raise TypeError("Pilot config field 'k_values' must be a list.")
    metadata = data.get("metadata", {})
    if not isinstance(metadata, dict):
        raise TypeError("Pilot config field 'metadata' must be a dict.")
    return PilotRunConfig(
        benchmark_path=str(data.get("benchmark_path", "benchmarks/mini_mag_bench")),
        provider=str(data["provider"]),
        model_name=str(data["model_name"]),
        api_base_url=_optional_str(data.get("api_base_url")),
        api_key_env=_optional_str(data.get("api_key_env")),
        temperature=float(data.get("temperature", 0.0)),
        top_p=_optional_float(data.get("top_p")),
        timeout_seconds=int(data.get("timeout_seconds", 60)),
        num_samples=int(data.get("num_samples", 3)),
        k_values=[int(value) for value in k_values],
        success_criterion=str(data.get("success_criterion", "mandatory_success")),
        output_dir=str(data["output_dir"]),
        notes=str(data.get("notes", "")),
        dry_run=bool(data.get("dry_run", False)),
        do_backend_request_check=bool(data.get("do_backend_request_check", False)),
        local_dummy_api_key_allowed=bool(
            data.get("local_dummy_api_key_allowed", False)
        ),
        input_modes=_str_list(data.get("input_modes"), ["structured", "graphical"]),
        task_ids=_str_list(data.get("task_ids"), []),
        ablation_ids=_str_list(
            data.get("ablation_ids"),
            [
                "graph_nl_with_refinement",
                "graph_only_with_refinement",
                "graph_nl_without_refinement",
            ],
        ),
        max_tokens=_optional_int(data.get("max_tokens")),
        stop=_optional_str_list(data.get("stop")),
        think=_optional_bool(data.get("think")),
        native_endpoint=_optional_str(data.get("native_endpoint")),
        num_predict=_optional_int(data.get("num_predict")),
        prompt_profile_id=_optional_str(data.get("prompt_profile_id")),
        prompt_profile_path=_optional_str(data.get("prompt_profile_path")),
        require_non_empty_code=bool(data.get("require_non_empty_code", False)),
        smoke_required=bool(data.get("smoke_required", False)),
        benchmark_smoke_required=bool(data.get("benchmark_smoke_required", False)),
        base_seed=_optional_int(data.get("base_seed")),
        cache_enabled=bool(data.get("cache_enabled", True)),
        resume=bool(data.get("resume", True)),
        run_id=_optional_str(data.get("run_id")),
        metadata=dict(metadata),
    )


def save_pilot_run_config(config: PilotRunConfig, path: str | Path) -> None:
    """Save a resolved pilot config without serializing any secret values."""

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file:
        yaml.safe_dump(config.to_dict(), file, sort_keys=False)


def run_pilot_from_config(
    config: PilotRunConfig,
    *,
    skip_benchmark_smoke_gate: bool = False,
) -> PilotRunBundle:
    """Run a config-driven pilot using the existing pass@k runner."""

    from magcoder.benchmark import load_benchmark_package
    from magcoder.evaluation.multisample import (
        MultiSamplePassKRunner,
        MultiSampleRunConfig,
    )

    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    pilot_config_path = output_dir / "pilot_config_resolved.yaml"
    backend_check_path = output_dir / "backend_check.json"
    save_pilot_run_config(config, pilot_config_path)
    if config.benchmark_smoke_required and not skip_benchmark_smoke_gate:
        _require_successful_benchmark_smoke(output_dir / "benchmark_smoke_result.json")

    dummy_key_enabled = _should_use_local_dummy_key(config)
    with _temporary_dummy_key(config.api_key_env, enabled=dummy_key_enabled):
        backend_result = check_generator_backend(
            BackendCheckConfig(
                provider=config.effective_provider,
                model_name=config.model_name,
                api_base_url=config.api_base_url,
                api_key_env=config.api_key_env,
                timeout_seconds=config.timeout_seconds,
                do_request=config.do_backend_request_check and not config.dry_run,
            )
        )
        save_backend_check_result(backend_result, backend_check_path)
        if not backend_result.ok:
            raise RuntimeError(f"Backend check failed: {backend_result.message}")

        package = load_benchmark_package(config.benchmark_path)
        run_config = MultiSampleRunConfig(
            run_id=config.resolved_run_id,
            benchmark_path=config.benchmark_path,
            provider=config.effective_provider,
            model_name=config.model_name,
            api_base_url=config.api_base_url,
            api_key_env=config.api_key_env,
            timeout_seconds=config.timeout_seconds,
            temperature=config.temperature,
            top_p=config.top_p,
            max_tokens=config.max_tokens,
            stop=config.stop,
            think=config.think,
            native_endpoint=config.native_endpoint,
            num_predict=config.num_predict,
            prompt_profile_id=config.prompt_profile_id,
            prompt_profile_path=config.prompt_profile_path,
            require_non_empty_code=config.require_non_empty_code,
            base_seed=config.base_seed,
            num_samples=config.num_samples,
            k_values=config.k_values,
            input_modes=config.input_modes,
            task_ids=config.task_ids,
            ablation_ids=config.ablation_ids,
            cache_enabled=config.cache_enabled,
            resume=config.resume,
            output_dir=config.output_dir,
            success_criterion=config.success_criterion,
            metadata=config.metadata,
        )
        result = MultiSamplePassKRunner().run(package, run_config)
    audit = audit_real_pilot(output_dir)
    return PilotRunBundle(
        output_dir=str(output_dir),
        pilot_config_path=str(pilot_config_path),
        backend_check_path=str(backend_check_path),
        passk_summary_csv_path=str(output_dir / "passk_summary.csv"),
        passk_summary_md_path=str(output_dir / "passk_summary.md"),
        sample_results_path=str(output_dir / "sample_results.jsonl"),
        pilot_audit_path=audit.audit_path,
        metadata={
            "num_samples_written": len(result.sample_results),
            "summary_rows": len(result.summary_rows),
            "backend_status": backend_result.status,
            "audit_secret_findings": len(audit.secret_findings),
        },
    )


def _require_successful_benchmark_smoke(path: Path) -> None:
    if not path.exists():
        raise RuntimeError(
            "Benchmark smoke gate failed: benchmark_smoke_result.json is missing. "
            "Run scripts/smoke_test_benchmark_prompt.py first, or pass "
            "--skip-benchmark-smoke-gate explicitly."
        )
    with path.open(encoding="utf-8") as file:
        data = json.load(file)
    if not isinstance(data, dict) or data.get("benchmark_smoke_success") is not True:
        raise RuntimeError(
            "Benchmark smoke gate failed: benchmark_smoke_success is not true."
        )


def _should_use_local_dummy_key(config: PilotRunConfig) -> bool:
    if not config.local_dummy_api_key_allowed:
        return False
    if config.effective_provider not in {"openai_compatible", "http"}:
        return False
    if not config.api_key_env or os.environ.get(config.api_key_env):
        return False
    parsed = urlparse(config.api_base_url or "")
    return parsed.hostname in {"127.0.0.1", "localhost", "::1"}


@contextmanager
def _temporary_dummy_key(api_key_env: str | None, *, enabled: bool) -> Iterator[None]:
    if not api_key_env or not enabled:
        yield
        return
    previous = os.environ.get(api_key_env)
    os.environ[api_key_env] = "dummy"
    try:
        yield
    finally:
        if previous is None:
            os.environ.pop(api_key_env, None)
        else:
            os.environ[api_key_env] = previous


def compare_pilot_runs(
    run_dirs: list[str | Path],
    *,
    output_path: str | Path,
) -> PilotComparisonResult:
    """Compare pass@k summaries across multiple pilot run directories."""

    markdown_path = Path(output_path)
    csv_path = markdown_path.with_suffix(".csv")
    if markdown_path.suffix != ".md":
        markdown_path = markdown_path.with_suffix(".md")
    rows = [_pilot_comparison_row(Path(run_dir)) for run_dir in run_dirs]
    _write_pilot_comparison_csv(rows, csv_path)
    _write_pilot_comparison_markdown(rows, markdown_path)
    return PilotComparisonResult(
        rows=rows,
        csv_path=str(csv_path),
        markdown_path=str(markdown_path),
    )


def check_generator_backend(config: BackendCheckConfig) -> BackendCheckResult:
    """Check whether a configured generator backend is ready to use."""

    if config.provider in {"fake", "dry_run"}:
        return BackendCheckResult(
            provider=config.provider,
            model_name=config.model_name,
            status="passed",
            ok=True,
            message=f"Provider {config.provider!r} does not require a backend.",
            request_attempted=False,
        )

    if config.provider == "ollama_native":
        if not config.api_base_url:
            return _failed_backend_result(config, "api_base_url is required.")
        if not config.do_request:
            return BackendCheckResult(
                provider=config.provider,
                model_name=config.model_name,
                status="ready",
                ok=True,
                message=(
                    "Ollama native configuration is present. No network request "
                    "was sent because --do-request was not set."
                ),
                api_base_url=config.api_base_url,
                api_key_env=config.api_key_env,
                env_var_present=bool(
                    config.api_key_env and os.environ.get(config.api_key_env)
                ),
                request_attempted=False,
            )
        return _request_ollama_native_backend(config)

    if config.provider not in {"openai_compatible", "http"}:
        return BackendCheckResult(
            provider=config.provider,
            model_name=config.model_name,
            status="failed",
            ok=False,
            message=f"Unsupported provider: {config.provider}",
            api_base_url=config.api_base_url,
            api_key_env=config.api_key_env,
        )

    if not config.api_base_url:
        return _failed_backend_result(config, "api_base_url is required.")
    if not config.api_key_env:
        return _failed_backend_result(config, "api_key_env is required.")

    api_key = os.environ.get(config.api_key_env)
    if not api_key:
        return _failed_backend_result(
            config,
            f"Environment variable {config.api_key_env!r} is not set.",
            env_var_present=False,
        )

    if not config.do_request:
        return BackendCheckResult(
            provider=config.provider,
            model_name=config.model_name,
            status="ready",
            ok=True,
            message=(
                "Configuration is present. No network request was sent because "
                "--do-request was not set."
            ),
            api_base_url=config.api_base_url,
            api_key_env=config.api_key_env,
            env_var_present=True,
            request_attempted=False,
        )

    return _request_backend(config, api_key)


def save_backend_check_result(
    result: BackendCheckResult,
    path: str | Path,
) -> None:
    """Save a backend check result as JSON."""

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file:
        json.dump(result.to_dict(), file, indent=2, sort_keys=True)
        file.write("\n")


def audit_real_pilot(run_dir: str | Path) -> PilotAuditResult:
    """Audit a pass@k pilot run directory and write pilot_audit.md."""

    root = Path(run_dir)
    passk_summary_path = root / "passk_summary.csv"
    sample_results_path = root / "sample_results.jsonl"
    samples = _load_sample_results(sample_results_path)
    total = len(samples)
    failed_samples = [
        _failed_sample_record(sample)
        for sample in samples
        if sample.get("generation_error") or not bool(sample.get("success"))
    ]
    secret_findings = scan_for_possible_secrets(root)
    result = PilotAuditResult(
        run_dir=str(root),
        passk_summary_exists=passk_summary_path.exists(),
        sample_results_exists=sample_results_path.exists(),
        total_samples=total,
        generation_error_rate=_rate(
            bool(sample.get("generation_error")) for sample in samples
        ),
        parse_success_rate=_rate(
            bool(sample.get("parse_success")) for sample in samples
        ),
        framework_pass_rate=_rate(
            bool(sample.get("framework_passed")) for sample in samples
        ),
        mandatory_success_rate=_rate(
            bool(sample.get("mandatory_success")) for sample in samples
        ),
        cache_hit_rate=_rate(bool(sample.get("cache_hit")) for sample in samples),
        failed_samples=failed_samples,
        secret_findings=secret_findings,
        audit_path=str(root / "pilot_audit.md"),
    )
    _write_pilot_audit(result, Path(result.audit_path))
    return result


def scan_for_possible_secrets(root: str | Path) -> list[dict[str, Any]]:
    """Scan text-like pilot artifacts for obvious API key patterns."""

    findings: list[dict[str, Any]] = []
    text_suffixes = {".json", ".jsonl", ".yaml", ".yml", ".md", ".csv"}
    for path in Path(root).rglob("*"):
        if not path.is_file() or path.suffix not in text_suffixes:
            continue
        try:
            lines = path.read_text(encoding="utf-8").splitlines()
        except UnicodeDecodeError:
            continue
        for line_number, line in enumerate(lines, start=1):
            for pattern in SECRET_PATTERNS:
                if pattern.search(line):
                    findings.append(
                        {
                            "path": str(path),
                            "line": line_number,
                            "pattern": pattern.pattern,
                        }
                    )
    return findings


def _failed_backend_result(
    config: BackendCheckConfig,
    message: str,
    *,
    env_var_present: bool = False,
) -> BackendCheckResult:
    return BackendCheckResult(
        provider=config.provider,
        model_name=config.model_name,
        status="failed",
        ok=False,
        message=message,
        api_base_url=config.api_base_url,
        api_key_env=config.api_key_env,
        env_var_present=env_var_present,
        request_attempted=False,
    )


def _request_backend(
    config: BackendCheckConfig,
    api_key: str,
) -> BackendCheckResult:
    endpoint = _chat_completions_endpoint(str(config.api_base_url))
    payload = {
        "model": config.model_name,
        "messages": [{"role": "user", "content": "Return the word ok."}],
        "temperature": 0.0,
        "max_tokens": 8,
    }
    request = urllib.request.Request(
        endpoint,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(
            request,
            timeout=config.timeout_seconds,
        ) as response:
            body = response.read().decode("utf-8", errors="replace")
    except urllib.error.HTTPError as error:
        body = error.read().decode("utf-8", errors="replace")
        return _request_failed_result(
            config,
            f"HTTP {error.code}: {_redact_secret(body, api_key)}",
        )
    except urllib.error.URLError as error:
        return _request_failed_result(config, f"Request failed: {error.reason}")
    except TimeoutError:
        return _request_failed_result(config, "Request timed out.")

    return BackendCheckResult(
        provider=config.provider,
        model_name=config.model_name,
        status="passed",
        ok=True,
        message="Tiny chat/completions request succeeded.",
        api_base_url=config.api_base_url,
        api_key_env=config.api_key_env,
        env_var_present=True,
        request_attempted=True,
        metadata={"response_preview": _redact_secret(body, api_key)[:200]},
    )


def _request_ollama_native_backend(config: BackendCheckConfig) -> BackendCheckResult:
    endpoint = str(config.api_base_url).rstrip("/") + "/api/tags"
    request = urllib.request.Request(endpoint, method="GET")
    try:
        with urllib.request.urlopen(
            request,
            timeout=config.timeout_seconds,
        ) as response:
            body = response.read().decode("utf-8", errors="replace")
    except urllib.error.HTTPError as error:
        body = error.read().decode("utf-8", errors="replace")
        return _request_failed_result(config, f"HTTP {error.code}: {body}")
    except urllib.error.URLError as error:
        return _request_failed_result(config, f"Request failed: {error.reason}")
    except TimeoutError:
        return _request_failed_result(config, "Request timed out.")

    return BackendCheckResult(
        provider=config.provider,
        model_name=config.model_name,
        status="passed",
        ok=True,
        message="Ollama native /api/tags request succeeded.",
        api_base_url=config.api_base_url,
        api_key_env=config.api_key_env,
        env_var_present=bool(config.api_key_env and os.environ.get(config.api_key_env)),
        request_attempted=True,
        metadata={"response_preview": body[:200]},
    )


def _request_failed_result(
    config: BackendCheckConfig,
    message: str,
) -> BackendCheckResult:
    return BackendCheckResult(
        provider=config.provider,
        model_name=config.model_name,
        status="failed",
        ok=False,
        message=message,
        api_base_url=config.api_base_url,
        api_key_env=config.api_key_env,
        env_var_present=True,
        request_attempted=True,
    )


def _redact_secret(text: str, secret: str) -> str:
    if not secret:
        return text
    return text.replace(secret, "[REDACTED]")


def _chat_completions_endpoint(api_base_url: str) -> str:
    normalized = api_base_url.rstrip("/")
    if normalized.endswith("/chat/completions"):
        return normalized
    return f"{normalized}/chat/completions"


def _load_sample_results(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    samples: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as file:
        for line in file:
            if not line.strip():
                continue
            data = json.loads(line)
            if isinstance(data, dict):
                samples.append(data)
    return samples


def _failed_sample_record(sample: dict[str, Any]) -> dict[str, Any]:
    return {
        "task_id": sample.get("task_id", ""),
        "input_mode": sample.get("input_mode", ""),
        "ablation_id": sample.get("ablation_id", ""),
        "sample_index": sample.get("sample_index", ""),
        "generation_error": sample.get("generation_error", ""),
        "success": sample.get("success", False),
    }


def _write_pilot_audit(result: PilotAuditResult, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Real Generator Pilot Audit",
        "",
        f"- Run directory: `{result.run_dir}`",
        f"- passk_summary.csv exists: {result.passk_summary_exists}",
        f"- sample_results.jsonl exists: {result.sample_results_exists}",
        f"- Total samples: {result.total_samples}",
        f"- Generation error rate: {result.generation_error_rate:.3f}",
        f"- Parse success rate: {result.parse_success_rate:.3f}",
        f"- Framework pass rate: {result.framework_pass_rate:.3f}",
        f"- Mandatory success rate: {result.mandatory_success_rate:.3f}",
        f"- Cache hit rate: {result.cache_hit_rate:.3f}",
        f"- Possible secret findings: {len(result.secret_findings)}",
        "",
        "## Failed Samples",
        "",
    ]
    if result.failed_samples:
        headers = [
            "task_id",
            "input_mode",
            "ablation_id",
            "sample_index",
            "generation_error",
            "success",
        ]
        lines.extend(
            [
                "| " + " | ".join(headers) + " |",
                "| " + " | ".join("---" for _ in headers) + " |",
            ]
        )
        for sample in result.failed_samples[:50]:
            lines.append(
                "| "
                + " | ".join(str(sample.get(header, "")) for header in headers)
                + " |"
            )
    else:
        lines.append("No failed samples.")

    lines.extend(["", "## Possible Secret Findings", ""])
    if result.secret_findings:
        for finding in result.secret_findings:
            lines.append(
                f"- `{finding['path']}` line {finding['line']}: "
                f"`{finding['pattern']}`"
            )
    else:
        lines.append("No obvious API key patterns were found.")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _rate(values: Iterable[Any]) -> float:
    items = list(values)
    if not items:
        return 0.0
    return sum(1 for item in items if item) / len(items)


def load_pilot_summary_csv(path: str | Path) -> list[dict[str, str]]:
    """Load a CSV summary as dictionaries for lightweight tests/tools."""

    with Path(path).open(encoding="utf-8", newline="") as file:
        return list(csv.DictReader(file))


def _pilot_comparison_row(run_dir: Path) -> dict[str, Any]:
    summary_path = run_dir / "passk_summary.csv"
    if not summary_path.exists():
        raise FileNotFoundError(f"Missing passk_summary.csv in {run_dir}")
    rows = load_pilot_summary_csv(summary_path)
    if not rows:
        raise ValueError(f"Empty passk_summary.csv in {run_dir}")
    first = rows[0]
    return {
        "run_dir": str(run_dir),
        "run_id": first.get("run_id", ""),
        "provider": first.get("provider", ""),
        "model_name": first.get("model_name", ""),
        "num_samples": first.get("num_samples", ""),
        "success_criterion": first.get("success_criterion", ""),
        "estimated_pass_at_1": _format_optional_float(
            _average_float(
                row.get("estimated_pass_at_k")
                for row in rows
                if row.get("k") == "1"
            )
        ),
        "estimated_pass_at_3": _format_optional_float(
            _average_float(
                row.get("estimated_pass_at_k")
                for row in rows
                if row.get("k") == "3"
            )
        ),
        "generation_error_rate": _format_optional_float(
            _average_float(row.get("generation_error_rate") for row in rows)
        ),
        "parse_success_rate": _format_optional_float(
            _average_float(row.get("parse_success_rate") for row in rows)
        ),
        "framework_pass_rate": _format_optional_float(
            _average_float(row.get("framework_pass_rate") for row in rows)
        ),
    }


def _write_pilot_comparison_csv(
    rows: list[dict[str, Any]],
    path: Path,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = _pilot_comparison_fieldnames()
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_pilot_comparison_markdown(
    rows: list[dict[str, Any]],
    path: Path,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = _pilot_comparison_fieldnames()
    lines = [
        "# Real Generator Pilot Comparison",
        "",
        "This table compares pilot runs using their pass@k summary files.",
        "",
        "| " + " | ".join(fieldnames) + " |",
        "| " + " | ".join("---" for _ in fieldnames) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(row[field]) for field in fieldnames) + " |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _pilot_comparison_fieldnames() -> list[str]:
    return [
        "run_dir",
        "run_id",
        "provider",
        "model_name",
        "num_samples",
        "success_criterion",
        "estimated_pass_at_1",
        "estimated_pass_at_3",
        "generation_error_rate",
        "parse_success_rate",
        "framework_pass_rate",
    ]


def _average_float(values: Iterable[str | None]) -> float | None:
    floats: list[float] = []
    for value in values:
        if value is None or value == "":
            continue
        floats.append(float(value))
    if not floats:
        return None
    return sum(floats) / len(floats)


def _format_optional_float(value: float | None) -> str:
    if value is None:
        return ""
    return f"{value:.3f}"


def _optional_str(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value)
    return text if text else None


def _optional_int(value: Any) -> int | None:
    if value is None:
        return None
    return int(value)


def _optional_bool(value: Any) -> bool | None:
    if value is None:
        return None
    return bool(value)


def _optional_str_list(value: Any) -> list[str] | None:
    if value is None:
        return None
    if not isinstance(value, list):
        raise TypeError("Expected a list of strings.")
    return [str(item) for item in value]


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    return float(value)


def _str_list(value: Any, default: list[str]) -> list[str]:
    if value is None:
        return list(default)
    if not isinstance(value, list):
        raise TypeError("Expected a list of strings.")
    return [str(item) for item in value]
