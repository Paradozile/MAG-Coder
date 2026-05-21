"""Safe local backend autostart and pilot orchestration."""

from __future__ import annotations

import os
import shutil
import socket
import subprocess
import sys
import time
import urllib.parse
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml

from magcoder.evaluation.pilot import (
    PilotRunBundle,
    PilotRunConfig,
    run_pilot_from_config,
)
from magcoder.utils.paths import (
    ensure_output_path_within_repo,
    is_path_within_repo,
)

LOCAL_DUMMY_API_KEY_VALUE = "magcoder-local-dummy-key"


class LocalBackendPilotError(RuntimeError):
    """Raised when a local backend pilot cannot safely proceed."""


@dataclass(slots=True)
class LocalBackendConfig:
    """Configuration for local backend autostart and pilot execution."""

    backend_id: str
    provider: str
    backend_kind: str
    model_name: str
    api_base_url: str
    api_key_env: str
    local_dummy_api_key_allowed: bool
    autostart: bool
    keep_server_running: bool
    server_start_timeout_seconds: int
    local_env_dir: str
    allow_create_local_env: bool
    server_command: list[str]
    benchmark_path: str
    temperature: float
    top_p: float | None
    num_samples: int
    k_values: list[int]
    success_criterion: str
    output_dir: str
    dry_run: bool = False
    timeout_seconds: int = 60
    max_tokens: int | None = None
    stop: list[str] | None = None
    prompt_profile_id: str | None = None
    prompt_profile_path: str | None = None
    require_non_empty_code: bool = False
    notes: str = ""
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
    read_only_reference_paths: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Render this config as plain Python containers."""

        return asdict(self)


@dataclass(slots=True)
class LocalBackendPreflightResult:
    """Safety and readiness checks for a local backend config."""

    ok: bool
    errors: list[str]
    warnings: list[str]
    output_dir: str
    server_command_found: bool
    server_executable: str | None
    endpoint_available: bool | None
    api_key_env: str
    api_key_available: bool
    dummy_api_key_used: bool
    read_only_reference_paths: list[str]

    def to_dict(self) -> dict[str, Any]:
        """Render this result as plain Python containers."""

        return asdict(self)


@dataclass(slots=True)
class LocalBackendPilotResult:
    """Result of a local backend check/autostart/pilot attempt."""

    config_path: str
    output_dir: str
    mode: str
    preflight: LocalBackendPreflightResult
    pilot_completed: bool
    server_started: bool
    server_stopped: bool
    bundle: PilotRunBundle | None = None
    audit_path: str | None = None
    report_path: str | None = None
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Render this result as plain Python containers."""

        data = asdict(self)
        data["bundle"] = None if self.bundle is None else self.bundle.to_dict()
        return data


def load_local_backend_config(path: str | Path) -> LocalBackendConfig:
    """Load a local backend YAML config."""

    with Path(path).open(encoding="utf-8") as file:
        data = yaml.safe_load(file)
    if not isinstance(data, Mapping):
        raise TypeError("Local backend config YAML root must be an object.")
    return local_backend_config_from_mapping(data)


def local_backend_config_from_mapping(
    data: Mapping[str, Any],
) -> LocalBackendConfig:
    """Build a local backend config from parsed YAML."""

    command = data.get("server_command", [])
    if not isinstance(command, list) or not command:
        raise TypeError("server_command must be a non-empty list.")
    k_values = data.get("k_values", [1, 3])
    if not isinstance(k_values, list):
        raise TypeError("k_values must be a list.")
    metadata = data.get("metadata", {})
    if not isinstance(metadata, dict):
        raise TypeError("metadata must be a dict.")
    return LocalBackendConfig(
        backend_id=str(data["backend_id"]),
        provider=str(data["provider"]),
        backend_kind=str(data["backend_kind"]),
        model_name=str(data["model_name"]),
        api_base_url=str(data["api_base_url"]),
        api_key_env=str(data["api_key_env"]),
        local_dummy_api_key_allowed=bool(
            data.get("local_dummy_api_key_allowed", False)
        ),
        autostart=bool(data.get("autostart", False)),
        keep_server_running=bool(data.get("keep_server_running", False)),
        server_start_timeout_seconds=int(
            data.get("server_start_timeout_seconds", 120)
        ),
        local_env_dir=str(data.get("local_env_dir", ".venv_mlx_local")),
        allow_create_local_env=bool(data.get("allow_create_local_env", False)),
        server_command=[str(item) for item in command],
        benchmark_path=str(data.get("benchmark_path", "benchmarks/mini_mag_bench")),
        temperature=float(data.get("temperature", 0.2)),
        top_p=_optional_float(data.get("top_p")),
        num_samples=int(data.get("num_samples", 3)),
        k_values=[int(value) for value in k_values],
        success_criterion=str(data.get("success_criterion", "mandatory_success")),
        output_dir=str(data["output_dir"]),
        dry_run=bool(data.get("dry_run", False)),
        timeout_seconds=int(data.get("timeout_seconds", 60)),
        max_tokens=_optional_int(data.get("max_tokens")),
        stop=_optional_str_list(data.get("stop")),
        prompt_profile_id=_optional_str(data.get("prompt_profile_id")),
        prompt_profile_path=_optional_str(data.get("prompt_profile_path")),
        require_non_empty_code=bool(data.get("require_non_empty_code", False)),
        notes=str(data.get("notes", "")),
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
        read_only_reference_paths=_str_list(
            data.get("read_only_reference_paths"),
            [],
        ),
        metadata=dict(metadata),
    )


def preflight_local_backend(
    config: LocalBackendConfig,
    *,
    repo_root: str | Path,
    check_endpoint: bool = False,
) -> LocalBackendPreflightResult:
    """Validate a local backend config without starting a server."""

    root = Path(repo_root).resolve()
    errors: list[str] = []
    warnings: list[str] = []
    try:
        output_dir = ensure_output_path_within_repo(config.output_dir, root)
    except ValueError as error:
        output_dir = (root / config.output_dir).resolve(strict=False)
        errors.append(str(error))
    try:
        local_env_dir = ensure_output_path_within_repo(config.local_env_dir, root)
    except ValueError as error:
        local_env_dir = (root / config.local_env_dir).resolve(strict=False)
        errors.append(f"local_env_dir is not allowed: {error}")

    if config.provider not in {"openai_compatible", "fake", "dry_run"}:
        errors.append(
            f"Unsupported provider for local backend pilot: {config.provider}"
        )

    api_key_available = bool(os.environ.get(config.api_key_env))
    dummy_api_key_used = False
    if not api_key_available and config.local_dummy_api_key_allowed:
        dummy_api_key_used = True
        warnings.append(
            f"{config.api_key_env} is missing; a process-local dummy value can be used."
        )
    elif not api_key_available:
        errors.append(f"Environment variable {config.api_key_env!r} is not set.")

    server_executable, command_errors = resolve_server_executable(
        config.server_command,
        repo_root=root,
        local_env_dir=local_env_dir,
    )
    errors.extend(command_errors)

    endpoint_available = None
    if check_endpoint:
        endpoint_available = is_openai_endpoint_listening(config.api_base_url)

    return LocalBackendPreflightResult(
        ok=not errors,
        errors=errors,
        warnings=warnings,
        output_dir=str(output_dir),
        server_command_found=server_executable is not None,
        server_executable=server_executable,
        endpoint_available=endpoint_available,
        api_key_env=config.api_key_env,
        api_key_available=api_key_available,
        dummy_api_key_used=dummy_api_key_used,
        read_only_reference_paths=[
            str(Path(path).resolve(strict=False))
            for path in config.read_only_reference_paths
        ],
    )


def resolve_server_executable(
    server_command: list[str],
    *,
    repo_root: str | Path,
    local_env_dir: str | Path,
) -> tuple[str | None, list[str]]:
    """Resolve the server executable without allowing external venv reuse."""

    root = Path(repo_root).resolve()
    executable = server_command[0]
    errors: list[str] = []
    executable_path = Path(executable)
    if executable_path.is_absolute() or len(executable_path.parts) > 1:
        resolved = executable_path.resolve(strict=False)
        if not is_path_within_repo(resolved, root):
            return None, [
                "Server executable path is outside the repository; "
                f"external virtual environments are not allowed: {resolved}"
            ]
        if resolved.exists():
            return str(resolved), []
        errors.append(f"Server executable does not exist: {resolved}")

    path_match = shutil.which(executable)
    if path_match:
        return path_match, errors

    local_bin = _local_env_bin(Path(local_env_dir).resolve(strict=False))
    local_match = local_bin / executable
    if local_match.exists():
        return str(local_match), errors

    errors.append(
        "Server command was not found in PATH or the mag-coder local env: "
        f"{executable}"
    )
    return None, errors


def is_openai_endpoint_listening(
    api_base_url: str,
    *,
    timeout_seconds: float = 1.0,
) -> bool:
    """Check whether the configured endpoint host and port are accepting TCP."""

    parsed = urllib.parse.urlparse(api_base_url)
    if parsed.scheme not in {"http", "https"} or parsed.hostname is None:
        return False
    port = parsed.port or (443 if parsed.scheme == "https" else 80)
    try:
        with socket.create_connection(
            (parsed.hostname, port),
            timeout=timeout_seconds,
        ):
            return True
    except OSError:
        return False


def run_local_backend_pilot(
    config: LocalBackendConfig,
    *,
    repo_root: str | Path,
    config_path: str | Path,
    check_only: bool = False,
    assume_running: bool = False,
) -> LocalBackendPilotResult:
    """Run check-only, assume-running, or autostart local backend pilot flow."""

    root = Path(repo_root).resolve()
    preflight = preflight_local_backend(
        config,
        repo_root=root,
        check_endpoint=assume_running or not check_only,
    )
    if (
        preflight.errors
        and _can_create_env_for_missing_command(
            config=config,
            preflight=preflight,
            check_only=check_only,
            assume_running=assume_running,
        )
    ):
        maybe_create_local_env(config, repo_root=root)
        preflight = preflight_local_backend(
            config,
            repo_root=root,
            check_endpoint=assume_running or not check_only,
    )
    output_dir = Path(preflight.output_dir)
    if preflight.errors:
        output_dir = _safe_failure_output_dir(output_dir, root)
        output_dir.mkdir(parents=True, exist_ok=True)
        result = _result(
            config_path=config_path,
            output_dir=output_dir,
            mode=_mode(check_only, assume_running),
            preflight=preflight,
            error="; ".join(preflight.errors),
        )
        _write_local_backend_report(
            result,
            output_dir / "local_backend_pilot_report.md",
        )
        raise LocalBackendPilotError(result.error or "Local backend preflight failed.")

    output_dir.mkdir(parents=True, exist_ok=True)
    if check_only:
        result = _result(
            config_path=config_path,
            output_dir=output_dir,
            mode="check-only",
            preflight=preflight,
        )
        _write_local_backend_report(
            result,
            output_dir / "local_backend_pilot_report.md",
        )
        return result

    process: subprocess.Popen[bytes] | None = None
    server_started = False
    env = _process_env(config, preflight)
    try:
        endpoint_available = bool(preflight.endpoint_available)
        if assume_running and not endpoint_available:
            raise LocalBackendPilotError(
                f"Endpoint is not reachable: {config.api_base_url}"
            )
        if not endpoint_available:
            if not config.autostart:
                raise LocalBackendPilotError(
                    f"Endpoint is not reachable and autostart is false: "
                    f"{config.api_base_url}"
                )
            process = _start_server(config, preflight, output_dir, root, env)
            server_started = True
            endpoint_available = _wait_for_endpoint(
                config.api_base_url,
                timeout_seconds=config.server_start_timeout_seconds,
            )
            if not endpoint_available:
                raise LocalBackendPilotError(
                    f"Server did not become reachable within "
                    f"{config.server_start_timeout_seconds} seconds."
                )

        with _temporary_env(env):
            bundle = run_pilot_from_config(_to_pilot_run_config(config))
        result = _result(
            config_path=config_path,
            output_dir=output_dir,
            mode=_mode(check_only, assume_running),
            preflight=preflight,
            pilot_completed=True,
            server_started=server_started,
            bundle=bundle,
            audit_path=bundle.pilot_audit_path,
        )
        _write_local_backend_report(
            result,
            output_dir / "local_backend_pilot_report.md",
        )
        return result
    except Exception as error:
        result = _result(
            config_path=config_path,
            output_dir=output_dir,
            mode=_mode(check_only, assume_running),
            preflight=preflight,
            server_started=server_started,
            error=str(error),
        )
        _write_local_backend_report(
            result,
            output_dir / "local_backend_pilot_report.md",
        )
        if isinstance(error, LocalBackendPilotError):
            raise
        raise LocalBackendPilotError(str(error)) from error
    finally:
        if process is not None and not config.keep_server_running:
            _stop_process(process)
            report_path = output_dir / "local_backend_pilot_report.md"
            if report_path.exists():
                _append_server_stop_note(report_path, server_stopped=True)


def maybe_create_local_env(
    config: LocalBackendConfig,
    *,
    repo_root: str | Path,
) -> Path:
    """Create a mag-coder-owned local environment and install server packages."""

    root = Path(repo_root).resolve()
    env_dir = ensure_output_path_within_repo(config.local_env_dir, root)
    env_dir.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [sys.executable, "-m", "venv", str(env_dir)],
        cwd=root,
        check=True,
    )
    pip = _local_env_bin(env_dir) / "pip"
    subprocess.run(
        [str(pip), "install", "mlx-openai-server"],
        cwd=root,
        check=True,
    )
    return env_dir


def _can_create_env_for_missing_command(
    *,
    config: LocalBackendConfig,
    preflight: LocalBackendPreflightResult,
    check_only: bool,
    assume_running: bool,
) -> bool:
    if check_only or assume_running:
        return False
    if not config.autostart or not config.allow_create_local_env:
        return False
    return all(
        "Server command was not found" in error
        or "Server executable does not exist" in error
        for error in preflight.errors
    )


def _start_server(
    config: LocalBackendConfig,
    preflight: LocalBackendPreflightResult,
    output_dir: Path,
    repo_root: Path,
    env: dict[str, str],
) -> subprocess.Popen[bytes]:
    executable = preflight.server_executable
    if executable is None:
        raise LocalBackendPilotError("Server executable is not available.")
    command = [executable, *config.server_command[1:]]
    log_path = output_dir / "server.log"
    log_file = log_path.open("ab")
    try:
        return subprocess.Popen(
            command,
            cwd=repo_root,
            env=env,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            shell=False,
        )
    except Exception:
        log_file.close()
        raise


def _wait_for_endpoint(api_base_url: str, *, timeout_seconds: int) -> bool:
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        if is_openai_endpoint_listening(api_base_url):
            return True
        time.sleep(1)
    return False


def _stop_process(process: subprocess.Popen[bytes]) -> None:
    if process.poll() is not None:
        return
    process.terminate()
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=10)


def _to_pilot_run_config(config: LocalBackendConfig) -> PilotRunConfig:
    return PilotRunConfig(
        benchmark_path=config.benchmark_path,
        provider=config.provider,
        model_name=config.model_name,
        api_base_url=config.api_base_url,
        api_key_env=config.api_key_env,
        timeout_seconds=config.timeout_seconds,
        temperature=config.temperature,
        top_p=config.top_p,
        max_tokens=config.max_tokens,
        stop=config.stop,
        prompt_profile_id=config.prompt_profile_id,
        prompt_profile_path=config.prompt_profile_path,
        require_non_empty_code=config.require_non_empty_code,
        num_samples=config.num_samples,
        k_values=config.k_values,
        success_criterion=config.success_criterion,
        output_dir=config.output_dir,
        notes=config.notes,
        dry_run=config.dry_run,
        local_dummy_api_key_allowed=config.local_dummy_api_key_allowed,
        input_modes=config.input_modes,
        task_ids=config.task_ids,
        ablation_ids=config.ablation_ids,
        metadata=config.metadata,
    )


def _process_env(
    config: LocalBackendConfig,
    preflight: LocalBackendPreflightResult,
) -> dict[str, str]:
    env = dict(os.environ)
    if preflight.dummy_api_key_used:
        env[config.api_key_env] = LOCAL_DUMMY_API_KEY_VALUE
    return env


@contextmanager
def _temporary_env(env: dict[str, str]) -> Iterator[None]:
    previous = dict(os.environ)
    os.environ.clear()
    os.environ.update(env)
    try:
        yield
    finally:
        os.environ.clear()
        os.environ.update(previous)


def _write_local_backend_report(
    result: LocalBackendPilotResult,
    path: Path,
) -> None:
    result.report_path = str(path)
    lines = [
        "# Local Backend Pilot Report",
        "",
        f"- Config: `{result.config_path}`",
        f"- Output directory: `{result.output_dir}`",
        f"- Mode: `{result.mode}`",
        f"- Pilot completed: {result.pilot_completed}",
        f"- Server started: {result.server_started}",
        f"- Server stopped: {result.server_stopped}",
        f"- Error: {result.error or ''}",
        "",
        "## Preflight",
        "",
        f"- OK: {result.preflight.ok}",
        f"- Server command found: {result.preflight.server_command_found}",
        f"- Server executable: `{result.preflight.server_executable or ''}`",
        f"- Endpoint available: {result.preflight.endpoint_available}",
        f"- API key env: `{result.preflight.api_key_env}`",
        f"- API key available: {result.preflight.api_key_available}",
        f"- Dummy key used: {result.preflight.dummy_api_key_used}",
        "",
        "## Errors",
        "",
    ]
    lines.extend(_list_or_none(result.preflight.errors))
    lines.extend(["", "## Warnings", ""])
    lines.extend(_list_or_none(result.preflight.warnings))
    lines.extend(["", "## Read-only References", ""])
    lines.extend(_list_or_none(result.preflight.read_only_reference_paths))
    if result.bundle is not None:
        lines.extend(
            [
                "",
                "## Pilot Bundle",
                "",
                f"- pass@k summary: `{result.bundle.passk_summary_csv_path}`",
                f"- sample results: `{result.bundle.sample_results_path}`",
                f"- audit: `{result.bundle.pilot_audit_path}`",
            ]
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _safe_failure_output_dir(output_dir: Path, repo_root: Path) -> Path:
    if is_path_within_repo(output_dir, repo_root):
        return output_dir
    return repo_root / "runs" / "local_backend_preflight_failed"


def _append_server_stop_note(path: Path, *, server_stopped: bool) -> None:
    with path.open("a", encoding="utf-8") as file:
        file.write(f"\n- Final server stopped: {server_stopped}\n")


def _result(
    *,
    config_path: str | Path,
    output_dir: Path,
    mode: str,
    preflight: LocalBackendPreflightResult,
    pilot_completed: bool = False,
    server_started: bool = False,
    server_stopped: bool = False,
    bundle: PilotRunBundle | None = None,
    audit_path: str | None = None,
    error: str | None = None,
) -> LocalBackendPilotResult:
    return LocalBackendPilotResult(
        config_path=str(config_path),
        output_dir=str(output_dir),
        mode=mode,
        preflight=preflight,
        pilot_completed=pilot_completed,
        server_started=server_started,
        server_stopped=server_stopped,
        bundle=bundle,
        audit_path=audit_path,
        error=error,
    )


def _mode(check_only: bool, assume_running: bool) -> str:
    if check_only:
        return "check-only"
    if assume_running:
        return "assume-running"
    return "autostart"


def _list_or_none(items: list[str]) -> list[str]:
    if not items:
        return ["No entries."]
    return [f"- {item}" for item in items]


def _local_env_bin(env_dir: Path) -> Path:
    return env_dir / ("Scripts" if os.name == "nt" else "bin")


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    return float(value)


def _optional_int(value: Any) -> int | None:
    if value is None:
        return None
    return int(value)


def _optional_str(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value)
    return text if text else None


def _optional_str_list(value: Any) -> list[str] | None:
    if value is None:
        return None
    if not isinstance(value, list):
        raise TypeError("Expected a list of strings.")
    return [str(item) for item in value]


def _str_list(value: Any, default: list[str]) -> list[str]:
    if value is None:
        return list(default)
    if not isinstance(value, list):
        raise TypeError("Expected a list of strings.")
    return [str(item) for item in value]
