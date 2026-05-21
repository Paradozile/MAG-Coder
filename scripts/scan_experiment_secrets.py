"""Scan experiment artifacts for accidental secret leakage."""

from __future__ import annotations

import argparse
import json
import os
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass(slots=True)
class SecretFinding:
    """One possible secret finding."""

    path: str
    line: int
    pattern_id: str
    preview: str


@dataclass(slots=True)
class SecretScanResult:
    """Secret scan report."""

    root_path: str
    scanned_files: int
    finding_count: int
    findings: list[SecretFinding] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Render this report as JSON-compatible containers."""

        return {
            "root_path": self.root_path,
            "scanned_files": self.scanned_files,
            "finding_count": self.finding_count,
            "findings": [asdict(finding) for finding in self.findings],
        }


SECRET_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("openai_key", re.compile(r"\bsk-[A-Za-z0-9_\-]{16,}\b")),
    ("bearer_token", re.compile(r"\bBearer\s+[A-Za-z0-9._\-]{16,}", re.I)),
    (
        "forbidden_secret_env_name",
        re.compile(r"\b(?:OPENAI_API_KEY|GEMINI_API_KEY|ANTHROPIC_API_KEY)\b"),
    ),
    (
        "api_key_assignment",
        re.compile(r"\b(?!LOCAL_LLM_API_KEY\b)[A-Z0-9_]*API_KEY\s*=\s*[^,\s]+"),
    ),
    (
        "json_api_key_value",
        re.compile(r'"api_key"\s*:\s*"[^"]{8,}"', re.I),
    ),
]

TEXT_SUFFIXES = {
    ".csv",
    ".json",
    ".jsonl",
    ".md",
    ".tex",
    ".txt",
    ".yaml",
    ".yml",
}


def scan_experiment_secrets(root: str | Path) -> SecretScanResult:
    """Scan a directory tree for common secret patterns."""

    root_path = Path(root)
    if not root_path.exists():
        raise FileNotFoundError(f"Scan path does not exist: {root_path}")
    findings: list[SecretFinding] = []
    scanned_files = 0
    paths = [root_path] if root_path.is_file() else list(root_path.rglob("*"))
    for path in paths:
        if not path.is_file() or path.suffix not in TEXT_SUFFIXES:
            continue
        scanned_files += 1
        findings.extend(_scan_file(path))
    return SecretScanResult(
        root_path=str(root_path),
        scanned_files=scanned_files,
        finding_count=len(findings),
        findings=findings,
    )


def save_secret_scan_report(
    result: SecretScanResult,
    output: str | Path,
) -> tuple[Path, Path]:
    """Save JSON and Markdown secret scan reports."""

    json_path = Path(output)
    md_path = json_path.with_suffix(".md")
    json_path.parent.mkdir(parents=True, exist_ok=True)
    with json_path.open("w", encoding="utf-8") as file:
        json.dump(result.to_dict(), file, indent=2, sort_keys=True)
        file.write("\n")
    _write_markdown_report(result, md_path)
    return json_path, md_path


def main() -> None:
    """CLI entry point."""

    parser = argparse.ArgumentParser(
        description="Scan a frozen experiment directory for possible secrets.",
    )
    parser.add_argument("--path", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = scan_experiment_secrets(args.path)
    json_path, md_path = save_secret_scan_report(result, args.output)
    print(f"Secret scan JSON: {json_path}")
    print(f"Secret scan Markdown: {md_path}")
    print(f"Findings: {result.finding_count}")


def _scan_file(path: Path) -> list[SecretFinding]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except UnicodeDecodeError:
        return []
    findings: list[SecretFinding] = []
    for line_number, line in enumerate(lines, start=1):
        if _line_is_allowed_reference(line):
            continue
        for pattern_id, pattern in SECRET_PATTERNS:
            if pattern.search(line):
                findings.append(
                    SecretFinding(
                        path=str(path),
                        line=line_number,
                        pattern_id=pattern_id,
                        preview=_redact_preview(line),
                    )
                )
    return findings


def _line_is_allowed_reference(line: str) -> bool:
    stripped = line.strip()
    if "api_key_env" in stripped:
        return True
    return stripped in {"LOCAL_LLM_API_KEY", "api_key_env: LOCAL_LLM_API_KEY"}


def _redact_preview(line: str) -> str:
    preview = line.strip()
    local_key = os.environ.get("LOCAL_LLM_API_KEY")
    if local_key:
        preview = preview.replace(local_key, "<redacted>")
    preview = re.sub(r"sk-[A-Za-z0-9_\-]{8,}", "sk-<redacted>", preview)
    preview = re.sub(r"Bearer\s+[A-Za-z0-9._\-]{8,}", "Bearer <redacted>", preview)
    return preview[:160]


def _write_markdown_report(result: SecretScanResult, path: Path) -> None:
    lines = [
        "# Secret Scan Report",
        "",
        f"- Root path: `{result.root_path}`",
        f"- Scanned files: {result.scanned_files}",
        f"- Finding count: {result.finding_count}",
        "",
    ]
    if not result.findings:
        lines.append("No possible secrets were found.")
    else:
        lines.extend(
            [
                "| path | line | pattern_id | preview |",
                "| --- | --- | --- | --- |",
            ]
        )
        for finding in result.findings:
            lines.append(
                f"| {finding.path} | {finding.line} | "
                f"{finding.pattern_id} | {finding.preview} |"
            )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
