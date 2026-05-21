"""Scan supplementary submission packages for local absolute path leakage."""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

TEXT_SUFFIXES = {
    ".csv",
    ".json",
    ".jsonl",
    ".md",
    ".sha256",
    ".tex",
    ".txt",
    ".yaml",
    ".yml",
}

PACKAGE_LEVEL_DOCS = {
    "README_SUBMISSION.md",
    "REPRODUCIBILITY.md",
    "RESULTS_INDEX.md",
    "CLAIM_BOUNDARY_SUMMARY.md",
    "PACKAGE_CONTENTS.md",
    "VALIDATION_COMMANDS.md",
    "scripts_index.md",
    "manifest.json",
}


@dataclass(slots=True)
class PathPrivacyFinding:
    """One path privacy finding."""

    path: str
    line: int
    pattern_id: str
    severity: str
    preview: str


@dataclass(slots=True)
class PathPrivacyScanResult:
    """Path privacy scan report."""

    root_path: str
    scanned_files: int
    finding_count: int
    package_level_failure_count: int
    warning_count: int
    findings: list[PathPrivacyFinding]

    def to_dict(self) -> dict[str, Any]:
        """Render as JSON-compatible containers."""

        return {
            "root_path": self.root_path,
            "scanned_files": self.scanned_files,
            "finding_count": self.finding_count,
            "package_level_failure_count": self.package_level_failure_count,
            "warning_count": self.warning_count,
            "findings": [asdict(finding) for finding in self.findings],
        }


def main() -> None:
    """CLI entry point."""

    parser = argparse.ArgumentParser(
        description="Scan a submission package for local absolute paths.",
    )
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = scan_submission_paths(args.root)
    save_path_privacy_scan_report(result, args.output)
    print(f"Path privacy scan JSON: {args.output}")
    print(f"Findings: {result.finding_count}")
    print(f"Package-level failures: {result.package_level_failure_count}")


def scan_submission_paths(root: str | Path) -> PathPrivacyScanResult:
    """Scan text files for local absolute path references."""

    root_path = Path(root)
    if not root_path.exists():
        raise FileNotFoundError(f"Submission package does not exist: {root_path}")
    findings: list[PathPrivacyFinding] = []
    scanned_files = 0
    patterns = _patterns(root_path)
    for path in sorted(root_path.rglob("*")):
        if not path.is_file() or path.suffix not in TEXT_SUFFIXES:
            continue
        scanned_files += 1
        findings.extend(_scan_file(root_path, path, patterns))
    package_failures = sum(
        1 for finding in findings if finding.severity == "failure"
    )
    warnings = sum(1 for finding in findings if finding.severity == "warning")
    return PathPrivacyScanResult(
        root_path=_portable_root(root_path),
        scanned_files=scanned_files,
        finding_count=len(findings),
        package_level_failure_count=package_failures,
        warning_count=warnings,
        findings=findings,
    )


def save_path_privacy_scan_report(
    result: PathPrivacyScanResult,
    output: str | Path,
) -> tuple[Path, Path]:
    """Save JSON and Markdown reports."""

    json_path = Path(output)
    md_path = json_path.with_suffix(".md")
    json_path.parent.mkdir(parents=True, exist_ok=True)
    with json_path.open("w", encoding="utf-8") as file:
        json.dump(result.to_dict(), file, indent=2, sort_keys=True)
        file.write("\n")
    _write_markdown(result, md_path)
    return json_path, md_path


def _patterns(root: Path) -> list[tuple[str, re.Pattern[str]]]:
    username = re.escape(Path.home().name)
    project_path = re.escape(str(Path.cwd()))
    root_abs = re.escape(str(root.resolve()))
    return [
        ("mac_users_path", re.compile(r"/Users/[^\\s\"'`<>]+")),
        ("linux_home_path", re.compile(r"/home/[^\\s\"'`<>]+")),
        ("windows_users_path", re.compile(r"C:\\\\Users\\\\[^\\s\"'`<>]+")),
        ("local_project_path", re.compile(project_path)),
        ("submission_absolute_path", re.compile(root_abs)),
        ("local_username", re.compile(rf"\b{username}\b")),
        ("virtualenv_path", re.compile(r"[/\\](?:\\.venv|venv)[/\\]")),
        ("temp_path", re.compile(r"(?:/private)?/tmp/[^\\s\"'`<>]+")),
        ("var_folders_path", re.compile(r"/private/var/folders/[^\\s\"'`<>]+")),
    ]


def _scan_file(
    root: Path,
    path: Path,
    patterns: list[tuple[str, re.Pattern[str]]],
) -> list[PathPrivacyFinding]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except UnicodeDecodeError:
        return []
    relative = path.relative_to(root)
    severity = _severity_for_path(relative)
    findings: list[PathPrivacyFinding] = []
    for line_number, line in enumerate(lines, start=1):
        for pattern_id, pattern in patterns:
            if pattern.search(line):
                findings.append(
                    PathPrivacyFinding(
                        path=str(relative),
                        line=line_number,
                        pattern_id=pattern_id,
                        severity=severity,
                        preview=_preview(line),
                    )
                )
    return findings


def _severity_for_path(relative: Path) -> str:
    if relative.name in {
        "path_privacy_scan_report.json",
        "path_privacy_scan_report.md",
        "secret_scan_report.json",
        "secret_scan_report.md",
    }:
        return "warning"
    if str(relative) in PACKAGE_LEVEL_DOCS:
        return "failure"
    if relative.parts and relative.parts[0] in {"frozen_results", "paper_tables"}:
        return "warning"
    return "failure"


def _preview(line: str) -> str:
    home = str(Path.home())
    preview = line.strip().replace(home, "<HOME>")
    return preview[:180]


def _portable_root(root: Path) -> str:
    if root.is_absolute():
        return f"<submission_package>/{root.name}"
    return str(root)


def _write_markdown(result: PathPrivacyScanResult, path: Path) -> None:
    lines = [
        "# Path Privacy Scan Report",
        "",
        f"- Root path: `{result.root_path}`",
        f"- Scanned files: {result.scanned_files}",
        f"- Findings: {result.finding_count}",
        f"- Package-level failures: {result.package_level_failure_count}",
        f"- Warnings: {result.warning_count}",
        "",
    ]
    if not result.findings:
        lines.append("No local absolute path findings were detected.")
    else:
        lines.extend(
            [
                "| severity | path | line | pattern_id | preview |",
                "| --- | --- | ---: | --- | --- |",
            ]
        )
        for finding in result.findings[:200]:
            lines.append(
                f"| {finding.severity} | {finding.path} | {finding.line} | "
                f"{finding.pattern_id} | {finding.preview} |"
            )
        if len(result.findings) > 200:
            lines.append(
                f"\nOnly the first 200 of {len(result.findings)} findings are shown."
            )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
