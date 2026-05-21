"""Validate a generated Full MAG-Bench v1 package."""

from __future__ import annotations

import argparse
from pathlib import Path

from magcoder.benchmark.full_builder.validators import (
    save_validation_report,
    validate_full_mag_bench,
)


def main() -> None:
    """CLI entry point."""

    parser = argparse.ArgumentParser(description="Validate Full MAG-Bench v1.")
    parser.add_argument("--bench", type=Path, required=True)
    args = parser.parse_args()
    report = validate_full_mag_bench(args.bench)
    json_path, md_path = save_validation_report(report, args.bench)
    print(f"Validation passed: {report.passed}")
    print(f"Validation JSON: {json_path}")
    print(f"Validation Markdown: {md_path}")
    if not report.passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
