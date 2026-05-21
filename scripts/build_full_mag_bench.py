"""Build a seed Full MAG-Bench v1 package from local source-task JSONL files."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from magcoder.benchmark.full_builder import build_full_mag_bench
from magcoder.benchmark.full_builder.validators import (
    save_validation_report,
    validate_full_mag_bench,
)
from scripts.validate_full_mag_bench_abms_oracles import validate_abms_oracles
from scripts.validate_full_mag_bench_oracles import validate_oracles


def main() -> None:
    """CLI entry point."""

    parser = argparse.ArgumentParser(
        description="Build Full MAG-Bench v1 from local HumanEval/MBPP-style JSONL.",
    )
    parser.add_argument(
        "--source",
        action="append",
        type=Path,
        required=True,
        help="Source-task JSONL file. May be passed multiple times.",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--max-cases",
        default=None,
        help="Maximum cases to build, or 'all'.",
    )
    parser.add_argument(
        "--formats",
        default="structured,graphical",
        help="Comma-separated output modes: structured,graphical,svg,png.",
    )
    parser.add_argument(
        "--split",
        default="none",
        choices=["none", "auto", "train", "dev", "test"],
    )
    parser.add_argument("--case-id-prefix", default="")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--skip-unsupported", action="store_true")
    parser.add_argument("--fail-on-oracle-failure", action="store_true")
    parser.add_argument("--write-source-loader-report", action="store_true")
    parser.add_argument("--variants-per-task", type=int, default=1)
    parser.add_argument("--max-variants", type=int, default=None)
    parser.add_argument("--target-cases", type=int, default=None)
    parser.add_argument(
        "--variant-policy",
        default="single",
        choices=["single", "paper2500"],
    )
    parser.add_argument("--deduplicate-variants", action="store_true")
    parser.add_argument("--drop-illogical", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    package, report = build_full_mag_bench(
        source_paths=args.source,
        output_dir=args.output,
        max_cases=None
        if args.target_cases is not None
        else _parse_max_cases(args.max_cases),
        formats=args.formats.split(","),
        overwrite=args.overwrite,
        split=args.split,
        case_id_prefix=args.case_id_prefix,
        seed=args.seed,
        skip_unsupported=args.skip_unsupported,
        write_source_loader_report=args.write_source_loader_report,
        variants_per_task=args.variants_per_task,
        max_variants=args.max_variants,
        target_cases=args.target_cases,
        variant_policy=args.variant_policy,
        deduplicate_variants=args.deduplicate_variants,
        drop_illogical=args.drop_illogical,
    )
    validation = validate_full_mag_bench(args.output)
    save_validation_report(validation, args.output)
    if args.fail_on_oracle_failure:
        oracle = validate_oracles(args.output)
        _write_json(oracle, args.output / "oracle_validation_report.json")
        abms_oracle = validate_abms_oracles(args.output)
        _write_json(abms_oracle, args.output / "abms_oracle_validation_report.json")
        if not (oracle["passed"] and abms_oracle["passed"]):
            raise SystemExit(1)
    print(f"Built {package.name} at {args.output}")
    print(f"Cases: {report.case_count}")
    print(f"Build report: {args.output / 'build_report.json'}")
    print(f"Validation passed: {validation.passed}")
    print(f"Validation report: {args.output / 'validation_report.json'}")


def _parse_max_cases(value: str | None) -> int | None:
    if value is None or value.lower() == "all":
        return None
    return int(value)


def _write_json(data: dict[str, object], path: Path) -> None:
    with path.open("w", encoding="utf-8") as file:
        json.dump(data, file, indent=2, sort_keys=True)
        file.write("\n")


if __name__ == "__main__":
    main()
