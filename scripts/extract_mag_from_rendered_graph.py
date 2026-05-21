"""Recover a MAGModel from a rendered graph smoke artifact."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from magcoder.graphics import (
    RenderedGraphArtifact,
    VLMExtractionConfig,
    extractor_from_name,
)
from magcoder.models import save_mag_to_json


def main() -> None:
    """CLI entry point."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--extractor", default="mock")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = extract_mag_from_rendered_graph(
        image_path=args.input,
        manifest_path=args.manifest,
        extractor_name=args.extractor,
        output_path=args.output,
    )
    print(f"Recovered MAG: {args.output}")
    print(f"Extractor: {result['extractor']}")


def extract_mag_from_rendered_graph(
    *,
    image_path: str | Path,
    manifest_path: str | Path,
    extractor_name: str,
    output_path: str | Path,
) -> dict[str, object]:
    """Recover MAG from a rendered graph and write JSON."""

    artifact = RenderedGraphArtifact(
        image_path=str(image_path),
        manifest_path=str(manifest_path),
    )
    extractor = extractor_from_name(extractor_name)
    result = extractor.extract(
        artifact,
        VLMExtractionConfig(extractor=extractor_name),
    )
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    save_mag_to_json(result.mag, output)
    metadata_path = output.with_suffix(".extraction.json")
    metadata_path.write_text(
        json.dumps(
            {
                "extractor": result.extractor,
                "used_manifest": result.used_manifest,
                "warnings": result.warnings,
                "metadata": result.metadata,
                "output": str(output),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return {
        "extractor": result.extractor,
        "used_manifest": result.used_manifest,
        "output": str(output),
    }


if __name__ == "__main__":
    main()
