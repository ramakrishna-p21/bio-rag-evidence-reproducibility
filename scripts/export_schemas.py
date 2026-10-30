"""Export the persistent build-manifest schema from its Pydantic model."""

import argparse
import json
from pathlib import Path

from biorag_repro.core.manifest import BuildManifest

SCHEMA_PATH = Path(__file__).resolve().parents[1] / "schemas" / "build_manifest.schema.json"


def schema_bytes() -> bytes:
    return (
        json.dumps(BuildManifest.model_json_schema(), ensure_ascii=False, sort_keys=True, indent=2)
        + "\n"
    ).encode("utf-8")


def check_schema(path: Path = SCHEMA_PATH) -> bool:
    return path.is_file() and path.read_bytes() == schema_bytes()


def main() -> int:
    parser = argparse.ArgumentParser(description="Export the BuildManifest JSON Schema")
    parser.add_argument("--check", action="store_true", help="fail if the generated schema differs")
    args = parser.parse_args()
    if args.check:
        if not check_schema():
            parser.exit(1, f"schema differs or is missing: {SCHEMA_PATH}\n")
        return 0
    SCHEMA_PATH.parent.mkdir(parents=True, exist_ok=True)
    SCHEMA_PATH.write_bytes(schema_bytes())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
