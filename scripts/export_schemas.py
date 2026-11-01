"""Export persistent JSON schemas from their Pydantic models."""

import argparse
import json
from pathlib import Path

from biorag_repro.core.manifest import BuildManifest
from biorag_repro.corpus.manifest import ReleaseRecord, ValidationReport

SCHEMA_DIR = Path(__file__).resolve().parents[1] / "schemas"
SCHEMAS = {
    "build_manifest.schema.json": BuildManifest,
    "corpus_validation_report.schema.json": ValidationReport,
    "corpus_release_record.schema.json": ReleaseRecord,
}
SCHEMA_PATH = SCHEMA_DIR / "build_manifest.schema.json"


def schema_bytes(model: type = BuildManifest) -> bytes:
    return (
        json.dumps(model.model_json_schema(), ensure_ascii=False, sort_keys=True, indent=2) + "\n"
    ).encode("utf-8")


def check_schema(path: Path = SCHEMA_PATH) -> bool:
    return path.is_file() and path.read_bytes() == schema_bytes()


def main() -> int:
    parser = argparse.ArgumentParser(description="Export persistent JSON Schemas")
    parser.add_argument("--check", action="store_true", help="fail if the generated schema differs")
    args = parser.parse_args()
    if args.check:
        stale = [
            str(SCHEMA_DIR / name)
            for name, model in SCHEMAS.items()
            if not (SCHEMA_DIR / name).is_file()
            or (SCHEMA_DIR / name).read_bytes() != schema_bytes(model)
        ]
        if stale:
            parser.exit(1, f"schema differs or is missing: {', '.join(stale)}\n")
        return 0
    SCHEMA_DIR.mkdir(parents=True, exist_ok=True)
    for name, model in SCHEMAS.items():
        (SCHEMA_DIR / name).write_bytes(schema_bytes(model))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
