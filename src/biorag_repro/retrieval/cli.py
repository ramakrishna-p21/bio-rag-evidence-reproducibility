"""Assemble or scientifically validate explicitly selected evidence contexts."""

import argparse
from pathlib import Path

from pydantic import TypeAdapter

from biorag_repro.core.manifest import ArtifactRef
from biorag_repro.retrieval import context
from biorag_repro.retrieval.models import ConfigurationRef


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("operation", choices=["assemble", "validate"])
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--configuration", type=Path, required=True)
    parser.add_argument("--configuration-root", type=Path, required=True)
    parser.add_argument("--qwen-tokenizer-root", type=Path, required=True)
    parser.add_argument("--bge-root", type=Path)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--record", type=Path)
    parser.add_argument("--manifest", type=Path)
    args = parser.parse_args()
    selected = TypeAdapter(context.Selection).validate_json(args.selection.read_bytes())
    if selected.encoder is not None or any(s.fixture_counter is not None for s in selected.sources):
        parser.error("command inputs cannot supply encoders or fixture counters")
    if selected.method == "bge":
        if args.bge_root is None:
            parser.error("BGE scientific verification requires --bge-root")
        from dataclasses import replace

        from biorag_repro.retrieval.dense import BgeEncoder

        selected = replace(selected, encoder=BgeEncoder(args.bge_root))
    cfg = ConfigurationRef.model_validate_json(args.configuration.read_bytes())
    if args.operation == "assemble":
        value = context.assemble(selected, cfg, args.configuration_root, args.qwen_tokenizer_root)
        refs = context.publish(
            value,
            selected,
            cfg,
            args.configuration_root,
            args.qwen_tokenizer_root,
            args.output_root,
        )
        for ref in refs:
            print(ref.model_dump_json())
    else:
        if args.record is None or args.manifest is None:
            parser.error("validation requires selected record and manifest reference JSON files")
        value = context.read(
            ArtifactRef.model_validate_json(args.record.read_bytes()),
            ArtifactRef.model_validate_json(args.manifest.read_bytes()),
            selected,
            cfg,
            args.configuration_root,
            args.qwen_tokenizer_root,
            args.output_root,
        )
        print(value.model_dump_json())


if __name__ == "__main__":
    main()
