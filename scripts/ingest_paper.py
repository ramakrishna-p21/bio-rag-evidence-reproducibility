"""Build and validate one frozen Phase 4 paper and representation."""

import argparse
import json
from pathlib import Path

from biorag_repro.core.config import resolve_config
from biorag_repro.core.manifest import ArtifactRef
from biorag_repro.documents.ingestion import (
    Arm,
    build_arm,
    load_frozen_paper,
    publish_validation,
    validate_build,
)
from biorag_repro.documents.models import JatsConfiguration
from biorag_repro.documents.pdf import PdfConfiguration


def main() -> int:
    parser = argparse.ArgumentParser(description="Build one verified frozen Phase 4 source pair")
    parser.add_argument("--publication", choices=["candidate", "complete"], default="candidate")
    preliminary, _ = parser.parse_known_args()
    if preliminary.publication == "complete":
        parser.error("independent human authorization is unavailable in production")
    parser.add_argument("--corpus-root", type=Path, required=True)
    parser.add_argument("--release-record", type=Path, required=True)
    parser.add_argument("--release-sha256", required=True)
    parser.add_argument("--corpus-version", required=True)
    parser.add_argument("--paper-id", required=True)
    parser.add_argument("--arm", choices=["JATS", "PDF_DOCLING", "pair"], required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--jats-config", type=Path, default=Path("configs/jats.toml"))
    parser.add_argument("--pdf-config", type=Path, default=Path("configs/pdf_docling.toml"))
    parser.add_argument("--assets-root", type=Path)
    parser.add_argument("--lock-path", type=Path)
    parser.add_argument("--reuse-attempt-id")
    args = parser.parse_args()
    if args.arm == "pair":
        parser.error(
            "paired gate publication is unavailable; use scripts/reproduce_pair.py for a candidate"
        )
    from biorag_repro.documents.authorization import PHASE2H_SHA256

    if args.release_sha256 != PHASE2H_SHA256 or args.corpus_version != "corpus_v1":
        parser.error("authorized Phase 2H release and corpus_v1 are required")
    release, paper = load_frozen_paper(
        args.corpus_root,
        args.release_record,
        args.release_sha256,
        args.corpus_version,
        args.paper_id,
    )
    arms: list[Arm] = [args.arm]
    evidence: dict[str, ArtifactRef] = {}
    for arm in arms:
        if arm == "JATS":
            config_hash = resolve_config(
                args.jats_config.read_bytes(), JatsConfiguration
            ).configuration_sha256
        else:
            config_hash = resolve_config(
                args.pdf_config.read_bytes(), PdfConfiguration
            ).configuration_sha256
        build = build_arm(
            paper,
            args.paper_id,
            arm,
            args.corpus_root,
            args.data_root,
            config_hash,
            assets_root=args.assets_root,
            lock_path=args.lock_path,
            reuse_attempt_id=args.reuse_attempt_id if arm == "PDF_DOCLING" else None,
        )
        if build.build_status != "complete":
            raise ValueError(f"{arm} build did not complete: {build.status}")
        report = validate_build(
            build,
            args.data_root,
            corpus_root=args.corpus_root,
            corpus_version=release.corpus_version,
        )
        evidence[arm] = publish_validation(report, args.data_root)
    print(
        json.dumps(
            {key: value.model_dump(mode="json") for key, value in evidence.items()}, sort_keys=True
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
