"""Build and validate one frozen Phase 4 paper and representation."""

import argparse
import json
import subprocess
from pathlib import Path

from biorag_repro.core.config import resolve_config
from biorag_repro.core.manifest import ArtifactRef
from biorag_repro.documents.models import JatsConfiguration
from biorag_repro.documents.pdf import PdfConfiguration
from biorag_repro.documents.phase4 import (
    Arm,
    Phase4GateIndex,
    build_arm,
    load_frozen_paper,
    publish_gate_index,
    publish_validation,
    validate_build,
)


def _committed_head() -> str:
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    for command in (["git", "diff", "--quiet"], ["git", "diff", "--cached", "--quiet"]):
        if subprocess.run(command, check=False).returncode:
            raise ValueError("gate evidence requires a clean tracked implementation")
    return head


def main() -> int:
    parser = argparse.ArgumentParser(description="Build one verified frozen Phase 4 source pair")
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
    args = parser.parse_args()
    release, paper = load_frozen_paper(
        args.corpus_root,
        args.release_record,
        args.release_sha256,
        args.corpus_version,
        args.paper_id,
    )
    if args.arm == "pair":
        implementation_sha = _committed_head()
    else:
        implementation_sha = None
    arms: list[Arm] = ["JATS", "PDF_DOCLING"] if args.arm == "pair" else [args.arm]
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
    if implementation_sha is not None:
        index = Phase4GateIndex(
            corpus_version=release.corpus_version,
            paper_id=paper.paper_id,
            corpus_release_sha256=args.release_sha256,
            implementation_git_sha=implementation_sha,
            jats_validation=evidence["JATS"],
            pdf_validation=evidence["PDF_DOCLING"],
        )
        evidence["gate_index"] = publish_gate_index(index, args.data_root)
    print(
        json.dumps(
            {key: value.model_dump(mode="json") for key, value in evidence.items()}, sort_keys=True
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
