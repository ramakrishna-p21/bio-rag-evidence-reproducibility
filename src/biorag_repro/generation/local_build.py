"""Candidate content verification for the exact retained Conda packaging build.

This establishes current payload equality to a publisher artifact. It does not reconstruct
historical installed files, approve an environment, or activate portable identity.
"""

from pathlib import Path

from biorag_repro.core.hashing import canonical_json_bytes, sha256_bytes
from biorag_repro.readiness.files import loads_strict, read_regular

CANDIDATE_REFERENCE_SHA256 = "df66aa39eecf044e897a111758ebcc032b73e91b9b31361b43e483b96cf70323"
CANDIDATE_ARCHIVE_SHA256 = "c432626b16768b8dab228bfb706f7060c2d462a21c516d240f68f2f902b5a044"
CANDIDATE_LOCK_SHA256 = "eb02933403d354ed90c2d6a2e191a6212aa6e2182091103df329daef7b8103d7"


def verify_packaging_payload(site_packages: Path, reference: bytes) -> dict:
    """Verify all payload files without importing or modifying the dependency.

    Unexpected files, including unverified generated caches, are refused. A future
    installed inventory must resolve such differences explicitly before adoption.
    """
    if sha256_bytes(reference) != CANDIDATE_REFERENCE_SHA256:
        raise ValueError("packaging reference differs from the genuine candidate artifact")
    value = loads_strict(reference)
    if (
        value["schema_version"] != "local_build_content_v1"
        or (value["distribution"], value["version"], value["build"])
        != ("packaging", "26.3", "pyhc364b38_0")
        or value["archive_sha256"] != CANDIDATE_ARCHIVE_SHA256
    ):
        raise ValueError("unexpected packaging build identity")
    roots = ("packaging", "packaging-26.3.dist-info")
    if site_packages.is_symlink():
        raise ValueError("installed package root cannot be a symlink")
    expected = value["files"]
    actual = set()
    for name in roots:
        root = site_packages / name
        if root.is_symlink():
            raise ValueError("package payload cannot be a symlink")
        for path in root.rglob("*"):
            if path.is_symlink():
                raise ValueError("package payload cannot contain symlinks")
            if path.is_file():
                actual.add(path.relative_to(site_packages).as_posix())
    if actual != set(expected):
        raise ValueError("packaging payload is missing files or contains unverified extra files")
    for name, record in expected.items():
        data = read_regular(site_packages, name)
        if len(data) != record["bytes"] or sha256_bytes(data) != record["sha256"]:
            raise ValueError(f"packaging content differs: {name}")
    direct_url = loads_strict(
        read_regular(site_packages, "packaging-26.3.dist-info/direct_url.json")
    )
    if direct_url != {"dir_info": {}, "url": value["source_url"]}:
        raise ValueError("packaging installation provenance differs")
    identity = {
        "distribution": value["distribution"],
        "version": value["version"],
        "build": value["build"],
        "archive_sha256": value["archive_sha256"],
        "payload_sha256": sha256_bytes(canonical_json_bytes(expected)),
    }
    return {
        "status": "candidate_content_verified_not_approved",
        "identity": identity,
        "reference_sha256": sha256_bytes(reference),
        "verified_files": len(expected),
        "observed_site_packages": str(site_packages.resolve()),
        "source_url": direct_url["url"],
        "historical_installed_content_verified": False,
    }
