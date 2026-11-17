"""The proposed portable identity is inert until the scientific rule is approved."""

import pytest

from biorag_repro.core.hashing import sha256_bytes
from scripts.run_generation import Refused, main, prospective_freeze_identity

HEAD = "f371d0da83a37843212074094b62be1522f0fc74"
LOCK = (
    b"numpy==2.4.6\n"
    b"packaging @ https://example.invalid/dist/packaging.whl#sha256=aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa\n"
    b"torch==2.15.0+cu132\n"
)


def freeze(path: str, lock: bytes = LOCK, head: str = HEAD) -> str:
    return lock.decode() + f"-e git+file://{path}/repo.bundle@{head}#egg=biorag_repro\n"


def test_different_install_paths_share_exact_dependency_identity():
    left = prospective_freeze_identity(freeze("/content/drive/A"), LOCK, HEAD)
    right = prospective_freeze_identity(freeze("/content/drive/B"), LOCK, HEAD)
    assert left["candidate_lock_sha256"] == right["candidate_lock_sha256"]
    assert left["candidate_lock_sha256"] == sha256_bytes(LOCK)
    assert left["raw_freeze_sha256"] != right["raw_freeze_sha256"]
    assert left["editable_install"] != right["editable_install"]
    assert left["portable_identity_sha256"] == right["portable_identity_sha256"]
    assert left["portable_identity"] == right["portable_identity"]
    assert "/content/" not in str(left["portable_identity"])
    assert left["activation_status"] == "pending_researcher_patch_review"


def test_locale_line_order_does_not_change_candidate_lock():
    raw = freeze("/content/drive/A").replace("numpy==2.4.6\n", "") + "numpy==2.4.6\n"
    result = prospective_freeze_identity(raw, LOCK, HEAD)
    assert result["candidate_lock_sha256"] == sha256_bytes(LOCK)
    assert (
        result["portable_identity_sha256"]
        == prospective_freeze_identity(freeze("/other/location"), LOCK, HEAD)[
            "portable_identity_sha256"
        ]
    )


def test_project_revision_remains_part_of_portable_identity():
    newer = "1" * 40
    left = prospective_freeze_identity(freeze("/drive"), LOCK, HEAD)
    right = prospective_freeze_identity(freeze("/drive", head=newer), LOCK, newer)
    assert left["portable_identity_sha256"] != right["portable_identity_sha256"]


def test_proposal_command_writes_only_a_candidate_record(tmp_path):
    freeze_file = tmp_path / "pip_freeze.txt"
    lock_file = tmp_path / "candidate_lock.txt"
    out = tmp_path / "proposal.json"
    freeze_file.write_text(freeze("/content/drive/A"))
    lock_file.write_bytes(LOCK)
    main(
        [
            "propose-environment-identity",
            "--freeze",
            str(freeze_file),
            "--candidate-lock",
            str(lock_file),
            "--git-head",
            HEAD,
            "--out",
            str(out),
        ]
    )
    assert out.is_file()
    assert (tmp_path / "batch.json").exists() is False
    with pytest.raises(FileExistsError):
        main(
            [
                "propose-environment-identity",
                "--freeze",
                str(freeze_file),
                "--candidate-lock",
                str(lock_file),
                "--git-head",
                HEAD,
                "--out",
                str(out),
            ]
        )


@pytest.mark.parametrize(
    "raw,lock,head",
    [
        (freeze("/drive", LOCK.replace(b"torch==2.15.0+cu132", b"torch==2.16.0")), LOCK, HEAD),
        (freeze("/drive", LOCK.replace(b"dist/packaging.whl", b"other/packaging.whl")), LOCK, HEAD),
        (freeze("/drive", head="0" * 40), LOCK, HEAD),
        (LOCK.decode(), LOCK, HEAD),
        (freeze("/drive") + "-e git+file:///other@" + HEAD + "#egg=biorag_repro\n", LOCK, HEAD),
    ],
)
def test_dependency_or_project_identity_drift_is_refused(raw, lock, head):
    with pytest.raises(Refused):
        prospective_freeze_identity(raw, lock, head)


def test_local_build_path_cannot_supply_a_portable_package_identity():
    local = b"packaging @ file:///host/build/work\n"
    moved = b"packaging @ file:///another/host/build/work\n"
    for lock in (local, moved):
        with pytest.raises(Refused, match="approved content identity"):
            prospective_freeze_identity(freeze("/drive", lock), lock, HEAD)
