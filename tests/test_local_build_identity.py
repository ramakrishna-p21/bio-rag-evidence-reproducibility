"""Small synthetic payloads test refusal rules; genuine build evidence is audited separately."""

import shutil
from types import SimpleNamespace

import pytest

from biorag_repro.core.hashing import canonical_json_bytes, sha256_bytes
from biorag_repro.generation import local_build
from biorag_repro.generation.environment import GitState
from scripts.run_generation import Refused, prospective_freeze_identity

SOURCE_URL = (
    "file:///home/conda/feedstock_root/build_artifacts/bld/rattler-build_packaging_1785888127/work"
)
HEAD = "f" * 40
LOCK = f"packaging @ {SOURCE_URL}\nnumpy==2.4.6\n".encode()


@pytest.fixture
def fixture_build(tmp_path, monkeypatch):
    payload = {
        "packaging/__init__.py": b'__version__ = "26.3"\n',
        "packaging-26.3.dist-info/direct_url.json": canonical_json_bytes(
            {"dir_info": {}, "url": SOURCE_URL}
        ),
    }
    root = tmp_path / "first"
    for name, data in payload.items():
        target = root / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
    reference = canonical_json_bytes(
        {
            "schema_version": "local_build_content_v1",
            "distribution": "packaging",
            "version": "26.3",
            "build": "pyhc364b38_0",
            "archive_sha256": local_build.CANDIDATE_ARCHIVE_SHA256,
            "source_url": SOURCE_URL,
            "files": {n: {"sha256": sha256_bytes(d), "bytes": len(d)} for n, d in payload.items()},
        }
    )
    # These pins belong only to this synthetic unit fixture, never to the real environment.
    monkeypatch.setattr(local_build, "CANDIDATE_REFERENCE_SHA256", sha256_bytes(reference))
    monkeypatch.setattr(local_build, "CANDIDATE_LOCK_SHA256", sha256_bytes(LOCK))
    from biorag_repro.generation import environment

    monkeypatch.setattr(
        environment,
        "system_probes",
        lambda repo: SimpleNamespace(
            git=lambda: GitState(status="ok", commit=HEAD, dirty=False, dirty_paths=0)
        ),
    )
    return root, reference


def freeze(path):
    return LOCK.decode() + f"-e git+file://{path}@{HEAD}#egg=biorag_repro\n"


def test_content_identity_is_independent_of_install_prefix_but_not_content(fixture_build, tmp_path):
    root, reference = fixture_build
    moved = tmp_path / "second"
    shutil.copytree(root, moved)
    left = prospective_freeze_identity(
        freeze("/first/project"),
        LOCK,
        HEAD,
        packaging_root=root,
        packaging_reference=reference,
        project_root=tmp_path,
    )
    right = prospective_freeze_identity(
        freeze("/second/project"),
        LOCK,
        HEAD,
        packaging_root=moved,
        packaging_reference=reference,
        project_root=tmp_path,
    )
    assert left["portable_identity_sha256"] == right["portable_identity_sha256"]
    assert (
        left["local_build_verification"]["observed_site_packages"]
        != right["local_build_verification"]["observed_site_packages"]
    )
    assert left["approved_for_execution"] is False
    assert left["local_build_verification"]["historical_installed_content_verified"] is False
    (moved / "packaging/__init__.py").write_bytes(b'__version__ = "26.4"\n')
    with pytest.raises(Refused, match="content differs"):
        prospective_freeze_identity(
            freeze("/second/project"),
            LOCK,
            HEAD,
            packaging_root=moved,
            packaging_reference=reference,
            project_root=tmp_path,
        )


def test_unverified_extra_files_and_revision_drift_fail(fixture_build, tmp_path, monkeypatch):
    root, reference = fixture_build
    (root / "packaging/extra.py").write_text("pass\n")
    with pytest.raises(ValueError, match="extra files"):
        local_build.verify_packaging_payload(root, reference)
    from biorag_repro.generation import environment

    monkeypatch.setattr(
        environment,
        "system_probes",
        lambda repo: SimpleNamespace(
            git=lambda: GitState(status="ok", commit="a" * 40, dirty=False, dirty_paths=0)
        ),
    )
    with pytest.raises(Refused, match="exact clean reviewed revision"):
        prospective_freeze_identity(
            freeze("/project"),
            LOCK,
            HEAD,
            packaging_root=root,
            packaging_reference=reference,
            project_root=tmp_path,
        )


def test_reference_and_lock_cannot_self_authorize(fixture_build, tmp_path):
    root, reference = fixture_build
    with pytest.raises(ValueError, match="genuine candidate artifact"):
        local_build.verify_packaging_payload(root, reference + b" ")
    changed_lock = LOCK.replace(b"numpy==2.4.6", b"numpy==2.4.5")
    with pytest.raises(Refused, match="exact retained"):
        prospective_freeze_identity(
            changed_lock.decode() + f"-e git+file:///project@{HEAD}#egg=biorag_repro\n",
            changed_lock,
            HEAD,
            packaging_root=root,
            packaging_reference=reference,
            project_root=tmp_path,
        )
