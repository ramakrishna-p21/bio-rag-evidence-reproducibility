"""Approval values and scientific source are separate identities (P9-D1).

The G09 digest covers executable source and configuration. The approval values live in one
tracked data file that is excluded from that digest by exact path, so a slot can hold the G09
digest without the digest depending on the slot. Everything here uses temporary repositories
and digests worked out independently with `git hash-object`.
"""

import hashlib
import inspect
import json
import os

import pytest
from readiness_world import REAL_REPO, build_world, git, write

from biorag_repro.core.hashing import canonical_json_bytes, sha256_bytes
from biorag_repro.readiness import cli
from biorag_repro.readiness.approvals import PolicyError, load_policy
from biorag_repro.readiness.evaluate import gather_all
from biorag_repro.readiness.gather import to_result
from biorag_repro.readiness.git_state import APPROVAL_FILE, probe
from biorag_repro.readiness.guard import AuthorizationRefused, authorize_final_test
from biorag_repro.readiness.policy import SLOT_NAMES, policy_sha256

G09 = "G09_scientific_source_revision"
# SHA-256 of the canonical JSON of version "1" with all 12 slots unset, worked out before the
# constant `REAL_POLICY` was removed (independent of the package: sorted keys, no whitespace).
ORIGINAL_POLICY_SHA256 = "aa4bbecf902a436deb6d725f302c01bccbd9ea93ac11912ae34a6e4c6e394026"


@pytest.fixture
def world(tmp_path, monkeypatch):
    return build_world(tmp_path, monkeypatch)


def values(world, **slots):
    data = json.loads((world.repo / APPROVAL_FILE).read_text())
    data["slots"].update(slots)
    return data


def write_values(repo, data):
    write(repo / APPROVAL_FILE, json.dumps(data, indent=2) + "\n")


def commit(repo, message="change"):
    git(repo, "add", "-A")
    git(repo, "commit", "-qm", message)


def g09(world):
    return gather_all(world.ctx)[G09]


def digest_of(gathered):
    return sha256_bytes(canonical_json_bytes(gathered.subject))


def expected_subject_digest(repo, excluded=APPROVAL_FILE):
    """Rebuild the G09 digest from `git hash-object`, without the package's Git code."""
    scope = ["src", "configs", "schemas", "scripts", "analysis", "pyproject.toml", "approvals"]
    listed = git(repo, "ls-files", "--", *scope).splitlines()
    entries = []
    for path in sorted(p for p in listed if p != excluded):
        mode = "100755" if os.access(repo / path, os.X_OK) else "100644"
        entries.append([mode, git(repo, "hash-object", path), path])
    specs = {
        name: hashlib.sha256((repo / "docs/frozen" / name).read_bytes()).hexdigest()
        for name in sorted(p.name for p in (repo / "docs/frozen").iterdir())
    }
    subject = {
        "kind": "scientific_source_v1",
        "tracked_scope": entries,
        "frozen_specifications": specs,
    }
    return hashlib.sha256(
        json.dumps(subject, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
    ).hexdigest()


# ---- the policy hash and the real values file --------------------------------------------


def test_migrated_policy_has_the_original_hash():
    assert policy_sha256(load_policy(REAL_REPO)) == ORIGINAL_POLICY_SHA256


def test_all_twelve_real_slots_are_unset():
    policy = load_policy(REAL_REPO)
    assert policy.policy_version == "1"
    assert set(policy.slots.model_dump()) == set(SLOT_NAMES) and len(SLOT_NAMES) == 12
    assert all(value is None for value in policy.slots.model_dump().values())


def test_real_values_file_is_tracked_clean_and_data_only():
    path = REAL_REPO / APPROVAL_FILE
    assert git(REAL_REPO, "ls-files", "--error-unmatch", APPROVAL_FILE) == APPROVAL_FILE
    assert git(REAL_REPO, "status", "--porcelain", "--", APPROVAL_FILE) == ""
    assert set(json.loads(path.read_text())) == {"policy_version", "slots"}


# ---- G09 does not depend on the values --------------------------------------------------


def test_approval_values_change_the_policy_hash_but_not_the_g09_subject(world):
    before = g09(world)
    policy_before = policy_sha256(load_policy(world.repo))
    write_values(world.repo, values(world, corpus_identity="a" * 64, gpu_authorization="b" * 64))
    commit(world.repo)
    after = g09(world)
    assert after.state == "ok"
    assert digest_of(after) == digest_of(before)
    assert policy_sha256(load_policy(world.repo)) != policy_before


def test_g09_digest_matches_an_independent_oracle(world):
    assert digest_of(g09(world)) == expected_subject_digest(world.repo)


def test_slot_holding_the_g09_digest_passes_across_a_values_only_commit(world):
    digest = digest_of(g09(world))
    write_values(world.repo, values(world, scientific_source_identity=digest))
    commit(world.repo, "approve source")
    again = g09(world)
    policy = load_policy(world.repo)
    result = to_result(G09, again, policy)
    assert digest_of(again) == digest == expected_subject_digest(world.repo)
    assert (result.status, result.expected_sha256) == ("passed", digest)
    # a further values-only commit still leaves the approved digest valid
    write_values(
        world.repo, values(world, scientific_source_identity=digest, gpu_authorization="c" * 64)
    )
    commit(world.repo, "another slot")
    assert to_result(G09, g09(world), load_policy(world.repo)).status == "passed"


def test_stale_approval_fails_after_scientific_code_changes(world):
    digest = digest_of(g09(world))
    write_values(world.repo, values(world, scientific_source_identity=digest))
    commit(world.repo)
    write(world.repo / "src/code.py", "x = 2\n")
    commit(world.repo, "code")
    result = to_result(G09, g09(world), load_policy(world.repo))
    assert (result.status, result.reason_code) == ("failed", "policy_mismatch")


def test_scientific_code_change_changes_the_subject(world):
    before = digest_of(g09(world))
    write(world.repo / "scripts/s.py", "y = 2\n")
    commit(world.repo)
    assert digest_of(g09(world)) not in (before,)


def test_policy_processing_code_is_inside_the_digest(world):
    path = world.repo / "src/biorag_repro/readiness/policy.py"
    write(path, "GATE_SLOT = {}\n")
    commit(world.repo, "policy code")
    before = digest_of(g09(world))
    write(path, "GATE_SLOT = {'x': 'y'}\n")
    commit(world.repo, "policy code changed")
    assert digest_of(g09(world)) != before
    assert any(p == "src/biorag_repro/readiness/policy.py" for _, _, p in probe(world.repo).entries)


def test_frozen_specification_change_changes_the_subject(world):
    before = digest_of(g09(world))
    write(world.repo / "docs/frozen/EXPERIMENT_PLAN.md", "edited\n")
    assert digest_of(g09(world)) != before


# ---- cleanliness of the values file ------------------------------------------------------


def test_uncommitted_value_edit_fails_cleanliness(world):
    write_values(world.repo, values(world, corpus_identity="a" * 64))
    result = g09(world)
    assert (result.state, result.reason) == ("failed", "scientific_source_not_clean")
    assert APPROVAL_FILE in probe(world.repo).modified


def test_staged_value_edit_fails_cleanliness(world):
    write_values(world.repo, values(world, corpus_identity="a" * 64))
    git(world.repo, "add", APPROVAL_FILE)
    assert g09(world).state == "failed"


def test_deleted_tracked_values_file_fails(world):
    (world.repo / APPROVAL_FILE).unlink()
    assert g09(world).state == "failed"
    assert APPROVAL_FILE in probe(world.repo).modified


def test_untracked_replacement_of_the_values_file_fails(world):
    git(world.repo, "rm", "-q", "--cached", APPROVAL_FILE)
    result = g09(world)
    assert result.state == "failed"
    state = probe(world.repo)
    assert APPROVAL_FILE in state.untracked or APPROVAL_FILE in state.modified


def test_values_file_that_git_does_not_track_fails(world):
    git(world.repo, "rm", "-q", "--cached", APPROVAL_FILE)
    write(world.repo / ".gitignore", "docs/frozen/\nout/\n" + APPROVAL_FILE + "\n")
    git(world.repo, "add", ".gitignore")
    git(world.repo, "commit", "-qm", "ignore the values file")
    result = g09(world)
    assert result.state == "failed"
    assert "not tracked" in (result.detail or "")


# ---- exact-path exclusion ----------------------------------------------------------------


@pytest.mark.parametrize(
    "name",
    [
        "approvals/policy.json.bak",
        "approvals/policy.JSON",
        "approvals/other.json",
        "approvals/sub/policy.json",
        "approvals/policy.json2",
    ],
)
def test_neighbouring_files_are_hashed(world, name):
    before = digest_of(g09(world))
    write(world.repo / name, "{}\n")
    commit(world.repo)
    after = g09(world)
    assert any(p == name for _, _, p in probe(world.repo).entries)
    assert digest_of(after) != before
    assert digest_of(after) == expected_subject_digest(world.repo)


def test_untracked_neighbour_under_approvals_fails(world):
    write(world.repo / "approvals/extra.json", "{}\n")
    assert g09(world).state == "failed"


def test_only_the_values_file_is_missing_from_the_tracked_scope(world):
    paths = [p for _, _, p in probe(world.repo).entries]
    assert APPROVAL_FILE not in paths
    assert "src/code.py" in paths and "pyproject.toml" in paths
    assert set(
        git(
            world.repo,
            "ls-files",
            "--",
            "src",
            "configs",
            "schemas",
            "scripts",
            "analysis",
            "pyproject.toml",
        ).splitlines()
    ) <= set(paths)


# ---- fail-closed loading ----------------------------------------------------------------


def good(repo_world):
    return json.loads((REAL_REPO / APPROVAL_FILE).read_text())


def put(repo, data):
    path = repo / APPROVAL_FILE
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_symlink() or path.exists():
        path.unlink()
    path.write_bytes(data if isinstance(data, bytes) else json.dumps(data).encode())


def test_missing_values_file_fails_closed(tmp_path):
    with pytest.raises(PolicyError):
        load_policy(tmp_path)


@pytest.mark.parametrize(
    "raw",
    [
        b"",
        b"{",
        b"[]",
        b"null",
        b'{"policy_version": "1", "slots": {}} trailing',
        b"\xff\xfe{}",
        b'{"policy_version": NaN, "slots": {}}',
    ],
)
def test_malformed_values_file_fails_closed(tmp_path, raw):
    put(tmp_path, raw)
    with pytest.raises(PolicyError):
        load_policy(tmp_path)


def test_duplicate_keys_fail(tmp_path):
    slots = ", ".join(f'"{n}": null' for n in SLOT_NAMES)
    put(
        tmp_path, f'{{"policy_version": "1", "slots": {{{slots}}}, "policy_version": "1"}}'.encode()
    )
    with pytest.raises(PolicyError):
        load_policy(tmp_path)
    put(
        tmp_path,
        f'{{"policy_version": "1", "slots": {{{slots}, "gpu_authorization": null}}}}'.encode(),
    )
    with pytest.raises(PolicyError):
        load_policy(tmp_path)


def test_unknown_missing_and_wrongly_typed_slots_fail(tmp_path):
    base = good(None)
    cases = []
    extra = json.loads(json.dumps(base))
    extra["slots"]["approved"] = None
    cases.append(extra)
    missing = json.loads(json.dumps(base))
    del missing["slots"]["gpu_authorization"]
    cases.append(missing)
    top = json.loads(json.dumps(base))
    top["note"] = "x"
    cases.append(top)
    no_slots = {"policy_version": "1"}
    cases.append(no_slots)
    for bad in ("ABC", "a" * 63, "g" * 64, "A" * 64, 5, True, ""):
        wrong = json.loads(json.dumps(base))
        wrong["slots"]["corpus_identity"] = bad
        cases.append(wrong)
    for case in cases:
        put(tmp_path, case)
        with pytest.raises(PolicyError):
            load_policy(tmp_path)


@pytest.mark.parametrize("version", ["2", "", "0", 1, None])
def test_wrong_policy_version_fails(tmp_path, version):
    data = good(None)
    data["policy_version"] = version
    put(tmp_path, data)
    with pytest.raises(PolicyError):
        load_policy(tmp_path)


def test_valid_populated_file_loads(tmp_path):
    data = good(None)
    data["slots"]["corpus_identity"] = "a" * 64
    put(tmp_path, data)
    assert load_policy(tmp_path).slots.corpus_identity == "a" * 64


def test_symlinked_values_file_or_directory_fails(tmp_path):
    real = tmp_path / "elsewhere.json"
    real.write_text((REAL_REPO / APPROVAL_FILE).read_text())
    (tmp_path / "approvals").mkdir()
    os.symlink(real, tmp_path / APPROVAL_FILE)
    with pytest.raises(PolicyError):
        load_policy(tmp_path)
    other = tmp_path / "repo2"
    other.mkdir()
    (tmp_path / "dir").mkdir()
    (tmp_path / "dir/policy.json").write_text((REAL_REPO / APPROVAL_FILE).read_text())
    os.symlink(tmp_path / "dir", other / "approvals")
    with pytest.raises(PolicyError):
        load_policy(other)


def test_values_file_must_be_a_regular_file(tmp_path):
    (tmp_path / APPROVAL_FILE).mkdir(parents=True)
    with pytest.raises(PolicyError):
        load_policy(tmp_path)


def test_loader_reads_no_environment_and_has_no_fallback():
    import ast
    import inspect

    from biorag_repro.readiness import approvals as policy

    tree = ast.parse(inspect.getsource(policy))
    attrs = {n.attr for n in ast.walk(tree) if isinstance(n, ast.Attribute)}
    assert not attrs & {"environ", "getenv", "argv"}
    assert not hasattr(policy, "REAL_POLICY")


# ---- production boundary ------------------------------------------------------------------


def test_cli_without_a_values_file_fails_before_publishing(world, capsys):
    (world.repo / APPROVAL_FILE).unlink()
    status = cli.run(
        ["evaluate", "--data-root", str(world.data), "--repo", str(world.repo)],
        _context=world.ctx,
    )
    assert status == 1
    assert not (world.data / "readiness").exists()


def test_cli_loads_the_policy_from_the_repo_it_evaluates(world):
    status = cli.run(
        ["evaluate", "--data-root", str(world.data), "--repo", str(world.repo)],
        _context=world.ctx,
    )
    assert status == 2
    report = next((world.data / "readiness/reports").glob("[0-9a-f]" * 64 + ".json"))
    stored = json.loads(report.read_text())
    assert stored["policy_sha256"] == ORIGINAL_POLICY_SHA256


def test_cli_verify_uses_a_changed_policy_and_refuses_the_old_report(world):
    cli.run(
        ["evaluate", "--data-root", str(world.data), "--repo", str(world.repo)], _context=world.ctx
    )
    sha = next((world.data / "readiness/reports").glob("[0-9a-f]" * 64 + ".json")).stem
    write_values(world.repo, values(world, corpus_identity="a" * 64))
    commit(world.repo)
    status = cli.run(
        [
            "verify",
            "--data-root",
            str(world.data),
            "--report-sha256",
            sha,
            "--repo",
            str(world.repo),
        ]
    )
    assert status == 1


def test_guard_refuses_when_the_policy_cannot_be_loaded(world):
    (world.repo / APPROVAL_FILE).unlink()
    with pytest.raises(AuthorizationRefused) as caught:
        authorize_final_test(world.data, "a" * 64, repo=world.repo)
    assert caught.value.reason == "policy_unavailable"


def test_guard_still_refuses_a_stored_not_ready_report(world):
    from biorag_repro.readiness import publish
    from biorag_repro.readiness.evaluate import evaluate

    policy = load_policy(world.repo)
    published = publish.publish_report(evaluate(world.ctx, policy), world.data)
    with pytest.raises(AuthorizationRefused) as caught:
        authorize_final_test(world.data, published.report_sha256, repo=world.repo)
    assert caught.value.reason == "stored_not_ready"


def test_guard_refuses_a_report_made_under_other_values(world):
    from biorag_repro.readiness import publish
    from biorag_repro.readiness.evaluate import evaluate

    published = publish.publish_report(evaluate(world.ctx, load_policy(world.repo)), world.data)
    write_values(world.repo, values(world, corpus_identity="a" * 64))
    commit(world.repo)
    with pytest.raises(AuthorizationRefused) as caught:
        authorize_final_test(world.data, published.report_sha256, repo=world.repo)
    assert caught.value.reason == "report_unverifiable"
    assert "different policy" in str(caught.value)


def test_old_report_still_verifies_under_its_own_policy(world):
    from biorag_repro.readiness import publish
    from biorag_repro.readiness.evaluate import evaluate

    policy = load_policy(world.repo)
    published = publish.publish_report(evaluate(world.ctx, policy), world.data)
    back = publish.read_report(world.data, published.report_sha256, policy)
    assert back.report.policy_sha256 == ORIGINAL_POLICY_SHA256


def test_no_command_line_option_selects_a_policy():
    parser = cli._parser()
    options = set()
    for sub in parser._subparsers._group_actions[0].choices.values():
        for action in sub._actions:
            options |= set(action.option_strings)
    assert options == {"-h", "--help", "--data-root", "--repo", "--report-sha256"}
    assert set(inspect.signature(authorize_final_test).parameters) == {
        "data_root",
        "report_sha256",
        "repo",
    }
