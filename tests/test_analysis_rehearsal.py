# ruff: noqa: F811
"""Offline Python-to-R rehearsal: preflight, R run, Python verification, record, read-back.

Everything is synthetic; no value is a scientific result. The primary tests run the real R
analysis. Failure tests replay the output of one real R run through the runner seam so that each
guard is tested on its own; the replay runner checks the exact argument vector it is given.
"""

import re
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

import pytest
from test_analysis_export import roster_of, world
from test_analysis_handoff import official_table, pair
from test_analysis_result import (
    ANALYSIS_DIR,
    GOLDEN,
    SEED,
    needs_r,
    writable_copy,
)
from test_generation_official import recognised, view  # noqa: F401  (fixtures)
from test_generation_prompt import needs_tokenizer
from test_r_input_fixture import CODE, build_table_with_papers

from biorag_repro.core.hashing import sha256_file
from biorag_repro.core.manifest import ArtifactRef
from biorag_repro.evaluation import analysis_cli as cli
from biorag_repro.evaluation import analysis_handoff as ah
from biorag_repro.evaluation import analysis_rehearsal as rh
from biorag_repro.evaluation import analysis_result as ar
from biorag_repro.evaluation.analysis_cli import main
from biorag_repro.evaluation.analysis_preflight import (
    PreflightError,
    publish_preflight,
    read_preflight,
)
from biorag_repro.evaluation.analysis_storage import publish_table
from biorag_repro.evaluation.analysis_table import build_table

LOCK = sha256_file(ANALYSIS_DIR / "renv.lock")
OUT_ROOT = "analysis/r_results"
FIXED_ID = "1" * 32
FIXED_TIME = "2026-11-06T10:00:00.000000Z"


def fixed_preflight_hooks():
    return {"now": lambda: FIXED_TIME, "new_id": lambda: FIXED_ID}


def publish_frozen_handoff(root: Path) -> ArtifactRef:
    _csv, _inventory, build = publish_table(build_table_with_papers(), root)
    _ref, handoff_build = ah.publish_handoff(
        ah.build_handoff(build, root, code=CODE, rehearsal=True), root
    )
    return handoff_build


def rehearse(root: Path, handoff_build: ArtifactRef, **changes) -> rh.Rehearsal:
    arguments = {
        "root": root,
        "handoff_build": handoff_build,
        "analysis_dir": ANALYSIS_DIR,
        "code": CODE,
        "seed": int(SEED),
        "lock_sha256": LOCK,
        "out_root": OUT_ROOT,
    } | changes
    return rh.rehearse(**arguments)


def scientific_bytes(root: Path, result: str) -> dict[str, bytes]:
    return {n: (root / result / n).read_bytes() for n in ar.SCIENTIFIC_FILES}


# ---- the real run -----------------------------------------------------------------------------


@pytest.fixture(scope="session")
def real_runs(tmp_path_factory):
    """Two independent real executions on the frozen table, into separate stores and one shared."""
    if shutil.which("Rscript") is None:
        pytest.skip("Rscript not available")
    root = tmp_path_factory.mktemp("rehearsal_store")
    build = publish_frozen_handoff(root)
    first = rehearse(root, build)
    second = rehearse(root, build)
    return root, build, first, second


@needs_r
def test_the_full_chain_runs_for_real_and_reads_back(real_runs):
    root, build, first, _ = real_runs
    assert first.scientific_result_sha256 == GOLDEN
    assert first.record.created
    checked = ar.read_result_record(first.record.build, root)
    assert checked.record.scientific_result_sha256 == GOLDEN
    assert checked.record.applied_bootstrap_seed == int(SEED)
    assert checked.record.preflight_build == first.preflight_build
    assert checked.record.handoff_build == build
    assert checked.record.scientific_authority == "none"
    assert checked.record.input_origin == "synthetic_rehearsal"
    assert checked.record.r_dependency_lock_sha256 == LOCK
    assert checked.record.rerun == "not_performed"
    assert read_preflight(first.preflight_build, root).preflight.handoff_build == build


@needs_r
def test_two_executions_agree_scientifically_and_keep_separate_chains(real_runs):
    root, _build, first, second = real_runs
    assert first.scientific_result_sha256 == second.scientific_result_sha256 == GOLDEN
    one, two = scientific_bytes(root, first.result_dir), scientific_bytes(root, second.result_dir)
    assert len(one) == 3 and all(one.values())
    assert one == two
    assert first.result_dir != second.result_dir
    assert first.preflight_build != second.preflight_build  # nonce
    assert first.run_record_sha256 != second.run_record_sha256  # run id and timestamps
    assert first.record.build != second.record.build
    for run in (first, second):
        checked = ar.read_result_record(run.record.build, root)
        assert checked.record.run_record_sha256 == run.run_record_sha256
        assert checked.record.preflight_build == run.preflight_build


@needs_r
def test_the_optional_rerun_is_recorded(tmp_path):
    build = publish_frozen_handoff(tmp_path)
    done = rehearse(tmp_path, build, rerun=True)
    assert ar.read_result_record(done.record.build, tmp_path).record.rerun in {
        "byte_identical",
        "numerically_equal_not_byte_identical",
    }


@needs_r
def test_a_zero_complete_case_handoff_runs_end_to_end(tmp_path):
    observations = pair(world())
    table = build_table(observations, roster_of(observations))
    _csv, _inventory, table_build = publish_table(table, tmp_path)
    _ref, build = ah.publish_handoff(
        ah.build_handoff(table_build, tmp_path, code=CODE, rehearsal=True), tmp_path
    )
    done = rehearse(tmp_path, build)
    record = ar.read_result_record(done.record.build, tmp_path).record
    by_outcome = {i.outcome: i for i in record.inference}
    zero = by_outcome["evidence_delivered"]
    assert zero.summary_status == "no_complete_pairs"
    assert zero.boot_status == "no_complete_pairs"
    assert zero.test_status == "inference_undefined"
    assert zero.holm_status == "holm_undefined"
    # Its hash is its own; it is not the frozen fixture's.
    assert done.scientific_result_sha256 != GOLDEN


# ---- replay: one real R run reused behind the runner seam ----------------------------------


@dataclass(frozen=True)
class Replay:
    root: Path
    build: ArtifactRef
    preflight_build: ArtifactRef
    result: str
    record_sha256: str
    scientific: str
    stdout: str


@pytest.fixture(scope="session")
def replay_base(tmp_path_factory):
    if shutil.which("Rscript") is None:
        pytest.skip("Rscript not available")
    root = tmp_path_factory.mktemp("replay_store")
    build = publish_frozen_handoff(root)
    calls = []

    def record_real(command, **kwargs):
        done = subprocess.run(command, **kwargs)  # noqa: PLW1510
        calls.append((command, done))
        return done

    done = rehearse(root, build, run=record_real, **fixed_preflight_hooks())
    return Replay(
        root,
        build,
        done.preflight_build,
        done.result_dir,
        done.run_record_sha256,
        done.scientific_result_sha256,
        calls[0][1].stdout,
    )


@pytest.fixture
def replay(replay_base, tmp_path):
    copy = writable_copy(replay_base.root, tmp_path / "store")
    drop_published_chain(copy)
    return Replay(
        copy,
        replay_base.build,
        replay_base.preflight_build,
        replay_base.result,
        replay_base.record_sha256,
        replay_base.scientific,
        replay_base.stdout.replace(str(replay_base.root), str(copy)),
    )


def drop_published_chain(root: Path) -> None:
    """Remove the preflight and record so a test republishes them. The run's own files stay.

    A preflight carries a nonce, so a real second run never repeats one; the fixed id and time of
    the replay base make the republished preflight identical to the one the R result cites.
    """
    for name in ("preflight", "result"):
        shutil.rmtree(root / "analysis" / name, ignore_errors=True)


def expected_command(w: Replay) -> list[str]:
    pre = read_preflight(w.preflight_build, w.root)
    handoff = pre.preflight.handoff
    csv = pre.preflight.csv
    pre_file = pre.preflight.model_dump()  # not used for the path; see below
    del pre_file
    stored = ar.read_manifest(w.root / w.preflight_build.logical_ref, w.preflight_build.sha256)
    pre_ref = stored.outputs[0].artifact
    return [
        "Rscript",
        "run_analysis.R",
        "run",
        "--handoff",
        str(w.root / handoff.logical_ref),
        "--handoff-sha256",
        handoff.sha256,
        "--csv",
        str(w.root / csv.logical_ref),
        "--lock-sha256",
        LOCK,
        "--seed",
        SEED,
        "--seed-label",
        "synthetic_rehearsal",
        "--out-root",
        str(w.root / OUT_ROOT),
        "--preflight",
        str(w.root / pre_ref.logical_ref),
        "--preflight-sha256",
        pre_ref.sha256,
    ]


class Runner:
    """Stands in for subprocess.run: checks the exact argument vector, then answers."""

    def __init__(self, w: Replay, *, returncode=0, stdout=None, stderr="", check=True):
        self.w = w
        self.returncode = returncode
        self.stdout = w.stdout if stdout is None else stdout
        self.stderr = stderr
        self.check = check
        self.calls: list[tuple[list[str], dict]] = []

    def __call__(self, command, **kwargs):
        self.calls.append((command, kwargs))
        if self.check:
            assert command == expected_command(self.w)
        return subprocess.CompletedProcess(command, self.returncode, self.stdout, self.stderr)


def replayed(w: Replay, runner: Runner | None = None, **changes) -> rh.Rehearsal:
    runner = runner or Runner(w)
    return rehearse(w.root, w.build, run=runner, **fixed_preflight_hooks(), **changes)


def nothing_published(w: Replay) -> bool:
    return not (w.root / "analysis" / "result").exists()


def result_files(w: Replay) -> Path:
    return w.root / w.result


def unseal(path: Path) -> Path:
    path.chmod(0o644)
    return path


def test_replay_reproduces_a_complete_rehearsal(replay):
    runner = Runner(replay)
    done = replayed(replay, runner)
    assert done.scientific_result_sha256 == GOLDEN
    assert done.result_dir == replay.result
    assert len(runner.calls) == 1
    kwargs = runner.calls[0][1]
    assert kwargs["cwd"] == ANALYSIS_DIR
    assert not kwargs.get("shell")
    assert ar.read_result_record(done.record.build, replay.root).record.preflight_build == (
        replay.preflight_build
    )


def test_the_r_command_is_an_argument_vector_with_the_verified_hashes(replay):
    runner = Runner(replay)
    replayed(replay, runner)
    command = runner.calls[0][0]
    assert isinstance(command, list) and all(isinstance(a, str) for a in command)
    assert command[command.index("--seed-label") + 1] == "synthetic_rehearsal"
    assert command[command.index("--seed") + 1] == SEED


# ---- ordering: an earlier failure stops everything after it ---------------------------------


class NeverCalled:
    def __call__(self, *args, **kwargs):  # pragma: no cover - failing is the point
        raise AssertionError("R was invoked")


def test_an_invalid_handoff_publishes_no_preflight_and_never_runs_r(tmp_path):
    build = publish_frozen_handoff(tmp_path)
    csv = tmp_path / ah.read_handoff(build, tmp_path).handoff.csv.logical_ref
    unseal(csv).write_bytes(csv.read_bytes() + b"x")
    with pytest.raises(rh.RehearsalError) as caught:
        rehearse(tmp_path, build, run=NeverCalled())
    assert caught.value.stage == "preflight"
    assert not (tmp_path / "analysis" / "preflight").exists()
    assert not (tmp_path / OUT_ROOT).exists()


def test_a_handoff_claiming_official_origin_is_refused_before_r(tmp_path):
    build = publish_frozen_handoff(tmp_path)
    manifest = ar.read_manifest(tmp_path / build.logical_ref, build.sha256)
    path = unseal(tmp_path / manifest.outputs[0].artifact.logical_ref)
    path.write_bytes(path.read_bytes().replace(b"synthetic_rehearsal", b"verified_official_run"))
    with pytest.raises(rh.RehearsalError) as caught:
        rehearse(tmp_path, build, run=NeverCalled())
    assert caught.value.stage == "preflight"
    assert not (tmp_path / "analysis" / "preflight").exists()
    assert not (tmp_path / OUT_ROOT).exists()


@needs_tokenizer
def test_a_verified_official_handoff_is_refused_before_r(tmp_path, view, tiny, recognised):
    verified, root, build = official_table(tmp_path, view, tiny)
    handoff = ah.build_handoff(build, root, code=CODE, official_runs=[verified])
    _, built = ah.publish_handoff(handoff, root, official_runs=[verified])
    with pytest.raises(ah.OfficialProvenanceNotVerified):
        rehearse(root, built, run=NeverCalled())
    assert not (root / "analysis" / "preflight").exists()
    assert not (root / OUT_ROOT).exists()


def test_a_wrong_handoff_hash_stops_before_r(tmp_path):
    build = publish_frozen_handoff(tmp_path)
    wrong = build.model_copy(update={"sha256": "0" * 64})
    with pytest.raises((rh.RehearsalError, ValueError)):
        rehearse(tmp_path, wrong, run=NeverCalled())
    assert not (tmp_path / OUT_ROOT).exists()


def test_a_preflight_that_fails_read_back_stops_before_r(tmp_path, monkeypatch):
    build = publish_frozen_handoff(tmp_path)

    def failing(*args, **kwargs):
        raise PreflightError("read-back disagreed")

    monkeypatch.setattr(rh, "read_preflight", failing)
    with pytest.raises(rh.RehearsalError) as caught:
        rehearse(tmp_path, build, run=NeverCalled())
    assert caught.value.stage == "preflight"
    assert not (tmp_path / OUT_ROOT).exists()


def test_r_is_given_the_verified_preflight_not_the_callers_handoff(tmp_path, monkeypatch):
    build = publish_frozen_handoff(tmp_path)
    other_root = tmp_path / "other"
    other_root.mkdir()
    other = publish_frozen_handoff(other_root)
    other_pre = read_preflight(publish_preflight(other, other_root, code=CODE)[1], other_root)
    monkeypatch.setattr(rh, "read_preflight", lambda build, root: other_pre)
    seen = []

    def runner(command, **kwargs):
        seen.append(command)
        return subprocess.CompletedProcess(command, 1, "", "stopped")

    with pytest.raises(rh.RehearsalError) as caught:
        rehearse(tmp_path, build, run=runner)
    assert caught.value.stage == "r_run"
    command = seen[0]
    assert command[command.index("--handoff") + 1].endswith(other_pre.preflight.handoff.logical_ref)
    assert command[command.index("--handoff-sha256") + 1] == other_pre.preflight.handoff.sha256


# ---- R execution failures -------------------------------------------------------------------


def test_a_failing_r_run_publishes_nothing(replay):
    runner = Runner(replay, returncode=1, stderr="error: boom")
    with pytest.raises(rh.RehearsalError) as caught:
        replayed(replay, runner)
    assert caught.value.stage == "r_run"
    assert "boom" in str(caught.value)
    assert nothing_published(replay)


def test_r_that_cannot_be_started_publishes_nothing(replay):
    def missing(command, **kwargs):
        raise FileNotFoundError("Rscript")

    with pytest.raises(rh.RehearsalError) as caught:
        replayed(replay, missing)
    assert caught.value.stage == "r_run"
    assert nothing_published(replay)


@pytest.mark.parametrize("damage", ["empty", "no_hash", "no_scientific_hash"])
def test_r_output_without_a_result_directory_or_hash_is_refused(replay, damage):
    stdout = {
        "empty": "",
        "no_hash": re.sub(r"^run_record.*$", "", replay.stdout, flags=re.MULTILINE),
        "no_scientific_hash": re.sub(
            r"^scientific_result.*$", "", replay.stdout, flags=re.MULTILINE
        ),
    }[damage]
    with pytest.raises(rh.RehearsalError) as caught:
        replayed(replay, Runner(replay, stdout=stdout))
    assert caught.value.stage == "r_run"
    assert nothing_published(replay)


def test_a_result_directory_outside_the_root_is_refused(replay):
    stdout = re.sub(
        r"^result directory: .*$",
        "result directory: /elsewhere/x",
        replay.stdout,
        flags=re.MULTILINE,
    )
    with pytest.raises(rh.RehearsalError) as caught:
        replayed(replay, Runner(replay, stdout=stdout))
    assert caught.value.stage == "r_run"
    assert nothing_published(replay)


def test_r_printing_another_scientific_hash_is_refused(replay):
    stdout = replay.stdout.replace(replay.scientific, "0" * 64)
    with pytest.raises(rh.RehearsalError) as caught:
        replayed(replay, Runner(replay, stdout=stdout))
    assert caught.value.stage == "postflight"
    assert nothing_published(replay)


def test_r_printing_another_run_record_hash_fails_verification(replay):
    stdout = replay.stdout.replace(replay.record_sha256, "0" * 64)
    with pytest.raises(rh.RehearsalError) as caught:
        replayed(replay, Runner(replay, stdout=stdout))
    assert caught.value.stage == "postflight"
    assert nothing_published(replay)


# ---- postflight failures --------------------------------------------------------------------


def mutate(path: Path, old: bytes, new: bytes) -> None:
    unseal(path).write_bytes(path.read_bytes().replace(old, new, 1))


@pytest.mark.parametrize("name", ar.DATA_FILES)
def test_a_modified_r_data_file_fails_postflight(replay, name):
    path = result_files(replay) / name
    unseal(path).write_bytes(path.read_bytes() + b"\n")
    with pytest.raises(rh.RehearsalError) as caught:
        replayed(replay)
    assert caught.value.stage == "postflight"
    assert nothing_published(replay)


def test_a_modified_run_record_fails_postflight(replay):
    path = result_files(replay) / "run_record.json"
    mutate(path, b"synthetic_rehearsal", b"synthetic_rehearsal ")
    with pytest.raises(rh.RehearsalError) as caught:
        replayed(replay)
    assert caught.value.stage == "postflight"
    assert nothing_published(replay)


def test_an_incomplete_r_directory_fails_postflight(replay):
    victim = next(p for p in result_files(replay).iterdir() if p.suffix == ".csv")
    unseal(victim).unlink()
    with pytest.raises(rh.RehearsalError) as caught:
        replayed(replay)
    assert caught.value.stage == "postflight"
    assert nothing_published(replay)


def test_a_result_that_cites_another_preflight_fails_postflight(replay):
    """A real R result of this handoff that cites a different, valid preflight is not accepted."""
    runner = Runner(replay, check=False)
    with pytest.raises(rh.RehearsalError) as caught:
        rehearse(
            replay.root,
            replay.build,
            run=runner,
            now=lambda: FIXED_TIME,
            new_id=lambda: "2" * 32,
        )
    assert caught.value.stage == "postflight"
    assert len(runner.calls) == 1
    assert nothing_published(replay)


def test_a_forged_scientific_value_with_a_consistent_hash_is_not_claimed_as_checked(replay):
    """Only the optional R rerun can see this; the record must still say it is R-reported."""
    done = replayed(replay)
    record = ar.read_result_record(done.record.build, replay.root).record
    statuses = {c.name: c.status for c in record.checks}
    assert "reported_by_r" in statuses.values()
    assert "not_established" in statuses.values()


# ---- publication and read-back --------------------------------------------------------------


def test_repeating_an_identical_publication_is_accepted(replay):
    first = replayed(replay)
    drop_published_chain_keep_record(replay)
    again = replayed(replay)
    assert first.record.created and not again.record.created
    assert first.record.build == again.record.build


def drop_published_chain_keep_record(w: Replay) -> None:
    shutil.rmtree(w.root / "analysis" / "preflight")


def test_a_preflight_is_never_republished_over_an_existing_one(replay):
    replayed(replay)
    before = {p: p.read_bytes() for p in (replay.root / "analysis" / "preflight").iterdir()}
    with pytest.raises(rh.RehearsalError) as caught:
        replayed(replay)
    assert caught.value.stage == "preflight"
    assert before == {p: p.read_bytes() for p in (replay.root / "analysis" / "preflight").iterdir()}


def test_a_conflicting_record_is_never_replaced(replay):
    first = replayed(replay)
    stored = unseal(replay.root / first.record.ref.logical_ref)
    stored.write_bytes(
        stored.read_bytes().replace(b"synthetic_rehearsal", b"synthetic_rehearsaX", 1)
    )
    tampered = stored.read_bytes()
    drop_published_chain_keep_record(replay)
    with pytest.raises(rh.RehearsalError) as caught:
        replayed(replay)
    assert caught.value.stage == "publication"
    assert stored.read_bytes() == tampered


def test_a_failed_read_back_is_an_unsuccessful_outcome(replay, monkeypatch):
    def failing(*args, **kwargs):
        raise ar.ResultError("read_back", "the record disagrees with its evidence")

    monkeypatch.setattr(rh, "read_result_record", failing)
    with pytest.raises(rh.RehearsalError) as caught:
        replayed(replay)
    assert caught.value.stage == "read_back"


def test_a_record_with_altered_references_fails_read_back(replay, monkeypatch):
    real = ar.read_result_record

    def altered(build, root):
        checked = real(build, root)
        record = checked.record.model_copy(update={"scientific_result_sha256": "0" * 64})
        return ar.VerifiedResultRecord(record, checked.verified)

    monkeypatch.setattr(rh, "read_result_record", altered)
    with pytest.raises(rh.RehearsalError) as caught:
        replayed(replay)
    assert caught.value.stage == "read_back"


def test_a_record_citing_another_preflight_fails_read_back(replay, monkeypatch):
    real = ar.read_result_record

    def altered(build, root):
        checked = real(build, root)
        record = checked.record.model_copy(
            update={"preflight_build": replay.build}  # not the preflight of this run
        )
        return ar.VerifiedResultRecord(record, checked.verified)

    monkeypatch.setattr(rh, "read_result_record", altered)
    with pytest.raises(rh.RehearsalError) as caught:
        replayed(replay)
    assert caught.value.stage == "read_back"


def test_a_stale_result_directory_is_not_a_completed_run(replay):
    """R printing a directory that holds only part of a run must not be recorded."""
    for path in list(result_files(replay).iterdir())[:2]:
        unseal(path).unlink()
    with pytest.raises(rh.RehearsalError):
        replayed(replay)
    assert nothing_published(replay)


def test_the_seed_and_label_are_not_defaulted():
    import inspect

    parameters = inspect.signature(rh.rehearse).parameters
    assert parameters["seed"].default is inspect.Parameter.empty
    assert parameters["lock_sha256"].default is inspect.Parameter.empty
    assert parameters["out_root"].default is inspect.Parameter.empty


def test_the_applied_seed_must_be_the_requested_seed(replay):
    # The replayed R output was produced with SEED; asking for another seed makes the argument
    # vector differ, which the replay runner rejects as a failed R run, so nothing is recorded.
    with pytest.raises((rh.RehearsalError, AssertionError)):
        replayed(replay, seed=int(SEED) + 1)
    assert nothing_published(replay)


def test_an_out_root_outside_the_store_is_refused(replay, tmp_path):
    with pytest.raises(rh.RehearsalError) as caught:
        replayed(replay, out_root="../escape")
    assert caught.value.stage == "arguments"


# ---- command line ----------------------------------------------------------------------------

REPOSITORY = ANALYSIS_DIR.parent


def cli_arguments(root: Path, build: ArtifactRef, **changes) -> list[str]:
    values = {
        "--root": str(root),
        "--handoff-build": build.logical_ref,
        "--handoff-sha256": build.sha256,
        "--seed": SEED,
        "--lock-sha256": LOCK,
        "--out-root": OUT_ROOT,
        "--repository": str(REPOSITORY),
    } | changes
    return ["rehearse", *[x for pair_ in values.items() for x in pair_]]


@needs_r
def test_the_command_runs_the_chain_and_its_record_verifies_separately(tmp_path, capsys):
    build = publish_frozen_handoff(tmp_path)
    assert main(cli_arguments(tmp_path, build)) == 0
    out = capsys.readouterr().out
    assert f"scientific_result_sha256 {GOLDEN}" in out
    assert "rehearsal seed, not the approved production seed" in out
    assert "no scientific authority" in out
    record = re.search(r"^result record build: (\S+) ([0-9a-f]{64})$", out, re.MULTILINE)
    assert record
    again = ["verify-result-record", "--root", str(tmp_path), "--result-build", record[1]]
    assert main([*again, "--result-sha256", record[2], "--repository", str(REPOSITORY)]) == 0


@needs_r
def test_the_command_stops_when_r_rejects_the_lock_hash(tmp_path, capsys):
    build = publish_frozen_handoff(tmp_path)
    assert main(cli_arguments(tmp_path, build, **{"--lock-sha256": "0" * 64})) == 1
    assert "rehearsal stopped: [r_run]" in capsys.readouterr().err
    assert not (tmp_path / "analysis" / "result").exists()


def test_the_command_has_no_default_seed_or_lock(tmp_path, capsys):
    build = publish_frozen_handoff(tmp_path)
    for missing in ("--seed", "--lock-sha256", "--out-root"):
        arguments = cli_arguments(tmp_path, build)
        at = arguments.index(missing)
        del arguments[at : at + 2]
        with pytest.raises(SystemExit) as caught:
            main(arguments)
        assert caught.value.code == 2
    capsys.readouterr()
    assert not (tmp_path / "analysis" / "preflight").exists()


def test_the_command_reports_an_official_origin_without_writing(tmp_path, monkeypatch, capsys):
    def refuse(*args, **kwargs):
        raise ah.OfficialProvenanceNotVerified("official")

    monkeypatch.setattr(cli, "rehearse", refuse)
    build = publish_frozen_handoff(tmp_path)
    assert main(cli_arguments(tmp_path, build)) == 2
    assert "Nothing is written" in capsys.readouterr().err


# ---- guards found missing by mutation testing ---------------------------------------------


def altered_read_back(monkeypatch, **changes):
    real = ar.read_result_record

    def altered(build, root):
        checked = real(build, root)
        return ar.VerifiedResultRecord(checked.record.model_copy(update=changes), checked.verified)

    monkeypatch.setattr(rh, "read_result_record", altered)


@pytest.mark.parametrize(
    "field, value",
    [
        ("preflight", None),
        ("handoff_build", None),
        ("run_record_sha256", "0" * 64),
        ("applied_bootstrap_seed", 1),
        ("result_dir", "analysis/elsewhere/x"),
    ],
)
def test_each_cited_reference_is_compared_on_read_back(replay, monkeypatch, field, value):
    if value is None:  # another valid reference of the same type
        value = replay.preflight_build
    altered_read_back(monkeypatch, **{field: value})
    with pytest.raises(rh.RehearsalError) as caught:
        replayed(replay)
    assert caught.value.stage == "read_back"


def test_the_seed_r_applied_must_be_the_seed_requested(replay):
    runner = Runner(replay, check=False)  # R "ran" with SEED although another was requested
    with pytest.raises(rh.RehearsalError, match="applied_bootstrap_seed") as caught:
        replayed(replay, runner, seed=int(SEED) + 1)
    assert caught.value.stage == "postflight"
    assert nothing_published(replay)


def test_the_lock_r_recorded_must_be_the_lock_requested(replay):
    runner = Runner(replay, check=False)
    with pytest.raises(rh.RehearsalError, match="r_dependency_lock_sha256") as caught:
        replayed(replay, runner, lock_sha256="0" * 64)
    assert caught.value.stage == "postflight"
    assert nothing_published(replay)


def test_a_result_directory_inside_the_store_but_outside_the_out_root_is_refused(replay):
    elsewhere = str(replay.root / "analysis" / "elsewhere" / "run")
    stdout = re.sub(
        r"^result directory: .*$",
        f"result directory: {elsewhere}",
        replay.stdout,
        flags=re.MULTILINE,
    )
    with pytest.raises(rh.RehearsalError, match="output root") as caught:
        replayed(replay, Runner(replay, stdout=stdout))
    assert caught.value.stage == "r_run"
    assert nothing_published(replay)


def test_clubsandwich_is_required_to_have_run_only_when_a_test_was_defined():
    row = {"test_status": "ok"}
    undefined = {"test_status": "inference_undefined"}
    assert ar._cr2_test_ran([row, row])
    assert ar._cr2_test_ran([undefined, row])
    assert not ar._cr2_test_ran([undefined, undefined])


def test_the_method_package_must_be_locked_and_installed_even_when_no_test_ran(replay_base):
    verified = ar.verify_r_result(
        replay_base.root, replay_base.result, replay_base.preflight_build, replay_base.record_sha256
    )
    run = verified.run
    handoff = verified.preflight.handoff.handoff
    files = ar.read_result_files(replay_base.root, replay_base.result)
    inventory = ar.parse_csv("package_inventory.csv", files["package_inventory.csv"])
    assert ar._check_environment(run, inventory, handoff, test_ran=True) == "0.7.0"
    # R did not use the package, and no test was defined: acceptable.
    unused = run.environment_actual.model_copy(
        update={
            "package_versions_used": {
                k: v
                for k, v in run.environment_actual.package_versions_used.items()
                if k != "clubSandwich"
            }
        }
    )
    run_unused = run.model_copy(update={"environment_actual": unused})
    assert ar._check_environment(run_unused, inventory, handoff, test_ran=False) == "0.7.0"
    with pytest.raises(ar.ResultError, match="used"):
        ar._check_environment(run_unused, inventory, handoff, test_ran=True)
    # Not locked and not installed: refused whether or not a test ran.
    locked = {
        k: v for k, v in run.environment_intended.lock_packages.items() if k != "clubSandwich"
    }
    intended = run.environment_intended.model_copy(update={"lock_packages": locked})
    without = [r for r in inventory if r["package"] != "clubSandwich"]
    run_unlocked = run_unused.model_copy(update={"environment_intended": intended})
    with pytest.raises(ar.ResultError, match="clubSandwich"):
        ar._check_environment(run_unlocked, without, handoff, test_ran=False)


def test_the_command_passes_the_rerun_flag_through(tmp_path, monkeypatch):
    seen = []

    def capture(*args, **kwargs):
        seen.append(kwargs["rerun"])
        raise rh.RehearsalError("preflight", "stop")

    monkeypatch.setattr(cli, "rehearse", capture)
    build = publish_frozen_handoff(tmp_path)
    main([*cli_arguments(tmp_path, build), "--rerun"])
    main(cli_arguments(tmp_path, build))
    assert seen == [True, False]
