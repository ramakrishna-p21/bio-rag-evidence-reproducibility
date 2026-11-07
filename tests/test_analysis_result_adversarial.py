# ruff: noqa: F811
"""Adversarial cases for the Python verification of an R result and of the result record.

Each case alters the real R result (or the store around it) and must fail at the verification
boundary it targets, so the tests assert the stage the error names, not merely that something
raised. Where a forgery is made hash-consistent on purpose, that is stated.
"""

import json
import os
import shutil
import subprocess

import pytest
from test_analysis_result import (
    ANALYSIS_DIR,
    GOLDEN,
    World,
    base_world,  # noqa: F401  (fixture)
    needs_r,
    publish,
    run_r,
    verify,
    world,  # noqa: F401  (fixture)
)
from test_r_input_fixture import CODE

from biorag_repro.core.hashing import HashMismatchError, canonical_json_bytes, sha256_bytes
from biorag_repro.core.manifest import ArtifactRef
from biorag_repro.core.storage import read_manifest
from biorag_repro.evaluation import analysis_result as ar
from biorag_repro.evaluation.analysis_preflight import publish_preflight, read_preflight

pytestmark = needs_r

DATA = ("coprimary_results.csv", "paper_counts.csv", "state_counts.csv", "package_inventory.csv")
SCIENTIFIC = DATA[:3]


# ---- helpers: alter a result directory and, when wanted, keep it hash-consistent --------------


def directory(w: World):
    return w.root / w.result


def read(w: World, name: str) -> bytes:
    return (directory(w) / name).read_bytes()


def write(w: World, name: str, data: bytes) -> None:
    (directory(w) / name).write_bytes(data)


def edit_text(w: World, name: str, old: str, new: str) -> None:
    text = read(w, name).decode()
    assert text.count(old) >= 1, (name, old)
    write(w, name, text.replace(old, new, 1).encode())


def load_record(w: World) -> dict:
    return json.loads(read(w, "run_record.json"))


def save_record(w: World, record: dict) -> str:
    """Write the record (formatting is not significant) and return its new SHA-256."""
    data = (json.dumps(record, indent=2) + "\n").encode()
    write(w, "run_record.json", data)
    return sha256_bytes(data)


def identity_of(record: dict, hashes: dict[str, str], *, order=None) -> str:
    names = order or sorted(SCIENTIFIC)
    return sha256_bytes(
        canonical_json_bytes(
            {
                "csv_sha256": record["inputs"]["csv_sha256"],
                "files": [{"name": n, "sha256": hashes[n]} for n in names],
                "handoff_sha256": record["inputs"]["handoff_sha256"],
                "seed": record["seed"]["value"],
            }
        )
    )


def forge(w: World, change=None, *, identity=True) -> str:
    """Make the record agree with the (already edited) files, as a forger who can rehash would.

    Returns the SHA-256 of the forged record, which the caller then supplies.
    """
    record = load_record(w)
    hashes = {n: sha256_bytes(read(w, n)) for n in DATA}
    for n in DATA:
        record["outputs"]["files"][n] = {"bytes": len(read(w, n)), "sha256": hashes[n]}
    record["environment_actual"]["installed_package_inventory_sha256"] = hashes[
        "package_inventory.csv"
    ]
    if change:
        change(record)
    if identity:
        record["outputs"]["scientific_result_sha256"] = identity_of(record, hashes)
    return save_record(w, record)


def fails(w: World, stage: str, **changes):
    with pytest.raises(ar.ResultError) as error:
        verify(w, **changes)
    assert error.value.stage == stage, error.value
    return error.value


def forged_fails(w: World, stage: str, change=None, *, identity=True):
    return fails(w, stage, run_record_sha256=forge(w, change, identity=identity))


# ---- the unaltered result is the control for every case below ----------------------------------


def test_the_untouched_copy_verifies(world):
    assert verify(world).claims["scientific_result_sha256"] == GOLDEN


# ---- file set, links and paths -----------------------------------------------------------------


@pytest.mark.parametrize("name", ar.RESULT_FILES)
def test_a_missing_result_file_is_a_file_set_failure(world, name):
    (directory(world) / name).unlink()
    fails(world, "file_set")


def test_an_unexpected_file_or_subdirectory_is_refused(world):
    (directory(world) / "notes.txt").write_text("x")
    fails(world, "file_set")
    (directory(world) / "notes.txt").unlink()
    (directory(world) / "sub").mkdir()
    fails(world, "file_set")


def test_a_partial_publication_with_only_some_files_is_refused(world):
    for name in ("run_record.json", "state_counts.csv"):
        (directory(world) / name).unlink()
    fails(world, "file_set")


def test_a_symlink_in_place_of_a_result_file_is_not_followed(world, tmp_path):
    outside = tmp_path / "outside.csv"
    outside.write_bytes(read(world, "paper_counts.csv"))  # same bytes: only the link is wrong
    (directory(world) / "paper_counts.csv").unlink()
    (directory(world) / "paper_counts.csv").symlink_to(outside)
    fails(world, "file_set")


def test_a_directory_fifo_or_empty_file_in_place_of_a_file_is_refused(world):
    target = directory(world) / "state_counts.csv"
    target.unlink()
    target.mkdir()
    fails(world, "file_set")
    target.rmdir()
    os.mkfifo(target)
    fails(world, "file_set")
    target.unlink()
    target.write_bytes(b"")
    fails(world, "file_set")


def test_a_symlinked_result_directory_or_parent_is_not_followed(world, tmp_path):
    real = directory(world)
    link = real.parent / "linked"
    link.symlink_to(real)
    fails(
        world,
        "result_location",
        result_dir=f"{real.parent.relative_to(world.root).as_posix()}/linked",
    )
    parent_link = world.root / "alias"
    parent_link.symlink_to(real.parent)
    fails(world, "result_location", result_dir=f"alias/{real.name}")


@pytest.mark.parametrize(
    "bad", ["", "/etc", "../x", "analysis/../analysis", "a//b", "a/./b", "a\\b", "analysis/\0x"]
)
def test_unsafe_result_paths_are_refused(world, bad):
    fails(world, "result_location", result_dir=bad)


def test_a_result_directory_that_does_not_exist_is_refused(world):
    fails(world, "result_location", result_dir="analysis/r_results/nothing_here")


# ---- the run record ----------------------------------------------------------------------------


def test_a_changed_record_fails_the_supplied_hash(world):
    record = load_record(world)
    record["execution"]["hostname"] = "elsewhere"
    save_record(world, record)
    fails(world, "record_hash")


def test_a_wrongly_supplied_hash_is_refused_for_an_intact_record(world):
    fails(world, "record_hash", run_record_sha256="0" * 64)


def rewrite_text(old: str, new: str):
    def edit(w: World) -> str:
        text = read(w, "run_record.json").decode()
        assert old in text
        data = text.replace(old, new, 1).encode()
        write(w, "run_record.json", data)
        return sha256_bytes(data)

    return edit


@pytest.mark.parametrize(
    "mutate",
    [
        lambda w: _raw(w, b"{not json"),
        lambda w: _raw(w, b"\xff\xfe"),
        lambda w: _raw(w, b"[]"),
        lambda w: _raw(w, b'{"kind": "r_run_record", "kind": "r_run_record"}'),
        rewrite_text('"value": 20260601', '"value": NaN'),
        rewrite_text('"synthetic_rehearsal"', '"verified_official_run"'),
        rewrite_text('"scientific_authority": "none"', '"scientific_authority": "full"'),
        rewrite_text('"lineage_verified_by_r": false', '"lineage_verified_by_r": true'),
        rewrite_text('"production_seed": "awaiting_approval"', '"production_seed": "approved"'),
        rewrite_text('"registry": "not_registered"', '"registry": "registered"'),
        rewrite_text('"n_resamples": 10000', '"n_resamples": 999'),
        rewrite_text('"r_version_required": "4.6.1"', '"r_version_required": "4.5.0"'),
        rewrite_text('"kind": "r_run_record"', '"kind": "r_run_record", "extra": 1'),
        rewrite_text('"schema_version": "1"', '"schema_version": "2"'),
        rewrite_text(
            '"python_preflight": "cited_bytes_hashed_not_interpreted"',
            '"python_preflight": "verified"',
        ),
    ],
)
def test_a_record_that_is_not_the_exact_synthetic_claim_set_is_refused_even_when_its_hash_is_supplied(
    world, mutate
):
    digest = mutate(world)
    fails(world, "record_structure", run_record_sha256=digest)


def _raw(w: World, data: bytes) -> str:
    write(w, "run_record.json", data)
    return sha256_bytes(data)


@pytest.mark.parametrize(
    "drop",
    [
        ("seed",),
        ("code", "git_commit"),
        ("environment_actual", "package_versions_used"),
        ("outputs", "files"),
    ],
)
def test_a_missing_required_record_field_is_refused(world, drop):
    record = load_record(world)
    holder = record
    for key in drop[:-1]:
        holder = holder[key]
    del holder[drop[-1]]
    fails(world, "record_structure", run_record_sha256=save_record(world, record))


@pytest.mark.parametrize(
    "path,value",
    [
        (("inputs", "handoff_sha256"), "A" * 64),
        (("inputs", "csv_sha256"), "abc"),
        (("code", "git_commit"), "xyz"),
        (("outputs", "scientific_result_sha256"), "0" * 63),
        (("code", "dirty"), "flip"),  # the flag must agree with the listed changed files
        (("provenance", "required"), ["handoff_sha256"]),
    ],
)
def test_malformed_or_inconsistent_record_values_are_refused(world, path, value):
    record = load_record(world)
    holder = record
    for key in path[:-1]:
        holder = holder[key]
    flipped = value == "flip"
    if flipped:
        value = not holder[path[-1]]  # whatever the tree was when R ran, say the opposite
    holder[path[-1]] = value
    error = fails(world, "record_structure", run_record_sha256=save_record(world, record))
    if flipped:
        assert "dirty flag disagrees" in str(error)


def _drop_a_file(r):
    del r["outputs"]["files"]["paper_counts.csv"]


def _wrong_size(r):
    r["outputs"]["files"]["state_counts.csv"]["bytes"] += 1


def _wrong_hash(r):
    r["outputs"]["files"]["coprimary_results.csv"]["sha256"] = "1" * 64


def _wrong_inventory_hash(r):
    r["environment_actual"]["installed_package_inventory_sha256"] = "2" * 64


@pytest.mark.parametrize("change", [_drop_a_file, _wrong_size, _wrong_hash, _wrong_inventory_hash])
def test_a_record_that_lists_other_files_or_hashes_is_an_output_hash_failure(world, change):
    record = load_record(world)
    change(record)
    fails(world, "output_hashes", run_record_sha256=save_record(world, record))


def test_changing_a_published_data_file_without_rehashing_is_an_output_hash_failure(world):
    edit_text(world, "coprimary_results.csv", "0.8571428571428571", "0.8571428571428572")
    fails(world, "output_hashes")


def test_changing_the_inventory_without_rehashing_is_an_output_hash_failure(world):
    edit_text(world, "package_inventory.csv", "R6,2.6.1", "R6,2.6.2")
    fails(world, "output_hashes")


# ---- scientific result identity ----------------------------------------------------------------


def test_a_hash_consistent_forgery_still_fails_the_scientific_identity(world):
    edit_text(world, "coprimary_results.csv", "ok,10000,0,-0.25", "ok,10000,0,-0.5")
    forged_fails(world, "result_identity", identity=False)


def test_a_wrong_stated_scientific_identity_alone_is_refused(world):
    def change(r):
        r["outputs"]["scientific_result_sha256"] = "3" * 64

    forged_fails(world, "result_identity", change, identity=False)


def test_a_modified_seed_changes_the_identity(world):
    def change(r):
        r["seed"]["value"] = 20260602

    forged_fails(world, "result_identity", change, identity=False)


@pytest.mark.parametrize(
    "construct",
    [
        "reversed_files",
        "without_csv",
        "without_handoff",
        "files_unnamed",
        "seed_as_text",
        "plain_concatenation",
    ],
)
def test_alternative_identity_constructions_are_not_accepted(world, construct):
    record = load_record(world)
    hashes = {n: sha256_bytes(read(world, n)) for n in SCIENTIFIC}
    inputs = record["inputs"]
    base = {
        "csv_sha256": inputs["csv_sha256"],
        "files": [{"name": n, "sha256": hashes[n]} for n in sorted(SCIENTIFIC)],
        "handoff_sha256": inputs["handoff_sha256"],
        "seed": record["seed"]["value"],
    }
    variants = {
        "reversed_files": base | {"files": base["files"][::-1]},
        "without_csv": {k: v for k, v in base.items() if k != "csv_sha256"},
        "without_handoff": {k: v for k, v in base.items() if k != "handoff_sha256"},
        "files_unnamed": base | {"files": [f["sha256"] for f in base["files"]]},
        "seed_as_text": base | {"seed": str(base["seed"])},
    }
    if construct == "plain_concatenation":
        forged = sha256_bytes("".join(hashes[n] for n in sorted(SCIENTIFIC)).encode())
    else:
        forged = sha256_bytes(canonical_json_bytes(variants[construct]))
    assert forged != GOLDEN

    def change(r):
        r["outputs"]["scientific_result_sha256"] = forged

    forged_fails(world, "result_identity", change, identity=False)


def test_the_identity_function_ignores_the_order_files_are_supplied_in():
    hashes = {
        "state_counts.csv": "a" * 64,
        "coprimary_results.csv": "b" * 64,
        "paper_counts.csv": "c" * 64,
    }
    forward = ar.scientific_result_id("d" * 64, "e" * 64, 7, hashes)
    assert forward == ar.scientific_result_id("d" * 64, "e" * 64, 7, dict(reversed(hashes.items())))
    assert forward != ar.scientific_result_id("d" * 64, "e" * 64, 8, hashes)
    assert forward != ar.scientific_result_id("e" * 64, "d" * 64, 7, hashes)


# ---- the CSV contract, with a hash-consistent forger -------------------------------------------


@pytest.mark.parametrize(
    "name,old,new",
    [
        ("coprimary_results.csv", "\n", "\r\n"),
        ("coprimary_results.csv", "evidence_delivered,9", "answer_correct,9"),  # duplicates / order
        ("coprimary_results.csv", ",ok,0.857", ",weird,0.857"),
        ("coprimary_results.csv", ",ok,10000,0,", ",bogus,10000,0,"),
        ("coprimary_results.csv", "0.8571428571428571", "0.857142857142857100"),
        ("coprimary_results.csv", "0.8571428571428571", "1.5"),
        ("coprimary_results.csv", "0.8571428571428571", "inf"),
        ("coprimary_results.csv", "evidence_delivered,9,7", "evidence_delivered,NA,7"),
        ("coprimary_results.csv", "evidence_delivered,9,7", "evidence_delivered,9,7.0"),
        ("coprimary_results.csv", "ok,NA,0.29160592175990208", "ok,mystery,0.29160592175990208"),
        ("paper_counts.csv", "evidence_delivered,P001,1,1", "evidence_delivered,P001,NA,1"),
        (
            "paper_counts.csv",
            "evidence_delivered,P001,1,1\nevidence_delivered,P002,2,2",
            "evidence_delivered,P002,2,2\nevidence_delivered,P001,1,1",
        ),
        ("paper_counts.csv", "evidence_delivered,P002,2,2", "evidence_delivered,P001,1,1"),
        ("state_counts.csv", "evidence_delivered", "unknown_outcome"),
        ("package_inventory.csv", "R6,2.6.1,NA,project,yes", "R6,2.6.1,NA,project,maybe"),
        ("package_inventory.csv", "KernSmooth", "R6dup"),
    ],
)
def test_csv_contract_violations_are_caught_at_the_csv_boundary(world, name, old, new):
    text = read(world, name).decode()
    assert old in text, (name, old)
    write(world, name, text.replace(old, new, 1 if old != "\n" else -1).encode())
    forged_fails(world, "csv")


def test_the_header_of_a_published_file_must_match_the_schema(world):
    text = (
        read(world, "state_counts.csv")
        .decode()
        .replace("outcome,condition,state,n", "outcome,state,condition,n", 1)
    )
    write(world, "state_counts.csv", text.encode())
    forged_fails(world, "csv")


# ---- binding to the Python preflight and handoff ----------------------------------------------


def second_run_citing(w: World, pre_ref: ArtifactRef):
    """A real R run that cites some other preflight file by its bytes (R only hashes them)."""
    return run_r(w.root, w.handoff, w.csv, pre_ref)


def test_an_unrelated_but_valid_preflight_is_refused_even_though_r_hashed_it(world):
    from test_analysis_handoff import pair, rehearsal
    from test_analysis_handoff import world as other_world

    from biorag_repro.evaluation import analysis_handoff as ah

    _table, _build, handoff = rehearsal(world.root, pair(other_world()))
    _ref, handoff_build = ah.publish_handoff(handoff, world.root)
    other_pre, other_pre_build = publish_preflight(handoff_build, world.root, code=CODE)
    result, record_sha = second_run_citing(world, other_pre)  # R accepts: bytes match its hash
    # Against the preflight R cited: its handoff is not the handoff R analysed.
    with pytest.raises(ar.ResultError) as error:
        verify(
            world, result_dir=result, preflight_build=other_pre_build, run_record_sha256=record_sha
        )
    assert error.value.stage == "preflight"
    # Against the preflight that matches the handoff R analysed: R cited other bytes.
    with pytest.raises(ar.ResultError) as error:
        verify(world, result_dir=result, run_record_sha256=record_sha)
    assert error.value.stage == "preflight"


def test_a_second_valid_preflight_for_the_same_handoff_is_not_the_one_r_cited(world):
    # A fresh preflight of the same handoff has its own nonce, hence different bytes.
    original = read_preflight(world.preflight_build, world.root).preflight
    _ref, other_build = publish_preflight(original.handoff_build, world.root, code=CODE)
    fails(world, "preflight", preflight_build=other_build)


def test_a_missing_preflight_manifest_or_changed_preflight_bytes_are_preflight_failures(world):
    ref = read_manifest_output(world)
    path = world.root / ref.logical_ref
    path.write_bytes(path.read_bytes().replace(b"synthetic_rehearsal", b"verified_official_run", 1))
    fails(world, "preflight")
    (world.root / world.preflight_build.logical_ref).unlink()
    fails(world, "preflight")


def read_manifest_output(w: World) -> ArtifactRef:
    return (
        read_manifest(w.root / w.preflight_build.logical_ref, w.preflight_build.sha256)
        .outputs[0]
        .artifact
    )


def test_a_handoff_or_table_changed_after_the_preflight_is_noticed(world):
    handoff_path = world.root / world.handoff.logical_ref
    handoff_path.write_bytes(handoff_path.read_bytes() + b" ")
    fails(world, "preflight")


@pytest.mark.parametrize(
    "path,value",
    [
        (("inputs", "handoff_sha256"), "4" * 64),
        (("inputs", "csv_sha256"), "5" * 64),
        (("inputs", "python_preflight_sha256"), "6" * 64),
        (("inputs", "python_preflight_bytes"), 1),
    ],
)
def test_facts_cited_by_r_that_differ_from_the_verified_chain_are_refused(world, path, value):
    def change(record):
        holder = record
        for key in path[:-1]:
            holder = holder[key]
        holder[path[-1]] = value

    forged_fails(world, "preflight", change)


# ---- arithmetic, with a hash-and-identity-consistent forger -----------------------------------


@pytest.mark.parametrize(
    "name,old,new",
    [
        ("paper_counts.csv", "evidence_delivered,P003,3,2", "evidence_delivered,P003,3,3"),
        ("paper_counts.csv", "answer_correct,P001,1,1", "answer_correct,P001,1,0"),
        ("state_counts.csv", "JATS,yes,7", "JATS,yes,8"),
        ("state_counts.csv", "JATS,no,1", "JATS,no,-1"),
        ("coprimary_results.csv", "evidence_delivered,9,7,5", "evidence_delivered,9,6,5"),
        ("coprimary_results.csv", "evidence_delivered,9,7,5", "evidence_delivered,10,7,5"),
        ("coprimary_results.csv", "evidence_delivered,9,7,5", "evidence_delivered,9,7,4"),
        ("coprimary_results.csv", ",ok,0.8571428571428571,", ",ok,0.7142857142857143,"),
        (
            "coprimary_results.csv",
            "0.14285714285714285,14.285714285714285",
            "0.14285714285714285,15",
        ),
        ("coprimary_results.csv", "0.14285714285714285", format(0.1428571429, ".17g")),
        ("coprimary_results.csv", "0.22222222222222221,1,1", "0.22222222222222221,2,1"),
        ("coprimary_results.csv", "ok,1,0,0.22222222222222221", "ok,1,0," + format(0.3, ".17g")),
    ],
)
def test_numbers_that_disagree_with_the_python_accounting_fail_the_oracle(world, name, old, new):
    text = read(world, name).decode()
    assert old in text, (name, old)
    write(world, name, text.replace(old, new, 1).encode())
    forged_fails(world, "arithmetic")


def test_last_digit_differences_within_tolerance_are_accepted_larger_ones_are_not(world):
    text = read(world, "coprimary_results.csv").decode()
    one_ulp = "%.17g" % (0.14285714285714285 + 2.8e-17)
    assert one_ulp != "0.14285714285714285"
    write(
        world,
        "coprimary_results.csv",
        text.replace("0.14285714285714285,14", f"{one_ulp},14", 1).encode(),
    )
    assert verify(world, run_record_sha256=forge(world)).claims["arithmetic"][0].n_complete == 7
    far = "%.17g" % (0.14285714285714285 + 1e-9)
    write(
        world,
        "coprimary_results.csv",
        text.replace("0.14285714285714285,14", f"{far},14", 1).encode(),
    )
    forged_fails(world, "arithmetic")


# ---- inference structure ---------------------------------------------------------------------


@pytest.mark.parametrize(
    "old,new",
    [
        ("ok,10000,0,-0.25", "ok,9999,0,-0.25"),
        ("ok,10000,0,-0.25", "single_cluster_degenerate,10000,0,-0.25"),
        ("ok,10000,0,-0.25", "ok,10000,20000,-0.25"),
        (
            "ok,10000,0,-0.25,0.80000000000000004",
            "ok,10000,0," + format(0.9, ".17g") + ",0.80000000000000004",
        ),
        (",ok,1,0,0.22", ",holm_undefined,1,0,0.22"),
        ("ok,NA,0.29160592175990208", "ok,NA,0"),
        ("ok,NA,0.29160592175990208,3.4883720930232562", "ok,NA,0.29160592175990208,0"),
        ("ok,1,0,0.22222222222222221,1,1", "ok,NA,0,0.22222222222222221,1,1"),
    ],
)
def test_inference_status_and_presence_violations_are_caught(world, old, new):
    text = read(world, "coprimary_results.csv").decode()
    assert old in text, old
    write(world, "coprimary_results.csv", text.replace(old, new, 1).encode())
    forged_fails(world, "inference")


def test_statistical_values_are_reported_by_r_and_are_not_recomputed(world):
    """The stated limit of this verifier: a p-value that is merely consistent passes.

    Only structure and relations Python can establish are checked (ranges, presence, Holm order,
    t times its standard error). The record therefore labels these values as R's own.
    """
    edit_text(
        world, "coprimary_results.csv", "0.65339209904915796,ok,1", format(0.7, ".17g") + ",ok,1"
    )
    claims = verify(world, run_record_sha256=forge(world)).claims
    assert claims["inference"][0].test_status == "ok"
    assert {name: status for name, status, _ in ar.CHECKS}["statistical_values"] == "reported_by_r"


UNDEFINED_ROW = "inference_undefined,{reason},NA,NA,NA,NA,holm_undefined,NA"


def undefine_first_outcome(w: World, reason: str) -> None:
    """Rewrite both rows so the family is consistently undefined, with the first test undefined for `reason`."""
    lines = read(w, "coprimary_results.csv").decode().splitlines()
    for index in (1, 2):
        cells = lines[index].split(",")
        head = cells[:15]  # through boot_ci_upper
        if index == 1:
            tail = UNDEFINED_ROW.format(reason=reason).split(",")
        else:
            tail = [*cells[15:21], "holm_undefined", "NA"]
        lines[index] = ",".join([*head, *tail, *cells[23:]])
    write(w, "coprimary_results.csv", ("\n".join(lines) + "\n").encode())


def test_an_undefined_test_with_a_numerical_reason_is_accepted_and_stays_undefined(world):
    undefine_first_outcome(world, "nonfinite_result")
    verified = verify(world, run_record_sha256=forge(world))
    assert verified.claims["inference"][0].test_status == "inference_undefined"
    assert verified.claims["inference"][0].test_reason == "nonfinite_result"
    assert verified.claims["inference"][0].holm_status == "holm_undefined"


def test_an_undefined_test_claiming_a_structural_cause_that_the_data_lacks_is_refused(world):
    undefine_first_outcome(world, "zero_variance")
    forged_fails(world, "inference")


def test_an_undefined_family_must_not_carry_an_adjusted_p_value(world):
    undefine_first_outcome(world, "nonfinite_result")
    edit_text(
        world, "coprimary_results.csv", "holm_undefined,NA,0,0.2222", "holm_undefined,0.5,0,0.2222"
    )
    forged_fails(world, "inference")


# ---- environment -------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "name,old,new",
    [
        ("package_inventory.csv", "clubSandwich,0.7.0", "clubSandwich,0.6.0"),
        ("package_inventory.csv", "R6,2.6.1,NA,project,yes", "R6,2.6.1,NA,project,no"),
    ],
)
def test_an_inventory_that_contradicts_the_recorded_lock_is_an_environment_failure(
    world, name, old, new
):
    edit_text(world, name, old, new)
    forged_fails(world, "environment")


@pytest.mark.parametrize(
    "change",
    [
        lambda r: r["environment_intended"]["lock_packages"].pop("clubSandwich"),
        lambda r: r["environment_intended"]["lock_packages"].update(R6="9.9.9"),
        lambda r: r["environment_actual"]["package_versions_used"].update(digest="0.0.1"),
        lambda r: r["environment_actual"]["package_versions_used"].pop("clubSandwich"),
        lambda r: r["environment_actual"].update(r_version="R version 4.5.0 (2025-04-11)"),
    ],
)
def test_a_record_that_contradicts_its_own_inventory_is_an_environment_failure(world, change):
    forged_fails(world, "environment", change)


def test_a_different_lock_hash_or_commit_is_recorded_not_rejected(world):
    """Historical results stay valid: the recorded lock and commit are assertions, not tests of today."""

    def change(record):
        record["environment_intended"]["renv_lock_sha256"] = "7" * 64
        record["code"]["git_commit"] = "8" * 40

    claims = verify(world, run_record_sha256=forge(world, change)).claims
    assert claims["r_dependency_lock_sha256"] == "7" * 64
    assert claims["recorded_by_r"].code.git_commit == "8" * 40


# ---- the result record and its publication -----------------------------------------------------


def published_dump(world) -> dict:
    published = publish(world)
    return json.loads((world.root / published.ref.logical_ref).read_text())


@pytest.mark.parametrize(
    "path,value",
    [
        (("purpose",), "official_run"),
        (("input_origin",), "verified_official_run"),
        (("scientific_authority",), "confirmed"),
        (("rerun",), "verified"),
        (("applied_bootstrap_seed",), 1),
        (("run_record_sha256",), "9" * 64),
        (("installed_package_inventory_sha256",), "9" * 64),
        (("r_dependency_lock_sha256",), "9" * 64),
        (("not_established",), []),
        (("result_dir",), "../elsewhere"),
        (("method", "bootstrap_resamples"), 1000),
        (("verifier", "tool"), "other"),
    ],
)
def test_the_record_model_refuses_stronger_or_inconsistent_claims(world, path, value):
    dump = published_dump(world)
    holder = dump
    for key in path[:-1]:
        holder = holder[key]
    holder[path[-1]] = value
    with pytest.raises(ValueError):
        ar.AnalysisResultRecord.model_validate(dump, strict=True)


def test_every_check_status_is_pinned(world):
    dump = published_dump(world)
    for index, check in enumerate(dump["checks"]):
        forged = json.loads(json.dumps(dump))
        forged["checks"][index]["status"] = (
            "reported_by_r" if check["status"] == "verified_by_python" else "verified_by_python"
        )
        with pytest.raises(ValueError):
            ar.AnalysisResultRecord.model_validate(forged, strict=True)


def test_a_self_consistent_but_untrue_stored_record_fails_the_read_back(world):
    published = publish(world)
    dump = json.loads((world.root / published.ref.logical_ref).read_text())
    dump["method"]["package_version"] = "9.9.9"
    forged = ar.AnalysisResultRecord.model_validate(dump, strict=True)
    stored = ar._store(forged, world.root)
    with pytest.raises(ar.ResultError) as error:
        ar.read_result_record(stored.build, world.root)
    assert error.value.stage == "read_back" and "method" in str(error.value)


def test_changed_record_bytes_after_publication_are_a_hash_failure(world):
    published = publish(world)
    path = world.root / published.ref.logical_ref
    path.chmod(0o644)
    path.write_bytes(path.read_bytes().replace(b'"not_performed"', b'"byte_identical"', 1))
    with pytest.raises(HashMismatchError):
        ar.read_result_record(published.build, world.root)


def test_r_files_tampered_after_publication_are_noticed_on_read_back(world):
    published = publish(world)
    edit_text(
        world, "paper_counts.csv", "evidence_delivered,P001,1,1", "evidence_delivered,P001,1,0"
    )
    with pytest.raises(ar.ResultError) as error:
        ar.read_result_record(published.build, world.root)
    assert error.value.stage == "output_hashes"


def test_the_whole_r_directory_replaced_after_publication_is_noticed(world):
    published = publish(world)
    for name in ar.RESULT_FILES:
        (directory(world) / name).unlink()
    for name in ar.RESULT_FILES:
        write(world, name, b"x\n")
    with pytest.raises(ar.ResultError):
        ar.read_result_record(published.build, world.root)


def test_a_conflicting_record_at_the_same_identity_is_never_replaced(world):
    published = publish(world)
    path = world.root / published.ref.logical_ref
    path.chmod(0o644)
    altered = path.read_bytes() + b" "
    path.write_bytes(altered)
    with pytest.raises(ar.ResultError) as error:
        publish(world)
    assert error.value.stage == "publication"
    assert path.read_bytes() == altered


def test_a_conflicting_build_manifest_is_never_replaced(world):
    published = publish(world)
    manifest = world.root / published.build.logical_ref
    manifest.chmod(0o644)
    altered = manifest.read_bytes() + b" "
    manifest.write_bytes(altered)
    with pytest.raises(ar.ResultError) as error:
        publish(world)
    assert error.value.stage == "publication"
    assert manifest.read_bytes() == altered


def test_a_record_file_without_its_manifest_is_completed_by_the_next_publication(world):
    first = publish(world)
    (
        world.root / first.build.logical_ref
    ).unlink()  # simulates an interruption between the two files
    again = publish(world)
    assert (again.ref, again.build) == (first.ref, first.build) and again.created
    assert ar.read_result_record(again.build, world.root).record.scientific_result_sha256 == GOLDEN


def test_nothing_is_published_when_any_verification_fails(world):
    edit_text(
        world, "paper_counts.csv", "evidence_delivered,P001,1,1", "evidence_delivered,P001,1,0"
    )
    with pytest.raises(ar.ResultError):
        publish(world)
    assert not (world.root / "analysis" / "result").exists()


def test_the_published_record_does_not_embed_machine_paths_or_the_host_name(world):
    published = publish(world)
    text = (world.root / published.ref.logical_ref).read_text()
    assert str(world.root) not in text
    assert load_record(world)["execution"]["hostname"] not in text


# ---- historical records stay verifiable ------------------------------------------------------


def test_the_current_checkout_is_compared_but_never_decides_validity(world, tmp_path):
    published = publish(world)
    repository = tmp_path / "other_checkout"
    (repository / "analysis" / "R").mkdir(parents=True)
    (repository / "analysis" / "renv.lock").write_text("{}")
    (repository / "analysis" / "run_analysis.R").write_text("# different\n")
    git = ["git", "-C", str(repository), "-c", "user.name=t", "-c", "user.email=t@example.org"]
    subprocess.run([*git, "init", "-q"], check=True)
    subprocess.run([*git, "add", "-A"], check=True)
    subprocess.run([*git, "commit", "-q", "-m", "x"], check=True)
    checked = ar.read_result_record(published.build, world.root)
    diagnostic = ar.compare_current_environment(checked.record, repository)
    assert diagnostic == {
        "git_commit": "different",
        "renv_lock": "different",
        "r_scripts": "different",
    }
    assert ar.read_result_record(published.build, world.root).record == checked.record
    empty = tmp_path / "no_checkout"
    empty.mkdir()
    assert set(ar.compare_current_environment(checked.record, empty).values()) == {"unavailable"}


def test_a_seed_other_than_the_rehearsal_seed_is_a_different_result_not_an_error(world):
    result, record_sha256 = run_r(world.root, world.handoff, world.csv, world.preflight, seed="7")
    verified = verify(world, result_dir=result, run_record_sha256=record_sha256)
    assert verified.claims["scientific_result_sha256"] != GOLDEN
    assert verified.claims["applied_bootstrap_seed"] == 7


def test_a_store_that_wrote_other_content_is_caught_by_the_read_back(world, monkeypatch):
    real_store = ar._store

    def store_with_other_content(record, root):
        other = record.model_copy(
            update={"method": record.method.model_copy(update={"package_version": "9.9.9"})}
        )
        return real_store(other, root)

    monkeypatch.setattr(ar, "_store", store_with_other_content)
    with pytest.raises(ar.ResultError) as error:
        publish(world)
    assert error.value.stage == "read_back" and "method" in str(error.value)


def test_a_matching_checkout_is_reported_as_the_same(world, tmp_path):
    published = publish(world)
    record = ar.read_result_record(published.build, world.root).record
    repository = tmp_path / "checkout"
    (repository / "analysis" / "R").mkdir(parents=True)
    for source in (ANALYSIS_DIR / "R").glob("*.R"):
        shutil.copy(source, repository / "analysis" / "R" / source.name)
    shutil.copy(ANALYSIS_DIR / "run_analysis.R", repository / "analysis" / "run_analysis.R")
    lock = repository / "analysis" / "renv.lock"
    lock.write_text("{}\n")
    git = ["git", "-C", str(repository), "-c", "user.name=t", "-c", "user.email=t@example.org"]
    subprocess.run([*git, "init", "-q"], check=True)
    subprocess.run([*git, "add", "-A"], check=True)
    subprocess.run([*git, "commit", "-q", "-m", "x"], check=True)
    head = subprocess.run(
        [*git, "rev-parse", "HEAD"], check=True, capture_output=True, text=True
    ).stdout.strip()
    code = record.recorded_by_r.code.model_copy(update={"git_commit": head})
    recorded = record.model_copy(
        update={
            "r_dependency_lock_sha256": sha256_bytes(lock.read_bytes()),
            "recorded_by_r": record.recorded_by_r.model_copy(update={"code": code}),
        }
    )
    assert ar.compare_current_environment(recorded, repository) == {
        "git_commit": "same",
        "renv_lock": "same",
        "r_scripts": "same",
    }


def test_the_size_limit_is_exact_and_refuses_an_oversized_file(world, monkeypatch):
    largest = max(len(read(world, name)) for name in ar.RESULT_FILES)
    monkeypatch.setattr(ar, "MAX_FILE_BYTES", largest)
    assert verify(world).claims["scientific_result_sha256"] == GOLDEN
    monkeypatch.setattr(ar, "MAX_FILE_BYTES", largest - 1)
    fails(world, "file_set")


# ---- guards found missing by mutation testing -------------------------------------------------


def test_a_race_that_swaps_a_link_in_after_the_type_check_is_still_refused(
    world, tmp_path, monkeypatch
):
    """The lstat says 'regular'; the no-follow open must still refuse the link that is there now."""
    outside = tmp_path / "same_bytes.csv"
    outside.write_bytes(read(world, "paper_counts.csv"))
    (directory(world) / "paper_counts.csv").unlink()
    (directory(world) / "paper_counts.csv").symlink_to(outside)
    real_stat = os.stat

    def lying_stat(path, *, dir_fd=None, follow_symlinks=True):
        if dir_fd is not None and path == "paper_counts.csv":
            return real_stat(outside)
        return real_stat(path, dir_fd=dir_fd, follow_symlinks=follow_symlinks)

    monkeypatch.setattr(ar.os, "stat", lying_stat)
    error = fails(world, "file_set")
    assert "cannot be opened as a regular file" in str(error)


def test_a_fifo_swapped_in_after_the_type_check_is_refused_by_the_descriptor_check(
    world, monkeypatch
):
    target = directory(world) / "state_counts.csv"
    target.unlink()
    os.mkfifo(target)
    real_stat = os.stat

    def lying_stat(path, *, dir_fd=None, follow_symlinks=True):
        if dir_fd is not None and path == "state_counts.csv":
            return real_stat(directory(world) / "paper_counts.csv")
        return real_stat(path, dir_fd=dir_fd, follow_symlinks=follow_symlinks)

    monkeypatch.setattr(ar.os, "stat", lying_stat)
    error = fails(world, "file_set")
    assert "not a regular file of acceptable size" in str(error)


def test_the_size_guard_before_and_the_read_cap_after_reading_both_hold(world, monkeypatch):
    monkeypatch.setattr(ar, "MAX_FILE_BYTES", 100)
    assert "acceptable size" in str(fails(world, "file_set"))
    real_fstat = os.fstat

    def understated(descriptor):
        info = real_fstat(descriptor)
        return type("Info", (), {"st_mode": info.st_mode, "st_size": 0})()

    monkeypatch.setattr(ar.os, "fstat", understated)  # a file that grew after it was measured
    assert "larger than" in str(fails(world, "file_set"))


@pytest.mark.parametrize("bad", ["a\\b", "a/./b", "../x", "", "/etc"])
def test_unsafe_paths_are_refused_by_the_path_check_itself(world, bad):
    assert "plain relative path" in str(fails(world, "result_location", result_dir=bad))


@pytest.mark.parametrize(
    "old,new",
    [
        ("evidence_delivered,9,7,5", "evidence_delivered,9,10,5"),  # more complete cases than items
        ("evidence_delivered,9,7,5", "evidence_delivered,9,7,8"),  # more papers than complete items
        (",0.80000000000000004,ok,NA,", ",0.80000000000000004,fine,NA,"),  # test status
        (",ok,1,0,0.22222222222222221", ",maybe,1,0,0.22222222222222221"),  # Holm status
    ],
)
def test_ranges_and_vocabularies_in_the_main_table_are_checked_at_the_csv_boundary(world, old, new):
    text = read(world, "coprimary_results.csv").decode()
    assert old in text, old
    write(world, "coprimary_results.csv", text.replace(old, new, 1).encode())
    forged_fails(world, "csv")


@pytest.mark.parametrize(
    "old,new",
    [
        # t no longer equals the risk difference over its standard error
        ("3.4883720930232562,0.48989794855663582", "3.4883720930232562," + format(0.6, ".17g")),
        # an adjusted p-value below its raw p-value
        ("0.65339209904915796,ok,1,0", "0.65339209904915796,ok,0.5,0"),
    ],
)
def test_relations_between_reported_statistics_are_checked_without_recomputing_them(
    world, old, new
):
    text = read(world, "coprimary_results.csv").decode()
    assert old in text, old
    write(world, "coprimary_results.csv", text.replace(old, new, 1).encode())
    forged_fails(world, "inference")


def test_an_unlocked_package_in_the_project_library_is_an_environment_failure(world):
    edit_text(
        world,
        "package_inventory.csv",
        "KernSmooth,2.23-26,recommended,r_other,no",
        "KernSmooth,2.23-26,recommended,project,no",
    )
    forged_fails(world, "environment")


def inference_row(**changes):
    row = {
        "outcome": "x",
        "boot_status": "ok",
        "boot_n_resamples": 10000,
        "boot_n_empty": 0,
        "boot_ci_lower": -0.25,
        "boot_ci_upper": 0.8,
        "test_status": "ok",
        "test_reason": None,
        "test_se": 0.3,
        "test_df": 3.5,
        "test_t": 0.5,
        "test_p": 0.6,
        "rd": 0.15,
    }
    return row | changes


UNDEFINED = {
    "test_status": "inference_undefined",
    "test_se": None,
    "test_df": None,
    "test_t": None,
    "test_p": None,
}


def structure_fails(row, n, papers, distinct):
    with pytest.raises(ar.ResultError) as error:
        ar._inference_structure(row, n, papers, distinct)
    assert error.value.stage == "inference"


def test_structural_inference_rules_follow_the_complete_cases():
    ar._inference_structure(inference_row(), 7, 5, {-1, 0, 1})
    # One observed paper: the interval is a mechanical artefact and the test cannot be run.
    single = {
        "boot_status": "single_cluster_degenerate",
        "test_reason": "single_cluster",
        **UNDEFINED,
    }
    ar._inference_structure(inference_row(**single), 3, 1, {0, 1})
    structure_fails(inference_row(**single | {"boot_status": "ok"}), 3, 1, {0, 1})
    structure_fails(inference_row(), 3, 1, {0, 1})  # a defined test with one paper
    structure_fails(inference_row(boot_status="single_cluster_degenerate"), 7, 2, {-1, 0, 1})
    # Identical differences: no variance, so the test is undefined for exactly that reason.
    zero = {"test_reason": "zero_variance", **UNDEFINED}
    ar._inference_structure(inference_row(**zero), 4, 2, {1})
    structure_fails(inference_row(), 4, 2, {1})
    structure_fails(inference_row(**zero | {"test_reason": "nonfinite_result"}), 4, 2, {1})
    # No complete pair: nothing was resampled and there is no interval.
    none = {
        "boot_status": "no_complete_pairs",
        "boot_n_resamples": 0,
        "boot_ci_lower": None,
        "boot_ci_upper": None,
        "test_reason": "no_complete_pairs",
        **UNDEFINED,
    }
    ar._inference_structure(inference_row(**none), 0, 0, set())
    structure_fails(inference_row(**none | {"boot_n_resamples": 10000}), 0, 0, set())
    structure_fails(
        inference_row(**none | {"boot_ci_lower": 0.0, "boot_ci_upper": 0.1}), 0, 0, set()
    )


def test_a_defined_test_needs_a_consistent_standard_error():
    structure_fails(inference_row(test_se=0.0), 7, 5, {-1, 0, 1})
    structure_fails(inference_row(test_df=0.0), 7, 5, {-1, 0, 1})
    structure_fails(inference_row(test_t=2.0), 7, 5, {-1, 0, 1})
    structure_fails(inference_row(boot_ci_lower=0.9), 7, 5, {-1, 0, 1})


def test_a_rerun_with_a_verdict_but_a_failing_exit_is_an_error(world):
    done = subprocess.CompletedProcess([], 1, "re-execution: byte_identical\n", "error: crashed")
    with pytest.raises(ar.ResultError) as error:
        ar.rerun_in_r(verify(world), world.root, ANALYSIS_DIR, run=lambda *a, **k: done)
    assert error.value.stage == "rerun"


def test_the_record_model_requires_its_files_in_the_fixed_order(world):
    dump = published_dump(world)
    dump["r_files"][0], dump["r_files"][1] = dump["r_files"][1], dump["r_files"][0]
    with pytest.raises(ValueError, match="five result files in order"):
        ar.AnalysisResultRecord.model_validate(dump, strict=True)


# ---- a build manifest that does not belong to this kind of record ------------------------------


def forged_manifest(w: World, published, name: str, **overrides) -> ArtifactRef:
    """A valid, lineage-consistent manifest for the published record with one property changed."""
    from biorag_repro.core.identity import ArtifactInput, derived_artifact_id
    from biorag_repro.core.manifest import BuildManifest, DerivedOutput
    from biorag_repro.core.storage import publish_manifest

    original = read_manifest(w.root / published.build.logical_ref, published.build.sha256)
    settings = overrides.get("settings", original.transformation_settings_sha256)
    kind = overrides.get("kind", "analysis_result_record")
    transformation = overrides.get("transformation", original.transformation_id)
    output = original.outputs[0].artifact
    identity = derived_artifact_id(
        artifact_kind=kind,
        inputs=[
            ArtifactInput(i.role, i.artifact.artifact_id, i.artifact.sha256)
            for i in original.inputs
        ],
        transformation_id=transformation,
        transformation_settings_sha256=settings,
        output_sha256=output.sha256,
        output_format="json",
        serialization_version="1",
    )
    ref = ArtifactRef(artifact_id=identity, logical_ref=output.logical_ref, sha256=output.sha256)
    manifest = BuildManifest(
        schema_version="1",
        inputs=original.inputs,
        transformation_id=transformation,
        transformation_settings_sha256=settings,
        outputs=[
            DerivedOutput(
                artifact=ref,
                artifact_kind=kind,
                output_format="json",
                serialization_version="1",
                identity_version="1",
            )
        ],
        status="complete",
    )
    path = f"analysis/result/{name}.build.json"
    digest = publish_manifest(manifest, w.root / path, {ref.logical_ref: w.root / ref.logical_ref})
    return ArtifactRef(artifact_id="build_" + digest, logical_ref=path, sha256=digest)


@pytest.mark.parametrize(
    "overrides,message",
    [
        ({"kind": "analysis_preflight"}, "does not describe an analysis result record"),
        ({"settings": "1" * 64}, "other settings"),
        ({"transformation": "analysis_result_v0"}, "other settings"),
    ],
)
def test_a_manifest_of_another_kind_or_settings_is_not_a_result_record(world, overrides, message):
    published = publish(world)
    forged = forged_manifest(world, published, "forged", **overrides)
    with pytest.raises(ar.ResultError) as error:
        ar.read_result_record(forged, world.root)
    assert error.value.stage == "read_back" and message in str(error.value)


def test_a_record_whose_lineage_differs_from_its_manifest_is_refused(world, monkeypatch):
    published = publish(world)
    record = ar.read_result_record(published.build, world.root).record
    original = read_preflight(world.preflight_build, world.root).preflight
    _ref, other_build = publish_preflight(original.handoff_build, world.root, code=CODE)
    forged = record.model_copy(update={"preflight_build": other_build})
    inputs = ar._inputs(record)
    with monkeypatch.context() as patch:
        patch.setattr(ar, "_inputs", lambda _record: inputs)  # the manifest keeps the real lineage
        stored = ar._store(forged, world.root)
    with pytest.raises(ar.ResultError) as error:
        ar.read_result_record(stored.build, world.root)
    assert error.value.stage == "read_back" and "lineage" in str(error.value)
