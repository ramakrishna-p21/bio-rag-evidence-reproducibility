# ruff: noqa: F811
"""The rehearsal-only handoff check of biorag-analysis.

It checks stored bytes, hashes, lineage and the published table of a *rehearsal* handoff. It can
never establish official provenance, and the tests pin that it says so and refuses to pass an
official handoff.
"""

import pytest
from test_analysis_handoff import CODE, official_table, pair, publish, rehearsal, world
from test_generation_official import recognised, view  # noqa: F401  (fixtures)
from test_generation_prompt import needs_tokenizer

from biorag_repro.evaluation import analysis_handoff as ah
from biorag_repro.evaluation.analysis_cli import main


def handoff_args(root, built, **extra):
    args = [
        "verify-handoff",
        "--root",
        str(root),
        "--handoff-build",
        built.logical_ref,
        "--handoff-sha256",
        built.sha256,
    ]
    return args + [a for k, v in extra.items() for a in (f"--{k.replace('_', '-')}", v)]


def published_rehearsal(tmp_path):
    table, _, handoff = rehearsal(tmp_path, pair(world()))
    ref, built = ah.publish_handoff(handoff, tmp_path)
    return table, handoff, ref, built


def test_a_valid_rehearsal_handoff_passes_and_says_what_it_does_not_prove(tmp_path, capsys):
    table, handoff, _, built = published_rehearsal(tmp_path)
    assert main(handoff_args(tmp_path, built)) == 0
    out = capsys.readouterr().out
    assert "synthetic rehearsal handoff" in out
    assert handoff.csv.sha256 in out and f"{len(table.rows)} rows" in out
    assert "cannot establish official" in out and "no scientific authority" in out
    assert "verified_official_run" not in out


def test_a_tampered_handoff_or_table_or_reference_fails(tmp_path, capsys):
    _, handoff, ref, built = published_rehearsal(tmp_path)
    good = handoff_args(tmp_path, built)
    wrong = handoff_args(tmp_path, built.model_copy(update={"sha256": "0" * 64}))
    assert main(wrong) == 1
    csv = tmp_path / handoff.csv.logical_ref
    csv.chmod(0o644)
    csv.write_bytes(csv.read_bytes().replace(b"TRUE", b"FALSE", 1))
    assert main(good) == 1
    assert "not valid" in capsys.readouterr().err
    stored = tmp_path / ref.logical_ref
    stored.chmod(0o644)
    stored.write_bytes(
        stored.read_bytes().replace(b"synthetic_rehearsal", b"verified_official_run")
    )
    assert main(good) == 1


def test_a_table_build_reference_is_not_accepted_as_a_handoff(tmp_path, capsys):
    _, build = publish(tmp_path, pair(world()))
    assert main(handoff_args(tmp_path, build)) == 1
    assert "does not describe an analysis handoff" in capsys.readouterr().err


def test_the_two_operations_take_their_own_references(tmp_path):
    _, _, _, built = published_rehearsal(tmp_path)
    with pytest.raises(SystemExit):
        main(["verify", "--root", str(tmp_path), "--handoff-build", built.logical_ref])
    with pytest.raises(SystemExit):
        main(["verify-handoff", "--root", str(tmp_path), "--build", built.logical_ref])


@needs_tokenizer
def test_an_official_handoff_is_not_accepted_by_the_standalone_check(
    tmp_path,
    view,
    tiny,
    recognised,
    capsys,
):
    verified, root, build = official_table(tmp_path, view, tiny)
    handoff = ah.build_handoff(build, root, code=CODE, official_runs=[verified])
    _, built = ah.publish_handoff(handoff, root, official_runs=[verified])
    assert main(handoff_args(root, built)) == 2
    captured = capsys.readouterr()
    assert "valid" not in captured.out
    assert "freshly verified official runs" in captured.err
