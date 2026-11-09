import ast
import inspect
import pathlib

import pytest
from readiness_support import REAL_POLICY, fixture_policy
from readiness_world import build_world

from biorag_repro.readiness import cli
from biorag_repro.readiness.policy import policy_sha256
from biorag_repro.readiness.publish import read_report


@pytest.fixture
def world(tmp_path, monkeypatch):
    return build_world(tmp_path, monkeypatch)


def sha_from(out: str) -> str:
    return next(l.split()[-1] for l in out.splitlines() if l.startswith("report sha256"))


def test_evaluate_publishes_and_exits_2_for_a_verified_not_ready(world, capsys):
    code = cli.run(
        ["evaluate", "--data-root", str(world.data), "--repo", str(world.repo)], _context=world.ctx
    )
    out = capsys.readouterr().out
    assert code == 2 and "decision NOT_READY" in out
    assert "G13_final_test_history_integrity: unavailable" in out
    sha = sha_from(out)
    assert cli.run(["verify", "--data-root", str(world.data), "--report-sha256", sha]) == 2


def test_main_is_bound_to_the_real_policy(world, monkeypatch, capsys):
    monkeypatch.setattr(cli, "system_context", lambda repo, data: world.ctx)
    assert cli.main(["evaluate", "--data-root", str(world.data), "--repo", str(world.repo)]) == 2
    sha = sha_from(capsys.readouterr().out)
    report = read_report(world.data, sha, REAL_POLICY).report
    assert report.policy_sha256 == policy_sha256(REAL_POLICY)


def test_verify_prints_current_comparison_without_changing_the_exit(world, capsys):
    cli.run(["evaluate", "--data-root", str(world.data)], _context=world.ctx)
    sha = sha_from(capsys.readouterr().out)
    code = cli.main(
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
    out = capsys.readouterr().out
    assert code == 2 and "current head: same" in out and "never retained" in out


def test_a_report_made_under_a_fixture_policy_is_invalid_for_the_production_command(world, capsys):
    cli.run(
        ["evaluate", "--data-root", str(world.data)], _policy=fixture_policy(), _context=world.ctx
    )
    sha = sha_from(capsys.readouterr().out)
    assert cli.main(["verify", "--data-root", str(world.data), "--report-sha256", sha]) == 1
    assert "different policy" in capsys.readouterr().err


def test_unknown_missing_and_malformed_reports_exit_1(world, capsys):
    assert cli.main(["verify", "--data-root", str(world.data), "--report-sha256", "0" * 64]) == 1
    assert (
        cli.main(["verify", "--data-root", str(world.data / "nope"), "--report-sha256", "x"]) == 1
    )
    capsys.readouterr()


def test_evaluation_error_exits_1_and_publishes_nothing(world, monkeypatch, capsys):
    from biorag_repro.readiness import evidence

    monkeypatch.setattr(
        evidence, "restart_recovery", lambda: (_ for _ in ()).throw(TypeError("bug"))
    )
    assert cli.run(["evaluate", "--data-root", str(world.data)], _context=world.ctx) == 1
    assert not (world.data / "readiness").exists()
    assert "readiness failed" in capsys.readouterr().err


def test_no_flag_or_environment_selects_a_policy_or_skips_checks():
    parser = cli._parser()
    parsers = [parser, *parser._subparsers._group_actions[0].choices.values()]  # type: ignore[union-attr]
    options = {o for p in parsers for a in p._actions for o in a.option_strings}
    assert options == {"-h", "--help", "--data-root", "--repo", "--report-sha256"}
    assert list(inspect.signature(cli.main).parameters) == ["argv"]
    keyword_only = [
        n for n, p in inspect.signature(cli.run).parameters.items() if p.kind is p.KEYWORD_ONLY
    ]
    assert keyword_only == ["_policy", "_context"]


def test_the_package_reads_no_environment_and_main_passes_no_policy():
    package = pathlib.Path(cli.__file__).parent
    for path in package.glob("*.py"):
        text = path.read_text()
        assert "os.environ" not in text and "getenv" not in text, path.name
    tree = ast.parse((package / "cli.py").read_text())
    main = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "main")
    assert ast.unparse(main.body[0]) == "return run(argv)"


def test_a_second_evaluation_is_a_new_report_and_the_first_stays_valid(world, capsys):
    cli.run(["evaluate", "--data-root", str(world.data)], _context=world.ctx)
    first = sha_from(capsys.readouterr().out)
    cli.run(["evaluate", "--data-root", str(world.data)], _context=world.ctx)
    second = sha_from(capsys.readouterr().out)
    assert first != second
    for sha in (first, second):
        assert cli.main(["verify", "--data-root", str(world.data), "--report-sha256", sha]) == 2
