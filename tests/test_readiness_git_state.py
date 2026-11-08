import subprocess

import pytest

from biorag_repro.core.hashing import sha256_bytes
from biorag_repro.readiness.git_state import GitUnavailable, probe


def git(repo, *args):
    subprocess.run(
        ["git", "-c", "user.name=t", "-c", "user.email=t@example.org", *args],
        cwd=repo,
        check=True,
        capture_output=True,
    )


@pytest.fixture
def repo(tmp_path):
    git(tmp_path, "init", "-q")
    for name, text in {
        "src/a.py": "a = 1\n",
        "configs/c.toml": "x = 1\n",
        "analysis/renv.lock": "{}\n",
        "pyproject.toml": "[project]\n",
        "README.md": "readme\n",
        ".gitignore": "out/\n*.log\n",
    }.items():
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
    git(tmp_path, "add", "-A")
    git(tmp_path, "commit", "-q", "-m", "init")
    return tmp_path


def test_clean_scope_lists_tracked_blobs(repo):
    state = probe(repo)
    assert state.modified == [] and state.untracked == []
    paths = [path for _, _, path in state.entries]
    assert paths == sorted(paths)
    assert set(paths) == {"src/a.py", "configs/c.toml", "analysis/renv.lock", "pyproject.toml"}
    blob = subprocess.check_output(["git", "hash-object", "src/a.py"], cwd=repo, text=True).strip()
    assert ("src/a.py" in paths) and any(b == blob for _, b, p in state.entries if p == "src/a.py")
    assert len(state.head) == 40


def test_tracked_modification_is_reported(repo):
    (repo / "src/a.py").write_text("a = 2\n")
    assert probe(repo).modified == ["src/a.py"]


def test_staged_only_change_is_reported(repo):
    (repo / "configs/c.toml").write_text("x = 2\n")
    git(repo, "add", "configs/c.toml")
    assert probe(repo).modified == ["configs/c.toml"]


def test_deleted_and_renamed_tracked_files_are_reported(repo):
    (repo / "src/a.py").unlink()
    git(repo, "mv", "pyproject.toml", "pyproject2.toml")
    state = probe(repo)
    assert "src/a.py" in state.modified and "pyproject.toml" in state.modified


@pytest.mark.parametrize(
    "name", ["src/new.py", "src/with space.py", "src/odd\nname.py", "src/ünï.txt", "src/noext"]
)
def test_untracked_files_in_scope_are_found_whatever_their_name(repo, name):
    (repo / name).write_text("x")
    assert probe(repo).untracked == [name]


def test_untracked_in_new_directory_lists_the_file_not_the_directory(repo):
    (repo / "analysis/R").mkdir()
    (repo / "analysis/R/x.R").write_text("1")
    assert probe(repo).untracked == ["analysis/R/x.R"]


def test_ignored_outputs_do_not_count(repo):
    (repo / "out").mkdir()
    (repo / "out/r.txt").write_text("x")
    (repo / "src/run.log").write_text("x")
    state = probe(repo)
    assert state.untracked == [] and state.modified == []


def test_untracked_outside_scope_is_counted_apart(repo):
    (repo / "notes.txt").write_text("x")
    state = probe(repo)
    assert state.untracked == [] and state.untracked_outside_scope == 1


def test_scope_identity_follows_content_not_commit(repo):
    first = probe(repo)
    (repo / "README.md").write_text("changed\n")
    git(repo, "commit", "-qam", "readme only")
    second = probe(repo)
    assert second.head != first.head and second.entries == first.entries
    (repo / "src/a.py").write_text("a = 3\n")
    git(repo, "commit", "-qam", "code")
    assert probe(repo).entries != first.entries


def test_not_a_repository_is_unavailable(tmp_path):
    with pytest.raises(GitUnavailable):
        probe(tmp_path / "missing")
    plain = tmp_path / "plain"
    plain.mkdir()
    with pytest.raises(GitUnavailable):
        probe(plain)


def test_snapshot_is_canonical(repo):
    state = probe(repo)
    assert sha256_bytes(state.snapshot_bytes()) == sha256_bytes(probe(repo).snapshot_bytes())
