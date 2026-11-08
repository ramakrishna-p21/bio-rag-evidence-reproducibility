"""Path-scoped Git state of the scientific source (gate G09).

Only the tracked source that defines the experiment is inspected: a modified, staged, deleted
or renamed tracked file, or an untracked file that Git does not ignore, inside the scope
fails the check. Ignoring is left to Git; no file-name pattern decides what counts. The scope
identity is the list of (mode, blob, path) Git records for the scope, so it follows content
and not the commit: a descendant commit with the same content has the same identity, and a
descendant with other content does not.
"""

import subprocess
from dataclasses import dataclass
from pathlib import Path

from biorag_repro.core.hashing import canonical_json_bytes

SCOPE = ("src", "configs", "schemas", "scripts", "analysis", "pyproject.toml")


class GitUnavailable(RuntimeError):
    """Git could not be run or the directory is not a repository."""


@dataclass(frozen=True)
class GitState:
    head: str
    entries: list[tuple[str, str, str]]  # (mode, blob, path) of tracked scope files, by path
    modified: list[str]
    untracked: list[str]
    untracked_outside_scope: int

    def snapshot_bytes(self) -> bytes:
        return canonical_json_bytes(
            {
                "kind": "git_scope_state",
                "scope": list(SCOPE),
                "head": self.head,
                "tracked": [list(e) for e in self.entries],
                "modified": self.modified,
                "untracked": self.untracked,
                "untracked_outside_scope": self.untracked_outside_scope,
            }
        )


def _git(repo: Path, *args: str) -> bytes:
    try:
        done = subprocess.run(
            ["git", "-c", "core.quotepath=off", *args],
            cwd=repo,
            capture_output=True,
            check=True,
            timeout=120,
        )
    except (OSError, subprocess.SubprocessError) as error:
        raise GitUnavailable(f"git {args[0]} failed: {error}") from error
    return done.stdout


def _records(output: bytes) -> list[str]:
    return [r.decode("utf-8", "surrogateescape") for r in output.split(b"\0") if r]


def probe(repo: Path) -> GitState:
    head = _git(repo, "rev-parse", "HEAD").decode().strip()
    listing = _records(_git(repo, "ls-files", "-z", "-s", "--", *SCOPE))
    entries = []
    for record in listing:
        meta, _, path = record.partition("\t")
        mode, blob, stage = meta.split(" ")
        if stage != "0":
            raise GitUnavailable(f"unmerged entry for {path}")
        entries.append((mode, blob, path))
    entries.sort(key=lambda e: e[2])

    modified: set[str] = set()
    untracked: list[str] = []
    outside = 0
    in_scope = set(
        _records(_git(repo, "ls-files", "-z", "--others", "--exclude-standard", "--", *SCOPE))
    )
    status = _records(_git(repo, "status", "--porcelain=v1", "-z", "--untracked-files=all"))
    skip_next = False
    for record in status:
        if skip_next:  # origin path of a rename or copy
            skip_next = False
            modified.add(record)
            continue
        code, path = record[:2], record[3:]
        if code == "??":
            if path in in_scope:
                untracked.append(path)
            else:
                outside += 1
            continue
        if code[0] in "RC" or code[1] in "RC":
            skip_next = True
        if _in_scope(path):
            modified.add(path)
    modified = {p for p in modified if _in_scope(p)}
    return GitState(
        head=head,
        entries=entries,
        modified=sorted(modified),
        untracked=sorted(untracked),
        untracked_outside_scope=outside,
    )


def _in_scope(path: str) -> bool:
    return any(path == s or path.startswith(s + "/") for s in SCOPE)


def is_ancestor(repo: Path, commit: str) -> bool:
    """True if `commit` is in the history of HEAD; False if Git says it is not."""
    try:
        subprocess.run(
            ["git", "merge-base", "--is-ancestor", commit, "HEAD"],
            cwd=repo,
            capture_output=True,
            check=True,
            timeout=60,
        )
    except subprocess.CalledProcessError:
        return False
    except (OSError, subprocess.SubprocessError) as error:
        raise GitUnavailable(f"git merge-base failed: {error}") from error
    return True
