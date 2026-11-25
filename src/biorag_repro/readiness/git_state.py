"""Path-scoped Git state of the scientific source (gate G09).

Only the tracked source that defines the experiment is inspected: a modified, staged, deleted
or renamed tracked file, or an untracked file that Git does not ignore, inside the scope
fails the check. Ignoring is left to Git; no file-name pattern decides what counts. The scope
identity is the list of (mode, blob, path) Git records for the scope, so it follows content
and not the commit: a descendant commit with the same content has the same identity, and a
descendant with other content does not.

One file is checked but not digested: `APPROVAL_FILE`, the approval values. The values include
the slot that holds this identity, so digesting them would make the identity a fixed point of
its own file. The exclusion is that exact path and nothing else under `approvals/`. The file
still has to be tracked and unmodified, and its content is pinned by the policy hash instead.
"""

import re
import subprocess
from dataclasses import dataclass
from pathlib import Path

from biorag_repro.core.hashing import canonical_json_bytes

SCOPE = ("src", "configs", "schemas", "scripts", "analysis", "pyproject.toml", "approvals")
APPROVAL_FILE = "approvals/policy.json"
EXECUTION_DECISION_FILE = "approvals/final_test_execution.json"


class GitUnavailable(RuntimeError):
    """Git could not be run or the directory is not a repository."""


@dataclass(frozen=True)
class GitState:
    head: str
    entries: list[tuple[str, str, str]]  # (mode, blob, path) of tracked scope files, by path
    modified: list[str]
    untracked: list[str]
    untracked_outside_scope: int
    approval_blob: str | None  # Git blob of APPROVAL_FILE; None if Git does not track it

    def snapshot_bytes(self) -> bytes:
        return canonical_json_bytes(
            {
                "kind": "git_scope_state",
                "scope": list(SCOPE),
                "head": self.head,
                "approval_file": APPROVAL_FILE,
                "approval_blob": self.approval_blob,
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
    approval_blob = None
    for record in listing:
        meta, _, path = record.partition("\t")
        mode, blob, stage = meta.split(" ")
        if stage != "0":
            raise GitUnavailable(f"unmerged entry for {path}")
        if path == APPROVAL_FILE:
            approval_blob = blob
        else:
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
        approval_blob=approval_blob,
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


def entries_at_revision(repo: Path, revision: str) -> list[tuple[str, str, str]]:
    """Return the G09 tracked entries at an existing full commit revision."""
    if not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise GitUnavailable("source revision is not a full lowercase Git SHA")
    try:
        resolved = _git(repo, "rev-parse", "--verify", revision + "^{commit}").decode().strip()
    except GitUnavailable as error:
        raise GitUnavailable("source revision is not an existing commit") from error
    if resolved != revision:
        raise GitUnavailable("source revision does not resolve to its exact commit")
    listing = _records(_git(repo, "ls-tree", "-r", "-z", revision, "--", *SCOPE))
    entries: list[tuple[str, str, str]] = []
    for record in listing:
        meta, _, path = record.partition("\t")
        mode, kind, blob = meta.split(" ")
        if kind != "blob":
            raise GitUnavailable(f"non-blob scientific source entry at {path}")
        if path != APPROVAL_FILE:
            entries.append((mode, blob, path))
    return sorted(entries, key=lambda entry: entry[2])


def execution_entries(entries: list[tuple[str, str, str]]) -> list[tuple[str, str, str]]:
    """The code/configuration entries, excluding only the prospective decision record.

    The full list remains the G09 subject. This filtered list is a second identity used to
    show that adding the authorization record did not alter executable scientific source.
    """
    return [entry for entry in entries if entry[2] != EXECUTION_DECISION_FILE]


def verify_execution_revision(repo: Path, state: GitState, revision: str) -> None:
    """Require an ancestor whose execution source is byte-identical to the clean live tree."""
    if not is_ancestor(repo, revision):
        raise GitUnavailable("source revision is not an ancestor of live HEAD")
    previous = execution_entries(entries_at_revision(repo, revision))
    current = execution_entries(state.entries)
    if previous != current:
        raise GitUnavailable("execution source or configuration changed since source revision")
