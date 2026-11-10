"""Conservative rejection checks for reviewer-attested answer-surface wording.

These checks only reject. Passing them does not establish that two wordings state the same
scientific fact; that remains the reviewer's attested judgment. Closed word lists, no
similarity scoring.
"""

import re

from biorag_repro.evaluation.models import AnswerSurface

_NUMBER = re.compile(r"(?:(?<![\w.])[-−])?\d+(?:,\d{3})*(?:\.\d+)?(?:[eE][-−+]?\d+)?")
_COMPARATOR = re.compile(
    r"<=|>=|≤|≥|<|>|=|\b(?:less than|greater than|more than|fewer than|at least|at most"
    r"|above|below|over|under)\b",
    re.IGNORECASE,
)
_UNIT = re.compile(
    r"\s*(%|fold|mg|kg|ml|mm|cm|kb|bp|years?|months?|days?|hours?|weeks?)(?![A-Za-z])",
    re.IGNORECASE,
)
_HIGHER = {
    "positive", "positively", "increase", "increased", "increases", "increasing", "higher",
    "high", "better", "improved", "improves", "gain", "gains", "up", "upregulated",
}  # fmt: skip
_LOWER = {
    "negative", "negatively", "decrease", "decreased", "decreases", "decreasing", "lower",
    "low", "worse", "worsened", "loss", "down", "downregulated", "reduced",
}  # fmt: skip
_NEGATION = {"not", "no", "without", "never", "non", "neither", "nor", "cannot"}
_STOP = {"with", "the", "and", "for", "from", "than", "that", "this", "are", "was", "were", "has"}


def _numbers(text: str) -> list[tuple[str, int]]:
    """Signed numeric tokens (digit-group commas removed) with the offset where each ends."""
    return [(m.group().replace("−", "-").replace(",", ""), m.end()) for m in _NUMBER.finditer(text)]


def _unit(text: str, end: int) -> str | None:
    found = _UNIT.match(text, end)
    return found.group(1).lower().rstrip("s") if found else None


def _words(text: str) -> list[str]:
    return re.findall(r"[a-z]+", text.lower())


def _direction(words: list[str]) -> tuple[bool, bool]:
    return any(w in _HIGHER for w in words), any(w in _LOWER for w in words)


def surface_problem(answer: str, surface: AnswerSurface, siblings: list[str]) -> str | None:
    """First reason the surface cannot be the benchmark answer, or None if no check objects.

    `siblings` are the benchmark texts of the other components of the same placement; they carry
    any condition or qualifier of the answer that the surface wording leaves out.
    """
    canonical = surface.canonical_text
    wanted, seen = _numbers(answer), _numbers(canonical)
    position, used = 0, []
    for value, end in wanted:
        match = next((k for k in range(position, len(seen)) if seen[k][0] == value), None)
        if match is None:
            return f"number {value} is not stated with the same value and sign, in order"
        used.append((end, match))
        position = match + 1
    for k, (value, _) in enumerate(seen):
        if k not in {m for _, m in used} and value.lstrip("-") not in surface.note:
            return f"extra number {value} is not explained in the reviewer note"
    for end, match in used:
        if _unit(answer, end) != _unit(canonical, seen[match][1]):
            return "unit differs from the benchmark answer"
    if [c.lower() for c in _COMPARATOR.findall(answer)] != [
        c.lower() for c in _COMPARATOR.findall(canonical)
    ]:
        return "comparator differs from the benchmark answer"
    given, shown = _words(answer), _words(canonical)
    if _direction(given) != _direction(shown):
        return "direction differs from the benchmark answer"
    if any(w in _NEGATION for w in given) != any(w in _NEGATION for w in shown):
        return "negation differs from the benchmark answer"
    carried = shown + [w for s in siblings for w in _words(s)]
    for word in given:
        if len(word) > 2 and word not in _STOP and not any(c.startswith(word) for c in carried):
            return (
                f"answer qualifier {word!r} is carried by neither the surface nor another component"
            )
    return None
