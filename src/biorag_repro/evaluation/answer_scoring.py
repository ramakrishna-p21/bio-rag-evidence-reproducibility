"""Deterministic comparison of one generated answer with benchmark gold.

Normalization (D2) is limited to Unicode NFC and whitespace collapsing. There is
no case folding, fuzzy matching, substring matching or embedding similarity: a
name is correct only if the item approves it. A difference of letter case alone
is routed to adjudication because it can separate real referents (BRCA1/Brca1).

Quantities (D3) are compared as exact rationals. There is no tolerance; a
rounding interval applies only when the gold carries one. Unit conversion uses a
small registry in which every conversion is checked for dimension.

Scoring rules are versioned. "v1" is the published behavior and stays the default.
"v2" additionally reads two numeric spellings and nothing else: English thousands
separators in a complete, strictly grouped number ("1,234.5"), and a plain-text
exponent whose sign directly follows "10" after a multiplication sign ("2.725 x 10-4").
"""

import re
import unicodedata
from dataclasses import dataclass
from decimal import Decimal
from fractions import Fraction

from biorag_repro.benchmark.models import (
    ConclusionGold,
    EntityGold,
    Gold,
    QuantityGold,
    SetGold,
)
from biorag_repro.evaluation.stage_c_models import Status

Answer = str | list[str]
RULES = ("v1", "v2")


@dataclass(frozen=True)
class Verdict:
    status: Status
    reason: str | None
    detail: str
    interpreted: tuple[str, ...] = ()


def _correct(detail: str, interpreted: tuple[str, ...]) -> Verdict:
    return Verdict("correct", None, detail, interpreted)


def _incorrect(reason: str, detail: str, interpreted: tuple[str, ...] = ()) -> Verdict:
    return Verdict("incorrect", reason, detail, interpreted)


def _unresolved(reason: str, detail: str, interpreted: tuple[str, ...] = ()) -> Verdict:
    return Verdict("non_evaluable", reason, detail, interpreted)


def normalize_text(text: str) -> str:
    """NFC composition and whitespace collapsing; nothing else."""
    return " ".join(unicodedata.normalize("NFC", text).split())


# --- quantities ---------------------------------------------------------------

# unit -> (dimension, size in the base unit of that dimension). Seed registry:
# conversions outside it go to adjudication instead of being guessed.
_UNITS: dict[str, tuple[str, Fraction]] = {}


def _register(
    dimension: str, base: str, prefixed: dict[str, int], extra: tuple[tuple[str, int], ...] = ()
) -> None:
    for prefix, power in prefixed.items():
        _UNITS[prefix + base] = (dimension, Fraction(10) ** power)
    for name, size in extra:
        _UNITS[name] = (dimension, Fraction(size))


_register("mass", "g", {"k": 3, "": 0, "m": -3, "μ": -6, "n": -9})
_register("length", "m", {"": 0, "c": -2, "m": -3, "μ": -6, "n": -9})
_register("volume", "L", {"": 0, "m": -3, "μ": -6})
_register("molar", "M", {"": 0, "m": -3, "μ": -6, "n": -9, "p": -12})
_register("time", "s", {"": 0, "m": -3}, extra=(("min", 60), ("h", 3600)))
_register("percent", "%", {"": 0})
for _ascii, _micro in (("ug", "μg"), ("um", "μm"), ("uL", "μL"), ("uM", "μM")):
    _UNITS[_ascii] = _UNITS[_micro]

_OPERATORS = {
    "<": "lt",
    "≤": "le",
    "<=": "le",
    "=": "eq",
    "==": "eq",
    "≥": "ge",
    ">=": "ge",
    ">": "gt",
}
_OP = r"<=|>=|==|≤|≥|<|>|="
_SUPERSCRIPT = str.maketrans("⁰¹²³⁴⁵⁶⁷⁸⁹⁺⁻", "0123456789+-")


def _quantity_pattern(adjacent_exponent: str) -> re.Pattern[str]:
    return re.compile(
        rf"""
        (?:(?P<label>[A-Za-z][A-Za-z0-9_\- ]*?)\s*(?={_OP}))?   # metric label, only before an operator
        (?P<op>{_OP})?\s*
        (?P<sign>[+-])?
        (?P<mantissa>\d+(?:\.\d*)?|\.\d+)
        (?:[eE](?P<e_plain>[+-]?\d+)
          |\s*[×xX·*]\s*10\s*(?:\^\s*(?P<e_caret>[+-]?\d+)|(?P<e_super>[⁺⁻]?[⁰¹²³⁴⁵⁶⁷⁸⁹]+))
          {adjacent_exponent}
        )?
        \s*(?P<unit>[A-Za-zμ%°][A-Za-zμ0-9%°/·^-]*)?
        """,
        re.VERBOSE,
    )


_QUANTITY = _quantity_pattern("")
# v2: the explicit sign must touch the 10 ("10-4", "10+2"); "10 - 4", "104" are not exponents.
_QUANTITY_V2 = _quantity_pattern(r"|\s*[×xX·*]\s*10(?P<e_adj>[+-]\d+)")
# v2: a whole number grouped in threes from a nonzero first group, optional decimal fraction.
# Digits, commas or points on either side (partial matches) and e-notation after it disqualify.
_GROUPED = re.compile(r"(?<![\d.,])[1-9]\d{0,2}(?:,\d{3})+(?:\.\d+)?(?![\d,.]|[eE][+-]?\d)")
_MAX_EXPONENT = 300


def _unit_name(unit: str | None) -> str:
    return normalize_text(unit or "").replace("µ", "μ")  # micro sign -> Greek mu


def _score_quantity(gold: QuantityGold, answer: str, rules: str) -> Verdict:
    text = normalize_text(answer).replace("−", "-")
    pattern = _QUANTITY
    if rules == "v2":
        text = _GROUPED.sub(lambda grouped: grouped.group().replace(",", ""), text)
        pattern = _QUANTITY_V2
    if not text:
        return _incorrect("empty_answer", "blank answer")
    if not re.search(r"\d", text):
        return _incorrect("no_numeric_value", "the answer contains no number")
    m = pattern.fullmatch(text.replace("µ", "μ"))
    if m is None:
        return _unresolved("uninterpretable_numeric_text", f"cannot interpret {text!r} exactly")
    exponent = int(
        m["e_plain"]
        or m["e_caret"]
        or m.groupdict().get("e_adj")
        or (m["e_super"] or "0").translate(_SUPERSCRIPT)
    )
    if abs(exponent) > _MAX_EXPONENT:
        return _unresolved("uninterpretable_numeric_text", "exponent outside the supported range")
    value = Fraction(Decimal(m["mantissa"])) * Fraction(10) ** exponent
    if m["sign"] == "-":
        value = -value
    operator = _OPERATORS[m["op"]] if m["op"] else "eq"
    unit = m["unit"] or ""
    shown = (f"{operator} {m['sign'] or ''}{m['mantissa']}E{exponent} {unit}").strip()
    seen = (shown,)

    if operator != gold.comparator:
        return _incorrect(
            "comparator_mismatch", f"answer is {operator}, gold is {gold.comparator}", seen
        )
    gold_unit = _unit_name(gold.unit)
    if unit == gold_unit:
        answer_size = gold_size = Fraction(1)
    elif not gold_unit:
        return _unresolved(
            "unit_not_in_gold", f"gold has no unit but the answer gives {unit}", seen
        )
    elif not unit:
        return _unresolved(
            "unit_missing", f"gold requires {gold_unit} but the answer has no unit", seen
        )
    elif unit in _UNITS and gold_unit in _UNITS:
        (dim_a, answer_size), (dim_g, gold_size) = _UNITS[unit], _UNITS[gold_unit]
        if dim_a != dim_g:
            return _incorrect(
                "incompatible_units", f"{unit} ({dim_a}) vs {gold_unit} ({dim_g})", seen
            )
    else:
        return _unresolved(
            "unknown_unit_conversion",
            f"no registered conversion between {unit} and {gold_unit}",
            seen,
        )

    in_base = value * answer_size
    exact = [gold.value, *gold.approved_equivalents]
    if any(in_base == Fraction(Decimal(v)) * gold_size for v in exact):
        return _correct("equals the reported value", seen)
    if gold.rounding_interval is not None:
        low, high = (Fraction(Decimal(v)) * gold_size for v in gold.rounding_interval)
        if low <= in_base <= high:
            return _correct("inside the authorized rounding interval", seen)
    return _incorrect("value_mismatch", "value differs from the reported value", seen)


# --- entities and sets --------------------------------------------------------


def _names(canonical: str, aliases: list[str]) -> dict[str, str]:
    return {normalize_text(name): canonical for name in [canonical, *aliases]}


def _case_only(text: str, names: dict[str, str]) -> str | None:
    """Canonical referent whose approved name differs from `text` by letter case only."""
    for name, canonical in names.items():
        if name.casefold() == text.casefold():
            return canonical
    return None


def _contains_name(text: str, names: dict[str, str]) -> str | None:
    """Referent whose approved name occurs in `text` as a whole token, case-exact.

    Covers punctuation-only differences ("TP53.") and approved names inside longer
    text. It is a routing test for review, never evidence of correctness: the text
    may contradict the name ("not TP53").
    """
    for name, canonical in names.items():
        if re.search(rf"(?<![A-Za-z0-9]){re.escape(name)}(?![A-Za-z0-9])", text):
            return canonical
    return None


def _score_entity(gold: EntityGold, answer: str) -> Verdict:
    text = normalize_text(answer)
    if not text:
        return _incorrect("empty_answer", "blank answer")
    names = _names(gold.canonical, gold.aliases)
    if text in names:
        return _correct("approved name", (names[text],))
    if _case_only(text, names):
        return _unresolved(
            "case_only_difference", f"{text!r} differs from an approved name by case"
        )
    if _contains_name(text, names):
        return _unresolved(
            "approved_name_in_longer_text", f"{text!r} contains an approved name among other text"
        )
    return _incorrect("entity_not_approved", f"{text!r} is not an approved name", (text,))


_DELIMITERS = (";", "\n", ",")
_CONJUNCTION = re.compile(r"\b(?:and|or)\b|&", re.IGNORECASE)


def _segment(answer: str, gold: SetGold) -> list[str] | None:
    """Split a string answer; None when a gold name contains the delimiter."""
    delimiter = next((d for d in _DELIMITERS if d in answer), None)
    if delimiter is None:
        return [answer]
    if any(delimiter in name for m in gold.members for name in [m.canonical, *m.aliases]):
        return None
    return answer.split(delimiter)


def _score_set(gold: SetGold, answer: Answer) -> Verdict:
    if isinstance(answer, str):
        pieces = _segment(answer, gold)
        if pieces is None:
            return _unresolved("set_segmentation_ambiguous", "a member name contains the delimiter")
    else:
        pieces = answer
    elements = [e for e in (normalize_text(p) for p in pieces) if e]
    if not elements:
        return _incorrect("empty_answer", "no members given")
    names: dict[str, str] = {}
    for member in gold.members:
        names |= _names(member.canonical, member.aliases)
    matched: set[str] = set()
    candidates: set[str] = set()
    extras: list[str] = []
    for element in elements:
        if element in names:
            matched.add(names[element])
        elif (referent := _case_only(element, names)) is not None:
            candidates.add(referent)
        else:
            extras.append(element)
    required = {m.canonical for m in gold.members}
    seen = tuple(sorted(matched))
    if extras:
        if isinstance(answer, str) and _CONJUNCTION.search(answer):
            return _unresolved("set_segmentation_ambiguous", "conjunction in an unsegmented answer")
        return _incorrect("set_extra_member", f"not approved members: {extras}", seen)
    if (matched | candidates) != required:
        return _incorrect(
            "set_incomplete", f"missing {sorted(required - matched - candidates)}", seen
        )
    if candidates:  # a case variant is never merged into an exact match of the same referent
        return _unresolved(
            "case_only_difference", "a member differs from approved names by case", seen
        )
    return _correct("complete approved set", seen)


# --- conclusions --------------------------------------------------------------


def normalized_answer(answer: Answer, answer_class: str) -> str | list[str]:
    """The answer in the scorer's own normalization (a categorical conclusion also drops one
    final full stop); used to tell distinct answers apart, never to decide equivalence."""
    if isinstance(answer, list):
        return [normalized_answer(a, answer_class) for a in answer]  # type: ignore[misc]
    return _sentence(answer) if answer_class == "categorical_conclusion" else normalize_text(answer)


def _sentence(text: str) -> str:
    return normalize_text(text).removesuffix(".").rstrip()


def _score_conclusion(gold: ConclusionGold, answer: str) -> Verdict:
    text = _sentence(answer)
    if not text:
        return _incorrect("empty_answer", "blank answer")
    wordings = [_sentence(w) for w in gold.equivalent_wording]
    if text in wordings:
        return _correct("curated wording", (text,))
    if text.casefold() in {w.casefold() for w in wordings}:
        return _unresolved("case_only_difference", "differs from curated wording by case", (text,))
    # An uncurated proposition is neither accepted nor failed automatically.
    return _unresolved(
        "wording_not_approved", "wording is not among the curated equivalents", (text,)
    )


def score_answer(gold: Gold, answer: Answer, rules: str = "v1") -> Verdict:
    if rules not in RULES:
        raise ValueError(f"unknown scoring rules {rules!r}")
    if isinstance(gold, SetGold):
        return _score_set(gold, answer)
    if isinstance(answer, list):
        if len(answer) != 1:
            return _incorrect("answer_shape_mismatch", "a single answer was required")
        answer = answer[0]
    if isinstance(gold, QuantityGold):
        return _score_quantity(gold, answer, rules)
    if isinstance(gold, EntityGold):
        return _score_entity(gold, answer)
    return _score_conclusion(gold, answer)
