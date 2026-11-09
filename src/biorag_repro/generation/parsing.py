"""Versioned short-answer parser. Malformed output is reported, never repaired.

Parser short_answer_json_v1 accepts one JSON object, with optional surrounding
whitespace, holding exactly the keys "answer" and "abstained". Code fences, prose around
the object, extra or duplicate keys, wrong types and conflicting fields are unparseable.
The raw output is stored separately; nothing here edits the answer text.
"""

import json
from dataclasses import dataclass
from typing import Any, Literal

PARSER_VERSIONS = ("short_answer_json_v1",)

# JSON Schema of the output the parser accepts. Prompt artifacts must carry this schema.
OUTPUT_SCHEMAS: dict[str, dict[str, Any]] = {
    "short_answer_json_v1": {
        "type": "object",
        "properties": {
            "answer": {"type": ["string", "array", "null"], "items": {"type": "string"}},
            "abstained": {"type": "boolean"},
        },
        "required": ["answer", "abstained"],
        "additionalProperties": False,
    }
}


@dataclass(frozen=True)
class ParsedOutput:
    parse_status: Literal["parsed", "unparseable"]
    answer: str | list[str] | None
    abstained: bool | None
    reason: str | None


def _unparseable(reason: str) -> ParsedOutput:
    return ParsedOutput("unparseable", None, None, reason)


def _no_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    keys = [key for key, _ in pairs]
    if len(set(keys)) != len(keys):
        raise ValueError("duplicate key")
    return dict(pairs)


def _reject_constant(name: str) -> Any:
    raise ValueError(f"nonfinite constant {name}")


def parse_output(raw: str, parser_version: str) -> ParsedOutput:
    if parser_version not in PARSER_VERSIONS:
        raise ValueError(f"unknown parser version {parser_version!r}")
    try:
        value = json.loads(
            raw, object_pairs_hook=_no_duplicate_keys, parse_constant=_reject_constant
        )
    except ValueError as error:
        return _unparseable("duplicate_key" if "duplicate key" in str(error) else "not_json")
    if not isinstance(value, dict):
        return _unparseable("not_an_object")
    if set(value) != {"answer", "abstained"}:
        return _unparseable("wrong_keys")
    answer, abstained = value["answer"], value["abstained"]
    if not isinstance(abstained, bool):
        return _unparseable("abstained_not_boolean")
    if abstained:
        if answer is not None:
            return _unparseable("answer_with_abstention")
        return ParsedOutput("parsed", None, True, None)
    if isinstance(answer, str):
        return ParsedOutput("parsed", answer, False, None)
    if isinstance(answer, list) and all(isinstance(member, str) for member in answer):
        return ParsedOutput("parsed", answer, False, None)
    return _unparseable("answer_missing" if answer is None else "answer_wrong_type")
