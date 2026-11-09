"""Short-answer parsing. Malformed output is preserved as unparseable, never repaired."""

import pytest

from biorag_repro.evaluation.stage_c_models import GenerationResult
from biorag_repro.generation.parsing import PARSER_VERSIONS, parse_output

V = "short_answer_json_v1"


def test_parser_version_is_registered():
    assert V in PARSER_VERSIONS


@pytest.mark.parametrize(
    "raw,answer",
    [
        ('{"answer": "12.5 µM", "abstained": false}', "12.5 µM"),
        ('{"answer": "membrane", "abstained": false}', "membrane"),
        ('{"answer": ["TP53", "BRCA1", "ATM"], "abstained": false}', ["TP53", "BRCA1", "ATM"]),
        ('{"answer": "n = 1,204 (≥ 18 y)", "abstained": false}', "n = 1,204 (≥ 18 y)"),
        ('{"abstained": false, "answer": "x"}', "x"),
        ('  \n{"answer": "x", "abstained": false}\n', "x"),
        ('{"answer": "", "abstained": false}', ""),  # empty answers are scored, not hidden
        ('{"answer": [], "abstained": false}', []),
        ('{"answer": "caf\\u00e9", "abstained": false}', "café"),
    ],
)
def test_well_formed_answers(raw, answer):
    parsed = parse_output(raw, V)
    assert (parsed.parse_status, parsed.abstained, parsed.answer) == ("parsed", False, answer)
    assert parsed.reason is None


def test_abstention():
    parsed = parse_output('{"answer": null, "abstained": true}', V)
    assert (parsed.parse_status, parsed.abstained, parsed.answer) == ("parsed", True, None)


@pytest.mark.parametrize(
    "raw",
    [
        "",
        "   ",
        "42",
        "The answer is 42.",
        '```json\n{"answer": "x", "abstained": false}\n```',  # fenced: not repaired
        'Sure! {"answer": "x", "abstained": false}',  # prose prefix
        '{"answer": "x", "abstained": false} Hope this helps.',  # prose suffix
        '{"answer": "x", "abstained": false}{"answer": "y", "abstained": false}',
        '{"answer": "x", "abstained": false',  # truncated
        "{'answer': 'x', 'abstained': false}",  # single quotes
        '{"answer": "x", "abstained": false,}',  # trailing comma
        '{"answer": "x"}',  # missing abstention
        '{"abstained": false}',  # missing answer
        '{"answer": "x", "abstained": false, "confidence": 0.9}',  # extra key
        '{"Answer": "x", "abstained": false}',  # wrong key case
        '{"answer": "x", "abstained": "false"}',  # not a boolean
        '{"answer": "x", "abstained": 0}',
        '{"answer": "x", "abstained": null}',
        '{"answer": null, "abstained": false}',  # no answer and not abstaining
        '{"answer": "x", "abstained": true}',  # answer and abstention conflict
        '{"answer": [], "abstained": true}',
        '{"answer": 12.5, "abstained": false}',  # number, not text
        '{"answer": ["a", 1], "abstained": false}',
        '{"answer": [["a"]], "abstained": false}',
        '{"answer": {"v": 1}, "abstained": false}',
        '{"answer": "x", "answer": "y", "abstained": false}',  # duplicate key
        '{"answer": NaN, "abstained": false}',
        '[{"answer": "x", "abstained": false}]',
        '"{\\"answer\\": \\"x\\"}"',
        '<think>\n\n</think>\n\n{"answer": "x", "abstained": false}',
        '{"answer": "x", "abstained": false}\x00',
    ],
)
def test_malformed_output_is_unparseable_and_carries_no_answer(raw):
    parsed = parse_output(raw, V)
    assert parsed.parse_status == "unparseable"
    assert parsed.answer is None and parsed.abstained is None
    assert parsed.reason


def test_reasons_distinguish_failure_kinds():
    reasons = {
        parse_output(raw, V).reason
        for raw in (
            "not json",
            "[1]",
            '{"answer": "x"}',
            '{"answer": "x", "abstained": "no"}',
            '{"answer": "x", "abstained": true}',
            '{"answer": "x", "answer": "y", "abstained": false}',
        )
    }
    assert len(reasons) == 6


def test_unknown_parser_version_is_an_error():
    with pytest.raises(ValueError):
        parse_output("{}", "short_answer_json_v0")


def test_parsing_is_deterministic_and_does_not_modify_the_answer_text():
    raw = '{"answer": "  padded \\n value  ", "abstained": false}'
    assert parse_output(raw, V).answer == "  padded \n value  "
    assert parse_output(raw, V) == parse_output(raw, V)


def test_parsed_fields_feed_the_stage_c_generation_record():
    from biorag_repro.core.hashing import sha256_bytes
    from biorag_repro.core.manifest import ArtifactRef

    raw = '{"answer": ["a", "b"], "abstained": false}'
    parsed = parse_output(raw, V)
    ref = ArtifactRef(
        logical_ref="gen/o1",
        artifact_id="art_o1",
        sha256=sha256_bytes(raw.encode()),
    )
    record = GenerationResult(
        phase="development",
        benchmark_version="b",
        item_id="i",
        paper_id="p",
        matrix_arm="confirmatory",
        representation_or_condition="JATS",
        repeat_index=1,
        resolved_config_sha256="a" * 64,
        observation_key="k",
        run_id="r",
        attempt_id="t",
        status="completed",
        output=ref,
        raw_output=raw,
        parse_status=parsed.parse_status,
        answer=parsed.answer,
        abstained=parsed.abstained,
    )
    assert record.answer == ["a", "b"]
    bad = parse_output("junk", V)
    GenerationResult(
        **{
            **record.model_dump(),
            "raw_output": "junk",
            "output": ref.model_copy(update={"sha256": sha256_bytes(b"junk")}),
            "parse_status": bad.parse_status,
            "answer": bad.answer,
            "abstained": bad.abstained,
        }
    )


def test_configuration_accepts_exactly_the_parser_versions_that_exist():
    from typing import get_args

    from biorag_repro.generation.config import GeneratorConfiguration

    literal = GeneratorConfiguration.model_fields["parser_version"].annotation
    assert set(get_args(literal)) == set(PARSER_VERSIONS)
