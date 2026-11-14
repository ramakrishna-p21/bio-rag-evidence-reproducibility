"""Supplementary analysis labels: answer-content identity and evidence form and location.

The frozen analysis table carries correctness labels only. This module adds a separate,
hash-linked artifact for the secondary analyses it cannot support (EXPERIMENT_PLAN section 9):
paired answer-content disagreement, stability of identical incorrect answers, table versus
prose evidence and evidence location. Python decides every label; R only counts them.

Four notions are kept apart and never merged:

* byte identity of the raw generator output;
* typed normalized answer identity (answer shape and each element after NFC and whitespace
  normalization, element order kept; nothing else is normalized);
* Stage C correctness (a correct and an incorrect answer of one pair necessarily differ in
  content; two correct answers are only both correct, not shown to be the same statement);
* established scientific equivalence, which the repository defines only through the gold of one
  item and is therefore never claimed here beyond "both correct".

Published tables are label-only: no question, gold, answer text or evidence text. Answers are
represented by opaque integers that are local to one item and arm, not by hashes (a short answer
such as a gene symbol is guessable from its hash).

Origin: only a verified synthetic rehearsal is supported. An official-origin supplement stays
unavailable until the production authorization and provenance requirements are satisfied; the
origin is read from the verified handoff, never from a caller-supplied label.
"""

import csv
import io
import re
from collections.abc import Sequence
from pathlib import Path
from typing import Annotated, Final, Literal, Self

from pydantic import Field, StringConstraints, model_validator

from biorag_repro.benchmark.models import BenchmarkItem, JatsRef, PdfRef
from biorag_repro.core.hashing import canonical_json_bytes, sha256_bytes
from biorag_repro.core.records import Record
from biorag_repro.core.storage import publish_bytes, serialize_project_json
from biorag_repro.documents.models import CanonicalDocument, CanonicalNode
from biorag_repro.evaluation.analysis_handoff import AnalysisHandoff, VerifiedHandoff
from biorag_repro.evaluation.analysis_models import MISSING, AnalysisRow
from biorag_repro.evaluation.analysis_result import ResultError, read_result_files
from biorag_repro.evaluation.analysis_table import (
    Observation,
    build_row,
    serialize_csv,
)
from biorag_repro.evaluation.answer_scoring import normalize_text
from biorag_repro.evaluation.stage_c import generation_digest

Sha256 = Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{64}$")]
BANNER: Final = "SYNTHETIC REHEARSAL — NOT SCIENTIFIC FINDINGS"
SUPPLEMENT_KIND: Final = "analysis_supplement"
RECORD_FILE: Final = "supplement_record.json"
DEFINITIONS_VERSION: Final = "1"

# Arms with three generations of one condition. Robustness arms have one, so no stability.
REPEATED_ARMS: Final = ("confirmatory", "closed_book")
REPEATS: Final = (1, 2, 3)

RESPONSE_COLUMNS: Final = (
    "observation_key",
    "item_id",
    "item_sha256",
    "matrix_arm",
    "condition",
    "repeat_index",
    "attempt_id",
    "stage_c_artifact",
    "answer_state",
    "stage_c_status",
    "answer_shape",
    "byte_group",
    "norm_group",
)
PAIR_COLUMNS: Final = (
    "item_id",
    "matrix_arm",
    "repeat_index",
    "jats_state",
    "pdf_state",
    "jats_status",
    "pdf_status",
    "pair_class",
)
STABILITY_COLUMNS: Final = (
    "item_id",
    "matrix_arm",
    "condition",
    "n_rows",
    "n_answers",
    "stability_class",
    "byte_identical",
)
EVIDENCE_ITEM_COLUMNS: Final = (
    "item_id",
    "item_sha256",
    "n_jats_refs",
    "n_pdf_refs",
    "n_evidence_sets",
    "n_sets_with_table_roles",
    "jats_form",
    "jats_location",
    "pdf_form",
    "form_relation",
    "item_form",
)
EVIDENCE_OBSERVATION_COLUMNS: Final = (
    "observation_key",
    "item_id",
    "matrix_arm",
    "condition",
    "repeat_index",
    "stage_a_label",
    "stage_a_form",
    "stage_a_sets_table",
    "stage_a_sets_prose",
    "stage_a_sets_mixed",
    "stage_a_sets_other",
    "stage_b_status",
    "stage_b_form",
)
TABLES: Final = {
    "supplement_response.csv": RESPONSE_COLUMNS,
    "supplement_pairs.csv": PAIR_COLUMNS,
    "supplement_stability.csv": STABILITY_COLUMNS,
    "supplement_evidence_items.csv": EVIDENCE_ITEM_COLUMNS,
    "supplement_evidence_observations.csv": EVIDENCE_OBSERVATION_COLUMNS,
}
ANSWER_STATES: Final = (
    "answer",
    "abstained",
    "unparseable",
    "operational_failure",
    "not_executed",
    "no_row",
)
PAIR_CLASSES: Final = (
    "identical_bytes",
    "identical_normalized",
    "both_correct",
    "differ_correctness_differs",
    "differ_both_incorrect_unresolved",
    "differ_correctness_unresolved",
    "not_comparable",
)
STABILITY_CLASSES: Final = (
    "incomplete",
    "contains_abstention_or_unparseable",
    "correctness_unresolved",
    "all_correct",
    "incorrect_identical_normalized",
    "incorrect_varying",
    "mixed",
)
FORMS: Final = ("table", "prose", "mixed", "other", "unresolved")


class SupplementError(ValueError):
    pass


class OfficialSupplementUnavailable(SupplementError):
    """Raised for any attempt to build a supplement from other than a verified rehearsal."""


# ---- answer identity ------------------------------------------------------------------------

NormKey = tuple[str, tuple[str, ...]]
Cells = dict[str, str]  # one table row: column name to canonical text


def answer_state(obs: Observation) -> str:
    result = obs.result
    if result is None:
        return "not_executed"
    if result.status == "operational_failure":
        return "operational_failure"
    if result.parse_status == "unparseable":
        return "unparseable"
    return "abstained" if result.abstained else "answer"


def normalized_key(answer: str | list[str]) -> NormKey:
    """Typed normalized answer: shape plus each element after NFC and whitespace collapsing.

    A string and a one-element list stay different, element order is kept and nothing is
    dropped, case-folded or re-segmented; the only normalization is the scoring rule D2.
    """
    if isinstance(answer, str):
        return ("string", (normalize_text(answer),))
    return ("list", tuple(normalize_text(member) for member in answer))


def _check_attempt(obs: Observation) -> None:
    result, score = obs.result, obs.score
    if result is None:
        if score is not None:
            raise SupplementError("a score exists for an observation without a generation")
        return
    if score is None:
        raise SupplementError("an executed observation needs its Stage C score")
    if (
        score.observation_key != result.observation_key
        or score.attempt_id != result.attempt_id
        or score.repeat_index != result.repeat_index
        or score.generation_sha256 != generation_digest(result)
        or score.output != result.output
    ):
        raise SupplementError("the score does not belong to this attempt's generation artifact")


def response_rows(observations: Sequence[Observation], rows: Sequence[AnalysisRow]) -> list[Cells]:
    """One label row per analysis-table row, bound to the exact attempt and score."""
    by_key = {o_row.observation_key: o_row for o_row in rows}
    if len(by_key) != len(rows):
        raise SupplementError("the analysis table repeats an observation key")
    paired: dict[str, tuple[Observation, AnalysisRow]] = {}
    for obs in observations:
        row, _ = build_row(obs)
        if by_key.get(row.observation_key) != row:
            raise SupplementError(f"observation {row.observation_key} differs from the table row")
        if row.observation_key in paired:
            raise SupplementError("an observation is supplied twice")
        _check_attempt(obs)
        paired[row.observation_key] = (obs, row)
    if set(paired) != set(by_key):
        raise SupplementError("the supplied observations are not exactly the table's rows")

    # Opaque groups are local to (item, arm) and numbered in (repeat, condition) order.
    ordered = sorted(
        paired.values(),
        key=lambda p: (p[1].item_id, p[1].matrix_arm, p[1].repeat_index, p[1].condition),
    )
    seen_bytes: dict[tuple[str, str], dict[str, int]] = {}
    seen_norm: dict[tuple[str, str], dict[NormKey, int]] = {}
    out: list[Cells] = []
    for obs, row in ordered:
        state = answer_state(obs)
        byte_group = norm_group = shape = MISSING
        if state == "answer":
            assert obs.result is not None and obs.result.raw_output is not None
            assert obs.result.answer is not None
            scope = (row.item_id, row.matrix_arm)
            raw_groups = seen_bytes.setdefault(scope, {})
            norm_groups = seen_norm.setdefault(scope, {})
            key = normalized_key(obs.result.answer)
            byte_group = str(raw_groups.setdefault(obs.result.raw_output, len(raw_groups) + 1))
            norm_group = str(norm_groups.setdefault(key, len(norm_groups) + 1))
            shape = key[0]
        out.append(
            {
                "observation_key": row.observation_key,
                "item_id": row.item_id,
                "item_sha256": row.item_sha256,
                "matrix_arm": row.matrix_arm,
                "condition": row.condition,
                "repeat_index": str(row.repeat_index),
                "attempt_id": row.attempt_id or MISSING,
                "stage_c_artifact": row.stage_c_artifact or MISSING,
                "answer_state": state,
                "stage_c_status": row.stage_c_status or MISSING,
                "answer_shape": shape,
                "byte_group": byte_group,
                "norm_group": norm_group,
            }
        )
    return sorted(out, key=lambda r: r["observation_key"])


# ---- pairs and stability (pure functions of the response rows) ------------------------------


def _by(rows: Sequence[Cells], *fields: str) -> dict[tuple[str, ...], Cells]:
    out: dict[tuple[str, ...], Cells] = {}
    for r in rows:
        key = tuple(r[f] for f in fields)
        if key in out:
            raise SupplementError("two response rows share one coordinate")
        out[key] = r
    return out


def pair_class(jats: Cells | None, pdf: Cells | None) -> str:
    if (
        jats is None
        or pdf is None
        or "answer" != jats["answer_state"]
        or "answer" != pdf["answer_state"]
    ):
        return "not_comparable"
    if jats["byte_group"] == pdf["byte_group"]:
        return "identical_bytes"
    if jats["norm_group"] == pdf["norm_group"]:
        return "identical_normalized"
    statuses = {jats["stage_c_status"], pdf["stage_c_status"]}
    if statuses == {"correct"}:
        return "both_correct"
    if statuses == {"incorrect"}:
        return "differ_both_incorrect_unresolved"
    if statuses == {"correct", "incorrect"}:
        return "differ_correctness_differs"
    return "differ_correctness_unresolved"


def pair_rows(response: Sequence[Cells]) -> list[Cells]:
    """JATS against PDF_DOCLING for one item, retrieval arm and repeat; closed book has no pair."""
    cells = _by(response, "item_id", "matrix_arm", "condition", "repeat_index")
    keys = sorted(
        {
            (r["item_id"], r["matrix_arm"], r["repeat_index"])
            for r in response
            if r["matrix_arm"] != "closed_book"
        },
        key=lambda k: (k[1], k[0], int(k[2])),
    )
    out = []
    for item_id, arm, repeat in keys:
        jats = cells.get((item_id, arm, "JATS", repeat))
        pdf = cells.get((item_id, arm, "PDF_DOCLING", repeat))
        out.append(
            {
                "item_id": item_id,
                "matrix_arm": arm,
                "repeat_index": repeat,
                "jats_state": jats["answer_state"] if jats else "no_row",
                "pdf_state": pdf["answer_state"] if pdf else "no_row",
                "jats_status": jats["stage_c_status"] if jats else MISSING,
                "pdf_status": pdf["stage_c_status"] if pdf else MISSING,
                "pair_class": pair_class(jats, pdf),
            }
        )
    return out


def stability_class(group: Sequence[Cells]) -> tuple[str, str]:
    """Stability class and byte flag for the (up to three) repeats of one item, arm, condition.

    Stable incorrect needs all three repeats completed, parsed, scored incorrect and with the same
    typed normalized answer. Abstentions, unparseable outputs, failures and absent rows never
    count as identical wrong answers.
    """
    by_repeat = {int(r["repeat_index"]): r for r in group}
    rows = [by_repeat.get(i) for i in REPEATS]
    if any(r is None or r["answer_state"] in {"operational_failure", "not_executed"} for r in rows):
        return "incomplete", MISSING
    present = [r for r in rows if r is not None]
    if any(r["answer_state"] != "answer" for r in present):
        return "contains_abstention_or_unparseable", MISSING
    byte_flag = "yes" if len({r["byte_group"] for r in present}) == 1 else "no"
    statuses = [r["stage_c_status"] for r in present]
    if any(s not in {"correct", "incorrect"} for s in statuses):
        return "correctness_unresolved", byte_flag
    if set(statuses) == {"correct"}:
        return "all_correct", byte_flag
    if set(statuses) == {"incorrect"}:
        same = len({r["norm_group"] for r in present}) == 1
        return ("incorrect_identical_normalized" if same else "incorrect_varying"), byte_flag
    return "mixed", byte_flag


def stability_rows(response: Sequence[Cells]) -> list[Cells]:
    groups: dict[tuple[str, str, str], list[Cells]] = {}
    for r in response:
        if r["matrix_arm"] in REPEATED_ARMS:
            groups.setdefault((r["matrix_arm"], r["item_id"], r["condition"]), []).append(r)
    out = []
    for (arm, item_id, condition), group in sorted(groups.items()):
        label, byte_flag = stability_class(group)
        out.append(
            {
                "item_id": item_id,
                "matrix_arm": arm,
                "condition": condition,
                "n_rows": str(len(group)),
                "n_answers": str(sum(r["answer_state"] == "answer" for r in group)),
                "stability_class": label,
                "byte_identical": byte_flag,
            }
        )
    return out


# ---- evidence form and location -------------------------------------------------------------

_SEGMENT = re.compile(r"^([A-Za-z][A-Za-z0-9_.\-]*)\[([1-9][0-9]*)\]$")
_TABLE_TAGS = frozenset(
    {"table-wrap", "table", "thead", "tbody", "tfoot", "tr", "th", "td", "table-wrap-foot"}
)
_NOT_PROSE_ANCESTORS = frozenset(
    {"fig", "fig-group", "caption", "supplementary-material", "ref-list", "boxed-text"}
)


def jats_tags(path: str) -> list[str] | None:
    """Element names of an indexed original-JATS path, or None when the path is not that format.

    The format is the one the benchmark validator resolves: `/article/` followed by
    `tag[n]` segments. A path outside it is not interpreted.
    """
    parts = path.split("/")
    if len(parts) < 3 or parts[0] != "" or parts[1] != "article":
        return None
    tags = []
    for part in parts[2:]:
        match = _SEGMENT.match(part)
        if match is None:
            return None
        tags.append(match.group(1))
    return tags


def jats_location(path: str) -> str:
    """Abstract, body or back by structural ancestry; anything else is explicit."""
    tags = jats_tags(path)
    if not tags:
        return "unresolved"
    if "abstract" in tags:
        return "abstract"
    return {"body": "body", "back": "back"}.get(tags[0], "other")


def jats_ref_form(ref: JatsRef) -> str:
    """Positive structural evidence only: a missing table id or table ancestor is not prose."""
    tags = jats_tags(ref.xml_path)
    if tags is None:
        return "unresolved"
    in_table = bool(_TABLE_TAGS & set(tags))
    if ref.table_id is not None and not in_table:
        return "unresolved"  # the reference contradicts its own path
    if in_table:
        return "table"
    if tags[-1] == "p" and not (_NOT_PROSE_ANCESTORS & set(tags)):
        return "prose"
    return "other"


def pdf_ref_form(ref: PdfRef) -> str:
    """A PDF reference establishes a table only through its table id; it never establishes prose."""
    return "table" if ref.table_id is not None else "unresolved"


def _combine(forms: Sequence[str]) -> str:
    if not forms or "unresolved" in forms:
        return "unresolved"
    kinds = set(forms)
    if kinds == {"table"} or kinds == {"prose"}:
        return forms[0]
    if {"table", "prose"} <= kinds:
        return "mixed"
    return "other"


def _item_location(refs: Sequence[JatsRef]) -> str:
    locations = {jats_location(r.xml_path) for r in refs}
    if "unresolved" in locations:
        return "unresolved"
    return locations.pop() if len(locations) == 1 else "multiple"


def evidence_item_row(item: BenchmarkItem, item_sha256: str) -> Cells:
    jats_form = _combine([jats_ref_form(r) for r in item.jats_refs])
    pdf_forms = {pdf_ref_form(r) for r in item.pdf_refs}
    pdf_form = "table" if pdf_forms == {"table"} else "unresolved"
    table_sets = sum(
        any(c.role in {"row", "column"} for c in s.components) for s in item.evidence_sets
    )
    established = {"table", "prose", "mixed"}
    if jats_form not in established:
        relation = "jats_not_established"
    elif jats_form == "prose" and table_sets:
        relation = "set_roles_conflict"
    elif pdf_form == "unresolved":
        relation = "pdf_not_established"
    elif jats_form == "prose":
        relation = "conflict"
    else:
        relation = "agree"
    item_form = "unresolved" if relation in {"set_roles_conflict", "conflict"} else jats_form
    return {
        "item_id": item.item_id,
        "item_sha256": item_sha256,
        "n_jats_refs": str(len(item.jats_refs)),
        "n_pdf_refs": str(len(item.pdf_refs)),
        "n_evidence_sets": str(len(item.evidence_sets)),
        "n_sets_with_table_roles": str(table_sets),
        "jats_form": jats_form,
        "jats_location": _item_location(item.jats_refs),
        "pdf_form": pdf_form,
        "form_relation": relation,
        "item_form": item_form,
    }


def container_form(node: CanonicalNode) -> str:
    """Form of one canonical evidence container; captions, titles and figures are `other`."""
    if node.table is not None:
        return "table"
    return "prose" if node.kind in {"paragraph", "list_item"} else "other"


def _container(document: CanonicalDocument, node_id: str) -> str:
    for node in document.nodes:
        if node.node_id == node_id:
            return container_form(node)
    raise SupplementError("an evidence container is not a node of the canonical document")


def _set_form(forms: Sequence[str]) -> str:
    kinds = set(forms)
    return forms[0] if len(kinds) == 1 else "mixed"


def evidence_observation_row(obs: Observation, row: AnalysisRow) -> Cells:
    out = {
        "observation_key": row.observation_key,
        "item_id": row.item_id,
        "matrix_arm": row.matrix_arm,
        "condition": row.condition,
        "repeat_index": str(row.repeat_index),
        "stage_a_label": row.stage_a_label or MISSING,
        "stage_a_form": "no_stage_a_record",
        "stage_a_sets_table": MISSING,
        "stage_a_sets_prose": MISSING,
        "stage_a_sets_mixed": MISSING,
        "stage_a_sets_other": MISSING,
        "stage_b_status": row.stage_b_status or MISSING,
        "stage_b_form": "no_stage_b_record",
    }
    evidence = obs.evidence
    if evidence is None:
        return out
    document = evidence.document
    set_forms = [
        _set_form([_container(document, p.container_node_id) for p in s.placements])
        for s in evidence.stage_a.sets
    ]
    counts = {f: set_forms.count(f) for f in ("table", "prose", "mixed", "other")}
    for form, n in counts.items():
        out[f"stage_a_sets_{form}"] = str(n)
    out["stage_a_form"] = "no_located_evidence" if not set_forms else _combine_sets(set_forms)
    b = evidence.stage_b
    if b is not None and b.rules == "v2" and b.basis == "reviewer":
        out["stage_b_form"] = "not_attributed"  # whole-context review: form is not inferred
    elif b is not None:
        delivered = [_container(document, d.container_node_id) for d in b.delivered]
        out["stage_b_form"] = "none_delivered" if not delivered else _combine_sets(delivered)
    return out


def _combine_sets(forms: Sequence[str]) -> str:
    kinds = set(forms)
    if len(kinds) == 1:
        return forms[0]
    return "mixed" if kinds <= {"table", "prose", "mixed"} else "other"


# ---- serialization --------------------------------------------------------------------------


def serialize_table(columns: Sequence[str], rows: Sequence[Cells]) -> bytes:
    buffer = io.StringIO(newline="")
    writer = csv.writer(buffer, lineterminator="\n")
    writer.writerow(columns)
    for r in rows:
        writer.writerow([r[c] for c in columns])
    return buffer.getvalue().encode("utf-8")


def parse_table(columns: Sequence[str], data: bytes) -> list[Cells]:
    records = list(csv.reader(io.StringIO(data.decode("utf-8"), newline="")))
    if not records or records[0] != list(columns):
        raise SupplementError("a supplement table header differs from its fixed columns")
    rows = []
    for record in records[1:]:
        if len(record) != len(columns):
            raise SupplementError("a supplement table row has the wrong number of cells")
        rows.append(dict(zip(columns, record, strict=True)))
    return rows


# ---- artifact -------------------------------------------------------------------------------


class SupplementFile(Record):
    bytes: int = Field(ge=1)
    sha256: Sha256


class SupplementRecord(Record):
    kind: Literal["analysis_supplement"] = "analysis_supplement"
    schema_version: Literal["1"] = "1"
    definitions_version: Literal["1"] = DEFINITIONS_VERSION
    purpose: Literal["synthetic_rehearsal"] = "synthetic_rehearsal"
    scientific_authority: Literal["none"] = "none"
    final: Literal[False] = False
    banner: Literal["SYNTHETIC REHEARSAL — NOT SCIENTIFIC FINDINGS"] = BANNER
    benchmark_version: str = Field(min_length=1)
    corpus_version: str = Field(min_length=1)
    analysis_table_sha256: Sha256
    handoff_sha256: Sha256
    files: dict[str, SupplementFile]
    supplement_sha256: Sha256

    @model_validator(mode="after")
    def consistent(self) -> Self:
        if set(self.files) != set(TABLES):
            raise SupplementError("the record must list exactly the supplement tables")
        if self.supplement_sha256 != supplement_id(
            self.handoff_sha256,
            self.analysis_table_sha256,
            {n: f.sha256 for n, f in self.files.items()},
        ):
            raise SupplementError("supplement_sha256 does not match its files and inputs")
        return self


def supplement_id(handoff: str, table: str, files: dict[str, str]) -> str:
    payload = {
        "analysis_table_sha256": table,
        "definitions_version": DEFINITIONS_VERSION,
        "files": [{"name": n, "sha256": files[n]} for n in sorted(files)],
        "handoff_sha256": handoff,
        "kind": SUPPLEMENT_KIND,
    }
    return sha256_bytes(canonical_json_bytes(payload))


class Supplement(Record):
    record: SupplementRecord
    tables: dict[str, bytes]


def build_supplement(verified: VerifiedHandoff, observations: Sequence[Observation]) -> Supplement:
    """Build the supplement for a verified synthetic rehearsal from the table's own observations."""
    handoff = verified.handoff
    if verified.official or handoff.input_origin != "synthetic_rehearsal":
        raise OfficialSupplementUnavailable(
            "a supplement is available only for a verified synthetic rehearsal; official-origin "
            "generation waits for the production authorization and provenance requirements"
        )
    rows = verified.rows
    table_sha256 = sha256_bytes(serialize_csv(rows))
    if table_sha256 != handoff.csv.sha256:
        raise SupplementError("the verified rows are not the table the handoff cites")
    obs_by_key = {build_row(o)[0].observation_key: o for o in observations}
    response = response_rows(observations, rows)
    items = {o.item.item_id: o.item for o in observations}
    item_sha = {r["item_id"]: r["item_sha256"] for r in response}
    row_by_key = {r.observation_key: r for r in rows}
    tables = {
        "supplement_response.csv": serialize_table(RESPONSE_COLUMNS, response),
        "supplement_pairs.csv": serialize_table(PAIR_COLUMNS, pair_rows(response)),
        "supplement_stability.csv": serialize_table(STABILITY_COLUMNS, stability_rows(response)),
        "supplement_evidence_items.csv": serialize_table(
            EVIDENCE_ITEM_COLUMNS,
            [evidence_item_row(items[i], item_sha[i]) for i in sorted(items)],
        ),
        "supplement_evidence_observations.csv": serialize_table(
            EVIDENCE_OBSERVATION_COLUMNS,
            [evidence_observation_row(obs_by_key[k], row_by_key[k]) for k in sorted(obs_by_key)],
        ),
    }
    files = {n: SupplementFile(bytes=len(b), sha256=sha256_bytes(b)) for n, b in tables.items()}
    record = SupplementRecord(
        benchmark_version=handoff.benchmark_version,
        corpus_version=handoff.corpus_version,
        analysis_table_sha256=table_sha256,
        handoff_sha256=handoff_sha256(handoff),
        files=files,
        supplement_sha256=supplement_id(
            handoff_sha256(handoff),
            table_sha256,
            {n: f.sha256 for n, f in files.items()},
        ),
    )
    return Supplement(record=record, tables=tables)


def handoff_sha256(handoff: AnalysisHandoff) -> str:
    """SHA-256 of the handoff file bytes (what the analysis cites as the handoff hash)."""
    return sha256_bytes(serialize_project_json(handoff, AnalysisHandoff))


def serialize_record(record: SupplementRecord) -> bytes:
    return canonical_json_bytes(record.model_dump(mode="json")) + b"\n"


def publish_supplement(supplement: Supplement, root: Path, relative: str) -> str:
    """Write the tables and the record into a new directory; return the record's SHA-256."""
    target = root / relative
    target.mkdir(parents=True, exist_ok=False)
    for name, data in supplement.tables.items():
        publish_bytes(target / name, data, sha256_bytes(data))
    record = serialize_record(supplement.record)
    publish_bytes(target / RECORD_FILE, record, sha256_bytes(record))
    for path in (*target.iterdir(), target):
        path.chmod(path.stat().st_mode & ~0o222)
    return sha256_bytes(record)


def read_supplement(
    root: Path, relative: str, record_sha256: str, *, verified: VerifiedHandoff
) -> Supplement:
    """Read a supplement by its trusted record hash and re-derive every dependent label.

    Pair and stability labels are recomputed from the response rows, and the response rows are
    checked against the verified analysis table. Origin and table linkage come from the verified
    handoff, not from the record.
    """
    if verified.official or verified.handoff.input_origin != "synthetic_rehearsal":
        raise OfficialSupplementUnavailable("only a verified synthetic rehearsal is supported")
    files = read_result_files(root, relative, (*TABLES, RECORD_FILE))
    if sha256_bytes(files[RECORD_FILE]) != record_sha256:
        raise ResultError("hash", "the supplement record differs from the trusted SHA-256")
    record = SupplementRecord.model_validate_json(files[RECORD_FILE], strict=True)
    if serialize_record(record) != files[RECORD_FILE]:
        raise ResultError("canonical_form", "the supplement record is not in canonical form")
    handoff = verified.handoff
    table_sha256 = sha256_bytes(serialize_csv(verified.rows))
    if (
        record.analysis_table_sha256 != table_sha256
        or table_sha256 != handoff.csv.sha256
        or record.handoff_sha256 != handoff_sha256(handoff)
        or (record.benchmark_version, record.corpus_version)
        != (handoff.benchmark_version, handoff.corpus_version)
    ):
        raise ResultError("citation", "the supplement does not cite the verified table and handoff")
    tables = {name: files[name] for name in TABLES}
    for name, data in tables.items():
        entry = record.files[name]
        if entry.sha256 != sha256_bytes(data) or entry.bytes != len(data):
            raise ResultError("hash", f"{name} differs from the hash recorded for it")
        if serialize_table(TABLES[name], parse_table(TABLES[name], data)) != data:
            raise ResultError("canonical_form", f"{name} is not in canonical form")
    check_tables(tables, verified.rows)
    return Supplement(record=record, tables=tables)


def check_tables(tables: dict[str, bytes], rows: Sequence[AnalysisRow]) -> None:
    """Cross-check the response rows with the table and re-derive the dependent labels."""
    response = parse_table(RESPONSE_COLUMNS, tables["supplement_response.csv"])
    by_key = {r.observation_key: r for r in rows}
    if {r["observation_key"] for r in response} != set(by_key) or len(response) != len(by_key):
        raise SupplementError("the response rows are not exactly the analysis-table rows")
    for r in response:
        row = by_key[r["observation_key"]]
        if (
            (r["item_id"], r["item_sha256"], r["matrix_arm"], r["condition"])
            != (row.item_id, row.item_sha256, row.matrix_arm, row.condition)
            or r["repeat_index"] != str(row.repeat_index)
            or r["attempt_id"] != (row.attempt_id or MISSING)
            or r["stage_c_artifact"] != (row.stage_c_artifact or MISSING)
            or r["stage_c_status"] != (row.stage_c_status or MISSING)
            or r["answer_state"] not in ANSWER_STATES
            or (r["answer_state"] == "answer") != (r["norm_group"] != MISSING)
            or (r["byte_group"] == MISSING) != (r["norm_group"] == MISSING)
        ):
            raise SupplementError("a response row disagrees with its analysis-table row")
    if tables["supplement_pairs.csv"] != serialize_table(PAIR_COLUMNS, pair_rows(response)):
        raise SupplementError("the pair labels differ from their derivation from the responses")
    if tables["supplement_stability.csv"] != serialize_table(
        STABILITY_COLUMNS, stability_rows(response)
    ):
        raise SupplementError("the stability labels differ from their derivation")
    evidence = parse_table(
        EVIDENCE_OBSERVATION_COLUMNS, tables["supplement_evidence_observations.csv"]
    )
    if {e["observation_key"] for e in evidence} != set(by_key) or len(evidence) != len(by_key):
        raise SupplementError("the evidence rows are not exactly the analysis-table rows")
    item_ids = {r.item_id for r in rows}
    listed = [
        r["item_id"]
        for r in parse_table(EVIDENCE_ITEM_COLUMNS, tables["supplement_evidence_items.csv"])
    ]
    if sorted(listed) != sorted(item_ids):
        raise SupplementError("the evidence items are not exactly the table's items")
