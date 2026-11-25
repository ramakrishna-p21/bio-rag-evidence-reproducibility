"""Stage B verification sheets: advisory proposals shown as anchors inside the full masked context.

The sheet is a reading aid. It never records a label. A proposal becomes a review record only
for an entry the researcher marked verified, and `apply` refuses any entry whose anchors do not
re-match the context exactly. Records are written into the queue's own review.json, which the
official ingestion (`evaluate_final_test.py stage-b-ingest`) validates again.

  plan   --queue Q --mapping M --scope-arm confirmatory --out DIR   (batches + advisory inputs)
  build  --plan DIR --queue Q --proposals-dir D                      (sheets, resumable)
  sheet  --queue Q --proposals P --handles B0001,B0002 --out FILE
  apply  --queue Q --proposals P --verification V --review R --log L

Batches keep the queue's own handles and order. Researcher-facing files carry no arm or
representation; the mapping is read only to choose handles. A batch sheet is built only when
every handle has a proposal, and an existing sheet is kept unless its inputs changed.
"""

import argparse
import hashlib
import html
import json
import time
from datetime import UTC, datetime
from pathlib import Path

NEGATIVE = {"not_delivered", "not_reachable_stage_a", "non_evaluable", "unresolved"}


def check(entry: dict, proposal: dict) -> list[str]:
    """Machine checks of one proposal against its masked entry; an empty list means none failed."""
    problems = []
    chunks = {c["chunk"]: c for c in entry["chunks"]}
    sets = entry["evidence_sets"]
    for cite in proposal["citations"] + proposal["conflict_citations"]:
        chunk = chunks.get(cite["chunk"])
        if chunk is None or not chunk["designated_paper"]:
            problems.append(f"{cite['chunk']}: not a designated-paper chunk")
            continue
        hits = chunk["text"].count(cite["quote"]) if cite["quote"] else 0
        if hits == 0:
            problems.append(f"{cite['chunk']}: quote not found")
        elif hits > 1 and not cite.get("occurrence"):
            problems.append(f"{cite['chunk']}: quote occurs {hits} times, no occurrence given")
        elif cite.get("occurrence") and not 1 <= cite["occurrence"] <= hits:
            problems.append(f"{cite['chunk']}: occurrence out of range")
    covered = {(c["set_index"], c["component_index"]) for c in proposal["citations"]}
    if proposal["status"] == "delivered":
        if not any(
            all((s, k) in covered for k in range(len(comps))) for s, comps in enumerate(sets)
        ):
            problems.append("delivered, but no evidence set is fully cited")
    elif proposal["citations"] and proposal["status"] in NEGATIVE - {"unresolved"}:
        problems.append("negative status carries supporting citations")
    if proposal["status"] in NEGATIVE - {"unresolved"}:
        named = {m["set_index"] for m in proposal["missing_components"]}
        if named != set(range(len(sets))):
            problems.append("negative status must name a missing component in every evidence set")
    return problems


def mark(text: str, spans: list[tuple[int, int, str]]) -> str:
    """Escape `text` and wrap each (start, end, label) span; overlapping spans are skipped."""
    out, at = [], 0
    for start, end, label in sorted(spans):
        if start < at:
            continue
        out += [
            html.escape(text[at:start]),
            f'<mark class="{label[0]}">',
            html.escape(text[start:end]),
            f"<sup>{html.escape(label)}</sup></mark>",
        ]
        at = end
    return "".join(out) + html.escape(text[at:])


def spans_in(text: str, cites: list[dict], kind: str, chunk: str) -> list[tuple[int, int, str]]:
    found = []
    for cite in cites:
        if cite["chunk"] != chunk or not cite["quote"] or text.count(cite["quote"]) == 0:
            continue
        nth, at = (cite.get("occurrence") or 1), -1
        for _ in range(nth):
            at = text.find(cite["quote"], at + 1)
        tag = f"{kind}{cite.get('set_index', '')}.{cite.get('component_index', '')}".rstrip(".")
        found.append((at, at + len(cite["quote"]), tag))
    return found


def render(entry: dict, proposal: dict) -> str:
    problems = check(entry, proposal)
    comps = "".join(
        f"<li>set {s} / {k} <b>{html.escape(c['role'])}</b>: {html.escape(c['text'])}</li>"
        for s, comps in enumerate(entry["evidence_sets"])
        for k, c in enumerate(comps)
    )
    chunks = "".join(
        f"<h4>{c['chunk']}{'' if c['designated_paper'] else ' (other paper)'}</h4><pre>"
        + mark(
            c["text"],
            spans_in(c["text"], proposal["citations"], "s", c["chunk"])
            + spans_in(c["text"], proposal["conflict_citations"], "c", c["chunk"]),
        )
        + "</pre>"
        for c in entry["chunks"]
    )
    miss = (
        "; ".join(
            f"set {m['set_index']} component {m['component_index']}"
            for m in proposal["missing_components"]
        )
        or "none"
    )
    flag = (
        f'<p class="bad">machine check: {html.escape("; ".join(problems))}</p>' if problems else ""
    )
    h = entry["handle"]
    return f"""<section id="{h}" data-handle="{h}">
<h3>{h} <span class="p {proposal.get("priority", "")}">{proposal.get("priority", "")}</span>
proposed: <b>{proposal["status"]}</b>{" / " + proposal["reason"] if proposal.get("reason") else ""}</h3>
<p><b>Q:</b> {html.escape(entry["question"])}</p>{flag}
<details open><summary>evidence components</summary><ul>{comps}</ul></details>
<p><b>missing:</b> {miss}. <b>flags:</b> {html.escape(", ".join(proposal["flags"]) or "none")}.
<b>unresolved:</b> {html.escape(proposal.get("unresolved_reason") or "none")}</p>
<p><b>rationale:</b> {html.escape(proposal["rationale"])}</p>
<p><b>conflicts/qualifiers:</b> {html.escape(proposal.get("conflicts_or_qualifiers") or "none")}</p>
<details open><summary>full masked context ({len(entry["chunks"])} chunks)</summary>{chunks}</details>
<p class="act"><label><input type="checkbox" class="v"> I checked every anchor and the unmarked
context for this entry</label>
<select class="a"><option value="accept">accept proposal</option>
<option value="own">I will record my own</option></select></p></section>"""


STYLE = """<style>body{font:14px sans-serif;max-width:1000px;margin:auto}pre{white-space:pre-wrap;
background:#f6f6f6;padding:6px}mark{background:#ffe58a}mark.c{background:#ffb3b3}sup{font-size:9px}
.bad{color:#b00;font-weight:bold}.p{padding:1px 6px;border-radius:4px;background:#ddd}
.difficult{background:#f4b6b6}.careful{background:#f7e3a1}section{border-top:2px solid #555;
margin-top:20px}</style>"""
SCRIPT = """<button onclick="exp()">Export verified entries (verification.json)</button>
<script>function exp(){const o={};document.querySelectorAll('section').forEach(s=>{
if(s.querySelector('.v').checked)o[s.dataset.handle]={action:s.querySelector('.a').value}});
const a=document.createElement('a');a.href=URL.createObjectURL(new Blob([JSON.stringify(
{verified:o,exported_utc:new Date().toISOString()},null,1)]));a.download='verification.json';
a.click()}</script>"""


def sheet(queue: Path, proposals: Path, handles: list[str], out: Path) -> dict:
    entries = {e["handle"]: e for e in json.loads(queue.read_bytes())}
    props = {p["handle"]: p for p in json.loads(proposals.read_bytes())["proposals"]}
    missing = [h for h in handles if h not in entries or h not in props]
    if missing:
        raise SystemExit(f"no entry or proposal for {missing[:5]}")
    body = "".join(render(entries[h], props[h]) for h in handles)
    index = "".join(f'<a href="#{h}">{h}</a> ' for h in handles)
    out.write_text(
        f"<html><head><meta charset=utf-8>{STYLE}</head><body><p>{index}</p>{body}{SCRIPT}"
        "</body></html>"
    )
    failed = [h for h in handles if check(entries[h], props[h])]
    return {"entries": len(handles), "machine_check_failures": failed}


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def plan(queue: Path, mapping: Path, arm: str, out: Path, size: int = 50) -> dict:
    """Batches of `size` handles in queue order, restricted to `arm`, with advisory inputs."""
    entries = json.loads(queue.read_bytes())
    chosen = {h for h, cs in json.loads(mapping.read_bytes()).items()
              if all(c.split("/")[1] == arm for c in cs)}  # fmt: skip
    selected = [e for e in entries if e["handle"] in chosen]
    batches = {f"batch_{n + 1:02d}": selected[n * size : (n + 1) * size]
               for n in range(-(-len(selected) // size))}  # fmt: skip
    out.mkdir(parents=True, exist_ok=True)
    (out / "inputs").mkdir(exist_ok=True)
    for name, group in batches.items():
        (out / "inputs" / f"{name}_entries.json").write_text(json.dumps(group, indent=1))
    plan = {"queue_sha256": sha(queue.read_bytes()), "mapping_sha256": sha(mapping.read_bytes()),
            "batches": {n: [e["handle"] for e in g] for n, g in batches.items()}}  # fmt: skip
    path = out / "plan.json"
    if path.exists() and json.loads(path.read_text()) != plan:
        raise SystemExit("plan.json exists with other content")
    path.write_text(json.dumps(plan, indent=1))
    return {"batches": {n: len(h) for n, h in plan["batches"].items()}, "entries": len(selected)}


def build(plan_dir: Path, queue: Path, proposals_dir: Path) -> dict:
    """One sheet per batch whose proposals are complete; finished sheets are not rebuilt."""
    plan = json.loads((plan_dir / "plan.json").read_bytes())
    if plan["queue_sha256"] != sha(queue.read_bytes()):
        raise SystemExit("queue differs from the plan")
    props = {}
    for file in sorted(proposals_dir.glob("*.json")):
        data = json.loads(file.read_bytes())
        for p in data if isinstance(data, list) else data.get("proposals", []):
            props[p["handle"]] = p  # later files supersede earlier ones
    ledger_path = plan_dir / "ledger.json"
    ledger = json.loads(ledger_path.read_text()) if ledger_path.exists() else {}
    report = {}
    for name, handles in plan["batches"].items():
        missing = [h for h in handles if h not in props]
        if missing:
            report[name] = {"state": "awaiting_proposals", "missing": len(missing)}
            continue
        digest = sha(json.dumps([props[h] for h in handles], sort_keys=True).encode())
        target = plan_dir / f"sheet_{name}.html"
        if ledger.get(name, {}).get("inputs_sha256") == digest and target.exists():
            report[name] = {"state": "kept"}
            continue
        started = time.monotonic()
        subset = plan_dir / f"proposals_{name}.json"
        subset.write_text(json.dumps({"proposals": [props[h] for h in handles]}))
        result = sheet(queue, subset, handles, target)
        ledger[name] = {"inputs_sha256": digest, "sheet_sha256": sha(target.read_bytes())}
        ledger_path.write_text(json.dumps(ledger, indent=1))
        report[name] = {"state": "built", "seconds": round(time.monotonic() - started, 2), **result}
    return report


def apply(queue, proposals, verification, review, log) -> dict:
    entries = {e["handle"]: e for e in json.loads(queue.read_bytes())}
    props = {p["handle"]: p for p in json.loads(proposals.read_bytes())["proposals"]}
    verified = json.loads(verification.read_bytes())["verified"]
    records = json.loads(review.read_bytes())
    by_handle = {r["handle"]: r for r in records}
    now = datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
    applied = []
    for handle, v in sorted(verified.items()):
        record = by_handle.get(handle)
        if record is None or record["status"] is not None:
            raise SystemExit(f"{handle}: no open review record")
        if v["action"] != "accept":
            continue  # the researcher records this one by hand
        p, problems = props[handle], check(entries[handle], props[handle])
        if problems:
            raise SystemExit(f"{handle}: {problems}")
        record.update(
            {
                k: p.get(k)
                for k in (
                    "status",
                    "reason",
                    "citations",
                    "conflict_citations",
                    "missing_components",
                    "unresolved_reason",
                    "flags",
                    "rationale",
                )
            },
            at_utc=now,
        )
        applied.append(handle)
    review.write_text(json.dumps(records, indent=1, sort_keys=True) + "\n")
    with log.open("a") as f:
        for handle in applied:
            f.write(json.dumps({"handle": handle, "action": "accept", "utc": now}) + "\n")
    return {"recorded": len(applied), "left_for_manual_record": len(verified) - len(applied)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("sheet", "apply"):
        c = sub.add_parser(name)
        c.add_argument("--queue", type=Path, required=True)
        c.add_argument("--proposals", type=Path, required=True)
    pl = sub.add_parser("plan")
    pl.add_argument("--queue", type=Path, required=True)
    pl.add_argument("--mapping", type=Path, required=True)
    pl.add_argument("--scope-arm", required=True)
    pl.add_argument("--out", type=Path, required=True)
    bu = sub.add_parser("build")
    bu.add_argument("--plan", type=Path, required=True)
    bu.add_argument("--queue", type=Path, required=True)
    bu.add_argument("--proposals-dir", type=Path, required=True)
    sub.choices["sheet"].add_argument("--handles", required=True)
    sub.choices["sheet"].add_argument("--out", type=Path, required=True)
    for flag in ("verification", "review", "log"):
        sub.choices["apply"].add_argument(f"--{flag}", type=Path, required=True)
    a = parser.parse_args()
    if a.command == "plan":
        result = plan(a.queue, a.mapping, a.scope_arm, a.out)
    elif a.command == "build":
        result = build(a.plan, a.queue, a.proposals_dir)
    elif a.command == "sheet":
        result = sheet(a.queue, a.proposals, a.handles.split(","), a.out)
    else:
        result = apply(a.queue, a.proposals, a.verification, a.review, a.log)
    print(json.dumps(result, indent=1))


if __name__ == "__main__":
    main()
