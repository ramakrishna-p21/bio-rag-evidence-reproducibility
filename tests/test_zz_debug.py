import hashlib

import test_generation_roster as r
from biorag_repro.benchmark.development_view import ArtifactRef
from biorag_repro.core.hashing import sha256_bytes
from biorag_repro.core.storage import serialize_project_json
from biorag_repro.benchmark.split import PaperSplit  # noqa


def test_dump():
    items_bytes = r.encode_jsonl(r.ITEMS)
    split_bytes = serialize_project_json(r.SPLIT, r.PaperSplit)
    print("DEBUGIN items", sha256_bytes(items_bytes), len(items_bytes))
    print("DEBUGIN split", sha256_bytes(split_bytes), len(split_bytes))
    print("DEBUGIN splitjson", split_bytes.decode()[:600].replace("\n", " "))
    for i, it in enumerate(r.ITEMS):
        print("DEBUGIN item", i, hashlib.sha256(r.encode_jsonl([it])).hexdigest()[:16])
