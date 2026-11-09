import json

from test_generation_roster import *  # noqa: F401,F403
from test_generation_roster import build, make_scope


def test_dump(world):
    directory, record_sha, _, _, _ = world
    built = build(world, scope=make_scope(directory, record_sha, {"q_a", "q_b"}, repeats=(1, 2)))
    rec = json.loads(built.record.model_dump_json())
    print("DEBUGREC", json.dumps({k: v for k, v in rec.items() if k != "rows"}, sort_keys=True))
    for r in rec["rows"]:
        print("DEBUGROW", json.dumps(r, sort_keys=True))
    print("DEBUGSHA", built.sha256)
