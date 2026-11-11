"""A synthetic rehearsal world for the guard-connected start. Nothing here is real content.

The renderer, the protected inputs and the readiness evidence are fixtures: a whitespace
tokenizer, invented questions that contain a canary string, and the Phase 9 fixture policy.
Everything they produce carries the rehearsal root kind, the mock backend and the
`synthetic_test` purpose, so none of it can be mistaken for production evidence.
"""

import hashlib
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

from finaltest_audit_support import ARM, CONFIGS, PLAN, settings_for
from finaltest_campaign_support import CANARY, FINAL_ITEMS, population, sha
from finaltest_support import made
from generation_support import to_toml
from test_generation_attempts import Clock, environment
from test_readiness_guard import Fixture

from biorag_repro.core.hashing import sha256_bytes
from biorag_repro.core.manifest import ArtifactRef
from biorag_repro.finaltest.layout import StorageAuthorization
from biorag_repro.finaltest.protected import ProtectedContext, ProtectedItem
from biorag_repro.finaltest.start import Assessed, StartRequest
from biorag_repro.generation.driver import MockBackend
from biorag_repro.generation.prompt import RenderedInput
from biorag_repro.generation.receipt import token_ids_sha256

__all__ = ["ARM", "CANARY", "Clock", "FakeRenderer", "SyntheticInputs", "World", "make_world"]


class FakeTokenizer:
    def __call__(self, text: str, **_options: object) -> dict[str, list[int]]:
        return {"input_ids": [sum(word.encode()) % 997 + 1 for word in text.split()] or [1]}


class FakeRenderer:
    """Renders like `InputRenderer` for the checks the start relies on, without a tokenizer."""

    def __init__(self, config):
        self.config = config
        self.tokenizer = FakeTokenizer()
        self.tokenizer_root = None

    def render(self, question: str, context: str | None, context_tokens: int | None):
        settings = self.config.settings
        retrieves = settings.matrix_arm != "closed_book"
        if retrieves != (context is not None) or (context is None) != (context_tokens is None):
            raise ValueError(
                "context and its token count are supplied exactly when the arm retrieves"
            )
        text = f"<user>{question if context is None else context + chr(10) + question}</user>"
        ids = self.tokenizer(text)["input_ids"]
        return RenderedInput(
            resolved_config_sha256=self.config.configuration_sha256,
            prompt_sha256=settings.prompt_sha256,
            chat_template_sha256=sha("template"),
            question=question,
            context=context,
            context_sha256=None if context is None else sha256_bytes(context.encode()),
            context_tokens=context_tokens,
            rendered_text=text,
            rendered_sha256=sha256_bytes(text.encode()),
            input_tokens=len(ids),
            input_token_ids_sha256=token_ids_sha256(ids),
        )


class SyntheticInputs:
    """Final-test-shaped inputs. `change` may rewrite one item or context to break a check."""

    def __init__(self, arm: str, *, change: Callable | None = None):
        self.arm = arm
        self.change = change
        self.calls = 0

    def items(self) -> list[ProtectedItem]:
        self.calls += 1
        out = [
            ProtectedItem(item, paper, f"{CANARY} {item}?", sha(CANARY + item))
            for item, paper in sorted(FINAL_ITEMS.items())
        ]
        return self.change("items", out) if self.change else out

    def context(self, item_id: str, condition: str):
        self.calls += 1
        if self.arm == "closed_book":
            found = None
        else:
            text = f"evidence for {item_id} from {condition}"
            found = ProtectedContext(
                exact_context=text,
                generator_tokens=len(text.split()),
                evidence_ids=(f"chunk-{item_id}-{condition}-1", f"chunk-{item_id}-{condition}-2"),
                source_identity_sha256=sha(f"source-{item_id}-{condition}"),
                reference=ArtifactRef(
                    artifact_id=f"ctx-{item_id}-{condition}",
                    logical_ref=f"contexts/{item_id}-{condition}.json",
                    sha256=sha(f"ctx-{item_id}-{condition}"),
                ),
            )
        return self.change("context", (item_id, condition, found)) if self.change else found


@dataclass
class World:
    base: Path
    root: Path
    anchors: Path
    witnesses: Path
    auth: StorageAuthorization
    fixture: Fixture
    arm: str
    request: StartRequest
    clock: Clock = field(default_factory=Clock)
    guard_calls: int = 0

    def assess(self) -> Assessed:
        self.guard_calls += 1
        return Assessed(self.fixture.authorize(), self.fixture.live.report)

    def with_request(self, **changes: object) -> StartRequest:
        values = {f: getattr(self.request, f) for f in StartRequest.__dataclass_fields__}
        return StartRequest(**(values | changes))

    def tree(self) -> dict[str, tuple]:
        """Path -> (mode, size, hash) of everything below the root, witnesses and anchors."""
        out = {}
        for top in (self.root, self.anchors, self.witnesses):
            if not top.exists():
                continue
            for path in sorted(top.rglob("*")):
                stat = path.lstat()
                digest = None if path.is_dir() else hashlib.sha256(path.read_bytes()).hexdigest()
                out[str(path)] = (stat.st_mode, stat.st_size, digest)
        return out


def make_world(base: Path, arm: str = ARM, **request_changes: object) -> World:
    (base / "storage").mkdir(parents=True)
    (base / "readiness").mkdir()
    root, anchors, auth = made(base / "storage")
    witnesses = base / "witnesses"

    request = StartRequest(
        root=root,
        authorization=auth,
        anchors=anchors,
        witnesses=witnesses,
        plan=PLAN,
        population=population(),
        matrix_arm=arm,
        config_toml=to_toml(settings_for(arm)).encode(),
        renderer=FakeRenderer(CONFIGS[arm]),
        seeds=None,
        inputs=SyntheticInputs(arm),
        backend_kind="mock",
    )
    world = World(base, root, anchors, witnesses, auth, Fixture(base / "readiness"), arm, request)
    if request_changes:
        world.request = world.with_request(**request_changes)
    return world


def mock_environment():
    return environment("mock")


__all__ += ["MockBackend", "mock_environment"]
