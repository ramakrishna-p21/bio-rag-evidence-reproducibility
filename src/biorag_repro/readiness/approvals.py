"""Read the tracked approval values (`approvals/policy.json`) into an `ApprovalPolicy`.

Nothing is substituted for a file that is missing or wrong: a missing policy must not read as
"nothing approved yet". The file is read once without following links, parsed strictly (no
duplicate keys, no NaN, UTF-8 only) and must hold exactly the approved structure. This module
is the only reader of the file; no option, environment variable or configuration selects
another path. It is part of the G09 source digest; the values file it reads is not.
"""

from pathlib import Path
from typing import Final

from pydantic import ValidationError

from biorag_repro.readiness.files import FileProblem, loads_strict, read_regular
from biorag_repro.readiness.git_state import APPROVAL_FILE
from biorag_repro.readiness.policy import LEGACY_SLOT_NAMES, SLOT_NAMES, ApprovalPolicy

POLICY_VERSION: Final = "3"


class PolicyError(ValueError):
    """The tracked approval values could not be read or are not the approved structure."""


def load_policy(repo: Path) -> ApprovalPolicy:
    """The tracked approval policy of the checkout at `repo`; any problem is an error."""
    try:
        value = loads_strict(read_regular(repo, APPROVAL_FILE))
    except FileProblem as error:
        raise PolicyError(f"{APPROVAL_FILE}: {error}") from error
    slots = value.get("slots") if isinstance(value, dict) else None
    version = value.get("policy_version") if isinstance(value, dict) else None
    expected = set(SLOT_NAMES if version == POLICY_VERSION else LEGACY_SLOT_NAMES)
    if not isinstance(slots, dict) or set(slots) != expected:
        raise PolicyError(f"{APPROVAL_FILE} must name exactly {len(expected)} active slots")
    if version not in ("1", "2", POLICY_VERSION):
        raise PolicyError(f"{APPROVAL_FILE}: unsupported policy_version")
    if version in ("2", POLICY_VERSION):
        try:
            authority = loads_strict(read_regular(repo, "approvals/researcher_decisions.json"))
            if (
                authority["protocol_version"] != "researcher_authority_v1"
                or authority["authority"] != "Rama Krishna Pudota"
                or authority["authority_role"] != "researcher_of_record"
            ):
                raise ValueError("researcher authority differs from the prospective amendment")
        except (FileProblem, KeyError, TypeError, ValueError) as error:
            raise PolicyError(f"researcher authority record is unusable: {error}") from error
    try:
        return ApprovalPolicy.model_validate(value)
    except ValidationError as error:
        raise PolicyError(f"{APPROVAL_FILE}: {error}") from error
