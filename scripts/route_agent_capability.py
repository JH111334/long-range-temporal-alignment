"""Fail-closed local capability gate; callers must use it before optional tools."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


REGISTRY = Path(__file__).resolve().parents[1] / ".agents" / "capabilities.json"


def decide(capability_id: str, phase: str, facts: set[str]) -> tuple[bool, str]:
    """Return allow/deny for one registry capability and known runtime facts."""
    entries = json.loads(REGISTRY.read_text(encoding="utf-8"))["capabilities"]
    entry = next((item for item in entries if item["id"] == capability_id), None)
    if entry is None:
        return False, "unknown_capability"
    if entry["implementation"] != "verified_local":
        return False, "implementation_unverified"
    if phase not in entry["allowed_phases"]:
        return False, "phase_forbidden"
    missing = set(entry["preconditions"]) - facts
    if missing:
        return False, "missing_precondition:" + ",".join(sorted(missing))
    forbidden = set(entry["forbidden_when"]) & facts
    if forbidden:
        return False, "forbidden_condition:" + ",".join(sorted(forbidden))
    return True, "allowed"


def main() -> int:
    """Check one proposed tool use without executing the provider."""
    parser = argparse.ArgumentParser()
    parser.add_argument("capability_id")
    parser.add_argument("phase")
    parser.add_argument("--fact", action="append", default=[])
    args = parser.parse_args()
    allowed, reason = decide(args.capability_id, args.phase, set(args.fact))
    print(json.dumps({"allowed": allowed, "reason": reason}, ensure_ascii=False))
    return 0 if allowed else 2


if __name__ == "__main__":
    raise SystemExit(main())
