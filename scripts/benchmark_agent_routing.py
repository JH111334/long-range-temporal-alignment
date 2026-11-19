"""Static routing contract checks; not a live skill-invocation benchmark."""

from __future__ import annotations

import json
from pathlib import Path

from route_agent_capability import decide


ROOT = Path(__file__).resolve().parents[1]
REGISTRY = ROOT / ".agents" / "capabilities.json"


def check(name: str, condition: bool) -> dict[str, str]:
    """Return one bounded scenario result."""
    return {"scenario": name, "result": "PASS" if condition else "FAIL"}


def main() -> int:
    """Verify the local routing and fail-closed contract."""
    agents = (ROOT / "AGENTS.md").read_text(encoding="utf-8")
    skills_root = ROOT / ".agents" / "skills"
    skill_names = (
        "project-context-router",
        "project-change-verification",
        "project-knowledge-writeback",
    )
    skills = {name: (skills_root / name / "SKILL.md").read_text(encoding="utf-8") for name in skill_names}
    registry = json.loads(REGISTRY.read_text(encoding="utf-8"))
    entries = registry["capabilities"]
    by_id = {entry["id"]: entry for entry in entries}
    fields = {
        "id", "provider", "implementation", "capability", "read_or_write", "side_effects",
        "allowed_phases", "preconditions", "forbidden_when", "timeout_seconds", "healthcheck",
        "evidence_output",
    }
    results = [
        check("unique_capability_ids", len(by_id) == len(entries)),
        check("registry_fields", all(fields <= entry.keys() for entry in entries)),
        check("mandatory_context_trigger", "IF bắt đầu task" in agents and "project-context-router" in agents),
        check("mandatory_change_trigger", "IF sửa code" in agents and "project-change-verification" in agents),
        check("mandatory_writeback_trigger", "IF có milestone" in agents and "project-knowledge-writeback" in agents),
        check("bounded_context", "Dừng đọc" in skills["project-context-router"]),
        check("exact_file_route", by_id["local.files.exact_read"]["implementation"] == "verified_local"),
        check("missing_docs_mcp_fail_closed", by_id["shared.research.docs.search"]["implementation"] == "unverified"),
        check("missing_repo_mcp_fail_closed", by_id["shared.research.repo.scan"]["implementation"] == "unverified"),
        check("missing_diff_guard_fail_closed", by_id["shared.audit.diff.guard"]["implementation"] == "unverified"),
        check("invalid_promotion_blocked", "Không lấy tên file" in skills["project-knowledge-writeback"]),
        check("conflict_not_auto_resolved", "CONFLICT" in skills["project-knowledge-writeback"]),
        check("writeback_destination", "Checkpoint/Error vào `General`" in skills["project-knowledge-writeback"]),
        check("adjacent_scope", "Không dùng thay evaluation" in skills["project-change-verification"]),
        check("route_known_exact_file", decide("local.files.exact_read", "DISCOVER", {"known_path", "within_workspace"}) == (True, "allowed")),
        check("reject_forbidden_file", decide("local.files.exact_read", "DISCOVER", {"known_path", "within_workspace", "outside_scope"})[0] is False),
        check("reject_missing_tool", decide("missing.tool", "AUDIT", set())[0] is False),
        check("reject_unverified_provider", decide("shared.audit.diff.guard", "AUDIT", {"provider_discovered", "git_diff_exists"})[0] is False),
        check("reject_wrong_phase", decide("local.git.diff", "IMPLEMENT", {"git_worktree"})[0] is False),
        check("reject_missing_precondition", decide("local.git.diff", "AUDIT", set())[0] is False),
    ]
    output = {
        "evidence_class": "static_routing_contract",
        "note": "This checks files and guards, not actual host implicit invocation or hook execution.",
        "results": results,
    }
    print(json.dumps(output, ensure_ascii=False, indent=2))
    return 0 if all(item["result"] == "PASS" for item in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
