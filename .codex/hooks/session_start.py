"""Emit a bounded context receipt for the current repository."""

from __future__ import annotations

import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
REQUIRED = (
    ROOT / "AGENTS.md",
    ROOT / "Documents" / "HOME" / "Trang chủ.md",
    ROOT / "Documents" / "Decision-Making" / "12_PROJECT-CONTEXT.md",
)


def main() -> int:
    """Check root and pointers without reading or writing project content."""
    event = json.load(sys.stdin)
    cwd = Path(event.get("cwd", "")).resolve()
    if cwd != ROOT and ROOT not in cwd.parents:
        print(json.dumps({"systemMessage": "Project hook: cwd ngoài repo; không cấp context."}))
        return 0
    missing = [str(path.relative_to(ROOT)) for path in REQUIRED if not path.is_file()]
    if missing:
        print(json.dumps({"systemMessage": "Project hook: thiếu pointer " + ", ".join(missing)}))
        return 0
    context = (
        "Repo temporal alignment: đọc AGENTS.md, Documents/HOME/Trang chủ.md, "
        "Documents/Decision-Making/12_PROJECT-CONTEXT.md; dùng project-context-router "
        "và dừng đọc khi đủ objective, state, authority, next action, verification."
    )
    print(json.dumps({"hookSpecificOutput": {"hookEventName": "SessionStart", "additionalContext": context}}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
