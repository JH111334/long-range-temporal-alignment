#!/usr/bin/env python3
"""Validate all discovered skills against Codex CLI / OpenAI skill requirements.
Enforces YAML frontmatter schema, naming conventions, and description constraints.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from datetime import datetime
from pathlib import Path

try:
    import yaml
except ImportError:
    # Fail-closed if yaml is not available
    print("PyYAML required for skill validation.", file=sys.stderr)
    sys.exit(1)

MAX_SKILL_NAME_LENGTH = 64
MAX_DESCRIPTION_LENGTH = 1024
ALLOWED_PROPERTIES = {"name", "description", "license", "allowed-tools", "metadata"}
NAME_PATTERN = re.compile(r"^[a-z0-9]+(-[a-z0-9]+)*$")

DEFAULT_ROOTS = [
    Path(r"C:\Users\GF63\.agents\skills"),
    Path(r"C:\Users\GF63\.codex\skills"),
    Path.cwd() / ".agents" / "skills",
    Path.cwd() / ".codex" / "skills",
]


def validate_skill_file(skill_md: Path) -> tuple[bool, str, dict]:
    """Validate a single SKILL.md against Codex constraints."""
    if not skill_md.is_file():
        return False, "File does not exist", {}

    try:
        content = skill_md.read_text(encoding="utf-8")
    except Exception as e:
        return False, f"UTF-8 decode error: {e}", {}

    if not content.startswith("---"):
        return False, "No YAML frontmatter found", {}

    normalized = content.replace("\r\n", "\n")
    match = re.match(r"^---\n(.*?)\n---", normalized, re.DOTALL)
    if not match:
        return False, "Invalid frontmatter format", {}

    fm_text = match.group(1)

    try:
        frontmatter = yaml.safe_load(fm_text)
        if not isinstance(frontmatter, dict):
            return False, "Frontmatter must be a YAML dictionary", {}
    except yaml.YAMLError as e:
        return False, f"Invalid YAML in frontmatter: {e}", {}

    unexpected_keys = set(frontmatter.keys()) - ALLOWED_PROPERTIES
    if unexpected_keys:
        allowed = ", ".join(sorted(ALLOWED_PROPERTIES))
        unexpected = ", ".join(sorted(unexpected_keys))
        return (
            False,
            f"Unexpected key(s) in SKILL.md frontmatter: {unexpected}. Allowed: {allowed}",
            frontmatter,
        )

    if "name" not in frontmatter:
        return False, "Missing 'name' in frontmatter", frontmatter
    if "description" not in frontmatter:
        return False, "Missing 'description' in frontmatter", frontmatter

    name = frontmatter.get("name", "")
    if not isinstance(name, str):
        return False, f"Name must be a string, got {type(name).__name__}", frontmatter
    name = name.strip()
    if not NAME_PATTERN.match(name):
        return (
            False,
            f"Name '{name}' should be hyphen-case (lowercase letters, digits, and hyphens only, no leading/trailing or double hyphens)",
            frontmatter,
        )
    if len(name) > MAX_SKILL_NAME_LENGTH:
        return (
            False,
            f"Name is too long ({len(name)} chars, max {MAX_SKILL_NAME_LENGTH})",
            frontmatter,
        )

    desc = frontmatter.get("description", "")
    if not isinstance(desc, str):
        return False, f"Description must be a string, got {type(desc).__name__}", frontmatter
    desc = desc.strip()
    if desc.startswith("[TODO:"):
        return False, "Description contains unfinished TODO placeholder", frontmatter
    if "<" in desc or ">" in desc:
        return False, "Description cannot contain angle brackets (< or >)", frontmatter
    if len(desc) > MAX_DESCRIPTION_LENGTH:
        return (
            False,
            f"Description is too long ({len(desc)} chars, max {MAX_DESCRIPTION_LENGTH})",
            frontmatter,
        )

    # Check body for unclosed code fences or TODO markers outside code blocks
    body = normalized[match.end() :]
    fence_marker = None
    fence_length = 0
    for line in body.splitlines():
        fence = re.match(r"^[ \t]*(?:(?:[-+*]|\d+[.)])[ \t]+)?(`{3,}|~{3,})(.*)$", line)
        if fence:
            marker = fence.group(1)
            if fence_marker is None:
                fence_marker = marker[0]
                fence_length = len(marker)
            elif (
                marker[0] == fence_marker
                and len(marker) >= fence_length
                and not fence.group(2).strip()
            ):
                fence_marker = None
                fence_length = 0
            continue

        if fence_marker is None and re.fullmatch(r"[ ]{0,3}\[TODO:[^\n]*\][ \t]*", line):
            return False, "Skill instructions contain an unfinished TODO placeholder", frontmatter

    return True, "Skill is valid!", frontmatter


def run_audit(output_json: Path | None = None) -> int:
    """Enumerate all active skills and output validation report."""
    results = []
    global_count = 0
    local_count = 0
    matt_wrapper_count = 0
    vendored_scanned_count = 0

    seen_paths = set()

    # 1. Active global and local roots
    for root in DEFAULT_ROOTS:
        if not root.is_dir():
            continue
        is_local = (root == Path.cwd() / ".agents" / "skills") or (root == Path.cwd() / ".codex" / "skills")
        for skill_dir in sorted(root.iterdir()):
            if not skill_dir.is_dir():
                continue
            skill_md = skill_dir / "SKILL.md"
            if not skill_md.is_file():
                continue
            if skill_md.resolve() in seen_paths:
                continue
            seen_paths.add(skill_md.resolve())

            valid, message, fm = validate_skill_file(skill_md)
            skill_name = fm.get("name", skill_dir.name)
            
            # Check if it's a wrapper for matt
            is_matt_wrapper = False
            if (skill_dir / "agents").is_dir() and any((skill_dir / "agents").glob("*.yaml")):
                is_matt_wrapper = True
                matt_wrapper_count += 1

            if is_local:
                local_count += 1
            else:
                global_count += 1

            results.append({
                "name": skill_name,
                "directory": skill_dir.name,
                "path": str(skill_md),
                "scope": "LOCAL" if is_local else "GLOBAL",
                "is_matt_wrapper": is_matt_wrapper,
                "valid": valid,
                "message": message,
            })

    # Count vendored matt skills if existing (dormant/scanned)
    vendored_root = Path(r"C:\Users\GF63\.skills\global\agent-system-v2\vendor\matt")
    if vendored_root.is_dir():
        for f in vendored_root.rglob("SKILL.md"):
            vendored_scanned_count += 1

    valid_count = sum(1 for r in results if r["valid"])
    invalid_count = sum(1 for r in results if not r["valid"])

    report = {
        "schema_version": "codex-skill-validation-v5",
        "timestamp": datetime.now().isoformat(),
        "total_discovered": len(results),
        "valid": valid_count,
        "invalid": invalid_count,
        "global": global_count,
        "local": local_count,
        "matt_wrappers": matt_wrapper_count,
        "vendored_scanned": vendored_scanned_count,
        "skills": results,
    }

    if output_json:
        output_json.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(f"Audit report written to: {output_json}")

    print(
        f"Validation Summary: Total={len(results)} | Valid={valid_count} | Invalid={invalid_count} | "
        f"Global={global_count} | Local={local_count} | MattWrappers={matt_wrapper_count} | "
        f"Vendored={vendored_scanned_count}"
    )

    if invalid_count > 0:
        print("\nInvalid skills detected:")
        for r in results:
            if not r["valid"]:
                print(f"  - [{r['directory']}] {r['path']}: {r['message']}")
        return 1

    return 0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        default=Path("artifacts/audits/agent-system/CODEX_SKILL_VALIDATION_V5.json"),
        help="Path to output JSON report",
    )
    args = parser.parse_args()
    code = run_audit(args.output)
    sys.exit(code)


if __name__ == "__main__":
    main()
