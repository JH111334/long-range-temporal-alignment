#!/usr/bin/env python3
"""
scripts/audit_cross_project_hygiene.py
Kiểm định xác định vệ sinh thư mục và quản trị tầng diễn ngôn phái sinh (Documenting)
trên toàn bộ 8 dự án nghiên cứu và đồ án UIT.

Tự động quét hệ thống tệp sống (live filesystem), không nhúng cứng số lượng hay giả định.
Xuất kết quả chi tiết kèm danh sách đường dẫn phân loại vào:
artifacts/audits/documenting/FINAL_CROSS_PROJECT_HYGIENE_AUDIT.json
"""

import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Set

BASE_DIR = Path(r"C:\Users\GF63\Documents")
OUTPUT_FILE = (
    Path(r"C:\Users\GF63\Documents\long-range-temporal-alignment")
    / "artifacts"
    / "audits"
    / "documenting"
    / "FINAL_CROSS_PROJECT_HYGIENE_AUDIT.json"
)

PROJECTS = [
    "DS111-Retrieval-Data-Analysis",
    "DS201-Temporal-Retrieval-Learning",
    "DS314-Evidence-Grounded-Retrieval",
    "DS335-VLM-Research",
    "DSC-TinDipLaPo",
    "Evidence-Grounded-Text-Agent",
    "general-reasoning-fact-checking",
    "long-range-temporal-alignment",
]

CANONICAL_ROOT_FILES: Set[str] = {
    ".gitignore",
    ".gitattributes",
    ".env.example",
    ".python-version",
    "LICENSE",
    "pyproject.toml",
    "README.md",
    "requirements.txt",
    "requirements-research.txt",
    "requirements-ui.txt",
    "uv.lock",
    "Report.md",
    "AGENTS.md",
    "CITATION.cff",
    "CONTRIBUTING.md",
    "SECURITY.md",
}

MISTAKE_REQUIRED_FIELDS = [
    "Context",
    "Competing branches / assumptions",
    "Failure",
    "Evidence cue",
    "Why the chosen assumption failed",
    "Change made",
    "Outcome",
    "Generalized prevention rule",
]

INSIGHT_REQUIRED_FIELDS = [
    "Context",
    "Alternatives considered",
    "Discriminating evidence",
    "Insight",
    "Decision / change",
    "Result",
    "Generalized rule",
]


def audit_project_root(proj_dir: Path) -> Dict[str, Any]:
    root_noncanonical = []
    for item in proj_dir.iterdir():
        if item.is_file():
            if item.name not in CANONICAL_ROOT_FILES and not item.name.startswith("requirements-"):
                root_noncanonical.append(str(item.relative_to(proj_dir)))
    return {
        "root_noncanonical_files": root_noncanonical,
        "root_clean": len(root_noncanonical) == 0,
    }


def audit_temp_and_scratch(proj_dir: Path) -> Dict[str, Any]:
    tmp_dir = proj_dir / ".tmp"
    scratch_dir = proj_dir / "scratch"

    tmp_files = []
    scratch_files = []
    manual_review_items = []

    if tmp_dir.exists():
        try:
            for root, dirs, files in os.walk(tmp_dir):
                for f in files:
                    fp = Path(root) / f
                    tmp_files.append(str(fp.relative_to(proj_dir)))
        except PermissionError as e:
            manual_review_items.append(f"{tmp_dir.name}: PermissionError access denied ({e})")

        # Specific check for locked directories like DSC pytest-run3-runtime
        for item in tmp_dir.iterdir():
            if item.is_dir() and "pytest" in item.name:
                try:
                    list(item.iterdir())
                except PermissionError:
                    manual_review_items.append(str(item.relative_to(proj_dir)))

    if scratch_dir.exists():
        try:
            for root, dirs, files in os.walk(scratch_dir):
                for f in files:
                    fp = Path(root) / f
                    scratch_files.append(str(fp.relative_to(proj_dir)))
        except PermissionError as e:
            manual_review_items.append(f"{scratch_dir.name}: PermissionError ({e})")

    return {
        "tmp_files": tmp_files,
        "tmp_files_count": len(tmp_files),
        "scratch_files": scratch_files,
        "scratch_files_count": len(scratch_files),
        "manual_review_items": manual_review_items,
    }


def audit_governance_files(proj_dir: Path) -> Dict[str, Any]:
    # Check RULE.md
    rule_path = proj_dir / "Documents" / "Documenting" / "RULE.md"
    rule_exists = rule_path.exists()
    rule_has_significance_gate = False
    rule_has_17_exclusions = False
    if rule_exists:
        text = rule_path.read_text(encoding="utf-8")
        rule_has_significance_gate = "Significance Admission Gate" in text
        rule_has_17_exclusions = "17 Lỗi Tầm Thường" in text or "17 Explicit Exclusions" in text

    # Check AGENTS.md
    agents_path = proj_dir / "AGENTS.md"
    agents_exists = agents_path.exists()
    agents_has_doc_block = False
    agents_has_rule_pointer = False
    if agents_exists:
        atext = agents_path.read_text(encoding="utf-8")
        agents_has_doc_block = "## Documenting" in atext
        agents_has_rule_pointer = "Documents/Documenting/RULE.md" in atext

    # Check local router / skill
    skills_dir = proj_dir / ".agents" / "skills"
    local_skills_found = []
    local_skill_has_rule_pointer = False
    router_status = "AGENTS.md + global skill sufficient"

    if skills_dir.exists():
        for s in skills_dir.iterdir():
            if s.is_dir():
                local_skills_found.append(s.name)
                skill_md = s / "SKILL.md"
                if skill_md.exists():
                    stext = skill_md.read_text(encoding="utf-8")
                    if "Documents/Documenting/RULE.md" in stext:
                        local_skill_has_rule_pointer = True

        if local_skills_found:
            if local_skill_has_rule_pointer:
                router_status = f"Local skills {local_skills_found} reconciled with RULE.md"
            else:
                router_status = f"Local skills {local_skills_found} present"

    return {
        "rule_md_exists": rule_exists,
        "rule_has_significance_gate": rule_has_significance_gate,
        "rule_has_17_exclusions": rule_has_17_exclusions,
        "agents_md_exists": agents_exists,
        "agents_has_doc_block": agents_has_doc_block,
        "agents_has_rule_pointer": agents_has_rule_pointer,
        "local_skills_found": local_skills_found,
        "router_status": router_status,
    }


def audit_documenting_content(proj_dir: Path) -> Dict[str, Any]:
    base_doc = proj_dir / "Documents" / "Documenting"
    evo_path = base_doc / "Project-Evolution.md"
    comp_dir = base_doc / "Components"

    files_to_check = []
    if evo_path.exists():
        files_to_check.append(evo_path)
    if comp_dir.exists():
        files_to_check.extend(sorted(list(comp_dir.glob("*.md"))))

    total_mistakes = 0
    total_insights = 0
    mistakes_compliant = 0
    insights_compliant = 0
    non_compliant_items = []
    exact_duplicate_lines = []

    evo_text = evo_path.read_text(encoding="utf-8") if evo_path.exists() else ""

    for f in files_to_check:
        text = f.read_text(encoding="utf-8")
        lines = text.splitlines()

        # Check duplicate lines between Project-Evolution and Components
        if f != evo_path and evo_text:
            for l in lines:
                if (l.startswith("  Failure:") or l.startswith("  Insight:")) and len(l) > 30:
                    if l in evo_text:
                        exact_duplicate_lines.append(f"{f.name}: {l[:60]}")

        i = 0
        while i < len(lines):
            line = lines[i]
            # Mistake bullet
            if re.match(r"^\s*-\s*\[.*?\]\s*\[LAYER:.*?\]\s*\[SEVERITY:.*?\]", line):
                total_mistakes += 1
                header = line.strip()
                block = [line]
                i += 1
                while i < len(lines) and not re.match(r"^\s*-\s*\[", lines[i]) and not re.match(r"^\s*##\s+", lines[i]):
                    block.append(lines[i])
                    i += 1
                btext = "\n".join(block)
                missing = [req for req in MISTAKE_REQUIRED_FIELDS if f"  {req}:" not in btext]
                if missing:
                    non_compliant_items.append({
                        "file": f.name,
                        "header": header,
                        "kind": "MISTAKE",
                        "missing_fields": missing,
                    })
                else:
                    mistakes_compliant += 1
                continue

            # Insight bullet
            elif re.match(r"^\s*-\s*\[.*?\]\s*\[LAYER:.*?\]", line):
                total_insights += 1
                header = line.strip()
                block = [line]
                i += 1
                while i < len(lines) and not re.match(r"^\s*-\s*\[", lines[i]) and not re.match(r"^\s*##\s+", lines[i]):
                    block.append(lines[i])
                    i += 1
                btext = "\n".join(block)
                missing = [req for req in INSIGHT_REQUIRED_FIELDS if f"  {req}:" not in btext]
                if missing:
                    non_compliant_items.append({
                        "file": f.name,
                        "header": header,
                        "kind": "INSIGHT",
                        "missing_fields": missing,
                    })
                else:
                    insights_compliant += 1
                continue
            else:
                i += 1

    return {
        "files_checked": len(files_to_check),
        "total_mistakes": total_mistakes,
        "mistakes_compliant": mistakes_compliant,
        "total_insights": total_insights,
        "insights_compliant": insights_compliant,
        "non_compliant_items": non_compliant_items,
        "exact_duplicate_lines": exact_duplicate_lines,
        "all_compliant": len(non_compliant_items) == 0 and len(exact_duplicate_lines) == 0,
    }


def main():
    report: Dict[str, Any] = {
        "audit_metadata": {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "scope": "All 8 UIT research and course repositories",
            "validator": "scripts/audit_cross_project_hygiene.py",
            "enforcement_standard": "Significance Admission Gate (Section 10-14)",
        },
        "projects": {},
        "summary": {},
    }

    grand_total_root_noncanonical = 0
    grand_total_tmp_files = 0
    grand_total_scratch_files = 0
    grand_total_manual_review = 0
    grand_total_mistakes = 0
    grand_total_insights = 0
    grand_total_compliant_mistakes = 0
    grand_total_compliant_insights = 0
    grand_total_non_compliant = 0
    grand_total_duplicates = 0

    all_rules_valid = True
    all_agents_valid = True

    for p in PROJECTS:
        proj_dir = BASE_DIR / p
        if not proj_dir.exists():
            continue

        root_audit = audit_project_root(proj_dir)
        temp_scratch_audit = audit_temp_and_scratch(proj_dir)
        gov_audit = audit_governance_files(proj_dir)
        doc_audit = audit_documenting_content(proj_dir)

        proj_result = {
            "root_audit": root_audit,
            "temp_scratch_audit": temp_scratch_audit,
            "governance_audit": gov_audit,
            "documenting_audit": doc_audit,
        }
        report["projects"][p] = proj_result

        grand_total_root_noncanonical += len(root_audit["root_noncanonical_files"])
        grand_total_tmp_files += temp_scratch_audit["tmp_files_count"]
        grand_total_scratch_files += temp_scratch_audit["scratch_files_count"]
        grand_total_manual_review += len(temp_scratch_audit["manual_review_items"])

        grand_total_mistakes += doc_audit["total_mistakes"]
        grand_total_insights += doc_audit["total_insights"]
        grand_total_compliant_mistakes += doc_audit["mistakes_compliant"]
        grand_total_compliant_insights += doc_audit["insights_compliant"]
        grand_total_non_compliant += len(doc_audit["non_compliant_items"])
        grand_total_duplicates += len(doc_audit["exact_duplicate_lines"])

        if not (gov_audit["rule_md_exists"] and gov_audit["rule_has_significance_gate"]):
            all_rules_valid = False
        if not (gov_audit["agents_md_exists"] and gov_audit["agents_has_rule_pointer"]):
            all_agents_valid = False

    # Check global skill
    global_skill_path = Path(r"C:\Users\GF63\.gemini\config\skills\documenting-governance\SKILL.md")
    global_skill_exists = global_skill_path.exists()
    global_skill_compliant = False
    if global_skill_exists:
        gstext = global_skill_path.read_text(encoding="utf-8")
        global_skill_compliant = "Significance Admission Gate" in gstext and "17 Explicit Exclusions" in gstext

    report["global_governance"] = {
        "global_skill_path": str(global_skill_path),
        "exists": global_skill_exists,
        "significance_gate_present": global_skill_compliant,
    }

    # Verdict
    has_hygiene_fail = grand_total_root_noncanonical > 0 or grand_total_tmp_files > 0
    has_doc_fail = (
        grand_total_non_compliant > 0
        or grand_total_duplicates > 0
        or not all_rules_valid
        or not all_agents_valid
        or not global_skill_compliant
    )

    if not has_hygiene_fail and not has_doc_fail:
        if grand_total_manual_review > 0:
            verdict = "PASS_WITH_MANUAL_REVIEW"
        else:
            verdict = "PASS_FINAL_CROSS_PROJECT_AUDIT"
    elif has_hygiene_fail:
        verdict = "FAIL_HYGIENE"
    elif has_doc_fail:
        verdict = "FAIL_DOCUMENTING_SIGNAL_QUALITY"
    else:
        verdict = "FAIL_REGRESSION"

    report["summary"] = {
        "verdict": verdict,
        "projects_scanned": len(report["projects"]),
        "root_noncanonical_files": grand_total_root_noncanonical,
        "tmp_files_remaining": grand_total_tmp_files,
        "scratch_files_remaining": grand_total_scratch_files,
        "manual_review_items": grand_total_manual_review,
        "total_documenting_items": grand_total_mistakes + grand_total_insights,
        "total_mistakes": grand_total_mistakes,
        "compliant_mistakes": grand_total_compliant_mistakes,
        "total_insights": grand_total_insights,
        "compliant_insights": grand_total_compliant_insights,
        "non_compliant_items": grand_total_non_compliant,
        "exact_duplicate_lines": grand_total_duplicates,
        "rules_updated_and_valid": all_rules_valid,
        "agents_updated_and_valid": all_agents_valid,
        "global_skill_valid": global_skill_compliant,
    }

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_FILE.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Audit completed. Verdict: {verdict}")
    print(f"Report written to: {OUTPUT_FILE}")
    print(json.dumps(report["summary"], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
