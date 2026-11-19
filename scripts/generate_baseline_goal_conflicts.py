"""Generate BASELINE_CONTEXT_AND_GOAL_CONFLICTS.json before any modifications.
"""

import hashlib
import json
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(r"C:\Users\GF63\Documents\long-range-temporal-alignment")

def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()

files_to_check = [
    "Documents/Decision-Making/19_STAGE03_REPRESENTATION_FIRST_CONTRACT.md",
    "Reports/03_TARGETED_OBJECTIVES_MCQA/STAGE_03_REPRESENTATION_FIRST_SPEC.md",
    "Documents/Branches/v2-targeted-continual-alignment/Current-State.md",
    "Documents/Global/Branch-Registry.md",
    "Documents/HOME/Trang chủ.md",
    "README.md",
    "Report.md",
    ".agents/skills/temporal-alignment-research/SKILL.md",
    ".agents/skills/temporal-research-planning/SKILL.md",
    ".agents/skills/project-context-router/SKILL.md",
    ".agents/skills/project-change-verification/SKILL.md",
    ".agents/skills/project-knowledge-writeback/SKILL.md"
]

conflicts = []

# Conflict 1: MCQA role (Representation-First Spec vs New Goal 1)
spec_path = REPO_ROOT / "Reports" / "03_TARGETED_OBJECTIVES_MCQA" / "STAGE_03_REPRESENTATION_FIRST_SPEC.md"
adr19_path = REPO_ROOT / "Documents" / "Decision-Making" / "19_STAGE03_REPRESENTATION_FIRST_CONTRACT.md"

conflicts.append({
    "conflict_id": "CONF-01-MCQA-DEFINITION-OF-SUCCESS",
    "files_involved": [
        {"path": str(spec_path.relative_to(REPO_ROOT)), "sha256": sha256_file(spec_path) if spec_path.exists() else "MISSING"},
        {"path": str(adr19_path.relative_to(REPO_ROOT)), "sha256": sha256_file(adr19_path) if adr19_path.exists() else "MISSING"}
    ],
    "current_implied_goal": "Representation-First (2026-10-19): MCQA is merely a downstream behavioral sanity probe, NOT the definition of success. Internal representation distance / logit delta was prioritized over closed-answer accuracy.",
    "proposed_new_goal": "Goal 1 (2026-11-05): YEAR-only closed-answer temporal QA correctness (MCQA 4-choice A/B/C/D + binary Yes/No) is the primary required outcome. Correct content and correct year required.",
    "authority": "User (Research Owner) + ChatGPT",
    "conflict_type": "GOAL_DEFINITION_SUPERSEDED",
    "decision": "Supersede ADR 19 with new canonical Goal Ladder ADR (DECISION-GOAL-LADDER-20261105-112). Keep historical files intact with superseding metadata."
})

# Conflict 2: Stage 03 5-Epoch Status Claim vs Evidence Gate
conflicts.append({
    "conflict_id": "CONF-02-STAGE03-5EPOCHS-STATUS-CLAIM",
    "files_involved": [
        {"path": "Reports/6-11Report/EVIDENCE-INDEX.json", "sha256": sha256_file(REPO_ROOT / "Reports/6-11Report/EVIDENCE-INDEX.json")},
        {"path": "Documents/Branches/v2-targeted-continual-alignment/Current-State.md", "sha256": sha256_file(REPO_ROOT / "Documents/Branches/v2-targeted-continual-alignment/Current-State.md")}
    ],
    "current_implied_goal": "Prior reports labeled 5-epoch run as 5EPOCHS_EXECUTED_AND_VERIFIED or implied Goal 1 gain achieved.",
    "proposed_new_goal": "Under Goal 1, 5-epoch results are quarantined as METRIC_ARITHMETIC_ONLY pending verified GPU replay on 128 fixed samples and independent test split.",
    "authority": "User (Research Owner) + ChatGPT",
    "conflict_type": "EVIDENCE_STATUS_OVERCLAIM",
    "decision": "Maintain QUARANTINED_PENDING_PROVENANCE and HOLD_NO_GO_TRAINING in Current-State.md and crosswalk."
})

# Conflict 3: Lack of Formal Goal Ladder in Branch Registry and Skills
conflicts.append({
    "conflict_id": "CONF-03-UNMAPPED-GOAL-LADDER-IN-REGISTRY",
    "files_involved": [
        {"path": "Documents/Global/Branch-Registry.md", "sha256": sha256_file(REPO_ROOT / "Documents/Global/Branch-Registry.md")},
        {"path": ".agents/skills/temporal-alignment-research/SKILL.md", "sha256": sha256_file(REPO_ROOT / ".agents/skills/temporal-alignment-research/SKILL.md")}
    ],
    "current_implied_goal": "Branches and skills lacked explicit G1 (YEAR-closed), G2 (YEAR+MONTH-closed), G3 (YEAR/MONTH-general/open QA) mapping.",
    "proposed_new_goal": "Align all branches, stages, and skills to reference the G1 -> G2 -> G3 goal ladder explicitly, separating outcomes from mechanisms.",
    "authority": "User (Research Owner) + ChatGPT",
    "conflict_type": "MISSING_TAXONOMY_ALIGNMENT",
    "decision": "Update Branch Registry, create Crosswalk, update local skills with Goal validation guards."
})

output = {
    "audit_type": "BASELINE_CONTEXT_AND_GOAL_CONFLICTS",
    "generated_at": datetime.now().isoformat(),
    "authority": "User (Research Owner) + ChatGPT",
    "active_branch_during_alignment": "GLOBAL_MAINTENANCE",
    "total_conflicts_identified": len(conflicts),
    "conflicts": conflicts,
    "baseline_files_snapshot": {
        f: sha256_file(REPO_ROOT / f) if (REPO_ROOT / f).exists() else "MISSING"
        for f in files_to_check
    }
}

staging_out = REPO_ROOT / "BASELINE_CONTEXT_AND_GOAL_CONFLICTS.json"
with open(staging_out, "w", encoding="utf-8") as f:
    json.dump(output, f, indent=2, ensure_ascii=False)

print(f"Generated {staging_out} with {len(conflicts)} conflicts documented.")
