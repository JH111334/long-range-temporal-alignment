"""Script to cleanly swap Arm 2 and Arm 3 designations across canonical receipts and reports.
- Old Arm 2 (Targeted Base V2) -> New Arm 3 (Targeted Base V2, User Scope)
- Old Arm 3 (MaT-LoRA FoNE) -> New Arm 2 (MaT-LoRA FoNE, Partner Scope)
"""

from __future__ import annotations

import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

def swap_four_arm_receipt():
    receipt_path = REPO_ROOT / "Documents" / "Documenting" / "Results" / "STAGE03_3B_V2_20261026" / "FOUR_ARM_YEAR_ONLY_RECEIPT.json"
    if not receipt_path.exists():
        print(f"File not found: {receipt_path}")
        return
    with open(receipt_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    models = data.get("models", {})
    old_arm2 = models.get("arm2_targeted_base")
    old_arm3 = models.get("arm3_matlora_fone")

    if old_arm2 and old_arm3:
        old_arm2["arm_name"] = "arm3_targeted_base"
        old_arm3["arm_name"] = "arm2_matlora_fone"
        
        new_models = {
            "arm1_ticktack_base": models["arm1_ticktack_base"],
            "arm2_matlora_fone": old_arm3,
            "arm3_targeted_base": old_arm2,
            "arm4_static_mlp_lora": models["arm4_static_mlp_lora"]
        }
        data["models"] = new_models

    # Update research division description
    data["research_division"] = {
        "user_scope": "Cải thiện LoRA và Targeted Continual Pretraining (Arms 3, 4, 5)",
        "partner_scope": "Biểu diễn thời gian sóng điều hòa và MaT-LoRA Dynamic Core (Arm 2)"
    }

    # Update McNemar keys
    old_mc = data.get("mcnemar_paired_tests", {})
    new_mc = {
        "Arm 1 (Ticktack Base) vs Arm 2 (MaT-LoRA FoNE)": old_mc.get("Arm 1 (Ticktack Base) vs Arm 3 (MaT-LoRA FoNE)"),
        "Arm 1 (Ticktack Base) vs Arm 3 (Targeted Base V2)": old_mc.get("Arm 1 (Ticktack Base) vs Arm 2 (Targeted Base V2)"),
        "Arm 1 (Ticktack Base) vs Arm 4 (Static MLP LoRA)": old_mc.get("Arm 1 (Ticktack Base) vs Arm 4 (Static MLP LoRA)"),
        "Arm 2 (MaT-LoRA FoNE) vs Arm 3 (Targeted Base V2)": old_mc.get("Arm 2 (Targeted Base V2) vs Arm 3 (MaT-LoRA FoNE)"),
        "Arm 2 (MaT-LoRA FoNE) vs Arm 4 (Static MLP LoRA)": old_mc.get("Arm 3 (MaT-LoRA FoNE) vs Arm 4 (Static MLP LoRA)"),
        "Arm 3 (Targeted Base V2) vs Arm 4 (Static MLP LoRA)": old_mc.get("Arm 2 (Targeted Base V2) vs Arm 4 (Static MLP LoRA)")
    }
    data["mcnemar_paired_tests"] = {k: v for k, v in new_mc.items() if v is not None}

    with open(receipt_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print("Updated FOUR_ARM_YEAR_ONLY_RECEIPT.json successfully.")

def swap_config_synthesis():
    config_path = REPO_ROOT / "Reports" / "03_TARGETED_OBJECTIVES_MCQA" / "ARM2_4_CONFIG_AND_OBJECTIVES_SYNTHESIS.json"
    if not config_path.exists():
        print(f"File not found: {config_path}")
        return
    with open(config_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    data["research_division_and_ownership"] = {
        "user_scope": "Cải thiện LoRA và Targeted Continual Pretraining (Arms 3, 4, 5)",
        "partner_scope": "Biểu diễn thời gian (Temporal Representation) và MaT-LoRA Dynamic Core (Arm 2)"
    }

    arms = data.get("arms_configuration", {})
    old_arm2 = arms.get("arm_2_targeted_continual_alignment")
    old_arm3 = arms.get("arm_3_matlora_fone")

    if old_arm2 and old_arm3:
        old_arm2["arm_id"] = "ARM_3"
        old_arm2["canonical_name"] = "Stage 03 Qwen2.5-3B Base Targeted Alignment (Arm 3)"
        
        old_arm3["arm_id"] = "ARM_2"
        old_arm3["canonical_name"] = "Stage 03 Qwen2.5-3B Base MaT-LoRA + FoNE Full (Arm 2)"

        new_arms = {
            "arm_2_matlora_fone": old_arm3,
            "arm_3_targeted_continual_alignment": old_arm2,
            "arm_4_static_mlp_lora_ablation": arms.get("arm_4_static_mlp_lora_ablation")
        }
        data["arms_configuration"] = new_arms

    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print("Updated ARM2_4_CONFIG_AND_OBJECTIVES_SYNTHESIS.json successfully.")

if __name__ == "__main__":
    swap_four_arm_receipt()
    swap_config_synthesis()
