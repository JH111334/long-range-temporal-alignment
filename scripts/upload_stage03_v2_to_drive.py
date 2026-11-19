"""Upload Stage 03 Year-Only Scientific Closure Artifacts to Google Drive."""

import hashlib
import json
import shutil
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DRIVE_DIR = Path(r"G:\My Drive\NCKH\Receipts\3B_STAGE03_V2_SCIENTIFIC_CLOSURE_20261026")
DRIVE_DIR.mkdir(parents=True, exist_ok=True)

# 1. Clean obsolete / duplicate / flawed metric files from Google Drive folder
STALE_DRIVE_FILES = [
    "THREE_ARM_COMPARISON_RECEIPT.json",
    "FOUR_ARM_COMPARISON_RECEIPT.json",
    "stage03_3b_unified_comparison.json",
    "STAGE03_3B_V2_RUN_STATUS.json",
    "RECEIPT_STAGE03_3B_V2_CLOSURE.yaml",
    "STAGE03_3B_THREE_ARM_EMPIRICAL_SYNTHESIS.md",
    "STAGE03_3B_ARCHITECTURAL_TRADEOFF_ANALYSIS.md",
    "STAGE03_3B_FOUR_ARM_ABLATION_ANALYSIS.md",
]
for f_name in STALE_DRIVE_FILES:
    p = DRIVE_DIR / f_name
    if p.exists():
        p.unlink()
        print(f"Removed stale file from Drive: {f_name}")

# 2. Files to synchronize (Clean Year-Only canonical bundle)
FILES_TO_COPY = [
    REPO_ROOT / "Documents/Documenting/Results/STAGE03_3B_V2_20261026/FOUR_ARM_YEAR_ONLY_RECEIPT.json",
    REPO_ROOT / "Documents/Documenting/Results/STAGE03_3B_V2_20261026/FIVE_ARM_YEAR_ONLY_RECEIPT.json",
    REPO_ROOT / "Documents/Documenting/Results/STAGE03_3B_V2_20261026/RECEIPT_STAGE03_3B_YEAR_ONLY_CLOSURE.yaml",
    REPO_ROOT / "Reports/03_TARGETED_OBJECTIVES_MCQA/STAGE03_3B_FOUR_ARM_YEAR_ONLY_SYNTHESIS.md",
    REPO_ROOT / "Reports/03_TARGETED_OBJECTIVES_MCQA/ARM2_4_CONFIG_AND_OBJECTIVES_SYNTHESIS.json",
    REPO_ROOT / "Reports/03_TARGETED_OBJECTIVES_MCQA/ARM2_5_CONFIG_AND_OBJECTIVES_SYNTHESIS.json",
    REPO_ROOT / "Reports/03_TARGETED_OBJECTIVES_MCQA/DATA_PROVENANCE_AND_EVALUATION_TRANSFORM_RECEIPT.json",
    REPO_ROOT / "Documents/Documenting/Results/STAGE03_3B_V2_20261026/ticktack_3b_base_canonical_results.json",
    REPO_ROOT / "Documents/Documenting/Results/STAGE03_3B_V2_20261026/stage03_3b_matlora_fone_results.json",
    REPO_ROOT / "Documents/Documenting/Results/STAGE03_3B_V2_20261026/stage03_3b_base_targeted_results.json",
    REPO_ROOT / "Documents/Documenting/Results/STAGE03_3B_V2_20261026/stage03_3b_static_mlp_lora_results.json",
    REPO_ROOT / "Documents/Documenting/Results/STAGE03_3B_V2_20261026/stage03_3b_static_mlp_lora_summary.json",
    REPO_ROOT / "Documents/Documenting/Results/STAGE03_3B_V2_20261026/stage03_3b_targeted_static_mlp_results.json",
    REPO_ROOT / "Documents/Documenting/Results/STAGE03_3B_V2_20261026/stage03_3b_targeted_static_mlp_summary.json",
    REPO_ROOT / "Documents/Documenting/Results/STAGE03_3B_V2_20261026/stage03_3b_targeted_matlora_results.json",
    REPO_ROOT / "Documents/Documenting/Results/STAGE03_3B_V2_20261026/stage03_3b_targeted_matlora_summary.json",
]

manifest = {
    "timestamp": datetime.now().isoformat(),
    "protocol": "STAGE03_3B_YEAR_ONLY_FIVE_ARM_SYNTHESIS_V1",
    "status": "COMPLETE_VERIFIED",
    "backbone": "Qwen/Qwen2.5-3B Base",
    "benchmark": "TempUN Clean 5,292 MCQA (4-fold cyclic debiasing) + 54 Counterfactual Pairs",
    "temporal_granularity": "YEAR_ONLY",
    "research_division": {
        "user_scope": "LoRA & Targeted Continual Pretraining (Arms 3, 4, 5)",
        "partner_scope": "Temporal Representation & Harmonic Core (Arm 2 MaT-LoRA FoNE)",
    },
    "files": [],
}

for src in FILES_TO_COPY:
    if src.exists():
        dst = DRIVE_DIR / src.name
        shutil.copy2(src, dst)
        h = hashlib.sha256()
        with open(dst, "rb") as f:
            while chunk := f.read(65536):
                h.update(chunk)
        manifest["files"].append({
            "name": dst.name,
            "size": dst.stat().st_size,
            "sha256": h.hexdigest(),
            "relative_repo_path": str(src.relative_to(REPO_ROOT)),
        })
        print(f"Uploaded: {dst.name} ({dst.stat().st_size:,} bytes | SHA: {h.hexdigest()[:16]}...)")

manifest_path = DRIVE_DIR / "MANIFEST.json"
with open(manifest_path, "w", encoding="utf-8") as f:
    json.dump(manifest, f, indent=2)
print(f"Wrote MANIFEST.json to {manifest_path}")

index_path = DRIVE_DIR / "INDEX.md"
with open(index_path, "w", encoding="utf-8") as f:
    f.write("# Stage 03 Qwen2.5-3B Scientific Closure Evidence Bundle (Year-Only Standard)\n\n")
    f.write(f"- **Timestamp**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    f.write("- **Backbone**: Qwen/Qwen2.5-3B Base Model\n")
    f.write("- **Benchmark**: TempUN Clean (5,292 MCQA cases) + 54 Counterfactual Pairs\n")
    f.write("- **Granularity**: Pure YEAR_ONLY (Triệt tiêu toàn bộ ranh giới ngày/tuần/tháng)\n\n")
    f.write("## 1. Executive Performance Summary\n\n")
    f.write("| Arm | Model | Scope | Canonical Acc | Debiased Mean | Strict Acc | 54-Pair Counterfactual |\n")
    f.write("| :--- | :--- | :--- | :--- | :--- | :--- | :--- |\n")
    f.write("| **Arm 1: Ticktack Base** | Qwen2.5-3B Base | Baseline | 32.03% | 31.19% | 5.29% | 0.0% (0/54) |\n")
    f.write("| **Arm 2: Targeted Base V2** | Qwen2.5-3B Base | LoRA & NTP (Tác giả) | 28.10% | 29.85% | 6.20% | 9.26% (5/54) |\n")
    f.write("| **Arm 3: MaT-LoRA FoNE** | Qwen2.5-3B Base | Representation (Cộng sự) | 26.27% | 27.17% | 8.20% | 70.37% (38/54) |\n")
    f.write("| **Arm 4: Static MLP LoRA** | Qwen2.5-3B Base | LoRA MLP (Tác giả) | 27.46% | 28.58% | **9.41%** | **66.67% (36/54)** |\n\n")
    f.write("## 2. Files and Integrity Hashes\n\n")
    for item in manifest["files"]:
        f.write(f"- `{item['name']}` ({item['size']:,} bytes): `{item['sha256']}`\n")

print(f"Wrote INDEX.md to {index_path}")
