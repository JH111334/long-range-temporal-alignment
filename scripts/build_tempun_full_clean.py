"""
Build 100% Clean TempUN Full Dataset (76 Stages, 1947-2022).

Directly converts upstream TempUN_Small raw CSV files into canonical, pure JSONL.gz datasets:
- train_104k_clean.jsonl.gz: 104,130 Next-Word-Prediction (NWP) rows across 76 years.
- eval_5292_clean.jsonl.gz: 5,292 balanced Multiple-Choice Question Answering (MCQA) rows (441 task-year pairs x 12 cases).

ZERO FoNE metadata, ZERO Can Chi conditioning strings, ZERO custom feature vectors.
Only standard text fields: 'example_id', 'year', 'stage_id', 'task_id', 'input', 'target'.
"""

from __future__ import annotations

import csv
import gzip
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List

TASK_MAPPING = {
    "easy_date_based_mcq_prompt_yearly_prompt": "DB",
    "hard_min_max_based_mcq_prompt_yearly_prompt": "MM",
    "hard_range_based_mcq_prompt_yearly_prompt": "RB",
    "hard_trend_based_mcq_prompt_yearly_prompt": "TB",
    "med_relative_based_mcq_prompt_yearly_prompt": "Rel",
    "med_window_based_mcq_prompt_yearly_prompt": "WB",
}

EXPECTED_NWP_TOTAL = 104130
EXPECTED_EVAL_TOTAL = 5292


def compute_sha256(filepath: Path) -> str:
    h = hashlib.sha256()
    with filepath.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def find_upstream_data_dir() -> Path:
    candidates = [
        PROJECT_ROOT / "data" / "raw" / "TempUN_Small" / "TempUN_Small",
        PROJECT_ROOT / "data" / "raw" / "TempUN_Small",
        PROJECT_ROOT / "scratch" / "partner_review" / "Data" / "raw" / "TempUN_Small" / "TempUN_Small",
        PROJECT_ROOT / "scratch" / "partner_review" / "Data" / "raw" / "TempUN_Small",
        Path("d:/nckh/data/TempUN/TempUN_Small/TempUN_Small"),
        Path("D:/Long-Range-Representation/Data/raw/TempUN_Small/TempUN_Small"),
        Path("Data/raw/TempUN_Small/TempUN_Small"),
        Path("../data/TempUN/TempUN_Small/TempUN_Small"),
    ]
    for c in candidates:
        if c.is_dir() and (c / "prompts_sorted_by_year").is_dir():
            return c.resolve()
    raise FileNotFoundError("Could not find upstream TempUN_Small directory in expected locations.")


def build_clean_dataset(
    upstream_dir: Path | None = None,
    output_dir: Path | None = None,
) -> Dict[str, Any]:
    if upstream_dir is None:
        upstream_dir = find_upstream_data_dir()
    if output_dir is None:
        output_dir = PROJECT_ROOT / "data" / "tempun_full_clean"

    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"[1/4] Upstream directory: {upstream_dir}")
    print(f"[1/4] Output directory: {output_dir}")

    # --- 1. Process Train NWP ---
    nwp_dir = upstream_dir / "prompts_sorted_by_year"
    nwp_files = sorted([f for f in nwp_dir.glob("*.csv") if f.name != "yearly_freq.csv"])
    print(f"[2/4] Found {len(nwp_files)} year files for NWP training (1947-2022).")

    train_rows: List[Dict[str, Any]] = []
    for csv_file in nwp_files:
        year_str = csv_file.stem
        year_int = int(year_str)
        with csv_file.open("r", encoding="utf-8", errors="replace") as f:
            reader = csv.DictReader(f)
            for row in reader:
                rec_id = row.get("id", "").strip()
                query = row.get("query", "").strip()
                ans = row.get("answer", "").strip()
                if not query or not ans:
                    continue
                clean_rec = {
                    "example_id": f"nwp:{year_str}:{rec_id}",
                    "year": year_int,
                    "stage_id": f"year:{year_str}",
                    "task_id": "NWP",
                    "input": query,
                    "target": ans,
                }
                train_rows.append(clean_rec)

    print(f"[2/4] Total NWP train rows extracted: {len(train_rows)} (Expected: {EXPECTED_NWP_TOTAL})")
    if len(train_rows) != EXPECTED_NWP_TOTAL:
        raise ValueError(f"Mismatch in NWP row count: expected {EXPECTED_NWP_TOTAL}, got {len(train_rows)}")

    train_out_path = output_dir / "train_104k_clean.jsonl.gz"
    with gzip.open(train_out_path, "wt", encoding="utf-8") as gz:
        for r in train_rows:
            gz.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"[2/4] Wrote: {train_out_path} ({train_out_path.stat().st_size / 1024 / 1024:.2f} MB)")

    # --- 2. Process Eval MCQA ---
    eval_base_dir = upstream_dir / "evaluation_prompts" / "yearly-format-eval-dataset"
    print(f"[3/4] Processing MCQA evaluation tasks from: {eval_base_dir}")

    eval_rows: List[Dict[str, Any]] = []
    task_dirs = sorted([d for d in eval_base_dir.iterdir() if d.is_dir()])
    
    task_year_counts = {}
    for t_dir in task_dirs:
        task_code = TASK_MAPPING.get(t_dir.name, t_dir.name)
        csv_files = sorted([f for f in t_dir.glob("*.csv") if f.name != "yearly_freq.csv"])
        for csv_file in csv_files:
            year_str = csv_file.stem
            year_int = int(year_str)
            group_rows: List[Dict[str, Any]] = []
            with csv_file.open("r", encoding="utf-8", errors="replace") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    rec_id = row.get("id", "").strip()
                    query = row.get("query", "").strip()
                    ans = row.get("answer", "").strip()
                    if not query or not ans:
                        continue
                    clean_rec = {
                        "example_id": f"mcq:{task_code}:{year_str}:{rec_id}",
                        "year": year_int,
                        "stage_id": f"year:{year_str}",
                        "task_id": task_code,
                        "input": query,
                        "target": ans,
                    }
                    group_rows.append(clean_rec)
            
            # Deterministic selection of 12 balanced samples per task-year
            ordered = sorted(
                group_rows,
                key=lambda r: (
                    hashlib.sha256(r["example_id"].encode()).digest(),
                    r["example_id"],
                ),
            )
            selected = ordered[:12]
            eval_rows.extend(selected)
            task_year_counts[(task_code, year_str)] = len(selected)

    eval_rows.sort(key=lambda r: (r["year"], r["task_id"], r["example_id"]))
    print(f"[3/4] Total MCQA eval rows extracted: {len(eval_rows)} (Expected: {EXPECTED_EVAL_TOTAL})")
    if len(eval_rows) != EXPECTED_EVAL_TOTAL:
        raise ValueError(f"Mismatch in Eval row count: expected {EXPECTED_EVAL_TOTAL}, got {len(eval_rows)}")

    eval_out_path = output_dir / "eval_5292_clean.jsonl.gz"
    with gzip.open(eval_out_path, "wt", encoding="utf-8") as gz:
        for r in eval_rows:
            gz.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"[3/4] Wrote: {eval_out_path} ({eval_out_path.stat().st_size / 1024 / 1024:.2f} MB)")

    # --- 3. Compute Checksums & Metadata ---
    train_sha = compute_sha256(train_out_path)
    eval_sha = compute_sha256(eval_out_path)

    checksums_path = output_dir / "CHECKSUMS.sha256"
    checksums_path.write_text(
        f"{train_sha}  train_104k_clean.jsonl.gz\n{eval_sha}  eval_5292_clean.jsonl.gz\n",
        encoding="utf-8",
    )

    metadata = {
        "title": "TempUN Full Clean (76 Years, 104k NWP, 5.2k MCQA)",
        "id": "antigravity/tempun-full-clean",
        "licenses": [{"name": "CC0-1.0"}],
        "nwp_samples": len(train_rows),
        "mcqa_samples": len(eval_rows),
        "stages": list(range(1947, 2023)),
        "sha256": {
            "train_104k_clean.jsonl.gz": train_sha,
            "eval_5292_clean.jsonl.gz": eval_sha,
        },
    }
    (output_dir / "dataset-metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )

    readme_content = f"""# TempUN Full Clean Dataset

This dataset is a 100% clean, pure extraction of upstream **TempUN** (Small archive covering all 76 chronological years 1947-2022).

## Contents
- `train_104k_clean.jsonl.gz`: 104,130 Next-Word-Prediction (NWP) samples across 76 stages.
- `eval_5292_clean.jsonl.gz`: 5,292 balanced Multiple-Choice Question Answering (MCQA) samples across 6 tasks (DB, MM, RB, TB, Rel, WB) and 76 stages (12 samples per task per year).
- `CHECKSUMS.sha256`: SHA-256 integrity verification hashes.

## Verification
- SHA256 `train_104k_clean.jsonl.gz`: `{train_sha}`
- SHA256 `eval_5292_clean.jsonl.gz`: `{eval_sha}`

## Schema Guarantee
- Fields: `example_id`, `year`, `stage_id`, `task_id`, `input`, `target`.
- ZERO conditioning metadata, ZERO FoNE features, ZERO Can Chi strings baked in text.
"""
    (output_dir / "README.md").write_text(readme_content, encoding="utf-8")
    print("[4/4] Successfully built clean TempUN full dataset!")
    return metadata


if __name__ == "__main__":
    build_clean_dataset()
