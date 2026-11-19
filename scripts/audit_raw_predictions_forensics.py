"""Forensic Audit of M0-FC40 Raw Predictions across 3 Seeds.

Answers:
1. Exact string matching / divergence across seeds 1, 2, 3 at stages after_00, after_01, after_02.
2. In-depth taxonomy of invalid predictions (empty, whitespace, verbose explanations, hallucination).
3. The exact 4 correct predictions and their characteristics.
4. Evaluation prompt formatting analysis.
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path


def load_json(path: Path | str) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def run_forensic_audit():
    base_dir = Path("artifacts/quarantine/M0_FC40/kaggle_3seed/artifacts")
    seeds = [1, 2, 3]
    stages = ["after_00", "after_01", "after_02"]

    print("=" * 80)
    print("M0-FC40 RAW PREDICTIONS FORENSIC AUDIT")
    print("=" * 80)

    # 1. Inspect seed receipts to verify different initialization / seed isolation
    print("\n--- 1. SEED RECEIPT & TRAINING LOSS COMPARISON ---")
    for s in seeds:
        receipt_path = base_dir / f"seed_{s}" / "seed_receipt.json"
        if receipt_path.exists():
            rc = load_json(receipt_path)
            print(f"Seed {s}: hash={rc.get('seed_receipt_hash', 'N/A')[:16]}..., status={rc.get('status')}")
            for st in rc.get("stages", []):
                print(f"  Stage {st['stage_idx']} ({st['stage_id']}): train_loss={st['train_loss']:.4f}, strict_acc={st['strict_accuracy']:.2f}%, invalid={st['invalid_syntax_rate']:.2f}%")

    # 2. Compare predictions across seeds for each stage
    print("\n--- 2. CROSS-SEED PREDICTION IDENTICALITY CHECK ---")
    for stg in stages:
        preds_by_seed = {}
        for s in seeds:
            p_file = base_dir / f"seed_{s}" / "predictions" / f"predictions_{stg}.json"
            if p_file.exists():
                preds_by_seed[s] = load_json(p_file)

        if len(preds_by_seed) == 3:
            total_cases = len(preds_by_seed[1])
            identical_1_2 = sum(1 for p1, p2 in zip(preds_by_seed[1], preds_by_seed[2]) if p1["raw_completion"] == p2["raw_completion"])
            identical_1_3 = sum(1 for p1, p3 in zip(preds_by_seed[1], preds_by_seed[3]) if p1["raw_completion"] == p3["raw_completion"])
            identical_all_3 = sum(1 for p1, p2, p3 in zip(preds_by_seed[1], preds_by_seed[2], preds_by_seed[3]) if p1["raw_completion"] == p2["raw_completion"] == p3["raw_completion"])

            print(f"Stage {stg} ({total_cases} cases):")
            print(f"  Identical Seed 1 vs Seed 2: {identical_1_2}/{total_cases} ({identical_1_2/total_cases*100:.1f}%)")
            print(f"  Identical Seed 1 vs Seed 3: {identical_1_3}/{total_cases} ({identical_1_3/total_cases*100:.1f}%)")
            print(f"  Identical across ALL 3 seeds: {identical_all_3}/{total_cases} ({identical_all_3/total_cases*100:.1f}%)")

            # Check if differences exist
            diffs = []
            for idx, (p1, p2, p3) in enumerate(zip(preds_by_seed[1], preds_by_seed[2], preds_by_seed[3])):
                if not (p1["raw_completion"] == p2["raw_completion"] == p3["raw_completion"]):
                    diffs.append((idx, p1["example_id"], p1["raw_completion"], p2["raw_completion"], p3["raw_completion"]))
            print(f"  Cases with cross-seed divergence: {len(diffs)}")
            if diffs:
                print("  Sample divergences:")
                for d in diffs[:3]:
                    print(f"    Case {d[0]} ({d[1]}): Seed1='{d[2]}' | Seed2='{d[3]}' | Seed3='{d[4]}'")

    # 3. Analyze Invalid Predictions Taxonomy on final stage (after_02) of Seed 1
    print("\n--- 3. TAXONOMY OF INVALID PREDICTIONS (Seed 1, after_02) ---")
    p_file = base_dir / "seed_1" / "predictions" / "predictions_after_02.json"
    preds = load_json(p_file)

    invalid_completions = Counter()
    invalid_patterns = defaultdict(list)

    correct_cases = []
    valid_incorrect_cases = []

    for item in preds:
        raw = item["raw_completion"]
        parsed = item["parsed_answer"]
        target = item["gold_target"]
        ex_id = item["example_id"]
        stage = item["stage_id"]

        if parsed == target:
            correct_cases.append(item)
        elif parsed is not None:
            valid_incorrect_cases.append(item)
        else:
            invalid_completions[raw] += 1
            # Classify pattern
            if raw == "":
                invalid_patterns["EMPTY"].append(item)
            elif raw.isspace():
                invalid_patterns["WHITESPACE"].append(item)
            elif any(c in raw for c in ("\n", "\\n")):
                invalid_patterns["MULTILINE"].append(item)
            else:
                invalid_patterns["NON_OPTION_STRING"].append(item)

    print(f"Total cases: {len(preds)}")
    print(f"Correct cases: {len(correct_cases)} ({len(correct_cases)/len(preds)*100:.2f}%)")
    print(f"Valid but Incorrect: {len(valid_incorrect_cases)} ({len(valid_incorrect_cases)/len(preds)*100:.2f}%)")
    print(f"Invalid Syntax: {sum(invalid_completions.values())} ({sum(invalid_completions.values())/len(preds)*100:.2f}%)")

    print("\nInvalid Completion Frequency (Top 15 most common completions):")
    for comp, count in invalid_completions.most_common(15):
        escaped = repr(comp)
        print(f"  Count={count:3d} : {escaped}")

    print("\nInvalid Pattern Breakdown:")
    for pat, items in invalid_patterns.items():
        print(f"  Pattern [{pat}]: {len(items)} cases")

    print("\n--- 4. THE 4 CORRECT CASES ---")
    for c in correct_cases:
        print(f"  Example: {c['example_id']} | Stage: {c['stage_id']} | Target: {c['gold_target']} | Raw: '{c['raw_completion']}'")

    print("\n--- 5. SAMPLE VALID BUT INCORRECT CASES ---")
    for v in valid_incorrect_cases[:8]:
        print(f"  Example: {v['example_id']} | Stage: {v['stage_id']} | Target: {v['gold_target']} | Parsed: {v['parsed_answer']} | Raw: '{v['raw_completion']}'")

    # 6. Check a sample prompt from eval dataset
    print("\n--- 6. EVAL PROMPT INSPECTION (First 2 cases) ---")
    eval_file = Path("artifacts/quarantine/eval_mcqa.jsonl")
    if not eval_file.exists():
        eval_file = Path("data/tempun/eval_mcqa.jsonl")
    if not eval_file.exists():
        # search
        matches = list(Path(".").glob("**/eval_mcqa.jsonl"))
        if matches:
            eval_file = matches[0]

    if eval_file.exists():
        print(f"Eval file located at: {eval_file}")
        lines = eval_file.read_text(encoding="utf-8").splitlines()[:2]
        for idx, line in enumerate(lines):
            row = json.loads(line)
            print(f"Case {idx + 1}:")
            print(f"  Input: {repr(row.get('input', ''))}")
            print(f"  Target: {repr(row.get('target', ''))}")


if __name__ == "__main__":
    run_forensic_audit()
