import gzip
import json
import re
from pathlib import Path
from collections import defaultdict
from typing import Dict, Any, List

def load_rel_cases():
    cases = []
    with gzip.open("data/tempun_full_clean/eval_5292_clean.jsonl.gz", "rt", encoding="utf-8") as f:
        for line in f:
            item = json.loads(line)
            if item.get("task_id") == "Rel":
                cases.append(item)
    return cases

def generate_counterfactual_rel_pairs(cases: List[Dict[str, Any]]):
    pattern = re.compile(r"In\s+(\d{4}),\s+(.*?)\s+(?:was|is|were)\s+higher than\s+(\d{4})\?\s+Answer as Yes or No", re.IGNORECASE)
    pairs = []
    unmatched = 0
    for case in cases:
        inp = case["input"].strip()
        m = pattern.search(inp)
        if not m:
            unmatched += 1
            continue
        y1, content, y2 = m.group(1), m.group(2), m.group(3)
        gold = case["target"].strip().lower()
        flipped_gold = "no" if gold == "yes" else "yes"
        # Preserve the exact verb used in original prompt if needed, or normalize to original
        verb_match = re.search(r"\b(was|is|were)\b\s+higher than", inp, re.IGNORECASE)
        verb = verb_match.group(1) if verb_match else "was"
        flipped_prompt = f"In {y2}, {content} {verb} higher than {y1}? Answer as Yes or No"
        
        pair = {
            "example_id": case.get("example_id", ""),
            "original_prompt": inp,
            "original_year": int(y1),
            "original_target": case["target"].strip(),
            "counterfactual_prompt": flipped_prompt,
            "counterfactual_year": int(y2),
            "counterfactual_target": "No" if case["target"].strip() == "Yes" else "Yes",
            "y1": int(y1),
            "y2": int(y2),
            "feature": content,
        }
        pairs.append(pair)
    return pairs, unmatched

if __name__ == "__main__":
    cases = load_rel_cases()
    pairs, unmatched = generate_counterfactual_rel_pairs(cases)
    print(f"Total Rel cases: {len(cases)}")
    print(f"Successfully generated CounterfactualV2 pairs: {len(pairs)} / {len(cases)}")
    print(f"Unmatched: {unmatched}")
    print("\nSample Pair 1:")
    print("  Original Prompt:", pairs[0]["original_prompt"])
    print("  Original Target:", pairs[0]["original_target"], f"(Year: {pairs[0]['original_year']})")
    print("  CF Prompt:      ", pairs[0]["counterfactual_prompt"])
    print("  CF Target:      ", pairs[0]["counterfactual_target"], f"(Year: {pairs[0]['counterfactual_year']})")
    print("\nSample Pair 2:")
    print("  Original Prompt:", pairs[1]["original_prompt"])
    print("  Original Target:", pairs[1]["original_target"], f"(Year: {pairs[1]['original_year']})")
    print("  CF Prompt:      ", pairs[1]["counterfactual_prompt"])
    print("  CF Target:      ", pairs[1]["counterfactual_target"], f"(Year: {pairs[1]['counterfactual_year']})")
