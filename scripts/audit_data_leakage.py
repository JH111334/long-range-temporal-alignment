import gzip
import json
from pathlib import Path

train_path = Path("data/tempun_full_clean/train_104k_clean.jsonl.gz")
eval_path = Path("data/tempun_full_clean/eval_5292_clean.jsonl.gz")
cf_path = Path("data/counterfactual/grounded_counterfactual_benchmark.jsonl")

train_inputs = set()
train_ids = set()

print("Loading train_104k...")
with gzip.open(train_path, "rt", encoding="utf-8") as f:
    for line in f:
        item = json.loads(line)
        train_inputs.add(item["input"].strip().lower())
        train_ids.add(item.get("example_id", item.get("id", "")))

print(f"Loaded {len(train_inputs)} unique train inputs, {len(train_ids)} train IDs.")

print("\nChecking eval_5292 overlap...")
eval_input_leak = 0
eval_id_leak = 0
total_eval = 0
with gzip.open(eval_path, "rt", encoding="utf-8") as f:
    for line in f:
        total_eval += 1
        item = json.loads(line)
        inp = item["input"].strip().lower()
        eid = item.get("example_id", item.get("id", ""))
        if inp in train_inputs:
            eval_input_leak += 1
        if eid in train_ids:
            eval_id_leak += 1

print(f"Eval Total: {total_eval} cases")
print(f"Eval Input Leak: {eval_input_leak} / {total_eval} ({eval_input_leak/total_eval*100:.2f}%)")
print(f"Eval ID Leak: {eval_id_leak} / {total_eval}")

print("\nChecking counterfactual benchmark overlap...")
cf_leak = 0
total_cf = 0
with open(cf_path, "r", encoding="utf-8") as f:
    for line in f:
        total_cf += 1
        item = json.loads(line)
        inp = item["input"].strip().lower()
        if inp in train_inputs:
            cf_leak += 1

print(f"CF Total: {total_cf} cases")
print(f"CF Leak into Train: {cf_leak} / {total_cf}")
