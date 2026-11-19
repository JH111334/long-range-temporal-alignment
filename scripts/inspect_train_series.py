import json
import re
from collections import defaultdict
from pathlib import Path

train_path = Path("data/tempun/train_nwp.jsonl")
eval_path = Path("data/tempun/eval_mcqa.jsonl")

records = []
with open(train_path, "r", encoding="utf-8") as f:
    for line in f:
        if line.strip():
            records.append(json.loads(line))

print(f"Total records in train: {len(records)}")

task_counts = defaultdict(int)
years = defaultdict(int)
series_map = defaultdict(list)
entity_set = set()

for r in records:
    task_id = r.get("task_id", "UNKNOWN")
    task_counts[task_id] += 1
    inp = r.get("input", "")
    tgt = r.get("target", "")
    src_id = r.get("source_record_id", "")
    year = r.get("canonical_year")
    years[year] += 1
    
    # Structure of source_record_id: e.g. "ZRIH-1-23-1984"
    parts = src_id.split("-")
    entity_code = parts[0] if parts else "UNKNOWN"
    entity_set.add(entity_code)
    
    if len(parts) >= 3:
        series_key = "-".join(parts[:-1])
    else:
        series_key = src_id
    series_map[series_key].append({
        "year": year,
        "target": tgt,
        "input": inp,
        "record_id": src_id,
        "entity": entity_code,
    })

print("Task IDs in train:", dict(task_counts))
print("Years in train:", dict(years))
print(f"Distinct entities in train: {len(entity_set)}: {sorted(list(entity_set))}")
print(f"Total distinct series: {len(series_map)}")

multi_year_series = {k: v for k, v in series_map.items() if len(v) >= 2}
print(f"Series with >= 2 years: {len(multi_year_series)}")

sample_keys = list(multi_year_series.keys())[:5]
for k in sample_keys:
    years_present = [item["year"] for item in multi_year_series[k]]
    print(f"Series {k}: years={years_present}")
