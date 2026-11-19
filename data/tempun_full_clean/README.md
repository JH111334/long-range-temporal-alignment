# TempUN Full Clean Dataset

This dataset is a 100% clean, pure extraction of upstream **TempUN** (Small archive covering all 76 chronological years 1947-2022).

## Contents
- `train_104k_clean.jsonl.gz`: 104,130 Next-Word-Prediction (NWP) samples across 76 stages.
- `eval_5292_clean.jsonl.gz`: 5,292 balanced Multiple-Choice Question Answering (MCQA) samples across 6 tasks (DB, MM, RB, TB, Rel, WB) and 76 stages (12 samples per task per year).
- `CHECKSUMS.sha256`: SHA-256 integrity verification hashes.

## Verification
- SHA256 `train_104k_clean.jsonl.gz`: `111d2d0908533a17ac10c98533802053631b708a23846524daa224bd1465697d`
- SHA256 `eval_5292_clean.jsonl.gz`: `a624d8d258f9676f00e504d16c35dc401cc1cce96613ca56af970daa32a12fc3`

## Schema Guarantee
- Fields: `example_id`, `year`, `stage_id`, `task_id`, `input`, `target`.
- ZERO conditioning metadata, ZERO FoNE features, ZERO Can Chi strings baked in text.
