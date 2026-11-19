# Chinh Sach Giu Lai Va Don Dep An Toan (Cleanup Retention Policy)

## 1. Phan Cap Tai Nguyen
- **MANDATORY_RETAIN (Bat buoc giu nguyen):**
  - Toan bo artifacts va receipts cua Run 02, Run 03, Run 04, Run 04.1, Run 04.2, Run 04.3.
  - Toan bo checkpoints va du lieu goc: data/tempun/, MCQA4_TRAIN_CLEAN.jsonl, MCQA4_DEV_CLEAN.jsonl, C1_COMPACT_ADAPTER_ONLY.pt.
  - Toan bo test suites hoi quy (regression tests) da chung minh loi cu khong quay lai.
  - Toan bo repo doi tac C:\Users\GF63\Documents\Long-Range-Representation (Read-Only 100%).
- **ELIGIBLE_FOR_PURGE (Cho phep don dep an toan):**
  - Project-local __pycache__ directories.
  - Stale .pytest_cache directories.
  - Temporary ephemeral logs in scratch/ not referenced by any canonical receipt.
