# FORENSIC AUDIT REPORT: MODAL 5 EPOCHS, BILLING, TRAIN/VAL/TEST, AND METRICS

- **Audit Date**: 2026-11-05 01:52:55
- **Audit Directory**: `Reports/6-11Report/_FORENSIC_AUDIT_20261104/`
- **Audit Classification**: `QUARANTINED_PENDING_PROVENANCE`
- **Authority**: Executor/Verifier only. Final research decisions reserved for User + ChatGPT.

---

## 1. Executive Summary & Forensic Verdicts

| Gate | Target Domain | Forensic Status | Key Verified Evidence |
| :--- | :--- | :---: | :--- |
| **G_DATA** | Dataset Integrity & Splits | **PASS** | `train_104k` (104,130 rows, SHA256 verified); `eval_5292` (5,292 rows, SHA256 verified); 0% leakage; `NO_INDEPENDENT_VAL_TEST_SPLIT`. |
| **G_METRICS** | Metric Arithmetic & Logic | **METRIC_ARITHMETIC_ONLY** | Arithmetic matches summaries across all 7 arms. Raw logits not saved in JSONs. Arm 1 dictionary collision (4,950 keys). |
| **G_RUN** | Modal Execution Lifecycle | **PASS** | App `ap-cNkSe2hf7vevvC9KjH1twj` confirmed 7h 39m 02s runtime on 1x A10G GPU. Sequential epoch progression proven across 8 hourly billing intervals. |
| **G_CHECKPOINT** | Checkpoint Lineage & Files | **PASS** | Volume `stage03-artifacts` contains ep1..ep5 checkpoints for both Arm 4 and Arm 5 (~22.8 MB each). |
| **G_BILLING** | Provider Cost Reconciliation | **BILLING_RECONCILED_CREDITS** | Gross compute: **$9.04 USD** (7.67 A10G hours across 8 continuous hourly intervals). Free credits: **-$9.03 USD**. Net cash: **$0.00 USD**. |
| **G_REPORT** | Manifest & Reporting Hygiene | **MANIFEST_DRIFT_FIXED** | Detached manifest generated; self-hash and 1,704 rotation pass overcounting documented in errata. |

---

## 2. Directory Contents & Artifact Catalog

This audit directory contains 10 detached forensic artifacts:
1. `README_AUDIT.md`: Executive summary and forensic findings.
2. `SOURCE_AND_SHA256_MANIFEST.json`: Detached SHA-256 and byte manifest of all 14 files in `Reports/6-11Report/`.
3. `SPLIT_AND_LEAKAGE_AUDIT.json`: Detailed split roles, leakage check, and CF V1 shortcut analysis.
4. `METRICS_RECOMPUTE_7_ARMS.json`: Recomputed canonical, debiased, strict, and task scores from case results across 7 arms.
5. `CASE_ID_AND_ROTATION_AUDIT.json`: Root cause of Arm 1 342 duplicate ID collision and corrected rotation pass count (19,464 strict / 21,276 total).
6. `MODAL_PROOF_OF_EXECUTION.json`: Modal CLI verification of app lifecycle, container timing, and hourly checkpoints.
7. `MODAL_BILLING_RECONCILIATION.json`: Redacted billing reconciliation table with 8-hour continuous interval meters explaining gross $9.04 vs net $0.00 cash.
8. `CHECKPOINT_LINEAGE_AND_REPLAY.json`: Modal Volume checkpoint inventory and local replay feasibility criteria.
9. `ERRATA_PRIOR_REPORTS.md`: Formal errata cataloging all 5 prior report inaccuracies without altering original history.
10. `GATE_STATUS.json`: Machine-readable gate evaluation records and quarantine status.

---

## 3. Unresolved Items & Decisions Reserved for User + ChatGPT
1. **Decision on Baseline Arm 1 Comparison**: Whether to re-evaluate Arm 1 with unique index keys to restore 5,292 keys, or use the 4,950 intersection subset for paired McNemar testing.
2. **Decision on Evaluation Split**: Whether to partition `eval_5292` into formal disjoint validation and test splits for future publication.
3. **Decision on CF V1 Balancing**: Whether to balance CF V1 with reverse-oriented pairs to eliminate the 66.67% static positional shortcut.
4. **Research Governance**: Next phase recommendation is `NO_GO_TRAINING` until research methodology decisions are formally approved.
