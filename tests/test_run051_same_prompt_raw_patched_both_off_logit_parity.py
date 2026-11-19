"""Unit Test for Logit Parity Between Raw Backbone and Patched Base with Both Off."""

from __future__ import annotations

import json
from pathlib import Path
import pytest

_REPO_ROOT = Path(__file__).resolve().parents[1]


def test_raw_and_both_off_exact_logit_parity():
    """Verify that RAW_AND_BOTH_OFF_LOGIT_PARITY.jsonl proves numerical identity."""
    parity_file = _REPO_ROOT / "artifacts" / "experiments" / "V2_CAUSAL_TEMPORAL_ALIGNMENT" / "run_05_1" / "RAW_AND_BOTH_OFF_LOGIT_PARITY.jsonl"
    assert parity_file.exists(), f"Missing parity log: {parity_file}"

    records = []
    with open(parity_file, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))

    assert len(records) >= 8, f"Expected at least 8 canary prompts tested, got {len(records)}"

    for rec in records:
        assert rec["parity_verdict"] == "PARITY_VERIFIED"
        # Exact numerical identity check (FP32 bit-for-bit or delta < 1e-5)
        assert rec["max_abs_diff"] < 1e-4, f"Prompt {rec['prompt_id']} failed parity: max_diff={rec['max_abs_diff']}"
        assert rec["top1_token_raw"] == rec["top1_token_both_off"], f"Top token mismatch on {rec['prompt_id']}"
        assert abs(rec["cosine_similarity"] - 1.0) < 1e-4, f"Cosine similarity degraded on {rec['prompt_id']}"


def test_factorial_conditioning_isolation():
    """Verify factorial isolation of Seam, LoRA, and Both-Off."""
    iso_file = _REPO_ROOT / "artifacts" / "experiments" / "V2_CAUSAL_TEMPORAL_ALIGNMENT" / "run_05_1" / "HOOK_AND_CONDITIONING_ISOLATION.json"
    assert iso_file.exists(), f"Missing isolation manifest: {iso_file}"

    with open(iso_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    interventions = data["factorial_interventions"]
    assert interventions["BOTH_OFF"]["verdict"] == "IDENTICAL_TO_RAW"
    assert interventions["BOTH_OFF"]["max_abs_diff_vs_raw"] < 1e-5

    assert interventions["SEAM_ONLY"]["verdict"] == "ACTIVE_ISOLATED_INJECTION"
    assert interventions["SEAM_ONLY"]["max_abs_diff_vs_raw"] > 0.01

    assert interventions["LORA_ONLY"]["verdict"] == "ACTIVE_ISOLATED_ADAPTATION"
    assert interventions["LORA_ONLY"]["max_abs_diff_vs_raw"] > 0.01

    assert interventions["BOTH_ON"]["verdict"] == "FULL_CONTINUOUS_ALIGNMENT"
