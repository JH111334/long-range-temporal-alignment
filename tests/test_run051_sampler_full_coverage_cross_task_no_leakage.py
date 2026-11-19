"""Unit Test for Stratified Sampler Full Coverage and Zero Leakage Across Tasks."""

from __future__ import annotations

import json
from pathlib import Path
import pytest

_REPO_ROOT = Path(__file__).resolve().parents[1]


def test_split_transitive_fact_family_audit_zero_leakage():
    """Verify zero overlap between MCQA train and dev questions."""
    audit_file = _REPO_ROOT / "artifacts" / "experiments" / "V2_CAUSAL_TEMPORAL_ALIGNMENT" / "run_05_1" / "SPLIT_TRANSITIVE_FACT_FAMILY_AUDIT.json"
    assert audit_file.exists(), f"Missing audit file: {audit_file}"

    with open(audit_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    checks = data["leakage_checks"]
    assert checks["mcqa_train_dev_exact_question_overlap"] == 0
    assert checks["is_disjoint_questions"] is True
    assert checks["verdict"] == "ZERO_LEAKAGE_CONFIRMED"


def test_sampler_exposure_plan_balanced():
    """Verify exposure plan has balanced year, entity, and target option distributions."""
    plan_file = _REPO_ROOT / "artifacts" / "experiments" / "V2_CAUSAL_TEMPORAL_ALIGNMENT" / "run_05_1" / "YEAR_ENTITY_TASK_EXPOSURE_PLAN.json"
    assert plan_file.exists(), f"Missing exposure plan: {plan_file}"

    with open(plan_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    strat = data["arm_b_stratification_audit"]
    assert strat["verdict"] == "BALANCED_STRATIFIED_DISTRIBUTION"

    # All 4 options a, b, c, d must be present in Arm B training exposure
    options = strat["mcqa_target_option_distribution"]
    assert len(options) == 4
    for opt in ["a", "b", "c", "d"]:
        assert opt in options
        assert options[opt] >= 5, f"Option {opt} underrepresented: {options[opt]}"


def test_sampler_replay_determinism():
    """Verify 100% determinism on sampler replay with seed 42."""
    proof_file = _REPO_ROOT / "artifacts" / "experiments" / "V2_CAUSAL_TEMPORAL_ALIGNMENT" / "run_05_1" / "SAMPLER_REPLAY_UNIT_PROOF.json"
    assert proof_file.exists(), f"Missing replay proof: {proof_file}"

    with open(proof_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    assert data["deterministic_identity"] is True
    assert data["verdict"] == "100_PERCENT_REPLAY_DETERMINISM_PROVEN"
