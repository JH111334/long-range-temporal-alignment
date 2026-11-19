"""Unit Test for Checkpoint Lineage and Tensor Fingerprint Divergence in Run 05."""

from __future__ import annotations

import json
from pathlib import Path
import pytest

_REPO_ROOT = Path(__file__).resolve().parents[1]


def test_checkpoint_parent_and_tensor_identity():
    """Verify that Checkpoint A and Checkpoint B diverged genuinely from parent and each other."""
    run05_dir = _REPO_ROOT / "artifacts" / "experiments" / "V2_CAUSAL_TEMPORAL_ALIGNMENT" / "run_05_ab_ntp"
    identity_path = run05_dir / "CHECKPOINT_PARENT_AND_TENSOR_IDENTITY.json"

    assert identity_path.exists(), f"Identity manifest {identity_path} missing!"

    with open(identity_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # 1. Parent verification
    parent = data["parent_checkpoint"]
    assert parent["sha256"] == "30cadb35504e7f15e12922ba4cb6274cb6ee880a16df5f1f7592de48889cbc82"
    assert len(parent["tensor_fingerprint"]) == 64

    # 2. Arms verification
    arms = data["arms_trained"]
    assert "R5_A_NTP_REL" in arms
    assert "R5_B_NTP_REL_MCQA" in arms

    fp_a = arms["R5_A_NTP_REL"]["tensor_fingerprint"]
    fp_b = arms["R5_B_NTP_REL_MCQA"]["tensor_fingerprint"]
    fp_parent = parent["tensor_fingerprint"]

    assert fp_a != fp_parent, "Arm A failed to diverge from parent!"
    assert fp_b != fp_parent, "Arm B failed to diverge from parent!"
    assert fp_a != fp_b, "Arm A and Arm B have identical weights!"

    # 3. Divergence verification verdict
    div = data["divergence_verification"]
    assert div["a_differs_from_parent"] is True
    assert div["b_differs_from_parent"] is True
    assert div["a_differs_from_b"] is True
    assert div["verdict"] == "GENUINE_INDEPENDENT_TRAINING_VERIFIED"
