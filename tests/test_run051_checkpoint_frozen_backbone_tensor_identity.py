"""Unit Test for Frozen Backbone and Compact Checkpoint Schema in Run 05.1."""

from __future__ import annotations

import json
from pathlib import Path
import pytest
import torch

_REPO_ROOT = Path(__file__).resolve().parents[1]


def test_model_state_checkpoint_identity():
    """Verify MODEL_STATE_CHECKPOINT_IDENTITY confirms 100% frozen backbone and pure adapter schema."""
    ident_file = _REPO_ROOT / "artifacts" / "experiments" / "V2_CAUSAL_TEMPORAL_ALIGNMENT" / "run_05_1" / "MODEL_STATE_CHECKPOINT_IDENTITY.json"
    assert ident_file.exists(), f"Missing checkpoint identity file: {ident_file}"

    with open(ident_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    arch = data["architecture_audit"]
    assert arch["adapted_modules_count"] == 72
    assert arch["base_layer_trainable_parameters"] == 0, "Base layer must have ZERO trainable parameters!"
    assert arch["canonical_parent_checkpoint"]["base_layer_keys_present"] == 0

    parity = data["logit_parity_summary"]
    assert parity["all_prompts_passed"] is True
    assert parity["verdict"] == "STRICT_NUMERICAL_PARITY_ESTABLISHED"


def test_parent_c1_checkpoint_pure_adapter():
    """Directly inspect C1_COMPACT_ADAPTER_ONLY.pt to verify no base_layer weights exist."""
    c1_path = _REPO_ROOT / "artifacts" / "experiments" / "V2_CAUSAL_TEMPORAL_ALIGNMENT" / "run_02" / "checkpoints" / "C1_COMPACT_ADAPTER_ONLY.pt"
    assert c1_path.exists(), f"Parent C1 checkpoint missing: {c1_path}"

    state = torch.load(c1_path, map_location="cpu", weights_only=True)
    assert "adapters_state" in state
    assert "gated_proj_state" in state

    # Verify all 72 modules have zero base_layer keys
    for mod_name, mod_state in state["adapters_state"].items():
        for k in mod_state.keys():
            assert "base_layer" not in k, f"Found base_layer key in {mod_name}: {k}"

    # Verify gated_proj keys
    for k in state["gated_proj_state"].keys():
        assert "base_layer" not in k
