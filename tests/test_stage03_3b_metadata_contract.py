"""Unit tests for Stage 03 Qwen2.5-3B-Instruct Metadata and Contract Invariants."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import pytest
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_3b_config_yaml_files_exist_and_valid():
    plan_path = PROJECT_ROOT / "Reports/03_TARGETED_OBJECTIVES_MCQA/QWEN25_3B_TRANSITION/config/3b_research_plan.yaml"
    contract_path = PROJECT_ROOT / "Reports/03_TARGETED_OBJECTIVES_MCQA/QWEN25_3B_TRANSITION/config/3b_model_and_resource_contract.yaml"

    assert plan_path.exists(), f"Missing {plan_path}"
    assert contract_path.exists(), f"Missing {contract_path}"

    with open(plan_path, "r", encoding="utf-8") as f:
        plan = yaml.safe_load(f)
    assert plan["research_plan"]["status"] == "PRE_3B_READY"
    assert plan["research_plan"]["prohibitions"]["no_go_large_training"] is True
    assert plan["research_plan"]["prohibitions"]["no_go_option_permutation"] is True

    with open(contract_path, "r", encoding="utf-8") as f:
        contract = yaml.safe_load(f)
    assert contract["model_contract"]["model_id"] == "Qwen/Qwen2.5-3B-Instruct"
    assert contract["model_contract"]["tensor_shapes"]["hidden_size"] == 2048
    assert contract["model_contract"]["tensor_shapes"]["num_hidden_layers"] == 36
    assert contract["model_contract"]["adapter_specification"]["total_adapted_modules"] == 108


def test_3b_audit_json_files_exist_and_consistent():
    trans_dir = PROJECT_ROOT / "Reports/03_TARGETED_OBJECTIVES_MCQA/QWEN25_3B_TRANSITION/audits"
    freeze_json = trans_dir / "PRE_3B_05B_EVIDENCE_FREEZE.json"
    lic_json = trans_dir / "3B_MODEL_CARD_LICENSE_AUDIT.json"
    shape_json = trans_dir / "3B_MODEL_SHAPE_ADAPTER_INVENTORY.json"
    budget_json = trans_dir / "3B_RUNTIME_CAPABILITY_AND_BUDGET.json"
    gate_json = trans_dir / "PRE_3B_GATE_MATRIX.json"

    for p in [freeze_json, lic_json, shape_json, budget_json, gate_json]:
        assert p.exists(), f"Missing audit file: {p}"
        with open(p, "r", encoding="utf-8") as f:
            data = json.load(f)
            assert isinstance(data, dict), f"Expected dict in {p}"

    with open(gate_json, "r", encoding="utf-8") as f:
        gates_data = json.load(f)
    assert gates_data["overall_verdict"] == "PRE_3B_READY"
    assert len(gates_data["gates"]) == 15
    assert gates_data["gates"]["G11_GPU_VRAM_PRECISION_MEASUREMENT"]["status"] == "NOT_VERIFIED"
    assert gates_data["gates"]["G01_REPO_BRANCH_AND_OWNERSHIP"]["status"] == "PASS"


def test_counterfactual_dataset_byte_hash_unchanged():
    cf_path = PROJECT_ROOT / "data/counterfactual/COUNTERFACTUAL_GOLD_AUDIT.jsonl"
    assert cf_path.exists(), f"Missing {cf_path}"

    hasher = hashlib.sha256()
    with open(cf_path, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    digest = hasher.hexdigest()

    expected_sha = "49eea7fe6fad69d4f3695e7cd8fbeac2c383e65b76d0802801666f9982cdeba5"
    assert digest == expected_sha, f"Dataset byte hash mutation detected! Expected {expected_sha}, got {digest}"


def test_option_permutation_default_remains_false():
    from temporal_alignment.training.option_permutation import OnlineOptionPermutation

    module_default = OnlineOptionPermutation()
    assert module_default.enabled is False, "OnlineOptionPermutation must be disabled by default"

    module_explicit = OnlineOptionPermutation(enabled=False)
    assert module_explicit.enabled is False


def test_cross_scale_dimensional_incompatibility():
    """Verify that 0.5B adapter weights (dim 896) cannot match 3B linear layer (dim 2048)."""
    import torch
    import torch.nn as nn

    # Simulating 3B gate_proj
    linear_3b = nn.Linear(2048, 11008, bias=False)
    # Simulating 0.5B LoRA matrix A
    lora_A_05b = torch.randn(8, 896)

    # Incompatible matrix multiplication must fail closed
    with pytest.raises(RuntimeError):
        _ = torch.matmul(linear_3b.weight[:8, :], lora_A_05b)
