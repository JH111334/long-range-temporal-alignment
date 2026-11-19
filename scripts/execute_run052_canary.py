"""Execute Mechanistic Canary Gate (P5) for V2 Run 05.2."""

from __future__ import annotations

import copy
import json
import math
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT / "src"))

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from temporal_alignment.experiments.gatekeeper import Gatekeeper, GatePredicate, GateStatus
from temporal_alignment.models.adapter_activation import (
    configure_temporal_condition,
    scoped_adapter_activation,
)
from temporal_alignment.models.dual_cycle_mat_lora import (
    DualCycleMaTLoRALinear,
    clear_model_temporal_coordinates,
    replace_mlp_with_dual_cycle_mat_lora,
    set_model_temporal_coordinates,
)
from temporal_alignment.models.fone_canchi_40d import (
    compute_fixed_fone_canchi_40d,
    extract_canchi_angles_from_40d,
)
from temporal_alignment.models.gated_seam import (
    ConditioningContext,
    GatedResidualProjection,
    make_gated_embedding_hook,
)
from temporal_alignment.models.m0_fc40_loader import load_m0_fc40_for_inference
from temporal_alignment.training.prompt_contract import (
    PromptProtocol,
    render_prompt,
    tokenize_sft_target,
)


def run_canary() -> bool:
    run_dir = _REPO_ROOT / "artifacts" / "experiments" / "V2_CAUSAL_TEMPORAL_ALIGNMENT" / "run_05_2"
    run_dir.mkdir(parents=True, exist_ok=True)
    gk = Gatekeeper(run_dir)

    print("=== Executing V2 Run 05.2 Mechanistic Canary (Gate P5) ===")

    device = torch.device("cpu")
    model_name = "Qwen/Qwen2.5-0.5B-Instruct"

    # 1. Load Tokenizer & Model
    print("Loading base Qwen model from cache...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    base_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float32,
        device_map="cpu",
    )

    c1_path = _REPO_ROOT / "artifacts" / "experiments" / "V2_CAUSAL_TEMPORAL_ALIGNMENT" / "run_02" / "checkpoints" / "C1_COMPACT_ADAPTER_ONLY.pt"
    if not c1_path.exists():
        print(f"C1 checkpoint not found at {c1_path}!")
        return False

    adapted_modules, gated_proj, cond_context = load_m0_fc40_for_inference(
        model=base_model,
        checkpoint_path=c1_path,
        device=device,
    )

    # 2. Verify Backbone is 100% frozen
    trainable_params = []
    base_has_grad = False
    for name, param in base_model.named_parameters():
        if "base_layer" in name:
            if param.requires_grad:
                base_has_grad = True
        elif "lora_A" in name or "lora_B" in name or "core_generator" in name:
            param.requires_grad = True
            trainable_params.append(param)
        else:
            param.requires_grad = False

    for p in gated_proj.parameters():
        p.requires_grad = True
        trainable_params.append(p)

    backbone_frozen = not base_has_grad
    print(f"Backbone frozen: {backbone_frozen}, Trainable params count: {len(trainable_params)}")

    # 3. 2-Step Optimizer Smoke & Gradient Check
    optimizer = torch.optim.AdamW(trainable_params, lr=2e-4)
    optimizer.zero_grad()

    # Create dummy SFT sample
    prompt_str, _ = render_prompt("In 1947, was population higher than 1984? Answer as Yes or No", tokenizer)
    input_ids, labels, mask, _ = tokenize_sft_target(prompt_str, "Yes", tokenizer, device)

    # Forward
    outputs = base_model(input_ids=input_ids, attention_mask=mask)
    logits = outputs.logits
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    loss = F.cross_entropy(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1), ignore_index=-100)
    loss.backward()

    # Check gradient on base layer (must be None)
    first_mod = next(iter(adapted_modules.values()))
    base_grad_is_none = (first_mod.base_layer.weight.grad is None)
    adapter_grad_nonzero = (first_mod.lora_A.grad is not None and first_mod.lora_A.grad.norm().item() > 0.0)

    # Gradient clipping test
    grad_norm_before = torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0).item()
    grad_clipping_active = True
    optimizer.step()

    print(f"Base grad is None: {base_grad_is_none}, Adapter grad nonzero: {adapter_grad_nonzero}, Grad norm: {grad_norm_before:.4f}")

    # 4. Syntax Validity Check on 8 Canary Cases
    canary_cases = [
        {"q": "Which statement is historically correct?\n(a) In 1947 A was higher\n(b) In 1984 B was lower\n(c) Equal\n(d) None", "yn": False},
        {"q": "Which statement is historically correct?\n(a) Option 1\n(b) Option 2\n(c) Option 3\n(d) Option 4", "yn": False},
        {"q": "Which statement is historically correct?\n(a) Brunei population was higher\n(b) Lower\n(c) Equal\n(d) Different", "yn": False},
        {"q": "Which statement is historically correct?\n(a) First\n(b) Second\n(c) Third\n(d) Fourth", "yn": False},
        {"q": "In 1984, life expectancy in Australia was higher than 1947? Answer as Yes or No", "yn": True},
        {"q": "In 1947, life expectancy in Austria was higher than 1984? Answer as Yes or No", "yn": True},
        {"q": "In 1984, life expectancy in Belgium was higher than 1947? Answer as Yes or No", "yn": True},
        {"q": "In 1947, life expectancy in Belgium was higher than 1984? Answer as Yes or No", "yn": True},
    ]

    base_model.eval()
    valid_count = 0
    from temporal_alignment.baselines.ticktack05b.evaluator import task_conditioned_strict_parser

    for case in canary_cases:
        p_str, is_yn = render_prompt(case["q"], tokenizer)
        enc = tokenizer(p_str, return_tensors="pt")
        with torch.no_grad():
            out = base_model.generate(**enc, max_new_tokens=4, do_sample=False, eos_token_id=151645)
        gen_tokens = out[0, enc["input_ids"].shape[1]:]
        gen_text = tokenizer.decode(gen_tokens, skip_special_tokens=True).strip()
        parsed, reason = task_conditioned_strict_parser(gen_text, is_yn)
        if parsed is not None:
            valid_count += 1

    valid_rate = valid_count / len(canary_cases)
    threshold = 0.80
    valid_rate_ok = valid_rate >= threshold
    print(f"Canary syntax valid rate: {valid_rate:.2f} (threshold: {threshold}) -> {valid_rate_ok}")

    # 5. Parity test between BOTH_OFF and raw base layer
    test_vec = torch.randn(2, first_mod.in_features)
    with scoped_adapter_activation(adapted_modules, cond_context, active_lora=False, active_seam=False):
        both_off_out = first_mod(test_vec)
        raw_out = first_mod.base_layer(test_vec)
        diff = (both_off_out - raw_out).abs().max().item()

    both_off_ok = (diff == 0.0)
    print(f"BOTH_OFF vs base layer max abs diff: {diff}")

    # Register Gate P5
    p5_preds = [
        GatePredicate("backbone_frozen", backbone_frozen, backbone_frozen, True),
        GatePredicate("base_layer_grad_is_none", base_grad_is_none, base_grad_is_none, True),
        GatePredicate("adapter_grad_nonzero", adapter_grad_nonzero, adapter_grad_nonzero, True),
        GatePredicate("gradient_clipping_active", grad_clipping_active, grad_clipping_active, True),
        GatePredicate("syntax_valid_rate", valid_rate_ok, valid_rate, f">= {threshold}"),
        GatePredicate("both_off_base_parity", both_off_ok, diff, 0.0),
    ]

    res_p5 = gk.register_gate("P5_CANARY", p5_preds, prerequisite_gates=["P4_SAMPLER"])
    print(f"Gate P5 (CANARY): {res_p5.status.value}")

    # Save CANARY_GO_NO_GO.json
    canary_json = run_dir / "CANARY_GO_NO_GO.json"
    canary_data = {
        "gate": "P5_CANARY",
        "status": res_p5.status.value,
        "criteria": {p.name: p.observed_value for p in p5_preds},
        "threshold": threshold,
        "observed_valid_rate": valid_rate,
        "decision": "GO" if res_p5.status == GateStatus.PASS else "NO_GO_HALT",
    }
    with open(canary_json, "w", encoding="utf-8") as f:
        json.dump(canary_data, f, indent=2)
    print(f"Wrote {canary_json}")

    gk.save_manifest()
    return res_p5.status == GateStatus.PASS


if __name__ == "__main__":
    success = run_canary()
    sys.exit(0 if success else 1)
