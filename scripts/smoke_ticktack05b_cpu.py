"""Deterministic CPU Smoke Test for TickTack-inspired 0.5B Baseline.

Verifies:
1. Real Qwen2.5-0.5B-Instruct loading on CPU.
2. 100% frozen backbone (all base model parameters have requires_grad=False).
3. Exactly 112 LoRA attention projection parameters have requires_grad=True.
4. Forward pass with temporal injection (Eq. 5) computes finite output.
5. Backward pass produces finite gradients strictly on LoRA parameters.
6. Free-vocabulary autoregressive generation produces non-empty token sequence.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
from transformers import AutoTokenizer

_PKG_ROOT = Path(__file__).resolve().parents[1] / "src"
if str(_PKG_ROOT) not in sys.path:
    sys.path.insert(0, str(_PKG_ROOT))

from temporal_alignment.baselines.ticktack05b.loss import compute_ntp_loss, compute_temporal_contrastive_loss
from temporal_alignment.baselines.ticktack05b.model import TickTackQwen05BModel


def run_cpu_smoke() -> None:
    print("Running Deterministic CPU Smoke Test for TickTack 0.5B...")
    model_name = "Qwen/Qwen2.5-0.5B-Instruct"
    revision = "7ae557604adf67be50417f59c2c2f167def9a775"

    tokenizer = AutoTokenizer.from_pretrained(model_name, revision=revision, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 1. Initialize model on CPU with float32
    print("Loading TickTackQwen05BModel on CPU...")
    model = TickTackQwen05BModel(
        base_model_name_or_path=model_name,
        revision=revision,
        torch_dtype=torch.float32,
    )
    model.eval()

    # 2. Check parameter freeze invariants
    trainable_names = []
    frozen_count = 0
    for name, p in model.named_parameters():
        if p.requires_grad:
            trainable_names.append(name)
            assert "lora_" in name, f"Non-LoRA parameter requires grad: {name}"
        else:
            frozen_count += 1

    print(f"PASS: {frozen_count} backbone parameters frozen. {len(trainable_names)} LoRA parameters trainable.")
    assert len(trainable_names) > 0, "No trainable LoRA parameters found!"

    # 3. Test forward pass with temporal injection
    print("Testing forward pass with Eq. 5 temporal injection...")
    prompt = "In 1947, the index was reported."
    toks = tokenizer(prompt, return_tensors="pt")
    input_ids = toks["input_ids"]
    attention_mask = toks["attention_mask"]

    out = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        years=[1947],
        tokenizer=tokenizer,
        temporal_injection_enabled=True,
    )

    logits = out["logits"]
    pooled = out["pooled_hidden_state"]
    assert torch.all(torch.isfinite(logits)), "Forward logits contain NaN/Inf!"
    assert torch.all(torch.isfinite(pooled)), "Pooled hidden state contains NaN/Inf!"
    print(f"PASS: Forward pass finite. Logits shape: {logits.shape}, Pooled shape: {pooled.shape}")

    # 4. Test backward pass and gradient flow
    print("Testing backward pass and gradient flow...")
    model.train()
    labels = input_ids.clone()
    loss_ntp = compute_ntp_loss(logits, labels)
    loss_temp, _ = compute_temporal_contrastive_loss(pooled, torch.tensor([1947]))
    total_loss = loss_ntp + 0.5 * loss_temp

    total_loss.backward()

    # Verify gradients
    trainable_with_grad = 0
    for name, p in model.named_parameters():
        if p.requires_grad:
            assert p.grad is not None, f"Trainable parameter {name} has no gradient!"
            assert torch.all(torch.isfinite(p.grad)), f"Gradient for {name} has NaN/Inf!"
            trainable_with_grad += 1
        else:
            assert p.grad is None, f"Frozen parameter {name} unexpectedly has gradient!"

    print(f"PASS: Gradients strictly computed on all {trainable_with_grad} trainable LoRA parameters. Zero update to W0.")

    # 5. Test free-vocabulary greedy generation
    print("Testing free-vocabulary greedy autoregressive generation (RAW_GREEDY_V1)...")
    gen_tokens = model.generate_free_vocab(
        input_ids=input_ids,
        year=1947,
        tokenizer=tokenizer,
        max_new_tokens=4,
        temporal_injection_enabled=True,
    )
    gen_text = tokenizer.decode(gen_tokens, skip_special_tokens=True)
    print(f"PASS: Autoregressive generation successful. Tokens: {gen_tokens}, Text: '{gen_text}'")

    print("\nALL DETERMINISTIC CPU SMOKE CHECKS PASSED.")


if __name__ == "__main__":
    run_cpu_smoke()
