"""Unit Test for ChatML Prefix Label Masking and EOT Token Supervision."""

from __future__ import annotations

import json
from pathlib import Path
import pytest
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

_REPO_ROOT = Path(__file__).resolve().parents[1]


def test_chatml_target_contains_eot_token():
    """Verify CHATML_ASSISTANT_TARGET_TOKEN_AUDIT confirms presence of <|im_end|>."""
    audit_file = _REPO_ROOT / "artifacts" / "experiments" / "V2_CAUSAL_TEMPORAL_ALIGNMENT" / "run_05_1" / "CHATML_ASSISTANT_TARGET_TOKEN_AUDIT.jsonl"
    assert audit_file.exists(), f"Missing audit file: {audit_file}"

    records = []
    with open(audit_file, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))

    assert len(records) >= 36
    for r in records:
        assert r["contains_im_end_151645"] is True
        assert 151645 in r["corrected_target_token_ids"]
        # Flawed Run 05 targets strictly lacked 151645
        assert 151645 not in r["flawed_target_token_ids"]


def test_label_mask_prefix_tokens():
    """Verify that prompt tokens are masked with -100 and only target tokens receive gradient."""
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct", local_files_only=True)
    prompt = "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\nTest question?<|im_end|>\n<|im_start|>assistant\n"
    target = "a<|im_end|>\n"
    full_text = prompt + target

    prompt_enc = tokenizer(prompt, return_tensors="pt")
    full_enc = tokenizer(full_text, return_tensors="pt")

    prompt_len = prompt_enc["input_ids"].shape[1]
    input_ids = full_enc["input_ids"]

    labels = input_ids.clone()
    labels[:, :prompt_len] = -100

    # Unmasked tokens count
    unmasked = (labels != -100).sum().item()
    target_len = full_enc["input_ids"].shape[1] - prompt_len
    assert unmasked == target_len, f"Expected {target_len} unmasked tokens, got {unmasked}"

    # Verify first unmasked token is target
    first_unmasked_idx = prompt_len
    assert labels[0, first_unmasked_idx].item() == tokenizer.encode("a")[0]
    # Verify second unmasked token is <|im_end|> (151645)
    assert labels[0, first_unmasked_idx + 1].item() == 151645
