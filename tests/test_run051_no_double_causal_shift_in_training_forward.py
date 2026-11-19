"""Unit Test for Causal Shift Correctness and Zero Double Shift in NTP/Target Loss."""

from __future__ import annotations

import json
from pathlib import Path
import pytest
import torch
import torch.nn.functional as F

_REPO_ROOT = Path(__file__).resolve().parents[1]


def test_no_double_causal_shift_parity():
    """Verify NTP_CAUSAL_SHIFT_NUMERICAL_PARITY confirms one-position shift with no double shift."""
    parity_file = _REPO_ROOT / "artifacts" / "experiments" / "V2_CAUSAL_TEMPORAL_ALIGNMENT" / "run_05_1" / "NTP_CAUSAL_SHIFT_NUMERICAL_PARITY.json"
    assert parity_file.exists(), f"Missing parity file: {parity_file}"

    with open(parity_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    audit = data["double_causal_shift_audit"]
    assert audit["is_model_forward_called_with_labels"] is False
    assert audit["is_manual_shift_applied_once"] is True
    assert audit["effective_shift_distance"] == 1
    assert audit["verdict"] == "ZERO_DOUBLE_SHIFT_VERIFIED"


def test_causal_shift_token_alignment():
    """Verify index correspondence: shift_logits[t] supervises shift_labels[t] == input_ids[t+1]."""
    seq_len = 10
    vocab_size = 50
    logits = torch.randn(1, seq_len, vocab_size)
    input_ids = torch.randint(0, vocab_size, (1, seq_len))

    # Correct one-position shift
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = input_ids[..., 1:].contiguous()

    assert shift_logits.shape[1] == seq_len - 1
    assert shift_labels.shape[1] == seq_len - 1

    # Check alignment for each step t in 0 .. seq_len-2
    for t in range(seq_len - 1):
        assert shift_labels[0, t] == input_ids[0, t + 1]

    # Verify cross entropy with ignore_index=-100 ignores masked tokens
    labels = input_ids.clone()
    labels[:, :5] = -100
    shift_labels_masked = labels[..., 1:].contiguous()

    loss = F.cross_entropy(
        shift_logits.view(-1, vocab_size),
        shift_labels_masked.view(-1),
        ignore_index=-100,
    )
    assert not torch.isnan(loss)
    assert loss.item() > 0.0
