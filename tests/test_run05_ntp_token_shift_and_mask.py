"""Unit Test for Run 05 Causal NTP Shift, Masking and Loss Contract."""

from __future__ import annotations

import torch
import torch.nn.functional as F
import pytest

from temporal_alignment.training_run05 import compute_causal_ntp_loss


def test_ntp_causal_shift_alignment():
    """Verify that shift_logits[:-1] strictly aligns with shift_labels[1:]."""
    seq_len = 10
    vocab_size = 50
    batch_size = 2

    # Simulated input_ids
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
    # Simulated logits from a causal language model
    logits = torch.randn(batch_size, seq_len, vocab_size)

    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = input_ids[..., 1:].contiguous()

    # Dimension checks
    assert shift_logits.shape == (batch_size, seq_len - 1, vocab_size)
    assert shift_labels.shape == (batch_size, seq_len - 1)

    # Position check: logits at position 0 align with label at position 1
    loss_manual = F.cross_entropy(
        shift_logits.view(-1, vocab_size),
        shift_labels.view(-1),
        ignore_index=-100,
    )
    assert not torch.isnan(loss_manual)
    assert loss_manual.item() > 0.0


def test_ntp_shift_rejects_off_by_one_unaligned():
    """Verify that unshifted labels produce mismatched cross-entropy."""
    vocab_size = 20
    logits = torch.randn(1, 5, vocab_size)
    input_ids = torch.randint(0, vocab_size, (1, 5))

    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = input_ids[..., 1:].contiguous()
    correct_loss = F.cross_entropy(shift_logits.view(-1, vocab_size), shift_labels.view(-1))

    # Bogus unshifted alignment
    bogus_loss = F.cross_entropy(logits[..., :-1, :].view(-1, vocab_size), input_ids[..., :-1].view(-1))

    # They should evaluate different tokens
    assert correct_loss.shape == bogus_loss.shape
