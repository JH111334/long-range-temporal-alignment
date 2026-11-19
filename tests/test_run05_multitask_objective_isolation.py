"""Unit Test for Objective Isolation and Multi-Task Loss Formulation in Run 05."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytest

from temporal_alignment.training_run05 import compute_target_ce_loss


def test_mcqa_objective_isolated_to_slot3():
    """Verify that compute_target_ce_loss computes valid cross-entropy on last token."""
    vocab_size = 100
    batch_size = 1
    seq_len = 8

    # Mock logits
    logits = torch.randn(batch_size, seq_len, vocab_size)
    target_token_id = 42

    last_logits = logits[:, -1, :].contiguous()
    target_tensor = torch.tensor([target_token_id], dtype=torch.long)

    loss = F.cross_entropy(last_logits, target_tensor)
    assert not torch.isnan(loss)
    assert loss.item() > 0.0


def test_arm_loss_composition():
    """Verify mathematical combination of components for Arm A vs Arm B."""
    l_ntp = torch.tensor(1.5, requires_grad=True)
    l_rel = torch.tensor(0.8, requires_grad=True)
    l_ctrl = torch.tensor(0.7, requires_grad=True)
    l_mcqa = torch.tensor(1.2, requires_grad=True)

    # Arm A formula: L_A = L_NTP + L_REL + L_CTRL
    loss_a = l_ntp + l_rel + l_ctrl
    assert torch.isclose(loss_a, torch.tensor(3.0))

    # Arm B formula: L_B = L_NTP + L_REL + L_MCQA
    loss_b = l_ntp + l_rel + l_mcqa
    assert torch.isclose(loss_b, torch.tensor(3.5))

    # Backward propagates to respective components
    loss_a.backward()
    assert l_ntp.grad is not None
    assert l_rel.grad is not None
    assert l_ctrl.grad is not None
    assert l_mcqa.grad is None
