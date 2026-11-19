"""Unit and integration tests for TickTack-inspired 0.5B baseline implementation.

Verifies:
1. Sexagenary mapping (Eq. 1) for CE and BCE, rejection of year zero.
2. Polar and Cartesian conversion (Eq. 2 & 3), literal vs adapted monotonic radius behavior.
3. Sinusoidal Temporal Encoding (Eq. 4) mathematical properties and independence.
4. Input embedding injection (Eq. 5) correctness and preservation of non-target tokens.
5. Numerical stability of NTP loss, temporal contrastive loss, and LoRA EWC loss.
6. Task-aware P12 prompt detection and strict parser validity guardrails.
"""

from __future__ import annotations

import math
import pytest
import torch
import torch.nn as nn

from temporal_alignment.baselines.ticktack05b.evaluator import (
    build_p12_prompt,
    detect_is_yes_no_task,
    task_conditioned_strict_parser,
)
from temporal_alignment.baselines.ticktack05b.loss import (
    LoRAEWCLoss,
    compute_ntp_loss,
    compute_temporal_contrastive_loss,
)
from temporal_alignment.baselines.ticktack05b.temporal_encoding import (
    compute_polar_coordinates,
    compute_sinusoidal_te,
    get_ticktack_temporal_embedding,
    inject_temporal_embedding,
    polar_to_cartesian,
    year_to_sexagenary_cycle,
)


def test_year_to_sexagenary_cycle() -> None:
    """Verifies Eq. 1 sexagenary mapping across known calendar years."""
    # Known benchmark anchors
    assert year_to_sexagenary_cycle(1947) == 23  # Đinh Hợi (index 23)
    assert year_to_sexagenary_cycle(1984) == 0   # Giáp Tý (index 0)
    assert year_to_sexagenary_cycle(2022) == 38  # Nhâm Dần (index 38)
    assert year_to_sexagenary_cycle(2024) == 40  # Giáp Thìn (index 40)

    # BCE astronomical mapping
    assert year_to_sexagenary_cycle(-1) == 56   # 1 BCE
    assert year_to_sexagenary_cycle(-100) == 17 # 100 BCE

    # Zero rejection
    with pytest.raises(ValueError):
        year_to_sexagenary_cycle(0)


def test_polar_and_cartesian_coordinates() -> None:
    """Verifies Eq. 2 & 3 and isolates the literal vs adapted radius ambiguity."""
    # Test Cartesian identity x^2 + y^2 = r^2
    r, theta = compute_polar_coordinates(1947, mode="literal_eq2")
    x, y = polar_to_cartesian(r, theta)
    assert math.isclose(x**2 + y**2, r**2, rel_tol=1e-6)

    # Literal Eq. 2 ambiguity verification: 1965 and 2025 have identical theta & radius
    r_1965_lit, th_1965_lit = compute_polar_coordinates(1965, mode="literal_eq2")
    r_2025_lit, th_2025_lit = compute_polar_coordinates(2025, mode="literal_eq2")
    assert math.isclose(th_1965_lit, th_2025_lit, rel_tol=1e-6)
    assert math.isclose(r_1965_lit, r_2025_lit, rel_tol=1e-6), "Literal Eq. 2 fails to distinguish 1965 and 2025"

    # Adapted monotonic radius distinction: 2025 must have larger radius than 1965
    r_1965_ada, _ = compute_polar_coordinates(1965, mode="adapted_monotonic")
    r_2025_ada, _ = compute_polar_coordinates(2025, mode="adapted_monotonic")
    assert r_2025_ada > r_1965_ada, "Adapted monotonic radius must distinguish distinct epochs"


def test_sinusoidal_te_properties() -> None:
    """Verifies Eq. 4 sinusoidal properties, shape, and bounds."""
    dim = 896
    val = 2.45
    te = compute_sinusoidal_te(val, dim=dim)

    assert te.shape == (dim,)
    assert te.dtype == torch.float32
    assert torch.all(te >= -1.0) and torch.all(te <= 1.0)

    # Independent verification of first two dimensions
    expected_te0 = math.sin(val)
    expected_te1 = math.cos(val)
    assert math.isclose(float(te[0]), expected_te0, rel_tol=1e-5)
    assert math.isclose(float(te[1]), expected_te1, rel_tol=1e-5)


def test_temporal_embedding_injection() -> None:
    """Verifies Eq. 5 embedding injection modifies target tokens and preserves others."""
    batch_size = 2
    seq_len = 8
    dim = 896

    embeds = torch.zeros(batch_size, seq_len, dim)
    te = get_ticktack_temporal_embedding(1947, dim=dim)
    target_indices = [2, 5]

    modified = inject_temporal_embedding(embeds, target_indices, te)

    # Check target indices received injection
    for idx in target_indices:
        assert torch.allclose(modified[0, idx, :], te)
        assert torch.allclose(modified[1, idx, :], te)

    # Check non-target indices remain strictly zero
    non_targets = [i for i in range(seq_len) if i not in target_indices]
    for idx in non_targets:
        assert torch.all(modified[0, idx, :] == 0.0)
        assert torch.all(modified[1, idx, :] == 0.0)


def test_loss_functions() -> None:
    """Verifies NTP loss, temporal contrastive loss, and LoRA EWC loss stability."""
    # 1. NTP Loss FP32
    vocab_size = 100
    logits = torch.randn(2, 5, vocab_size, requires_grad=True)
    labels = torch.tensor([[10, 20, 30, -100, -100], [5, 15, 25, 35, -100]])
    loss_ntp = compute_ntp_loss(logits, labels)
    assert loss_ntp > 0.0
    assert torch.isfinite(loss_ntp)
    loss_ntp.backward()
    assert logits.grad is not None and torch.all(torch.isfinite(logits.grad))

    # 2. Temporal Contrastive Loss (Eq. 8, delta=1.0)
    h_states = torch.randn(4, 64)
    # 2 samples from 1947, 2 samples from 1984
    year_labels = torch.tensor([1947, 1947, 1984, 1984])
    loss_temp, n_pos = compute_temporal_contrastive_loss(h_states, year_labels, delta=1.0)
    assert n_pos == 4  # (0,1), (1,0), (2,3), (3,2)
    assert loss_temp > 0.0
    assert torch.isfinite(loss_temp)

    # 3. LoRA EWC Loss (Eq. 9)
    linear = nn.Linear(10, 10)
    ewc = LoRAEWCLoss(lambda_ewc=50.0)
    # Before registration, penalty is 0.0
    pen_init, _ = ewc(linear)
    assert pen_init.item() == 0.0

    # Register Fisher
    fisher_fake = {"weight": torch.ones_like(linear.weight)}
    ewc.update_task_fisher(linear, fisher_fake)

    # Shift weights
    with torch.no_grad():
        linear.weight.add_(0.5)

    pen_shifted, count = ewc(linear)
    assert count == 100
    assert pen_shifted.item() > 0.0
    assert torch.isfinite(pen_shifted)


def test_task_aware_evaluator() -> None:
    """Verifies task detection from question text and strict parsing rules."""
    # Task detection without gold target
    yn_q = "In 1947, Private Civil Liberties Index was higher than 1952? Answer as Yes or No"
    mc_q = "In 1984, what was the primary economic indicator? (a) GDP (b) CPI (c) HDI (d) Debt"

    assert detect_is_yes_no_task(yn_q) is True
    assert detect_is_yes_no_task(mc_q) is False

    # Parser: Valid outputs
    assert task_conditioned_strict_parser("Yes", is_yn=True) == ("Yes", "VALID")
    assert task_conditioned_strict_parser("  no  ", is_yn=True) == ("No", "VALID")
    assert task_conditioned_strict_parser("(a)", is_yn=False) == ("a", "VALID")
    assert task_conditioned_strict_parser("Answer: b", is_yn=False) == ("b", "VALID")

    # Parser: Invalid outputs (Task type mismatches)
    # Outputting 'Yes' on MCQA must be strictly INVALID
    pred_mc, stat_mc = task_conditioned_strict_parser("Yes", is_yn=False)
    assert stat_mc == "INVALID_WRONG_TYPE_YN_ON_MCQA"
    assert pred_mc is None

    # Outputting letter on Yes/No must be strictly INVALID
    pred_yn, stat_yn = task_conditioned_strict_parser("c", is_yn=True)
    assert stat_yn == "INVALID_WRONG_TYPE_LETTER_ON_YN"
    assert pred_yn is None
