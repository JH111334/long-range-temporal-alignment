"""Unit Test for Factorial Inference Conditioning Interventions (Arm C)."""

from __future__ import annotations

import torch
import pytest

from temporal_alignment.models.dual_cycle_mat_lora import (
    clear_model_temporal_coordinates,
    set_model_temporal_coordinates,
)
from temporal_alignment.models.fone_canchi_40d import (
    compute_fixed_fone_canchi_40d,
    extract_canchi_angles_from_40d,
)
from temporal_alignment.models.gated_seam import (
    ConditioningContext,
    GatedResidualProjection,
)


def test_factorial_conditioning_mechanisms():
    device = torch.device("cpu")
    cond_context = ConditioningContext()
    gated_proj = GatedResidualProjection(in_features=40, out_features=896)

    # 1. C_TRUE
    vec_true = compute_fixed_fone_canchi_40d(1947)
    theta_true, z_true = extract_canchi_angles_from_40d(vec_true)
    vec_t = torch.tensor(vec_true, dtype=torch.float32, device=device).unsqueeze(0)
    proj_true = gated_proj(vec_t).squeeze(0)
    cond_context.set_vector(proj_true)
    assert cond_context.current_vector is not None
    assert cond_context.current_vector.shape == (896,)
    cond_context.clear()
    assert cond_context.current_vector is None

    # 2. C_WRONG_YEAR
    vec_wrong = compute_fixed_fone_canchi_40d(1947 + 30)
    theta_wrong, z_wrong = extract_canchi_angles_from_40d(vec_wrong)
    assert theta_true != theta_wrong or z_true != z_wrong

    # 3. C_ZERO_VECTOR
    vec_zero = [0.0] * 40
    vec_z_t = torch.tensor(vec_zero, dtype=torch.float32, device=device).unsqueeze(0)
    proj_zero = gated_proj(vec_z_t).squeeze(0)
    cond_context.set_vector(proj_zero)
    assert cond_context.current_vector is not None
    cond_context.clear()

    # 4. C_SEAM_OFF
    # Seam is cleared, LoRA takes theta/z
    cond_context.clear()
    assert cond_context.current_vector is None

    # 5. C_LORA_OFF
    # Seam is set, LoRA theta/z are zero/cleared
    cond_context.set_vector(proj_true)
    assert cond_context.current_vector is not None
    cond_context.clear()

    # 6. C_BOTH_OFF
    cond_context.clear()
    assert cond_context.current_vector is None
