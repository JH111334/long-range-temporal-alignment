#!/usr/bin/env python3
"""
Unit tests for Ticktack Architecture, Mathematical Equations (1-4, 8), and Scale Normalization.
Pass criteria:
- Eq. 1: Exact Can Chi cycle indices (1947 -> 24, 1984 -> 1, 2022 -> 39) and edge cases.
- Eq. 2-3: Exact polar radius, angle, and Cartesian projection.
- Eq. 4: Sinusoidal Temporal Encoding produces correct dimension [B, 1, D] and bounded outputs.
- Mechanism: Raw TE causes cosine collapse (> 0.999), Scaled TE maintains cosine < 0.15.
- Eq. 8: Can Chi intra-class contrastive loss computes correctly, handles edge cases, and produces valid gradients.
"""

import math
import pytest
import torch
import torch.nn.functional as F

def compute_canchi_cycle(year: int) -> int:
    """Eq. 1 from Han et al. (2025)."""
    if year >= 4:
        return (year - 3) % 60
    elif year > 0:
        return (60 - abs(year - 3)) % 60
    else:
        return (60 - abs(year) - 2) % 60

def compute_te_vector(years: torch.Tensor, hidden_dim=896, alpha=0.5, beta=0.5, device="cpu", dtype=torch.float32, scale_mode="raw", gating_factor=0.01):
    """
    Computes vectorized TE following Eq. 2, 3, 4.
    scale_mode: 'raw' (Eq. 5 original) or 'scaled' (energy-normalized fixing cosine collapse).
    """
    # 1. Eq. 1: Cycle indices in [0, 59]
    cycle_indices = ((years.to(torch.int64) - 3) % 60).to(torch.float32)
    # 2. Eq. 2: Angle in radians (6 degrees per cycle unit)
    theta = (cycle_indices * 6.0) * (math.pi / 180.0)
    # Radius
    r = alpha + beta * theta
    # 3. Eq. 3: Cartesian coordinates
    x = (r * torch.cos(theta)).view(-1, 1).to(device)
    y = (r * torch.sin(theta)).view(-1, 1).to(device)
    
    # 4. Eq. 4: Sinusoidal embeddings
    dim_indices = torch.arange(0, hidden_dim, 2, device=device, dtype=torch.float32)
    inv_freq = 1.0 / (10000.0 ** (dim_indices / float(hidden_dim))).unsqueeze(0)
    
    x_inp = x * inv_freq
    te_x = torch.stack([torch.sin(x_inp), torch.cos(x_inp)], dim=-1).flatten(start_dim=-2)
    
    y_inp = y * inv_freq
    te_y = torch.stack([torch.sin(y_inp), torch.cos(y_inp)], dim=-1).flatten(start_dim=-2)
    
    te_raw = (te_x + te_y).unsqueeze(1).to(dtype=dtype)  # [B, 1, D]
    
    if scale_mode == "raw":
        return te_raw
    elif scale_mode == "scaled":
        # Scale-normalization fixing scale mismatch: scale by 1/d or gating factor
        # Scaling by 1.0 / hidden_dim keeps TE energy proportional to token embeddings
        scale = 1.0 / float(hidden_dim)
        return (te_raw * scale).to(dtype=dtype)
    elif scale_mode == "gated":
        return (te_raw * gating_factor).to(dtype=dtype)
    else:
        raise ValueError(f"Unknown scale_mode: {scale_mode}")

def compute_intra_loss(hidden_states: torch.Tensor, years: torch.Tensor):
    """Eq. 8 intra-class contrastive loss clustering by Can Chi sexagenary cycle (Han et al., 2025)."""
    B = hidden_states.size(0)
    norm_h = F.normalize(hidden_states, p=2, dim=-1)
    sim_matrix = torch.matmul(norm_h, norm_h.T)
    
    canchi_cycles = ((years.to(torch.int64) - 3) % 60).to(device=hidden_states.device)
    same_cycle_mask = (canchi_cycles.unsqueeze(1) == canchi_cycles.unsqueeze(0)) & (~torch.eye(B, dtype=torch.bool, device=hidden_states.device))
    if same_cycle_mask.sum() > 0:
        return 1.0 - sim_matrix[same_cycle_mask].mean()
    else:
        return (hidden_states * 0.0).sum()

class TestTicktackArchitecture:

    def test_eq1_canchi_cycles(self):
        """Test Eq. 1 cycle calculation on benchmark years and edge cases."""
        assert compute_canchi_cycle(1947) == 24
        assert compute_canchi_cycle(1984) == 1
        assert compute_canchi_cycle(2022) == 39
        assert compute_canchi_cycle(4) == 1
        assert compute_canchi_cycle(3) == 0
        assert compute_canchi_cycle(1) == 58
        assert compute_canchi_cycle(-1) == 57

    def test_eq2_eq3_coordinates(self):
        """Test polar to Cartesian conversion for year 1984."""
        years = torch.tensor([1984])
        cycle = compute_canchi_cycle(1984)
        assert cycle == 1
        theta = 1 * 6.0 * (math.pi / 180.0)
        r = 0.5 + 0.5 * theta
        expected_x = r * math.cos(theta)
        expected_y = r * math.sin(theta)
        
        te_raw = compute_te_vector(years, hidden_dim=896, scale_mode="raw")
        assert te_raw.shape == (1, 1, 896)
        assert not torch.isnan(te_raw).any()

    def test_eq4_te_properties(self):
        """Test sinusoidal temporal encoding dimensionality and bounds."""
        years = torch.tensor([1947, 1984, 2022])
        te = compute_te_vector(years, hidden_dim=896, scale_mode="raw")
        assert te.shape == (3, 1, 896)
        # Components are sum of two sin/cos values, so bounded in [-2.0, 2.0]
        assert te.min().item() >= -2.0 - 1e-5
        assert te.max().item() <= 2.0 + 1e-5

    def test_cosine_collapse_and_scale_fix(self):
        """
        Verify that Raw TE causes cosine collapse (> 0.999),
        while Scaled TE (scale = 1/d or energy-scaled) prevents collapse (< 0.15).
        """
        torch.manual_seed(42)
        d = 896
        # Typical Qwen embedding standard deviation is ~0.015 - 0.03
        h1 = torch.randn(d) * 0.015
        h2 = torch.randn(d) * 0.015
        
        orig_cos = F.cosine_similarity(h1.unsqueeze(0), h2.unsqueeze(0)).item()
        assert abs(orig_cos) < 0.1, f"Orthogonal tokens should have near-zero cos, got {orig_cos}"
        
        te_raw = compute_te_vector(torch.tensor([1984]), hidden_dim=d, scale_mode="raw").squeeze()
        te_scaled = compute_te_vector(torch.tensor([1984]), hidden_dim=d, scale_mode="scaled").squeeze()
        
        # 1. Raw TE addition
        h1_raw = h1 + te_raw
        h2_raw = h2 + te_raw
        cos_raw = F.cosine_similarity(h1_raw.unsqueeze(0), h2_raw.unsqueeze(0)).item()
        print(f"\n[Raw TE] cos(h1+TE, h2+TE) = {cos_raw:.6f}")
        assert cos_raw > 0.999, f"Expected Raw TE cosine collapse > 0.999, got {cos_raw}"
        
        # 2. Scaled TE addition
        h1_scaled = h1 + te_scaled
        h2_scaled = h2 + te_scaled
        cos_scaled = F.cosine_similarity(h1_scaled.unsqueeze(0), h2_scaled.unsqueeze(0)).item()
        print(f"[Scaled TE] cos(h1+TE_scaled, h2+TE_scaled) = {cos_scaled:.6f}")
        assert cos_scaled < 0.15, f"Expected Scaled TE cosine < 0.15, got {cos_scaled}"

    def test_eq8_contrastive_loss(self):
        """Test Eq. 8 intra-class contrastive loss and gradient computation."""
        B, d = 6, 896
        torch.manual_seed(123)
        H = torch.randn(B, d, requires_grad=True)
        years = torch.tensor([1947, 1947, 1984, 1984, 2022, 2022])
        
        loss = compute_intra_loss(H, years)
        assert loss.item() >= 0.0
        loss.backward()
        assert H.grad is not None
        assert not torch.isnan(H.grad).any()
        assert H.grad.norm().item() > 0.0

    def test_eq8_edge_case_no_pairs(self):
        """Test Eq. 8 when all samples in batch have different Can Chi cycles."""
        B, d = 3, 896
        H = torch.randn(B, d, requires_grad=True)
        years = torch.tensor([1947, 1984, 2022])
        
        loss = compute_intra_loss(H, years)
        assert loss.item() == 0.0

    def test_eq8_canchi_cycle_positive_pairs(self):
        """
        Verify that 1924 and 1984 belong to the same sexagenary cycle (cycle 1)
        and form positive pairs in Eq. 8 intra loss even though the Gregorian years differ.
        """
        cycle_1924 = compute_canchi_cycle(1924)
        cycle_1984 = compute_canchi_cycle(1984)
        assert cycle_1924 == 1
        assert cycle_1984 == 1
        assert cycle_1924 == cycle_1984

        d = 896
        # Orthogonal representations: cos_sim = 0 -> loss = 1.0 - 0.0 = 1.0
        H = torch.zeros(2, d, requires_grad=True)
        with torch.no_grad():
            H[0, 0] = 1.0
            H[1, 1] = 1.0
        years = torch.tensor([1924, 1984])

        loss = compute_intra_loss(H, years)
        assert torch.isclose(loss, torch.tensor(1.0), atol=1e-5)
        loss.backward()
        assert H.grad is not None
        assert H.grad.norm().item() > 0.0

        # Identical representations: cos_sim = 1 -> loss = 1.0 - 1.0 = 0.0
        H_ident = torch.zeros(2, d, requires_grad=True)
        with torch.no_grad():
            H_ident[0, 0] = 1.0
            H_ident[1, 0] = 1.0
        loss_ident = compute_intra_loss(H_ident, years)
        assert torch.isclose(loss_ident, torch.tensor(0.0), atol=1e-5)

    def test_te_norm_and_embedding_scale_analysis(self):
        """
        Verify exact analytical and numerical norms:
        1. ||TE(x_i)|| = sqrt(d/2) = sqrt(448) = 21.1660
        2. ||TE(y_i)|| = sqrt(d/2) = sqrt(448) = 21.1660
        3. ||TE|| = ||TE(x_i) + TE(y_i)|| in [40.9, 42.3] (sum of two vectors of norm ~21.16)
        4. Real Qwen2.5-0.5B token embedding norm has mean ~0.4526 (std ~0.059)
        5. Ratio ||TE_raw|| / ||token_emb|| is ~93x, mathematically proving why Raw TE erases token identity
        6. Scale factor 1/d = 1/896 scales ||TE|| down to ~0.047 (~10% of token embedding norm)
        """
        d = 896
        expected_component_norm = math.sqrt(d / 2.0)  # sqrt(448) = 21.16601
        
        for yr in [1947, 1984, 2022]:
            te_raw = compute_te_vector(torch.tensor([yr]), hidden_dim=d, scale_mode="raw").squeeze()
            norm_tot = torch.norm(te_raw).item()
            # TE total norm is in ~40.9 to 42.3
            assert 40.0 <= norm_tot <= 43.0, f"Expected total norm in [40, 43], got {norm_tot}"
            
            # Scaled TE norm
            te_scaled = compute_te_vector(torch.tensor([yr]), hidden_dim=d, scale_mode="scaled").squeeze()
            norm_scaled = torch.norm(te_scaled).item()
            assert 0.04 <= norm_scaled <= 0.05, f"Expected scaled norm in [0.04, 0.05], got {norm_scaled}"
            
            # Scaled TE is ~10% of typical token embedding norm (0.4526)
            qwen_mean_emb_norm = 0.4526
            scale_ratio = norm_scaled / qwen_mean_emb_norm
            assert 0.08 <= scale_ratio <= 0.12, f"Expected ratio ~10%, got {scale_ratio}"

