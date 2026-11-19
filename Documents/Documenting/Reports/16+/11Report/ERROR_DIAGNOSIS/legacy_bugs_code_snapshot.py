"""STATIC ARCHIVAL CODE SNAPSHOT — LEGACY STAGE 03 SETUP (DOES NOT EXECUTE)

Preserves verbatim the 3 legacy code/config bugs documented in `ERROR_DIAGNOSIS`
prior to V3 repair (`G1` YEAR-only, `v2-targeted-continual-alignment`, MLP-only LoRA).
"""

# ==============================================================================
# BUG 1: Hardcoded `token_idx=14`, padding-mask collision, and missing 2nd year
# Source A (Train): scripts/run_stage03_targeted_arms4_5_5epochs_modal.py:160-169, 494
# Source B (Eval):  scripts/run_stage03_targeted_replay_128_modal.py:132-141, 593, 696
# ==============================================================================

def legacy_inject_seam_prefill_safe(model, seam_proj, input_ids, v40, token_idx: int = 14):
    embed_tokens = model.get_input_embeddings()
    inputs_embeds = embed_tokens(input_ids)
    B, S, D = inputs_embeds.shape
    if S > token_idx:
        inj = seam_proj(v40.to(inputs_embeds.device, dtype=inputs_embeds.dtype))
        delta = torch.zeros_like(inputs_embeds)
        delta[:, token_idx, :] = inj  # BUG 1a: fixed token index 14 hits non-year token or PAD
        inputs_embeds = inputs_embeds + delta
    return inputs_embeds


# Training call site (right-padded batch, no year-span lookup):
#   inputs_embeds = inject_seam_prefill_safe(model, seam_projection, input_ids, v40, token_idx=14)
#
# Eval call site (tokenizer.padding_side = "left", annihilates injection when pad_len >= 15):
#   tokenizer.padding_side = "left"
#   inputs_embeds = inject_seam_prefill_safe(model, seam_proj, inputs["input_ids"], v40_t, token_idx=14)
#
# V2 partial fix (`TemporalPolicy.YEAR_SPAN`) only injected `anchor_mentions[b]` (first year),
# leaving comparison years in Rel/WB/RB tasks uninjected. Replaced in V3 by `MULTI_YEAR_SPAN`.


# ==============================================================================
# BUG 2: Legacy Sigmoid Gate mismatch (~75.95x explosion + sign inversion)
# Source (Legacy Eval): scripts/run_stage03_targeted_replay_128_modal.py:119-130
# Ground Truth (Train): scripts/run_stage03_targeted_arms4_5_5epochs_modal.py:146-158
# ==============================================================================

class LegacyBrokenGatedResidualProjection:
    """Eval replay used torch.sigmoid(self.alpha) while training used linear self.alpha."""

    def forward_legacy_broken_eval(self, v_time):
        h = self.linear(v_time)
        h = self.norm(h)
        gate = torch.sigmoid(self.alpha)  # BUG 2: alpha=-0.006561 -> sigmoid=+0.498360 (+76x, flipped sign)
        return gate * h

    def forward_training_ground_truth(self, v_time):
        proj = self.linear(v_time)
        normed = self.norm(proj)
        return self.alpha * normed  # Fixed in V2/V3 with gate_mode="linear"


# ==============================================================================
# BUG 3: Unbounded linear z, linear Can-Chi radius, dyadic-only FoNE & constant LR
# Source: scripts/run_stage03_targeted_arms4_5_5epochs_modal.py:110-143, 461
# ==============================================================================

def legacy_compute_fone_32d(year_f: float):
    # Dyadic span [10 .. 10000]; V3 adds powers-of-10 log-spaced periods 10^(4*i/15) [1 .. 10000]
    periods = [10.0 * (1000.0 ** (i / 15.0)) for i in range(16)]
    feats = []
    for p in periods:
        omega = 2.0 * math.pi / p
        feats.append(math.sin(omega * year_f))
        feats.append(math.cos(omega * year_f))
    return feats


def legacy_compute_canchi_8d(year_int: int):
    offset = year_int - 4
    stem_idx = offset % 10
    branch_idx = offset % 12
    cycle_idx = offset % 60
    theta_stem = 2.0 * math.pi * float(stem_idx) / 10.0
    theta_branch = 2.0 * math.pi * float(branch_idx) / 12.0
    theta_cycle = 2.0 * math.pi * float(cycle_idx) / 60.0
    radius = 1.0 + float(year_int - 2000) / 60.0  # BUG 3a: unbounded linear radius (BCE -500 -> -40.67)
    return [
        math.sin(theta_stem), math.cos(theta_stem),
        math.sin(theta_branch), math.cos(theta_branch),
        math.sin(theta_cycle), math.cos(theta_cycle),
        float(radius), float(cycle_idx) / 60.0,
    ]


def legacy_compute_theta_z(year: int):
    offset = year - 3
    cycle = offset % 60
    theta = float(cycle) * 6.0 * (math.pi / 180.0)
    z = float(year - 1947) / 75.0  # BUG 3b: unbounded linear z -> (z/10)^2 = 10.64 at BCE -500
    return theta, z


# Legacy optimizer setup (scripts/run_stage03_targeted_arms4_5_5epochs_modal.py:461):
#   optimizer = torch.optim.AdamW(trainable_params, lr=2e-4, weight_decay=0.01)
#   # NO lr_scheduler (constant 2e-4 across all 4,065 optimizer steps, no warmup, no cosine decay)
