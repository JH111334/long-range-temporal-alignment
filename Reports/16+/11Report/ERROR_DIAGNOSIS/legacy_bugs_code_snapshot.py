"""STATIC ARCHIVAL CODE SNAPSHOT — UNIFIED REFERENCE FRAME (DOES NOT EXECUTE)

Tách bạch rõ 2 hệ quy chiếu trong `11Report/`:
- Frame 1 (Eval Replay V2 trong 11Report): Lỗi `sigmoid(alpha)` thuộc code Eval V1 cũ
  (đã được sửa về `linear` trong V2 và chứng nhận ở `02_LINEAR_GATE_PARITY_RECEIPT.json`).
- Frame 2 (Huấn luyện 5 Epochs: Baseline 5ep cũ vs V3 5ep mới):
  3 điểm lệch thực sự tồn tại trong script 5-epoch training cũ
  (`scripts/run_stage03_targeted_arms4_5_5epochs_modal.py`) được khắc phục trong V3:
  (1) `token_idx=14` -> `MULTI_YEAR_SPAN`, (2) `z` tuyến tính -> `asinh` + FoNE `powers10`,
  (3) `AdamW` hằng số `2e-4` -> `AdamW` phân nhóm + `Warmup/Cosine LR Scheduler`.
"""

# ==============================================================================
# FRAME 1 (EVAL V1 BUG — ĐÃ ĐÓNG TRONG V2, KHÔNG PHẢI CODE 5-EPOCH TRAINING)
# Source: scripts/run_stage03_targeted_replay_128_modal.py:126-130
# ==============================================================================

def frame1_legacy_eval_sigmoid_bug(self, v_time):
    h = self.linear(v_time)
    h = self.norm(h)
    gate = torch.sigmoid(self.alpha)  # alpha=-0.006561 -> +0.498360 (+75.95x, đảo dấu)
    return gate * h


# ==============================================================================
# FRAME 2 (BASELINE 5-EPOCH TRAINING CŨ — HỆ QUY CHIẾU SO SÁNH VỚI V3)
# Source: scripts/run_stage03_targeted_arms4_5_5epochs_modal.py
# ==============================================================================

# [2.1] Tiêm cố định tại token_idx=14 trong lúc train (dòng 160-169, 494)
def frame2_5ep_train_inject_seam_prefill_safe(model, seam_proj, input_ids, v40, token_idx: int = 14):
    embed_tokens = model.get_input_embeddings()
    inputs_embeds = embed_tokens(input_ids)
    B, S, D = inputs_embeds.shape
    if S > token_idx:
        inj = seam_proj(v40.to(inputs_embeds.device, dtype=inputs_embeds.dtype))
        delta = torch.zeros_like(inputs_embeds)
        delta[:, token_idx, :] = inj  # Trượt khỏi token năm trong 104k mẫu & bỏ sót năm thứ 2
        inputs_embeds = inputs_embeds + delta
    return inputs_embeds


# [2.2] Trục z tuyến tính, bán kính tuyến tính & FoNE dyadic (dòng 110-143)
def frame2_5ep_train_compute_theta_z(year: int):
    offset = year - 3
    cycle = offset % 60
    theta = float(cycle) * 6.0 * (math.pi / 180.0)
    z = float(year - 1947) / 75.0  # Không nén phi tuyến (BCE -500 -> z = -32.63)
    return theta, z


def frame2_5ep_train_canchi_radius(year_int: int) -> float:
    return 1.0 + float(year_int - 2000) / 60.0  # Tuyến tính không bị chặn


# [2.3] AdamW hằng số 2e-4 không LR Scheduler (dòng 461, 515)
# optimizer = torch.optim.AdamW(trainable_params, lr=2e-4, weight_decay=0.01)
# (Không có Warmup, không có Cosine/WSD Decay; gộp chung LR cho LoRA và Seam/Core)
