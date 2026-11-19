# Phán Quyết Cổng Kiểm Định G1: Tính Tương Đương Tuyệt Đối Của Mô Hình Nền (Base Equivalence Verdict)

## 1. Mục Tiêu Kiểm Định
Xác minh rằng việc bọc 72 module MLP bằng `DualCycleMaTLoRALinear` và gắn hook `make_gated_embedding_hook` không làm biến tính mô hình nền Qwen2.5-0.5B-Instruct khi tắt các cơ chế thích ứng (`BOTH_OFF`).

## 2. Kết Quả Số Học Định Lượng
- **Số lượng prompt canary kiểm tra**: 8 mẫu đa dạng (bao gồm MCQA, Relation, NTP).
- **Sai lệch tuyệt đối lớn nhất ($\\max |\\text{logits}_{\\text{raw}} - \\text{logits}_{\\text{both\\_off}}|$)**: `0.00000000e+00` (ngưỡng chấp thuận: $< 1\\times 10^{-4}$).
- **Cosine similarity trên token cuối cùng**: `1.00000274` trên tất cả các prompt.
- **Top-1 token ID**: Trùng khớp 100% giữa mô hình gốc và mô hình bọc adapter khi tắt can thiệp.

## 3. Kiểm Toán Đóng Băng Trọng Số Nền
- Tổng số tham số `base_layer.weight` có `requires_grad = True`: **0** (khóa cứng 100%).
- Cấu trúc checkpoint cha $C_1$ ([`C1_COMPACT_ADAPTER_ONLY.pt`](file:///artifacts/experiments/V2_CAUSAL_TEMPORAL_ALIGNMENT/run_02/checkpoints/C1_COMPACT_ADAPTER_ONLY.pt)): Dung lượng 13.71 MB, chứa đúng 508 tensor adapters, hoàn toàn không chứa khóa `base_layer`.

## 4. Phán Quyết
**ĐẠT CHUẨN (PASS)**: Tính tương đương số học giữa mô hình bọc và mô hình nền nguyên bản được xác lập vững chắc. Cổng G1 chính thức được phê duyệt để tiến hành sang Cổng G2 (ChatML Target, NTP Shift và EOS Stop).
