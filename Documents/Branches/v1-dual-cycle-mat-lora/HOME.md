---
loại: "tổng-quan"
dự_án: "long-range-temporal-alignment"
trạng_thái: "đang-dùng"
chuẩn_hiện_hành: true
cập_nhật_lần_cuối: "2026-10-13"
---

# Nhánh V1: Điểm Xuất Phát Chuẩn Tắc (Periodic Dual-Cycle MaT-LoRA & Targeted Continual Pretraining)

Nhánh V1 đóng vai trò là mốc đối chứng nền tảng (Starting Anchor) của nhánh Cơ chế học tập (Learning Track) trong đề tài `Long-Range-Representation`.

## 1. Cấu hình kiến trúc và biểu diễn

- **Mô hình nền (Backbone)**: `Qwen/Qwen2.5-0.5B-Instruct` tại commit `7ae557604adf67be50417f59c2c2f167def9a775` (đóng băng $100\%$ trọng số $W_0$, không mở rộng dung lượng).
- **Bộ thích ứng (Adapter)**: `Periodic Dual-Cycle MaT-LoRA` ($r=8, \alpha=16.0, K=2$ sóng hài, $F_{\text{base}} = I_8$) can thiệp trên toàn bộ 72 module MLP (`gate_proj`, `up_proj`, `down_proj`).
- **Khế ước biểu diễn**: `FIXED_FONE_CANCHI_V1` gồm FoNE 32D ghép nối Can-Chi 8D thành vector 40D cố định (mã băm golden vectors: `6c35fc55cbdfd571c03cd6bf56d3d4b65b367730b8bb3bc7d89c9dc82f78eddc`).
- **Giao diện điều kiện (Safe Seam)**: Phép chiếu số dư liên tục qua cổng $\alpha$ (khởi tạo tại $0.0$) tại vị trí token 14 prefill, bảo toàn tính toàn vẹn cú pháp ChatML.
- **Hàm mất mát huấn luyện**: `MASKED_TARGET` Causal Cross-Entropy (chỉ tính loss trên phần target sinh ra và thẻ kết thúc EOS).
- **Tập dữ liệu**: TempUN Small ($2.461$ mẫu huấn luyện qua 3 chặng: `1947` [733], `1984` [1338], `2022` [390]).
- **Khế ước đánh giá**: 180 câu hỏi trắc nghiệm MCQA đóng băng đo theo `GENERATIVE_STRICT_PARSE_V1`.

## 2. Bằng chứng kiểm định và kết quả đo đạc thực tế

### A. Kiểm toán pháp y số học Canary (1 seed, 45 steps)
- Khắc phục thành công lỗi NaN của phiên bản trước nhờ: khởi tạo $B=0$ bảo toàn parity, giữ tham số khả huấn ở FP32, tính loss FP32 và loại bỏ `GradScaler`.
- Kết quả Canary: 45/45 bước hữu hạn, loss giảm đều ($4.20 \to 1.73 \to 2.43$), 3 checkpoints đạt $3.387.649/3.387.649$ phần tử hữu hạn ($0$ NaN/Inf), Strict Accuracy $13.33\% - 20.00\%$, Invalid Rate $20.00\%$. Minh chứng tại `artifacts/diagnostics/M0_FC40/CANARY_FORENSIC_VERDICT.json`.

### B. Huấn luyện toàn phần 3 Seed ($7.383$ steps)
- Thực thi hoàn tất $100\%$ không lỗi trên GPU Tesla T4 (`qh0205/lrr-fc105-m0-fc40-3seed`).
- Hàm mất mát hội tụ ổn định: Chặng 1 avg loss $1.866$, Chặng 2 avg loss $1.723$, Chặng 3 avg loss $1.741$.
- Hiệu năng sinh suy luận (Generative Strict MCQA): Strict Accuracy đạt $2.22\% \pm 0.00\%$, Tỷ lệ lỗi cú pháp (Invalid Syntax Rate) đạt $93.33\% \pm 0.00\%$, Chuyển giao ngược BWT đạt $-3.125\%$.

## 3. Bài học khoa học cốt lõi từ Nhánh V1

1. **Hiện tượng đánh đổi căn chỉnh (Alignment Tax)**: Khi mô hình Instruct được huấn luyện liên tục với hàm mất mát Next-Word Prediction thông thường trên văn bản dài, trọng số adapter bị trôi dạt về phân phối ngôn ngữ mở, làm mất khả năng tuân thủ cấu trúc sinh đáp án một ký tự của bài toán trắc nghiệm.
2. **Tối ưu cục bộ khác tối ưu toàn cục**: Loss đào tạo giảm đều đặn và hội tụ không đồng nghĩa với việc năng lực suy luận thời gian được cải thiện ở đầu ra cuối cùng.
3. **Quyết định chuyển giao**: Nhánh V1 được đóng băng hoàn toàn làm mốc xuất phát chuẩn tắc (Starting Baseline). Mọi can thiệp cải tiến tiếp theo nhằm giải quyết hiện tượng suy giảm cú pháp và chống quên tri thức được chuyển sang **Nhánh V2**.

## 4. Liên kết điều hướng

- [[Documents/Branches/v1-dual-cycle-mat-lora/Current-State|Trạng thái chi tiết V1]]
- [[Documents/Branches/v2-targeted-continual-alignment/HOME|Nhánh V2: Working Branch hiện hành]]
- [[Documents/Decision-Making/FIXED_FONE_CANCHI_LEARNING_TRACK_105|Quyết định nền tảng FCLEARN_105]]
- [[Documents/Documenting/Experiment/LEARNING_TRACK_EXPERIMENT_HISTORY|Lịch sử thực nghiệm]]
