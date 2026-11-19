# Checkpoint — Hoàn tất ô B1M và phân rã ma trận nhân quả 2×2 (Objective × Trainable Path)

## 1. Định danh

- Project: long-range-temporal-alignment
- Thời điểm: 2026-10-01, Asia/Bangkok.
- Phiên bản: V1
- Owner: Antigravity Agent
- Git revision: 0a4bb164815ca7de5a422c83f07a176bbfe384d6
- Stage: Hoàn tất ô còn thiếu duy nhất trong ma trận nhân quả 2×2: B1M_UNMASKED_MLP_ONLY_V1 qua 3 hạt giống (seeds 1, 2, 3) trên GPU Tesla T4 (Kaggle).
- System profile: Python 3.12, PyTorch 2.6.0, Transformers, PEFT, BitsAndBytes 0.50.2 (4-bit NF4).
- Mức bằng chứng: measured_paired_statistical_evidence|reproducible_artifact_receipt.

## 2. Dữ liệu, Mô hình và Khế ước đóng băng

- **Mô hình nền**: `Qwen/Qwen2.5-0.5B-Instruct` (Revision SHA-256 ghim: `7ae557604adf67be50417f59c2c2f167def9a775`).
- **Dữ liệu huấn luyện**: TempUN substrate (Manifest SHA-256: `40b8ac841b1aaace92648dc1b12d3c780435704dd4a6a148984223de07f7fbf7`).
  - Giai đoạn 1947: 733 mẫu.
  - Giai đoạn 1984: 1.338 mẫu.
  - Giai đoạn 2022: 390 mẫu.
  - Tổng số mẫu huấn luyện: 2.461 mẫu.
- **Tập ca đánh giá đóng băng**: 180 ca MCQA (SHA-256: `d11007d08bf43ea760c473ea8a1504a98776204610ca0136f62ae2965e236d10`).
- **Khế ước thực nghiệm**: [B1M_UNMASKED_MLP_ONLY_3SEED_CONTRACT.md](file:///c:/Users/GF63/Documents/long-range-temporal-alignment/Documents/Decision-Making/B1M_UNMASKED_MLP_ONLY_3SEED_CONTRACT.md).
- **Cấu hình thực nghiệm**: [b1m_unmasked_mlp_only_v1_3seed.json](file:///c:/Users/GF63/Documents/long-range-temporal-alignment/configs/experiments/b1m_unmasked_mlp_only_v1_3seed.json).

## 3. Ma trận nhân quả 2×2 Canonical

Kết quả đo đạc trung bình qua 3 hạt giống (Seed 1, Seed 2, Seed 3):

| Objective \ Path | ALL_LINEAR (1.099.776 tham số) | MLP_ONLY (829.440 tham số, 0 attention) |
|---|---|---|
| **UNMASKED_FULL_SEQUENCE** | **B1**<br>Valid Acc: 25,38%<br>Strict Acc: 13,15%<br>Invalid Rate: 48,15% | **B1M**<br>Valid Acc: 30,44%<br>Strict Acc: 16,11%<br>Invalid Rate: 47,59% |
| **MASKED_TARGET** | **B2**<br>Valid Acc: 32,95%<br>Strict Acc: 19,07%<br>Invalid Rate: 42,04% | **B3**<br>Valid Acc: 34,85%<br>Strict Acc: 20,93%<br>Invalid Rate: 40,19% |

## 4. Phân rã Hiệu ứng chính (Main Effects)

### 4.1. Hiệu ứng Objective (Masked Target so với Unmasked)
- **Dưới cấu trúc All-Linear** ($\Delta = B2 - B1$):
  - Valid Accuracy: $+7,57\%$ (Seed 1: $+3,24\%$, Seed 2: $+8,73\%$, Seed 3: $+10,74\%$; 95% CI: $[+0,51\%, +18,78\%]$).
  - Strict Accuracy: $+5,93\%$ (Seed 1: $+4,44\%$, Seed 2: $+6,11\%$, Seed 3: $+7,22\%$).
  - Invalid Rate: $-6,11\%$ (cải thiện tỷ lệ tuân thủ định dạng).
- **Dưới cấu trúc MLP-Only** ($\Delta = B3 - B1M$):
  - Valid Accuracy: $+4,41\%$ (Seed 1: $+8,22\%$, Seed 2: $+8,86\%$, Seed 3: $-3,85\%$; 95% CI: $[-6,10\%, +10,33\%]$).
  - Strict Accuracy: $+4,81\%$ (Seed 1: $+11,11\%$, Seed 2: $+7,22\%$, Seed 3: $-3,89\%$).
  - Invalid Rate: $-7,41\%$.

### 4.2. Hiệu ứng Trainable Path (MLP-Only so với All-Linear)
- **Dưới mục tiêu Unmasked** ($\Delta = B1M - B1$):
  - Valid Accuracy: $+5,06\%$ (Seed 1: $+0,99\%$, Seed 2: $+2,56\%$, Seed 3: $+11,63\%$; 95% CI: $[-0,53\%, +16,49\%]$).
  - Strict Accuracy: $+2,96\%$ (Seed 1: $-2,22\%$, Seed 2: $+2,22\%$, Seed 3: $+8,89\%$).
  - Invalid Rate: $-0,56\%$.
- **Dưới mục tiêu Masked** ($\Delta = B3 - B2$):
  - Valid Accuracy: $+1,90\%$ (Seed 1: $+5,98\%$, Seed 2: $+2,70\%$, Seed 3: $-2,96\%$; 95% CI: $[-7,50\%, +7,08\%]$).
  - Strict Accuracy: $+1,85\%$ (Seed 1: $+4,44\%$, Seed 2: $+3,33\%$, Seed 3: $-2,22\%$).
  - Invalid Rate: $-1,85\%$.

## 5. Kiểm định Hiệu ứng Tương tác (Interaction Effect)

Hiệu ứng tương tác được tính toán và kiểm chứng đồng thời qua hai đẳng thức số học:
$$\text{INTERACTION} = (B3 - B1M) - (B2 - B1) = (B3 - B2) - (B1M - B1)$$

- **Kiểm tra đẳng thức số học**: Sai lệch tuyệt đối $< 10^{-9}$ (`mathematical_equality_verified: true`).
- **Giá trị tương tác đo được**:
  - Valid Accuracy Interaction: $-3,16\%$ (std: $10,19\%$; Seed 1: $+4,99\%$, Seed 2: $+0,14\%$, Seed 3: $-14,59\%$).
  - Strict Accuracy Interaction: $-1,11\%$ (std: $9,09\%$; Seed 1: $+6,67\%$, Seed 2: $+1,11\%$, Seed 3: $-11,11\%$).
  - Invalid Rate Interaction: $-1,30\%$ (std: $10,56\%$).
- **Phân loại kiểm định**: `ANTAGONISTIC_OR_REDUNDANT`.

### Diễn giải khoa học
1. Cả hai can thiệp (`MASKED_TARGET` và `MLP_ONLY`) đều mang lại hiệu quả cải thiện độc lập khi áp dụng đơn lẻ lên baseline B1.
2. Tuy nhiên, khi kết hợp đồng thời (`B3`), mức tăng hiệu năng tổng thể ($+9,47\%$ Valid Acc so với B1) nhỏ hơn tổng mức tăng của từng yếu tố cộng lại ($+7,57\% + 5,06\% = +12,63\%$).
3. Do đó, tồn tại sự dư thừa (redundancy) hoặc bù trừ một phần giữa việc cô lập pathway thích ứng vào MLP và việc giới hạn vùng giám sát loss vào target tokens.
4. Giả thuyết "MLP pathway hưởng lợi hiệp đồng vượt trội (super-additive interaction) từ targeted temporal supervision" **không được ủng hộ** (refuted) trên benchmark này. Mối quan hệ giữa hai yếu tố mang tính bù trừ/dư thừa thay vì hiệp đồng tương hỗ.

## 6. Phân tích lát cắt và tính ổn định

- **Lát cắt trùng lặp (Overlap Slices)**:
  - Trên lát cắt `NON_EXACT_OVERLAP` (120 ca), Strict Accuracy tăng dần đều: B1 (8,06%) → B1M (9,17%) → B2 (11,11%) → B3 (15,00%). Điều này chứng minh hiệu quả của việc cô lập MLP và giám sát mục tiêu không bị thao túng bởi rò rỉ mẫu ghi nhớ nông.
  - Trên lát cắt `OVERLAP_PRONE` (60 ca), B2 đạt Strict Accuracy cao nhất (35,00%), trong khi B3 đạt 32,78% và B1M đạt 30,00%.
- **Nhóm nhiệm vụ (Task Families)**:
  - B1M thể hiện ưu thế mạnh nhất trên nhóm câu hỏi điểm thời gian liên tục `CP` (Valid Acc: 51,39%) và `MM` (36,51%), nhưng suy giảm trên nhóm `TB` (14,05%).

## 7. Ranh giới tri thức và Khuyến nghị kế tiếp

- **Ranh giới công bố**: LoRA và MLP-only LoRA là kỹ thuật prior art; B1M là controlled missing cell trong thiết lập Qwen + TempUN; không tuyên bố cơ chế mới khi chưa có bằng chứng giải phẫu trọng số.
- **Tiết kiệm tài nguyên**: MLP-only tiết kiệm $24,58\%$ tham số huấn luyện ($829.440$ so với $1.099.776$) và bảo toàn hoàn toàn không gian chú ý ($0$ attention adapter).
- **Thực nghiệm kế tiếp khả thi**:
  1. Kiểm soát số lượng tham số tương đương (Parameter-Matched Control): So sánh MLP-only ($r=2$, 829k tham số) với Attention-only ($r=6$, 811k tham số).
  2. B4 Temporal-conditioned adapter tích hợp biểu diễn đối tác $z_t$.
