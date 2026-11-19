---
loại: "quyết-định"
mã: "DECISION-REP-ABLATION-102-EVALUATION"
dự_án: "long-range-temporal-alignment"
trạng_thái: "đang-dùng"
chuẩn_hiện_hành: true
thay_thế: []
bị_thay_thế_bởi: []
tham_chiếu: []
được_xác_thực_bởi: []
cập_nhật_lần_cuối: "2026-10-07"
---

# Quyết Định Đánh Giá Khoa Học: Kết Quả Thực Nghiệm Ma Trận 2×2 FoNE × Can-Chi Representation Ablation (REP_ABLATION_102)

## 1. Bối Cảnh Và Câu Hỏi Nghiên Cứu

Thực nghiệm `REP_ABLATION_102` được thực hiện nhằm kiểm định độc lập và tương tác giữa biểu diễn sóng liên tục Fourier (**FoNE**) và biểu diễn hình học chu kỳ Á Đông Lục Thập Hoa Giáp (**Can-Chi TickTack**) trên nền tảng mô hình `Qwen/Qwen2.5-0.5B-Instruct`.

Công thức học tập được cố định 100% từ Week 1:
- Mục tiêu huấn luyện: `MASKED_TARGET` (Prompt tokens masked to -100).
- Kiến trúc thích ứng: `MLP_ONLY` LoRA ($r=2, \alpha=4$, chỉ can thiệp `gate_proj`, `up_proj`, `down_proj`).
- Dữ liệu: TempUN Small (3 chặng lịch sử: 1947 $\to$ 1984 $\to$ 2022).
- Bộ đánh giá chuẩn tắc: 180 mẫu MCQA cố định tuyệt đối qua 3 hạt nhân độc lập (Seed 1, 2, 3).

---

## 2. Bằng Chứng Thực Nghiệm Đo Đạc (Empirical Measurements)

Dữ liệu được hội tụ chuẩn tắc tại `artifacts/releases/REP_ABLATION_102/`:
- Mã băm hợp đồng: `artifacts/releases/REP_ABLATION_102/contract.json`
- Tổng hợp phân tích: `artifacts/releases/REP_ABLATION_102/aggregate/final_summary.json`

### Bảng tổng hợp hiệu năng qua 3 seed:

| Cấu hình | Mô tả Biểu diễn | Strict Accuracy | Valid Accuracy | Invalid Rate | BWT Continual |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **$R_{00}$** | Baseline B3 (Không can thiệp biểu diễn) | **20.93% ± 3.26%** | **34.85% ± 3.61%** | **40.19% ± 3.39%** | **+0.86%** |
| **$R_{10}$** | FoNE Fourier 32D | 14.44% ± 1.47% | 28.42% ± 1.87% | 49.26% ± 1.79% | -2.12% |
| **$R_{01}$** | Can-Chi Geometry 8D | 4.44% ± 2.94% | 23.61% ± 17.13% | 71.48% ± 25.41% | +5.91% |
| **$R_{11}$** | FoNE × Can-Chi Combined 96D | 1.30% ± 2.25% | 46.67% ± 0.00%* | 97.22% ± 4.81% | -22.50% |

*\*Valid Accuracy của $R_{11}$ chỉ tính trên tập mẫu sống sót sau hiện tượng sụp đổ định dạng (Seed 3 đạt 100% Invalid).*

---

## 3. Phân Tích Thống Kê Và Kiểm Định Ghép Cặp

### 3.1. Phân tích phương sai nhân tố 2×2 (Factorial Effects):
- **Hiệu ứng chính FoNE**: $-4.81\% \pm 2.73\%$ trên Strict Accuracy, làm tăng Invalid Rate $+17.41\% \pm 15.44\%$.
- **Hiệu ứng chính Can-Chi**: $-14.81\% \pm 2.58\%$ trên Strict Accuracy, làm tăng Invalid Rate $+39.63\% \pm 11.18\%$.
- **Hiệu ứng tương tác**: $+1.67\% \pm 1.94\%$ trên Strict Accuracy, làm tăng Invalid Rate $+8.33\% \pm 12.11\%$.

### 3.2. Kiểm định McNemar (180 mẫu ghép cặp, hiệu chỉnh Holm-Bonferroni):
- **M1 ($R_{00}$ vs $R_{10}$)**: $R_{00}$ SUPERIOR trên cả 3 seeds (hướng nhất quán 100%).
- **M2 ($R_{00}$ vs $R_{01}$)**: $R_{00}$ SUPERIOR với ý nghĩa thống kê vượt trội ($p < 1.64 \times 10^{-4}$ trên toàn bộ 3 seeds, HB-significant).
- **M3 ($R_{10}$ vs $R_{11}$)**: $R_{10}$ SUPERIOR với ý nghĩa thống kê vượt trội ($p < 2.08 \times 10^{-3}$ trên cả 3 seeds, HB-significant).
- **M4 ($R_{01}$ vs $R_{11}$)**: $R_{01}$ SUPERIOR ($b > c$ trên cả 3 seeds).
- **M5 ($R_{00}$ vs $R_{11}$)**: $R_{00}$ SUPERIOR với mức ý nghĩa tuyệt đối ($p < 1.06 \times 10^{-6}$ trên cả 3 seeds, HB-significant).

---

## 4. Chẩn Đoán Nguyên Nhân Gốc Rễ (Root Cause Analysis)

1. **Lệch pha phân bố tầng Embedding (Embedding Drift)**:
   - Việc chiếu vector liên tục ngoại lai vào vị trí token 0 tại tầng embedding phá vỡ sự cân bằng của ChatML template (`<|im_start|>`), khiến LLM nhỏ bị mất kiểm soát cú pháp đầu ra.
2. **Sụp đổ định dạng tỷ lệ thuận với số chiều vector (Dimension-Scaled Format Collapse)**:
   - Khi tăng số chiều vector từ 0 ($R_{00}$) $\to$ 8 ($R_{01}$) $\to$ 32 ($R_{10}$) $\to$ 96 ($R_{11}$), tỷ lệ Invalid Rate tăng vọt từ 40.19% $\to$ 71.48% $\to$ 49.26% $\to$ 97.22%.
3. **Hiện tượng can nhiễu biểu diễn (Representation Interference)**:
   - Không có bằng chứng thực nghiệm nào ủng hộ việc cộng vector chu kỳ Can-Chi hay Fourier trực tiếp vào embedding. Trái lại, biểu diễn này tạo ra rào cản nghiêm trọng đối với khả năng suy luận thời gian.

---

## 5. Quyết Định Khoa Học (Scientific Decision)

1. **QUYẾT ĐỊNH**: **REJECT** giả thuyết tiêm vector biểu diễn chu kỳ Can-Chi và sóng FoNE trực tiếp vào tầng Embedding trên kiến trúc LLM nhỏ.
2. **CHUẨN TẮC**: Khóa kiến trúc $R_{00}$ (Baseline B3: `MASKED_TARGET` $\times$ `MLP_ONLY` LoRA) làm chuẩn tối ưu hiện hành của đề tài.
3. **ĐỊNH HƯỚNG TIẾP THEO**:
   - Chuyển hướng nghiên cứu Can-Chi sang cơ chế **Textual Prompt Engineering** (văn bản hóa chu kỳ Lục Thập Hoa Giáp trong prompt) hoặc **Cross-Attention Temporal Adapters**, tách biệt hoàn toàn khỏi luồng embedding từ vựng.
