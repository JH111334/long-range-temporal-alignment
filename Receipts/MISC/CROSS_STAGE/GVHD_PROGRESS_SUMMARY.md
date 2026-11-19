# Báo Cáo Tiến Độ Nghiên Cứu Cô Đọng Dành Cho GVHD (Executive Summary)

**Dự án:** Nghiên cứu Biểu diễn và Căn chỉnh Thời gian Dài hạn trong Mô hình Ngôn ngữ (Long-Range Temporal Alignment)  
**Phân nhánh:** Cơ chế học tập và thích ứng tham số (Learning Track)  
**Ngày cập nhật:** 16/10/2026 (Phiên bản đã kiểm toán pháp y toàn diện)  
**Chủ trì thực hiện:** Nhóm nghiên cứu Learning Track  
**Mô hình cơ sở:** `Qwen/Qwen2.5-0.5B-Instruct`

---

## 1. Bản Đồ Hiện Trạng 3 Stage: VERIFIED vs DIAGNOSTIC vs UNKNOWN

| Phân tầng nghiên cứu | Mức độ xác thực (Evidence Grade) | Kết quả đo lường chuẩn tắc | Phán quyết khoa học & Trạng thái |
|---|---|---|---|
| **STAGE 01 — MLP-Only LoRA** | **`MEASURED_VERIFIED`** | Bảng $2 \times 2$ factorial chuẩn: $B_1$ ($13.15\%$), $B_{1M}$ ($16.11\%$), $B_2$ ($19.07\%$), $B_3$ ($20.93\%$). Giảm $24.58\%$ tham số trainable ($1,099,776 \to 829,440$). Tương tác âm nhẹ ($-1.11\text{ pp}$). FoNE McNemar đạt $p < 0.05$ trên $2/3$ seeds. $B_3, R_{00}, C_0$ là các alias của cùng Anchor Config. | **`NO_RETRAIN`** (Bảo toàn trọng số lịch sử; loại bỏ toàn bộ số liệu nhầm lẫn $35\%/39\%/31\%$). |
| **STAGE 02 — Dual-Cycle MaT-LoRA** | **`MEASURED_VERIFIED`**<br>(Full N180 đã xác thực) | **Phân giải dứt điểm N24 vs N180:**<br>- Điểm số $37.50\% \pm 4.17\%$ là kết quả chẩn đoán trên **$n=24$ mẫu** ($10/24, 9/24, 8/24$).<br>- Kết quả đo đạc thực tế trên **Full 180 câu TempUN** với bộ nạp chuẩn tắc: **$29.81\% \pm 2.80\%$** ($49/180, 59/180, 53/180$) so với Base Model $32.22\%$ ($\Delta = -2.41\text{ pp}$, McNemar $p > 0.05$, không có ý nghĩa thống kê). Invalid Rate $= 0.00\%$. | **`NO_RETRAIN`**<br>(`N180_P12_CONFIRMED`; bóc tách lỗi loader bypass cũ). |
| **STAGE 03 — Causal Multitask Objectives** | **`ACTIVE_STAGE`**<br>(Chẩn đoán hoàn tất) | - DEV Relation ($n=482$): Arm B đạt **$62.03\%$**, Arm A đạt $50.0\%$.<br>- DEV MCQA Clean ($n=120$): Arm B đạt **$50.83\%$**.<br>- TempUN 180: Arm B đạt $24.44\%$ ($22.50\%$ trên lát cắt sạch 120 câu).<br>- **Proof of Treatment (36 mẫu):** $\Delta_{\text{logits}} = \mathbf{0.118655}$, $4/36$ flips (xác thực can thiệp thời gian hoạt động).<br>- **Rotated-Option MCQA (624 biến thể):** Acc gộp $24.20\%$, thiên lệch cực đoan nhãn `b` ($78.2\%$), Semantic Consistency Rate đạt đúng **$0 / 156$ ($0.00\%$)**.<br>- **Counterfactuals (5 cặp):** Đoán $100\%$ nhãn 'No' ($0/5$ flips). | **`NO-GO (HALT)`**<br>(Đình chỉ full GPU retraining; bắt buộc triển khai Online Choice Permutation). |

---

## 2. Các Bài Học Pháp Y và Khắc Phục Sai Lệch Trọng Yếu

1. **Thu hồi số liệu không có căn cứ ở Stage 01:** Đã loại bỏ các con số suy diễn "B2 38.89% / 0% invalid" (thực chất là invalid rate của baseline F0) và xác lập lại bảng $2 \times 2$ factorial chuẩn tắc qua 3 hạt giống từ tệp gốc `01_factorial_2x2_results.json`.
2. **Khắc phục lỗi ngộ nhận N24 thành N180 ở Stage 02:** Đã phân định rạch ròi tập chẩn đoán $n=24$ ($37.50\% \pm 4.17\%$) và phục hồi kết quả đo đạc thực tế trên toàn bộ 180 câu TempUN Small ($29.81\% \pm 2.80\%$). Bóc tách lỗi loader cũ (tìm sai key `adapted_modules` thay vì `adapters_state`) khiến script cũ chạy nhầm mô hình nền.
3. **Phá vỡ nghi vấn pseudo-immunity $\Delta=0.0$ ở Stage 03:** Chứng minh hiện tượng nhãn không đổi ở Run 05.2 là do caller cũ chưa thực hiện phép cộng năm. Khi sửa caller, biến thiên logit đạt $0.1187$, khẳng định vector thời gian có tác động thực chất lên không gian biểu diễn.
4. **Bóc trần sự sụp đổ nhãn và shortcut chữ cái qua phép kiểm hoán vị:** Phép kiểm xoay vòng 4 phương án trắc nghiệm trên 624 biến thể bộc lộ tỷ lệ nhất quán ngữ nghĩa bằng đúng $0.00\%$ ($0/156$). Mô hình hiện tại chọn đáp án dựa trên vị trí ký tự chữ cái thay vì suy luận ngữ nghĩa độc lập.

---

## 3. Kế Hoạch Khoa Học và Rào Cản Quyết Định (Next Steps)

- **Cổng Go/No-Go:** **`HALT FULL GPU TRAINING`**. Nhóm nghiên cứu không khởi chạy bất kỳ lượt huấn luyện GPU lớn nào khi bài toán sụp đổ nhãn chữ cái chưa được giải quyết.
- **Can thiệp tối thiểu bắt buộc:** Thiết kế cơ chế **Hoán vị phương án ngẫu nhiên trong quá trình huấn luyện (Online Choice Permutation)** và điều hòa phân bố nhãn (Label Smoothing) nhằm triệt tiêu hoàn toàn shortcut vị trí chữ cái trước khi tái đánh giá năng lực suy luận thời gian.
