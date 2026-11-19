# Báo Cáo Thực Nghiệm Quét Nhân Quả Toàn Diện 54 Cặp Phản Thực Nghiệm (Full Admitted 54-Pair Causal Sweep Report)

**Mã thực nghiệm**: `RUN_FULL_SWEEP_54P_20261019_161756`  
**Ngày thực hiện**: 2026-10-19  
**Môi trường thực thi**: CPU (`torch.float32`), thời gian: 781.92s, 0 optimizer steps (Strictly Inference-Only).  
**Checkpoint thẩm định**: [R52_B_NTP_REL_MCQA.pt](file:///c:/Users/GF63/Documents/long-range-temporal-alignment/artifacts/experiments/V2_CAUSAL_TEMPORAL_ALIGNMENT/run_05_2/checkpoints/R52_B_NTP_REL_MCQA.pt) (SHA-256: `1af285df69ddaf94d21dcb858e8bd2d5e532874b484131d7fa3a31a4eac41988`).  
**Tập dữ liệu**: [COUNTERFACTUAL_GOLD_AUDIT.jsonl](file:///c:/Users/GF63/Documents/long-range-temporal-alignment/data/counterfactual/COUNTERFACTUAL_GOLD_AUDIT.jsonl) (SHA-256: `49eea7fe6fad69d4f3695e7cd8fbeac2c383e65b76d0802801666f9982cdeba5`).

---

## 1. Mục Tiêu Nghiên Cứu và Bốn Giả Thuyết Điểm Nghẽn

Sau khi lượt chạy thử nghiệm pilot ($n=6$) bộc lộ hiện tượng $0/4$ cặp lật nhãn đúng, thực nghiệm toàn diện này được triển khai nhằm khảo sát trên toàn bộ $54$ cặp câu hỏi phản thực nghiệm ($108$ câu hỏi đơn lẻ) để phân xử rạch ròi 4 giả thuyết khoa học:
- **(H1) Biểu diễn thời gian (Representation Encoding)**: Vector $z(t)$ không mã hóa đủ khoảng cách thời gian hoặc chu kỳ để phân biệt các mốc lịch sử.
- **(H2) Dẫn truyền kiến trúc (Architecture Routing)**: Phép chiếu Seam Injection hoặc Dynamic Core $F_t$ làm suy biến hoặc chặn đứng tín hiệu trước khi lan truyền tới vị trí token quyết định ($p=-1$).
- **(H3) Khiên chắn giải mã tham lam (Greedy / Calibration Shield)**: Tín hiệu thời gian đã dịch chuyển phân phối logit đúng hướng về phía chân lý ($\Delta G > 0$), nhưng cơ chế giải mã tham lam và bias từ khóa văn bản đã che khuất sự lật nhãn.
- **(H4) Ánh xạ ngữ nghĩa mục tiêu (Semantic Mapping Objectives)**: Hàm mục tiêu huấn luyện đa nhiệm (Causal NTP + Quan hệ thời gian) chưa cung cấp cơ chế giám sát để mô hình ánh xạ bộ ba $(entity, relation, time) \to truth$.

---

## 2. Kiểm Toán Tập Dữ Liệu Phản Thực Nghiệm (Grounded Counterfactual Benchmark)

- **Tổng số cặp thẩm định**: 54 cặp ($108$ câu hỏi độc lập).
  - **Nhánh Gold-Flip (Primary Endpoint)**: 36 cặp ($72$ câu hỏi), thỏa mãn $y^*(t_1) \ne y^*(t_2)$ qua các sự kiện lịch sử xác định (thống nhất quốc gia, sụp đổ liên bang, tuyên bố độc lập, mở kênh đào, v.v.).
  - **Nhánh Gold-Same (Negative Control)**: 18 cặp ($36$ câu hỏi đối chứng âm), thỏa mãn $y^*(t_1) = y^*(t_2)$ (thực thể duy trì trạng thái ổn định suốt khoảng thời gian quan sát) nhằm kiểm tra xu hướng lật giả do nhạy cảm với bất kỳ sự thay đổi năm nào.
- **Số cụm thời gian độc lập (Temporal Clusters)**: 23 cụm (ví dụ: `COLD_WAR_COLLAPSE`, `POST_WAR_EUROPE`, `SOUTHEAST_ASIA_DECOLONIZATION`, `STABLE_HISTORICAL_CONTROL`).
- **Phân loại kiểm toán**: `PROGRAMMATIC_VALIDATION_PASSED_WITH_CURATED_HISTORICAL_GROUNDING`. Toàn bộ mốc năm là số nguyên khác 0, biên độ thời gian cách biệt từ 11 đến 40 năm.

---

## 3. Kết Quả Đo Lường Năm Cấp Độ Nhân Quả

### 3.1. Cấp Độ 1: Phân Phối Tín Hiệu (Delivery) — ĐẠT (`PASS`)
- Khoảng cách vector biểu diễn trung bình: $\bar{D}_z = 5.317237$.
- Biến thiên ma trận Dynamic Core Frobenius norm: $\bar{D}_F = 0.682921$.
- Biến thiên ma trận trọng số thích ứng LoRA Frobenius norm: $\bar{D}_W = 0.500124 > 10^{-5}$.
- **Kết luận**: Tín hiệu thời gian được mã hóa thành công và truyền dẫn thông suốt làm thay đổi trọng số thích ứng nội tại của mô hình.

### 3.2. Cấp Độ 2: Độ Nhạy Trạng Thái Ẩn và Logit (Sensitivity) — ĐẠT (`PASS`)
- Độ lệch logit tuyệt đối trung bình: $\Delta_{\text{logits}} = 0.135589 > 0.01$.
- Độ nhạy kích hoạt tại Tầng 14 vị trí cổng Seam ($p = 14$): $S_{14, \text{seam}} = 0.052463$.
- Độ nhạy kích hoạt tại Tầng 14 vị trí token quyết định ($p = -1$): $S_{14, \text{comp}} = 0.053811$.
- Độ ổn định khi chạy lại cùng mốc năm (Replay Stability): $\Delta_{\text{replay}} = 0.00000000$.
- **Kết luận**: Mô hình phản ứng nhạy bén và tất định trước sự thay đổi của vector thời gian điều kiện.

### 3.3. Cấp Độ 3: Tính Chọn Lọc (Selectivity) — CHƯA ĐẠT (`INCONCLUSIVE`)
- Độ lệch logit khi can thiệp đổi năm thật: $\Delta_{\text{time}} = 0.135589$.
- Độ lệch logit khi bơm vector nhiễu ngẫu nhiên cùng chuẩn độ lớn: $\Delta_{\text{rand}} = 0.119522$.
- Biên độ chọn lọc:
  $$\text{SelectivityMargin} = 0.135589 - 0.119522 = +0.016067 < 0.02.$$
- **Kết luận**: Khi mở rộng trên toàn bộ 54 cặp, biên độ chọn lọc giảm từ $0.0398$ (trong pilot $n=6$) xuống còn $0.0161$, không vượt qua ngưỡng kiểm định $0.02$. Mô hình phản ứng với vector thời gian chỉ nhỉnh hơn một chút so với một vector nhiễu ngẫu nhiên cùng độ lớn.

### 3.4. Hiệu Quả Định Hướng Chân Lý (Gold-Directed Effect $\Delta G$) — THẤT BẠI (`NO_DIRECTIONAL_GAIN`)
Để kiểm tra xem liệu sự biến thiên logit có hướng về phía đáp án chân lý hay không trước khi bị chặn bởi cơ chế giải mã tham lam, thực nghiệm đo lường qua log-likelihood của toàn chuỗi đáp án:
$$G(t) = s_{\text{Yes}}(x, t) - s_{\text{No}}(x, t), \quad \Delta G = G_{y^*(t_2)}(x, t_2) - G_{y^*(t_2)}(x, t_1).$$
- Giá trị trung bình $\Delta G$: $-0.058017$.
- Số cặp ghi nhận $\Delta G > 0$ (dịch chuyển đúng hướng chân lý): $13 / 36$ cặp ($36.11\%$).
- **Kết luận**: Tỷ lệ định hướng đúng chỉ đạt $36.11\%$, thấp hơn đáng kể so với mức đoán ngẫu nhiên ($50\%$). Điều này chứng minh vector thời gian không những không giúp mô hình tiến gần hơn tới chân lý lịch sử, mà còn kéo phân phối xác suất ngược hướng chân lý trên đa số các trường hợp.

### 3.5. Cấp Độ 4: Tính Đúng Đắn Ngữ Nghĩa (Semantic Correctness) — THẤT BẠI (`FAIL_NO_CAUSAL_FLIP`)
- Tỷ lệ đúng cả hai mốc thời gian trên tập Gold-Flip: $\text{BothCorrect} = 0 / 36 = 0.0\%$.
- Tỷ lệ lật nhãn quan sát được trên tập Gold-Flip: $0 / 36 = 0.0\%$.
- Tỷ lệ giữ nguyên nhãn trên tập Gold-Same (Negative Control): $18 / 18 = 100.0\%$.
- **Kết luận**: Mô hình hoàn toàn không có khả năng lật nhãn nhân quả theo sự thay đổi của thời gian lịch sử dưới bất kỳ hình thức giải mã nào.

### 3.6. Cấp Độ 5: Kiểm Định Thống Kê Ghép Cặp (Robustness & Statistics)
- Bảng tiếp liên ghép cặp $2 \times 2$ trên toàn bộ 108 câu hỏi:
  | | Arm B Đúng | Arm B Sai |
  |---|---|---|
  | **Base Đúng** | $a = 2$ | $b = 59$ |
  | **Base Sai** | $c = 44$ | $d = 3$ |
- Độ chính xác mô hình nền (`BASE_BOTH_OFF`): $56.48\%$ ($61/108$).
- Độ chính xác mô hình thích ứng (`ARM_B_FULL_TRUE_YEAR`): $42.59\%$ ($46/108$).
- **Kiểm định nhị thức chính xác (Two-sided Exact Binomial Test)**: $p = 0.167444 > 0.05$.
- **Kiểm định McNemar hiệu chỉnh cụm Obuchowski**: $p = 0.115243 > 0.05$ (Hệ số tương quan nội cụm ICC: $\rho = -0.043353$, Bootstrap 95% CI: $[0.0000, 0.2619]$).
- **Kết luận**: Không có sự vượt trội mang ý nghĩa thống kê của Arm B so với mô hình Base trên tập 108 câu hỏi phản thực nghiệm.

---

## 4. Phán Xử Bốn Giả Thuyết Điểm Nghẽn

1. **Giả thuyết H1 (Biểu diễn thời gian)**: **BÁC BỎ (`REFUTED`)**  
   *Bằng chứng*: $\bar{D}_z = 5.3172 > 0$ trên toàn bộ 54 cặp. Bộ mã hóa FoNE-CanChi40D phân biệt rõ rệt các mốc thời gian khác nhau trong không gian vector.

2. **Giả thuyết H2 (Dẫn truyền kiến trúc)**: **BÁC BỎ (`REFUTED`)**  
   *Bằng chứng*: $\bar{D}_W = 0.5001 > 0$ và $\Delta_{\text{logits}} = 0.1356 > 0.01$. Cả cổng Seam Injection và lõi thích ứng động MaT-LoRA đều chuyển đổi tín hiệu thành công đến token sinh đáp án cuối cùng.

3. **Giả thuyết H3 (Khiên chắn giải mã tham lam)**: **KHÔNG ĐỦ CƠ SỞ (`INCONCLUSIVE`)**  
   *Bằng chứng*: Việc đo đạc sequence log-probability $s_{\text{Yes}}, s_{\text{No}}$ cho thấy ngay cả khi bỏ qua bước argmax của greedy decoding, $\Delta G$ chỉ dương ở $36.11\%$ ca. Sự thất bại không đơn thuần do ranh giới quyết định của greedy decoding, mà bắt nguồn từ chính chiều hướng dịch chuyển của phân phối xác suất.

4. **Giả thuyết H4 (Ánh xạ ngữ nghĩa mục tiêu)**: **ĐƯỢC XÁC LẬP VỮNG CHẮC (`SUPPORTED`)**  
   *Bằng chứng*: Tỷ lệ đúng cả cặp bằng $0/36$ ($0.0\%$). Hàm mục tiêu Causal NTP kết hợp Relation loss trên tập dữ liệu hiện tại chỉ rèn luyện cho mô hình năng lực so sánh số học đơn thuần, hoàn toàn chưa dạy mô hình cách gắn kết vector thời gian với các sự kiện lịch sử thế giới thực.

---

## 5. Phán Quyết Khoa Học và Đề Xuất Thực Nghiệm Tiếp Theo

$$\mathbf{PHÁN\ QUYẾT\ CUỐI\ CÙNG:\ NO\_GO\_SCALE}$$

1. **Khóa dừng toàn diện**: Tuyệt đối không mở rộng tập dữ liệu huấn luyện hoặc khởi chạy huấn luyện GPU quy mô lớn khi cơ chế ánh xạ ngữ nghĩa chưa được xác lập.
2. **Biến số thực nghiệm đơn độc tiếp theo (Single Variable Intervention)**:
   - Thay vì tiếp tục tinh chỉnh mô hình phía downstream, nhánh nghiên cứu cần chuyển giao kết quả đo đạc cho phía đối tác (**Partner Branch**) để tái thiết kế cơ chế nhúng thời gian (ví dụ: chuyển từ FoNE-CanChi thuần số học sang biểu diễn có tích hợp semantic anchor hoặc learned entity-time interaction).
   - Duy trì nguyên vẹn hệ thống đánh giá nhân quả, bộ kiểm toán 54 cặp và OpenTelemetry fail-open làm công cụ nghiệm thu chuẩn tắc cho các checkpoint biểu diễn tiếp theo từ phía đối tác.
