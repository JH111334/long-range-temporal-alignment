# Ma Trận Quyết Định Điểm Nghẽn Biểu Diễn Thời Gian (Stage 03)

**Mã thực nghiệm (Run ID)**: `RUN_FULL_SWEEP_54P_20261019_161756`  
**Thời điểm ghi nhận**: `2026-10-19T16:17:56+07:00`  
**Checkpoint thẩm định**: [R52_B_NTP_REL_MCQA.pt](file:///c:/Users/GF63/Documents/long-range-temporal-alignment/artifacts/experiments/V2_CAUSAL_TEMPORAL_ALIGNMENT/run_05_2/checkpoints/R52_B_NTP_REL_MCQA.pt) (SHA-256: `1af285df69ddaf94d21dcb858e8bd2d5e532874b484131d7fa3a31a4eac41988`)  
**Tập dữ liệu phản thực nghiệm thẩm định**: [COUNTERFACTUAL_GOLD_AUDIT.jsonl](file:///c:/Users/GF63/Documents/long-range-temporal-alignment/data/counterfactual/COUNTERFACTUAL_GOLD_AUDIT.jsonl)  
**Quy mô mẫu thẩm định**: 54 cặp hợp lệ (36 cặp Gold-Flip, 18 cặp Gold-Same đối chứng âm; tổng cộng 108 câu hỏi, 23 cụm thời gian độc lập).  
**Số bước huấn luyện (Optimizer Steps)**: `0` (Chế độ thuần suy luận trên CPU, đóng băng 100% backbone và adapter).

---

## 1. Bảng Quyết Định Năm Cấp Độ (Five-Level Causal Matrix)

| Cấp Độ Khảo Sát | Chỉ Số Đo Lường Chính | Giá Trị Thực Nghiệm | Ngưỡng Tối Thiểu (Gate Threshold) | Trạng Thái Phán Quyết | Điểm Nghẽn (Bottleneck Status) |
|---|---|---|---|---|---|
| **Cấp độ 1: Delivery** | Frobenius norm biến thiên trọng số $\bar{D}_W$ | `0.500124` | $> 10^{-5}$ | **PASS** | Thông suốt ($\bar{D}_z = 5.317237, \bar{D}_F = 0.682921$, tín hiệu đến trọng số adapter) |
| **Cấp độ 2: Sensitivity** | Độ lệch logit tuyệt đối trung bình $\Delta \text{logits}$ | `0.135589` | $> 0.01$ | **PASS** | Nhạy cảm ($S_{14, \text{seam}} = 0.052463, S_{14, \text{comp}} = 0.053811$, phân rã 2 trục vật lý) |
| **Cấp độ 3: Selectivity** | Biên độ chọn lọc so với nhiễu ngẫu nhiên $\text{SelectivityMargin}$ | `0.016067` | $> 0.02$ | **INCONCLUSIVE** | Chưa đạt tính chọn lọc ($\Delta_{\text{time}} = 0.135589, \Delta_{\text{rand}} = 0.119522$; phản ứng tương tự nhiễu) |
| **Gold-Directed Effect** | Tỷ lệ tăng log-prob đáp án đúng $\Delta G > 0$ và trung bình $\bar{\Delta G}$ | Rate: `0.3611` (13/36)<br>Mean: `-0.058017` | Rate $> 0.50$<br>Mean $> 0.0$ | **NO_DIRECTIONAL_GAIN** | Vector thời gian làm suy giảm xác suất đáp án đúng lịch sử ở 63.89% số cặp |
| **Cấp độ 4: Semantic Correctness** | Tỷ lệ đúng cả cặp phản thực nghiệm $\text{BothCorrectRate}$ | `0.0` (0/36) | $\ge 0.30$ ($30\%$) | **FAIL_NO_CAUSAL_FLIP** | **ĐIỂM NGHẼN CỐT LÕI (Không lật nhãn theo sự thật lịch sử; Gold-Same Stability = 18/18 = 100%)** |
| **Cấp độ 5: Robustness** | McNemar chính xác hai phía / hiệu chỉnh cụm Obuchowski | Exact $p = 0.167444$<br>Clustered $p = 0.115243$ | $p < 0.05$ | **KHÔNG CÓ GAIN CÓ Ý NGHĨA** | Base Acc: $56.48\%$ ($61/108$), Arm B Acc: $42.59\%$ ($46/108$); ICC $\rho = -0.043353$ |

---

## 2. Phán Quyết Bốn Giả Thuyết Điểm Nghẽn Khoa Học

| Giả Thuyết Khoa Học | Nội Dung Giả Thuyết | Bằng Chứng Kiểm Chứng Thực Nghiệm | Trạng Thái Phán Quyết |
|---|---|---|---|
| **(H1) Biểu diễn thời gian** | Năm / representation không encode đủ quan hệ ngữ nghĩa / thứ tự / chu kỳ. | Khoảng cách biểu diễn $\bar{D}_z = 5.317237 > 0$ trên toàn bộ 54 cặp; $z(t)$ phân biệt rõ rệt từng năm lịch sử. | **BÁC BỎ (REFUTED)** |
| **(H2) Dẫn truyền kiến trúc** | Projection, Seam, dynamic core, adapter không chuyển thông tin hữu ích tới answer position. | Biến thiên trọng số $\bar{D}_W = 0.500124 > 10^{-5}$; độ nhạy tầng 14 đạt $0.0538$; độ lệch logit token quyết định đạt $0.135589 > 0.01$. | **BÁC BỎ (REFUTED)** |
| **(H3) Khiên chắn giải mã tham lam** | Tín hiệu đã tới answer scores đúng hướng nhưng greedy / parser / shortcut che khuất. | Dịch chuyển log-probability chuỗi ép giáo viên (Teacher-Forced Sequence Log-Prob) cho thấy $\Delta G > 0$ chỉ đạt $36.11\%$ ($13/36$), trung bình $\bar{\Delta G} = -0.058017$. Ngay ở mức log-prob, vector thời gian đã kéo phân phối xác suất ngược hướng chân lý ở $63.89\%$ số ca. | **CHƯA ĐỦ CĂN CỨ ĐƠN BIẾN (INCONCLUSIVE)** |
| **(H4) Ánh xạ ngữ nghĩa mục tiêu** | Objectives / training data semantics chưa dạy mô hình ánh xạ $(entity, relation, time) \to truth$. | Tỷ lệ đúng cả cặp phản thực nghiệm $\text{BothCorrect} = 0/36$ ($0.0\%$). Dù tín hiệu thời gian được bơm đầy đủ qua kiến trúc, mô hình hoàn toàn không liên kết được vector thời gian với việc chọn câu trả lời đúng theo ngữ nghĩa lịch sử. | **ĐƯỢC CHỨNG MINH LÀ NGUYÊN NHÂN CỐT LÕI (SUPPORTED)** |

---

## 3. Phân Tích Cơ Chế Hai Trục: Tầng (Layer) vs Vị Trí (Position)

1. **Vị trí tiêm cổng Seam ($p = 14$, token `' selecting'`)**: Tầng 14 ghi nhận độ nhạy trạng thái ẩn $S_{14, \text{seam}} = 0.052463$.
2. **Vị trí sinh từ tiếp theo ($p = -1$, vị trí quyết định nhãn)**: Tầng 14 ghi nhận độ nhạy trạng thái ẩn $S_{14, \text{comp}} = 0.053811$.
3. **Tính ổn định suy luận lại (Replay Stability)**: Sai khác logit cực đại giữa hai lần suy luận độc lập trên cùng dữ liệu và checkpoint đạt đúng `0.00000000` ($100\%$ tái lập số học xác định).
4. **Kết luận giải phẫu**: Tín hiệu thời gian được truyền bảo toàn từ cổng Seam ($p=14$) tới token quyết định ($p=-1$) mà không bị suy giảm theo chiều sâu tầng. Tuy nhiên, vector thời gian tạo ra biến thiên tương tự như nhiễu ngẫu nhiên ($\text{SelectivityMargin} = 0.016067 < 0.02$), phản ánh rằng các lớp biểu diễn chưa được tối ưu hóa để tương thích với không gian ngữ nghĩa của mô hình ngôn ngữ nền.

---

## 4. Quyết Định Quản Trị Nghiên Cứu và Rào Cản Phê Duyệt

$$\mathbf{PHÁN\ QUYẾT:\ NO_GO_SCALE}$$

1. **Tuyệt đối không mở rộng quy mô dữ liệu huấn luyện hoặc huấn luyện GPU (NO-GO SCALE)**: Khi biểu diễn thời gian chưa đạt tính chọn lọc (Level 3 Selectivity Inconclusive) và hoàn toàn thất bại ở tính đúng đắn ngữ nghĩa (Level 4 BothCorrect = 0.0%), việc mở rộng tập dữ liệu hay tăng số bước huấn luyện chỉ làm tăng nguy cơ học vẹt phân phối bề mặt của các token văn bản mà không giải quyết được khiếm khuyết biểu diễn cốt lõi.
2. **Thí nghiệm can thiệp tối thiểu tiếp theo (Minimal Single-Variable Next Step)**:
   - Giữ nguyên toàn bộ hạ tầng đánh giá xác định và 54 cặp phản thực nghiệm đã kiểm toán.
   - Yêu cầu phía đối tác cung cấp checkpoint biểu diễn thay thế (hoặc ma trận chiếu $W_{\text{proj}}$ mới) có tính phân tách cao hơn giữa các mốc năm trước khi tiến hành chu kỳ kiểm định suy luận tiếp theo.
   - Thử nghiệm can thiệp hoán vị phương án trực tuyến (Online Option Permutation) trong hàm mục tiêu để triệt tiêu thiên kiến ký tự trước khi xem xét mở lại cổng huấn luyện.
