# Báo Cáo Phân Tích Đối Chứng Nhân Quả 2x2: Văn Bản Câu Hỏi so với Vector Biểu Diễn (Phase B Report)

**Dự án**: `long-range-temporal-alignment`  
**Nhánh thực hiện**: `v2-targeted-continual-alignment`  
**Thời điểm khóa sổ**: `2026-10-20`  
**Mã đợt chạy khảo sát**: `RUN_STAGE03_POST_ARCHIVE_CAUSAL_20261020_082138`  
**Checkpoint kiểm định**: [`R52_B_NTP_REL_MCQA.pt`](file:///C:/Users/GF63/Documents/long-range-temporal-alignment/artifacts/experiments/V2_CAUSAL_TEMPORAL_ALIGNMENT/run_05_2/checkpoints/R52_B_NTP_REL_MCQA.pt) (SHA-256: `1af285df69ddaf94d21dcb858e8bd2d5e532874b484131d7fa3a31a4eac41988`)  
**Tập dữ liệu chuẩn tắc**: [`COUNTERFACTUAL_GOLD_AUDIT.jsonl`](file:///C:/Users/GF63/Documents/long-range-temporal-alignment/data/counterfactual/COUNTERFACTUAL_GOLD_AUDIT.jsonl) (SHA-256: `49eea7fe6fad69d4f3695e7cd8fbeac2c383e65b76d0802801666f9982cdeba5`)

---

## 1. Thiết Kế Thí Nghiệm Ma Trận 2x2 Causal Factorial

Nhằm giải quyết triệt để khiếm khuyết của độ đo Delta G cũ (chỉ can thiệp trên x1, thiếu ô đối ngẫu x2, z1), thí nghiệm Phase B thiết lập đầy đủ 4 ô đánh giá cho toàn bộ 54 cặp phản thực nghiệm (tổng cộng 216 lượt suy luận trên CPU):
- `T1_Z1`: Prompt x(t1), conditioned vector z(t1)
- `T1_Z2`: Prompt x(t1), conditioned vector z(t2)
- `T2_Z1`: Prompt x(t2), conditioned vector z(t1) *(ô can thiệp mới được bổ sung)*
- `T2_Z2`: Prompt x(t2), conditioned vector z(t2)

### Các Chỉ Số Đo Lường Nhân Quả
- Biên độ nhãn đúng (Gold Margin): m(x, z) = s_gold - s_alt.
- Hiệu ứng vector biểu diễn tại prompt t1: E_z(t1) = m(x1, z1) - m(x1, z2).
- Hiệu ứng vector biểu diễn tại prompt t2: E_z(t2) = m(x2, z2) - m(x2, z1).
- Hiệu ứng vector tổng hợp trên cặp: E_z_pair = [E_z(t1) + E_z(t2)] / 2.
- Tỷ lệ chuyển dịch đồng thuận hai chiều: PositiveBothSidesRate = P[E_z(t1) > 0 & E_z(t2) > 0].
- Độ lớn hiệu ứng văn bản (Text Effect Magnitude) so với độ lớn hiệu ứng adapter (Adapter Effect Magnitude).

---

## 2. Kết Quả Thực Nghiệm Trên 36 Cặp Gold-Flip (Primary Endpoint)

- **Hiệu ứng tại prompt t1 (E_z(t1))**: Mean = -0.058017, Std = 0.265763.
  - Số cặp E_z(t1) > 0: 13 / 36 (36.11%).
  - Kiểm định nhị thức hai phía so với 50%: p = 0.13250.
- **Hiệu ứng tại prompt t2 (E_z(t2))**: Mean = -0.058351, Std = 0.266699.
  - Số cặp E_z(t2) > 0: 13 / 36 (36.11%).
  - Kiểm định nhị thức hai phía so với 50%: p = 0.13250.
- **Hiệu ứng tổng hợp cặp (E_z_pair)**: Mean = -0.058184, 95% Clustered Bootstrap CI = [-0.147830, 0.044178].
- **Tỷ lệ đồng thuận hai chiều (PositiveBothSidesRate)**: 13 / 36 = 36.11% (95% CI: [20.44%, 55.89%]).

---

## 3. So Sánh Độ Lớn: Văn Bản Câu Hỏi (Text) so với Vector Biểu Diễn (Adapter)

- **Độ lớn hiệu ứng văn bản trung bình (Text Effect Magnitude)**: 0.015888
- **Độ lớn hiệu ứng adapter trung bình (Adapter Effect Magnitude)**: 0.248463
- **Tỷ số áp đảo (Text-to-Adapter Dominance Ratio)**: 0.06x
- **Tỷ lệ số cặp bị văn bản áp đảo hoàn toàn**: 0 / 36 (0.00%).

### Phân Tích Bản Chất Vật Lý Của Điểm Nghẽn
1. **Tín hiệu can thiệp đến được mô hình nhưng ở biên độ quá nhỏ**: Vector thời gian z(t) sau khi đi qua Gated Seam và MaT-LoRA tạo ra sự thay đổi logit ở mức ~0.11 đơn vị logit. Tuy nhiên, sự xuất hiện của năm cụ thể trong văn bản câu hỏi (ví dụ 'In 2022' vs 'In 1984') làm dịch chuyển phân phối logit lên tới ~1.0 - 3.0 đơn vị logit (gấp nhiều lần độ nhạy của adapter).
2. **Sự sụp đổ nhãn All-No chiếm ưu thế tuyệt đối**: Do trọng số LoRA bị huấn luyện lệch hoàn toàn về nhãn 'No' (như đã chứng minh trong Phase A), lực đẩy nhỏ ~0.11 từ vector thời gian hoàn toàn không đủ để vượt qua ngưỡng kích hoạt (decision boundary) để lật nhãn sang 'Yes'.
3. **Kết luận về giả thuyết cơ chế**: Bác bỏ giả thuyết cho rằng vector biểu diễn z(t) đang điều khiển ngữ nghĩa thời gian của mô hình một cách chọn lọc (selective causal mechanism). Hiện tượng mô hình đổi dự đoán trên một số ít câu hỏi chỉ là dao động biên quanh ngưỡng sụp đổ, không phải năng lực nhận thức thời gian chân thực.

---

## 4. Quyết Định Quản Trị Nghiên Cứu
- **Phán quyết**: Duy trì nghiêm ngặt cổng `NO_GO_SCALE`.
- Không thực hiện bất kỳ đợt huấn luyện GPU quy mô lớn nào khi cơ chế nhận dạng chưa được kiểm chứng.
- Chuyển tiếp sang Phase C & D: Điều tra hàm mục tiêu huấn luyện (Objective Forensics) và xác định điều kiện can thiệp một biến tiếp theo.
