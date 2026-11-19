# Biên Bản Hiệu Chỉnh Khoa Học Thực Nghiệm Pilot (Pilot 6-Pair Scientific Correction)

Tài liệu này ghi nhận việc kiểm toán độc lập, tái tính toán số học và khắc phục các sai lệch trong diễn giải kết quả từ lượt chạy thử nghiệm pilot `RUN_BOTTLENECK_20261019_150638`.

## 1. Bảng Đối Soát Sai Lệch (Claim vs Empirical Reality Matrix)

| Hạng Mục | Tuyên Bố / Ghi Nhận Ban Đầu | Giá Trị Log Thực Tế | Giá Trị Tái Tính Chuẩn Xác | Trạng Thái Phân Xử | Đường Dẫn Bằng Chứng | Hành Động Hiệu Chỉnh |
|---|---|---|---|---|---|---|
| **Mẫu số thực nghiệm (Denominator)** | Đại diện quét biểu diễn Stage 03 | 6 cặp câu hỏi (4 flip + 2 same) | $N = 6$ cặp ($12$ mốc thời gian) | `CORRECTED_TO_PILOT` | [RAW_PREDICTIONS.jsonl](file:///c:/Users/GF63/Documents/long-range-temporal-alignment/artifacts/stage03/representation_bottleneck/RUN_BOTTLENECK_20261019_150638/RAW_PREDICTIONS.jsonl) | Phân loại chính thức là PILOT SANITY CHECK; không sử dụng làm kết luận quét toàn diện. |
| **Kiểm định McNemar** | $p = 0.0000$, Bootstrap CI $[1.0, 1.0]$ | Bảng tiếp liên: $a=0, b=6, c=0, d=0$ | Exact Binomial: $p = 0.03125$; Edwards: $p \approx 0.0412$ | `CORRECTED_TO_EXACT` | [clustered_mcnemar.py](file:///c:/Users/GF63/Documents/long-range-temporal-alignment/src/temporal_alignment/causal_evaluation/clustered_mcnemar.py) | Hủy bỏ $p=0.0000$ suy biến do cỡ mẫu quá nhỏ; thay bằng kiểm định nhị thức chính xác; gắn nhãn `INSUFFICIENT_CLUSTER_POWER`. |
| **Mâu thuẫn Tính chọn lọc (Selectivity)** | Báo cáo văn bản ghi FAIL trong khi log ghi PASS | $\Delta_{\text{time}} = 0.1096$, $\Delta_{\text{rand}} = 0.0697$, Margin $= 0.0398$ | Margin $= +0.039834 > 0.02$ (PASS số học) | `NUMERICALLY_RECONCILED` | [SCOREBOARD.json](file:///c:/Users/GF63/Documents/long-range-temporal-alignment/artifacts/stage03/representation_bottleneck/RUN_BOTTLENECK_20261019_150638/SCOREBOARD.json) | Điều hòa thống nhất: Số học đạt ngưỡng pilot $> 0.02$, nhưng ghi chú rõ biên độ $0.0398$ trên $n=6$ chưa đủ lực chứng minh tính chọn lọc toàn cục. |
| **Thẩm quyền Kiểm toán Gold** | Kiểm toán độc lập nguồn gốc bên ngoài (External Grounding) | 54 cặp kiểm tra qua assertions mã nguồn | 54 cặp thẩm định quy chuẩn nội bộ | `PROGRAMMATIC_VALIDATION` | [COUNTERFACTUAL_GOLD_AUDIT.jsonl](file:///c:/Users/GF63/Documents/long-range-temporal-alignment/data/counterfactual/COUNTERFACTUAL_GOLD_AUDIT.jsonl) | Đổi phân loại thành `PROGRAMMATIC_VALIDATION_ONLY` kèm `CURATED_HISTORICAL_GROUNDING`. |
| **Trạng thái OpenTelemetry / Phoenix** | Hệ thống tracing phân tán hoạt động | 15 spans ghi nhận cục bộ, 0 span export thành công | Toàn bộ spans nằm trong bộ nhớ cục bộ | `OTEL_LOCAL_ONLY` | [TRACE_INDEX.json](file:///c:/Users/GF63/Documents/long-range-temporal-alignment/artifacts/stage03/representation_bottleneck/RUN_BOTTLENECK_20261019_150638/traces/TRACE_INDEX.json) | Xác nhận Phoenix collector offline; cơ chế fail-open kích hoạt an toàn; gắn nhãn `OTEL_EXPORT_NOT_VERIFIED`. |
| **Diễn giải Suy giảm qua Attention** | Tín hiệu suy giảm qua Attention do $S_{14,14} > S_{14,-1}$ | $S_{14,14} = 0.0588$, $S_{14,-1} = 0.0523$ | Khác biệt vị trí token (token 14 vs token -1) | `UNFOUNDED_RETRACTED` | [mechanism_tracer.py](file:///c:/Users/GF63/Documents/long-range-temporal-alignment/src/temporal_alignment/causal_evaluation/mechanism_tracer.py) | Rút lại kết luận suy giảm qua Attention; ghi nhận đây là phép đo tại hai vị trí token chuỗi khác nhau chưa chuẩn hóa alignment. |

---

## 2. Phân Tích Chi Tiết Các Điểm Hiệu Chỉnh

### 2.1. Cỡ Mẫu và Ý Nghĩa Thống Kê Cụm
Trong lượt chạy `RUN_BOTTLENECK_20261019_150638`, tham số `--max_pairs 6` được áp dụng nhằm thẩm định nhanh sự thông suốt của pipeline. Việc tính toán phương sai điều chỉnh cụm Obuchowski trên $N=6$ mẫu thuộc 6 cụm thời gian khác nhau ($K=6$, mỗi cụm $m_k=1$) dẫn đến ma trận hiệp phương sai nội cụm bị suy biến, sản sinh ra giá trị $p=0.0000$ và khoảng tin cậy bootstrap $[1.0, 1.0]$.
Theo chuẩn mực thống kê:
- Với bảng tiếp liên $2 \times 2$ có $b=6, c=0$, kiểm định nhị thức chính xác hai phía (Two-sided Exact Binomial Test) cho:
  $$p = 2 \times \left(\frac{1}{2}\right)^6 = \frac{2}{64} = 0.03125.$$
- Kiểm định McNemar có hiệu chỉnh liên tục Edwards cho $\chi^2 = 4.1667 \implies p = 0.0412$.
Do đó, không được công bố giá trị $p=0.0000$ hay khoảng tin cậy suy biến, mà phải ghi nhận `INSUFFICIENT_CLUSTER_POWER` cho các phân tích cụm khi $N < 30$.

### 2.2. Tính Chọn Lọc và Ngưỡng Pilot
Số liệu đo đạc thực tế ghi nhận:
- Độ lệch logit khi thay đổi năm thật: $\Delta_{\text{time}} = 0.109567$.
- Độ lệch logit khi bơm vector ngẫu nhiên cùng chuẩn độ lớn: $\Delta_{\text{rand}} = 0.069733$.
- Biên độ chọn lọc: $\text{SelectivityMargin} = 0.109567 - 0.069733 = +0.039834$.
Ngưỡng quy định trong `thresholds.yaml` là $0.02$. Về mặt số học, giá trị $0.039834 > 0.02$ thỏa mãn tiêu chí vượt ngưỡng của pilot. Tuy nhiên, sự vượt trội này có biên độ hẹp ($+0.0398$) trên tập mẫu nhỏ ($n=6$), chưa đủ để khẳng định tính chọn lọc nhân quả vững chắc cho toàn bộ miền năm lịch sử.

### 2.3. Trạng Thái Vận Hành Của OpenTelemetry
Biên bản trace ghi nhận 15 spans trong [OTEL_SPANS.jsonl](file:///c:/Users/GF63/Documents/long-range-temporal-alignment/artifacts/stage03/representation_bottleneck/RUN_BOTTLENECK_20261019_150638/traces/OTEL_SPANS.jsonl). Do máy chủ Arize Phoenix không hoạt động tại cổng 6006 trong lúc thực thi, mô-đun `Stage03Tracer` đã kích hoạt cơ chế fail-open, ghi nhận dữ liệu hoàn toàn vào bộ nhớ cục bộ (`otlp_export_success_count: 0`). Trạng thái này được phân loại chính xác là `OTEL_LOCAL_ONLY`, đảm bảo tính minh bạch tuyệt đối về việc chưa có dữ liệu nào được đẩy lên máy chủ từ xa.
