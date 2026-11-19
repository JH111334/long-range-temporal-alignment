# Phán Quyết Thực Nghiệm Quét Nhân Quả Toàn Diện (Run Verdict: RUN_FULL_SWEEP_54P_20261019_161756)

- **Mã thực nghiệm**: `RUN_FULL_SWEEP_54P_20261019_161756`
- **Số cặp thẩm định**: 54 cặp (36 Gold-Flip, 18 Gold-Same), 108 câu hỏi, 23 cụm thời gian độc lập.
- **Thời gian thực thi**: 781.92s trên CPU (`torch.float32`, 0 optimizer steps).
- **Mã băm Checkpoint**: `1af285df69ddaf94d21dcb858e8bd2d5e532874b484131d7fa3a31a4eac41988`.

## 1. Kết Quả Đo Lường Cốt Lõi

1. **Delivery (Cấp độ 1)**: ĐẠT (`PASS`). $\bar{D}_z = 5.3172$, $\bar{D}_F = 0.6829$, $\bar{D}_W = 0.5001 > 10^{-5}$. Tín hiệu thời gian được mã hóa và truyền hoàn toàn qua lõi ma trận thích ứng.
2. **Sensitivity (Cấp độ 2)**: ĐẠT (`PASS`). Độ lệch logit trung bình $\Delta_{\text{logits}} = 0.135589 > 0.01$. Kích hoạt ẩn tại Tầng 14 ghi nhận $S_{14, \text{seam}} = 0.0525$ và $S_{14, \text{comp}} = 0.0538$.
3. **Selectivity (Cấp độ 3)**: KHÔNG ĐẠT NGƯỠNG (`INCONCLUSIVE`). $\text{SelectivityMargin} = 0.135589 - 0.119522 = 0.016067 < 0.02$. Mô hình biến thiên trước tín hiệu thời gian thật tương tự như trước nhiễu ngẫu nhiên cùng chuẩn độ lớn.
4. **Gold-Directed Effect ($\Delta G$)**: KHÔNG CÓ HIỆU QUẢ ĐỊNH HƯỚNG (`NO_DIRECTIONAL_GAIN`). Chỉ có 13/36 cặp flip ($36.11\%$) ghi nhận $\Delta G > 0$. Trung bình $\Delta G = -0.0580$. Tín hiệu thời gian kéo phân phối xác suất ngược hướng chân lý lịch sử trên đa số các câu hỏi.
5. **Semantic Correctness (Cấp độ 4)**: THẤT BẠI (`FAIL_NO_CAUSAL_FLIP`). Tỷ lệ đúng cả cặp $\text{BothCorrect} = 0/36$ ($0.0\%$). Không ghi nhận bất kỳ ca lật nhãn nào thành công theo chân lý. Tỷ lệ giữ nguyên nhãn trên đối chứng âm đạt $18/18$ ($100.0\%$).
6. **Robustness & McNemar (Cấp độ 5)**: Độ chính xác Base là $56.48\%$ ($61/108$), Arm B là $42.59\%$ ($46/108$). Kiểm định Exact Binomial McNemar cho $p = 0.167444 > 0.05$. Không có độ lợi thống kê so với mô hình nền.

## 2. Phán Quyết Bốn Giả Thuyết Điểm Nghẽn

- **(H1) Biểu diễn thời gian**: BÁC BỎ (`REFUTED`). $\bar{D}_z = 5.3172 > 0$, vector $z(t)$ hoàn toàn phân biệt các năm.
- **(H2) Dẫn truyền kiến trúc**: BÁC BỎ (`REFUTED`). $\bar{D}_W = 0.5001$, $\Delta_{\text{logits}} = 0.1356$, tín hiệu truyền tới token quyết định.
- **(H3) Khiên chắn giải mã tham lam**: KHÔNG ĐỦ CƠ SỞ (`INCONCLUSIVE`). Vì ngay cả ở mức độ sequence log-probability, $\Delta G$ chỉ dương ở $36.11\%$ ca.
- **(H4) Ánh xạ ngữ nghĩa mục tiêu**: ĐƯỢC ỦNG HỘ (`SUPPORTED`). Objectives huấn luyện NTP + Relation chưa bao giờ dạy mô hình ánh xạ bộ ba $(entity, relation, time) \to truth$.

## 3. Quyết Định Quản Trị Nghiên Cứu

$$\mathbf{PHÁN\ QUYẾT:\ NO\_GO\_SCALE}$$

Dừng toàn diện việc mở rộng quy mô dữ liệu và huấn luyện GPU. Chuyển giao yêu cầu nghiên cứu biểu diễn sang phía đối tác.
