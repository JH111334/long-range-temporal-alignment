# Hợp Đồng Bàn Giao Liên Nhánh (Branch Handoff Contract)

Tài liệu này xác lập ranh giới trách nhiệm, thẩm quyền và giao thức trao đổi tạo tác giữa hai nhánh nghiên cứu độc lập.

## 1. Phân Định Trách Nhiệm

1. **Nhánh Biểu Diễn Thời Gian (Partner Branch)**:
   - Thư mục làm việc: `C:\Users\GF63\Documents\Long-Range-Representation` (chế độ **STRICT READ-ONLY** đối với nhánh hiện tại).
   - Trách nhiệm: Nghiên cứu, thiết kế và huấn luyện các bộ mã hóa biểu diễn thời gian dài hạn ($z(t)$) bao gồm: FoNE, FoNE+CanChi 40D, Ticktack, learned year embedding.
   - Đầu ra bàn giao: Checkpoint bộ mã hóa, bảng tra cứu hoặc hàm sinh biểu diễn, tài liệu đặc tả toán học và mã băm SHA-256 đã niêm phong.

2. **Nhánh Mục Tiêu Huấn Luyện và Đánh Giá Nhân Quả (User Branch)**:
   - Thư mục làm việc: `C:\Users\GF63\Documents\long-range-temporal-alignment` (nhánh hoạt động `v2-targeted-continual-alignment`).
   - Trách nhiệm:
     - Xây dựng kiến trúc ghép nối (Seam Injection tại index 14, Dynamic Core $F_t$ qua `DualCycleMaTLoRA`).
     - Thiết kế mục tiêu tối ưu (NTP, Temporal Relation, MCQA/CP objective).
     - Thiết lập hệ thống kiểm toán độc lập, đo đạc nhân quả 5 cấp độ (Delivery, Sensitivity, Selectivity, Semantic Correctness, Robustness).
     - Hệ thống quan sát và theo vết phân tán (OpenTelemetry + Arize Phoenix).
   - Đầu ra bàn giao: Báo cáo ma trận điểm nghẽn, bảng chỉ số tiếp liên $2 \times 2$, kiểm định McNemar hiệu chỉnh cụm và quyết định GO/NO-GO mở rộng.

## 2. Giao Thức Trao Đổi Tạo Tác (Artifact Handoff Protocol)

1. Mọi checkpoint từ phía đối tác đưa vào sử dụng phải được đăng ký trong `registries/MODEL_REGISTRY.json` với đầy đủ:
   - Đường dẫn tuyệt đối của checkpoint.
   - Mã băm SHA-256 tính toán độc lập.
   - Ngày tiếp nhận và phiên bản cam kết git của bộ mã hóa nguồn.
2. Không thực hiện lệnh `git merge`, `git cherry-pick`, hay sửa đổi trực tiếp vào cây thư mục của đối tác.
3. Khi đối tác cung cấp biểu diễn ứng viên mới, nhánh hiện tại chỉ nạp qua giao diện chuẩn tắc quy định tại `representation_interface.schema.json`.

## 3. Tiêu Chí Nghiệm Thu Bàn Giao (Acceptance Criteria)

Biểu diễn của đối tác chỉ được xác nhận là đạt yêu cầu tích hợp khi:
1. Đạt kiểm định hình thức: Kích thước tensor, kiểu dữ liệu và miền giá trị thỏa mãn schema.
2. Đạt Cấp độ 1 (Delivery): Ma trận trọng số thích ứng $\Delta W(t)$ biến thiên với Frobenius norm $> 10^{-5}$.
3. Đạt Cấp độ 2 (Sensitivity): Độ nhạy kích hoạt $S_{14, 14} > 0.005$ và độ lệch logit trung bình $> 0.01$.
4. Đạt Cấp độ 3 (Selectivity): Độ lệch logit của biểu diễn vượt trội so với vector ngẫu nhiên cùng chuẩn độ lớn với biên an toàn $> 0.02$.
5. Đạt Cấp độ 4 (Semantic Correctness): Tỷ lệ lật nhãn đúng theo cặp phản thực nghiệm vượt ngưỡng $30\%$.
