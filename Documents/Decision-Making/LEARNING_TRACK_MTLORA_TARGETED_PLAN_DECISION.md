---
loại: "quyết-định"
mã: "DECISION-LEARNING-TRACK-MATLORA-ANCHOR-104"
dự_án: "long-range-temporal-alignment"
trạng_thái: "bị-thay-thế"
chuẩn_hiện_hành: false
thay_thế: ["DECISION-TEMPORAL-CONDITIONING-RESCUE-103"]
bị_thay_thế_bởi: ["[[FIXED_FONE_CANCHI_LEARNING_TRACK_105]]"]
tham_chiếu: ["DECISION-TEMPORAL-CONDITIONING-RESCUE-103", "DECISION-REP-ABLATION-102-EVALUATION"]
được_xác_thực_bởi: []
cập_nhật_lần_cuối: "2026-10-12"
---

# Quyết Định Thiết Kế: Xác Lập MaT-LoRA Làm Điểm Xuất Phát (Anchor M0) và Định Hướng Thực Nghiệm M1

## 1. Bối Cảnh và Mục Tiêu Quyết Định
- **Mã định danh:** `DECISION-LEARNING-TRACK-MATLORA-ANCHOR-104`
- **Ngày ban hành:** 12/10/2026
- **Trạng thái:** `APPROVED_ACTIVE`
- **Thẩm quyền chỉ đạo:** Giảng viên Hướng dẫn và Ban Cố vấn Đề tài (Tháng 10/2026)
- **Kế thừa và Thay thế:** Quyết định này thay thế (supersedes) dự thảo sơ bộ `L0=standard MLP-LoRA -> L1=generic MTLoRA`. Kiến trúc `Periodic Dual-Cycle MaT-LoRA` của đối tác được lựa chọn chính thức làm **Điểm xuất phát (Starting Anchor $M_0$)** cho nhánh Cơ chế học tập (Learning Track).

---

## 2. Nguyên Tắc Cốt Lõi Bất Biến (Invariants)
1. **Khóa cứng mô hình nền tảng:** Duy trì mô hình `Qwen/Qwen2.5-0.5B-Instruct` tại commit cố định `7ae557604adf67be50417f59c2c2f167def9a775`. Tuyệt đối không mở rộng kích thước mô hình lên 1.5B hoặc 3B.
2. **Phân định hai đường đua độc lập (Track Separation):**
   - **Nhánh Biểu Diễn Hình Học (Partner Track):** Đối tác sở hữu đa tạp hình học thời gian (hình trụ $\mathbb{S}^1 \times \mathbb{R}$, ứng viên hình cầu $\mathbb{S}^2$) và mã nguồn biểu diễn.
   - **Nhánh Cơ Chế Học Tập (Learning Track - Nhánh Hiện Tại):** Sở hữu thích ứng tham số, hàm mục tiêu thời gian (`MASKED_TARGET` + $\mathcal{L}_{\text{TEMPORAL}}$) và cơ chế chống quên lãng liên tục.
3. **Đóng băng giao diện hình học:** Trong toàn bộ các thực nghiệm của nhánh Learning, vector biểu diễn tọa độ trụ và cơ chế tích hợp của partner MaT-LoRA được giữ nguyên trạng thái đóng băng; không làm biến đổi hình học trong quá trình đánh giá hàm mục tiêu.

---

## 3. Xác Lập Anchor Xuất Phát $M_0$ và Ranh Giới Bằng Chứng

1. **Đặc tả $M_0$:**
   - Kiến trúc: `Ticktack_v2_DualCycle_MaT_LoRA` kết hợp vi chu kỳ Can-Chi $F_{\text{cycle}}(\theta)$ và đại kỷ nguyên $F_{\text{epoch}}(z)$ trên 72 khối MLP của Qwen2.5-0.5B ($r=8, \alpha=16.0$).
   - Hàm mục tiêu: `MASKED_TARGET` Cross-Entropy.
   - Kết quả đo đạc chuẩn tắc (Khế ước `GENERATIVE_STRICT_PARSE_V1`, 3 seed): Strict Acc $= 20.51\% \pm 3.06\%$, Invalid Rate $= 26.85\% \pm 4.63\%$, Valid Acc $= 28.01\% \pm 2.79\%$.
2. **Ranh giới tuyên bố khoa học:**
   - Số liệu $36.93\%$ trong báo cáo sơ bộ được đo theo giao thức unmasked/lenient; khi kiểm toán lại theo khế ước nghiêm ngặt, Strict Acc đạt $20.51\%$ và kiểm định McNemar so với $F_0$ Baseline ($20.93\%$) đạt phán quyết `INCONCLUSIVE` ($p > 0.15$).
   - MaT-LoRA được chọn làm điểm xuất phát chiến lược vì khả năng giảm mạnh tỷ lệ lỗi cú pháp (từ $38.89\%$ xuống $26.85\%$) và tính tương thích cao với chu kỳ Can-Chi, không tuyên bố vượt trội thuần túy adapter khi chưa có matched ablation.

---

## 4. Định Hướng Thực Nghiệm Tiếp Theo: $M_1$ Causal Targeted Temporal Objective

1. **Thiết kế thực nghiệm $M_1$:**
   - Giữ nguyên toàn bộ cấu hình mô hình, adapter MaT-LoRA và dữ liệu của $M_0$.
   - Bổ sung hàm mục tiêu phụ định hướng thời gian nhân quả:
     $$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{MASKED\_TARGET}} + \lambda \cdot \mathcal{L}_{\text{TEMPORAL}}$$
     với $\lambda = 0{,}1$ làm ứng viên sàng lọc định sẵn.
   - Nhiệm vụ phụ: Phân biệt sự kiện bị xáo trộn mốc thời gian (Year-Corruption Discrimination) trên các sự kiện lịch sử thuộc tập huấn luyện, có quy tắc tạo mẫu âm nghiêm ngặt (lệch $\Delta \in \{\pm 5, \pm 10, \pm 60\}$ năm) nhằm triệt tiêu nguy cơ rò rỉ tập đánh giá.
2. **Đối chứng cấu trúc bắt buộc:**
   - Triển khai song song nhánh đối chứng cấu trúc $M_1\text{-SHUFFLE}$ (xáo trộn nhãn mốc thời gian theo seed cố định) để loại trừ giả thuyết cải thiện do hiệu ứng điều hòa ngẫu nhiên.
3. **Cổng dừng (Stop Gates):**
   - Chỉ kết luận có hiệu quả khi $M_1$ vượt trội có ý nghĩa thống kê ($p < 0{,}05$) so với cả $M_0$ và $M_1\text{-SHUFFLE}$, đồng thời không làm tăng tỷ lệ lỗi cú pháp quá $30\%$.

---

## 5. Khế Ước Ghép Nối Với Đối Tác (Merge Contract)

Hai nhánh chỉ tiến hành hợp nhất theo ma trận nhân tố $2 \times 2$ sau khi từng nhánh chứng minh được hiệu quả thống kê độc lập:
$$\begin{matrix}
R_0 + M_0 & \text{(Baseline đối chứng: Cylinder + MaT-LoRA MASKED_TARGET)} \\
R_1 + M_0 & \text{(Đóng góp đơn lẻ của hình học mới, ví dụ: Sphere)} \\
R_0 + M_1 & \text{(Đóng góp đơn lẻ của hàm mục tiêu thời gian mới)} \\
R_1 + M_1 & \text{(Hiệu ứng tương tác kết hợp)}
\end{matrix}$$
