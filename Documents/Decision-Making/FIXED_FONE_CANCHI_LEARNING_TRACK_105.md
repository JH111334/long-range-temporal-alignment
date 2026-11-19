---
loại: "quyết-định"
mã: "DECISION-FIXED-FONE-CANCHI-LEARNING-TRACK-105"
dự_án: "long-range-temporal-alignment"
trạng_thái: "đang-dùng"
chuẩn_hiện_hành: true
thay_thế: ["[[LEARNING_TRACK_MTLORA_TARGETED_PLAN_DECISION]]"]
bị_thay_thế_bởi: []
tham_chiếu: ["DECISION-LEARNING-TRACK-MATLORA-ANCHOR-104", "DECISION-TEMPORAL-CONDITIONING-RESCUE-103", "DECISION-REP-ABLATION-102-EVALUATION"]
được_xác_thực_bởi: []
cập_nhật_lần_cuối: "2026-10-12"
---

# Quyết Định Thiết Kế: Khóa Cứng Biểu Diễn FoNE × Can-Chi, Xác Lập M0-FC40 Làm Starting Anchor và Khóa Thí Nghiệm M0-FC40 vs M1-TEMP vs M1-NC

## 1. Bối Cảnh và Mục Tiêu Quyết Định

- **Mã định danh:** `DECISION-FIXED-FONE-CANCHI-LEARNING-TRACK-105`
- **Thời điểm ban hành:** 12/10/2026 (Hiệu chỉnh toàn diện lúc 18:45 +07:00)
- **Trạng thái:** `APPROVED_ACTIVE`
- **Thẩm quyền chỉ đạo:** Quyết định chính thức của Người nghiên cứu (Learning Track Lead) và Ban Cố vấn Đề tài (Tháng 10/2026).
- **Kế thừa và Đính chính (Correction & Supersession):**
  - Quyết định này thay thế hoàn toàn phạm vi hình học hình trụ tại `DECISION-LEARNING-TRACK-MATLORA-ANCHOR-104` và bãi bỏ kế hoạch sơ bộ định tuyến `FC0 (Standard MLP-LoRA) vs FC1 (MaT-LoRA)` làm thí nghiệm đầu tiên.
  - **Kiến trúc xuất phát chính thức (Starting Architecture):** Bắt buộc là **Periodic Dual-Cycle MaT-LoRA** của đối tác (thích ứng trên giao diện 40D độc lập của Learning Track, định danh là **`M0-FC40`**). Không quay lại bước phân vân lựa chọn adapter; Standard MLP-LoRA chỉ giữ vai trò đối chứng lịch sử (Historical Baseline từ Week 1 $B_3 = C_0$) hoặc kiểm soát dự phòng khi phản biện yêu cầu.
  - Toàn bộ các thực nghiệm thuộc nhánh Learning Track chuyển sang hoạt động trên Biểu Diễn Đóng Băng FoNE × Can-Chi (`FIXED_FONE_CANCHI_V1`). Mọi khái niệm hình học đa tạp (hình trụ $\mathbb{S}^1 \times \mathbb{R}$, hình cầu $\mathbb{S}^2$) thuộc quyền sở hữu độc quyền của đối tác (Partner Track).

---

## 2. Nguyên Tắc Cốt Lõi Bất Biến (Invariants)

1. **Khóa cứng mô hình nền tảng:** Duy trì mô hình `Qwen/Qwen2.5-0.5B-Instruct` tại commit cố định `7ae557604adf67be50417f59c2c2f167def9a775`. Tuyệt đối không mở rộng kích thước mô hình lên 1.5B hoặc 3B.
2. **Phân định ranh giới quyền sở hữu (Ownership Boundary):**
   - **Nhánh Biểu Diễn Hình Học (Partner / Representation Track):** Đối tác sở hữu độc quyền không gian hình học đa tạp (cylinder $\mathbb{S}^1 \times \mathbb{R}$, sphere $\mathbb{S}^2$) và mã nguồn tại `SourceCode/long_range_representation/representation/`. Nhánh Learning không can thiệp, không biến đổi mã nguồn của đối tác.
   - **Nhánh Cơ Chế Học Tập (My / Learning Track):** Sở hữu độc quyền nghiên cứu cơ chế thích ứng tham số Dual-Cycle MaT-LoRA, hàm mục tiêu thời gian nhân quả (`MASKED_TARGET` kết hợp $\mathcal{L}_{\text{temporal}}$), và cơ chế chống quên lãng liên tục trên cùng một biểu diễn FoNE × Can-Chi đã khóa.
3. **Đóng băng tuyệt đối biểu diễn đầu vào:** Trong toàn bộ chuỗi thực nghiệm của nhánh Learning, vector thời gian $\mathbf{z}_{40}$ và cơ chế giao diện tiêm (seam) được đóng băng bất biến; không thay đổi số chiều hoặc động lực học trích xuất đặc trưng thời gian.

---

## 3. Đặc Tả Khế Ước Biểu Diễn Đóng Băng (FIXED_FONE_CANCHI_V1)

Biểu diễn thời gian đầu vào của nhánh Learning Track được chuẩn hóa bất biến theo cấu trúc ghép nối (concatenation) 40 chiều:

$$\mathbf{z} = [\mathbf{e}_{\text{FoNE}} \,\|\, \mathbf{e}_{\text{CanChi}}] \in \mathbb{R}^{40}$$

1. **Fourier Numerical Embedding (FoNE số học 32 chiều):**
   - Số lượng tần số: $K = 16$.
   - Chu kỳ logarit: $T_k = T_{\min} \cdot \left(\frac{T_{\max}}{T_{\min}}\right)^{\frac{k}{K-1}}$ với $T_{\min} = 10{,}0$ năm, $T_{\max} = 10000{,}0$ năm.
   - Tọa độ: $[\cos(\omega_k \cdot y), \sin(\omega_k \cdot y)]_{k=0}^{15}$ với $\omega_k = \frac{2\pi}{T_k}$ và $y$ là mốc năm lịch sử trực tiếp.
   - Mã băm cấu hình FoNE: `4ebea847bd63b21a7b553f0724394e73331e1bc844763d31d5ec81508307fb50`.
2. **Can-Chi Sexagenary Embedding (Can-Chi 8 chiều):**
   - Mã hóa vi chu kỳ Lục Thập Hoa Giáp (60 năm), Thập Thiên Can (10 năm), Thập Nhị Địa Chi (12 năm) và thành phần trôi dạt thế kỷ (century drift).
   - Chiều đặc trưng: 8 chiều số thực chuẩn hóa.
3. **Tổng số chiều biểu diễn:** $\dim(\mathbf{z}) = 32 + 8 = 40$.
4. **Giao diện tiêm an toàn (Safe Conditioning Seam):**
   - Cơ chế: Biến đổi trạng thái ẩn prefill qua cổng số dư liên tục (Gated Continuous Residual Prefill).
   - Vị trí tiêm: Token thứ 14 (`token_index=14`), tại ranh giới kết thúc phần dẫn nhập câu hỏi.
   - Lớp chiếu: $\operatorname{Linear}(40, 896) \to \operatorname{LayerNorm}(896)$.
   - Hệ số cổng khởi tạo: $\alpha_0 = 0{,}0$ (bảo toàn nguyên vẹn phân phối biểu diễn gốc tại bước khởi đầu).
   - Virtual tokens: $0$ (tuyệt đối không chèn virtual token, không phá vỡ cấu trúc ChatML token-0).
5. **Chứng thực bất biến (Golden Vectors Manifest):**
   - Vị trí lưu trữ: [feature_manifest.json](file:///c:/Users/GF63/Documents/long-range-temporal-alignment/artifacts/releases/FCLEARN_105/feature_manifest.json).
   - Mã băm kiểm tra (SHA256): `6c35fc55cbdfd571c03cd6bf56d3d4b65b367730b8bb3bc7d89c9dc82f78eddc`.

---

## 4. Đặc Tả Kỹ Thuật Starting Anchor M0-FC40

Khi chuyển giao kiến trúc Periodic Dual-Cycle MaT-LoRA sang hoạt động trên biểu diễn độc lập FoNE × Can-Chi 40D:

1. **Lan truyền xuôi (Forward Computation):**
   - Không gọi module tọa độ trụ của đối tác. Góc pha Lục Thập Hoa Giáp $\theta$ và trục kỷ nguyên $z$ được trích xuất trực tiếp từ 8 chiều Can-Chi có sẵn trong vector đặc trưng $\mathbf{z}_{40}$ để kích hoạt lõi động `DualCycleCoreGenerator`:
     $$F(\theta, z) = F_{\text{base}} + \sum_{k=1}^K [\cos(k\theta) C_k + \sin(k\theta) S_k] + z E_{\text{lin}} + \left(\frac{z}{10}\right)^2 E_{\text{quad}}$$
   - Cùng lúc đó, toàn bộ vector 40D được đưa qua Safe Seam tiêm tại token 14 prefill.
2. **Tham số hóa và Khả năng tương thích:**
   - Cấu hình: Hạng $r=8$, hệ số tỷ lệ $\alpha=16{,}0$, $K=2$ sóng điều hòa, $F_{\text{base}} = I_8$ trên 72 khối MLP.
   - **Tương thích checkpoint:** Checkpoint gốc của đối tác (huấn luyện trên pipeline trụ riêng tại `TICKTACK_V2_CONTINUAL`) **không tương thích nhị phân trực tiếp** với giao diện mới.
    - Trạng thái kiểm toán thực nghiệm: Bản chạy đầu tiên trên Kaggle GPU (Version 2) bị phân loại **`REJECT`** do lỗi bất ổn định số học (Quality Gate FAIL: `train_loss = NaN`, `strict_accuracy = 0.00%`). Toàn bộ 9 tệp checkpoint đã được lưu trữ và kiểm toán tại [artifacts/audits/M0_FC40_ACCEPTANCE_REPORT.json](file:///c:/Users/GF63/Documents/long-range-temporal-alignment/artifacts/audits/M0_FC40_ACCEPTANCE_REPORT.json).
    - Cần áp dụng bản vá tối thiểu **`M0-FC40-R1`** (khởi tạo $B=0$, upcast FP32 CrossEntropy, kích hoạt `GradScaler`) trước khi sử dụng làm mốc đối chứng chính thức.

---

## 5. Thí Nghiệm Đã Khóa: Đối Chứng Ghép Cặp M0-FC40-R1 vs M1-TEMP vs M1-NC

Thí nghiệm duy nhất được phê duyệt cho giai đoạn tiếp theo:

| Mã nhánh | Cấu hình Adapter & Biểu diễn | Hàm mục tiêu huấn luyện | Cơ chế kiểm chứng âm | Mục đích khoa học / Câu hỏi nghiên cứu | Trạng thái |
| :---: | :--- | :---: | :---: | :--- | :---: |
| **`M0-FC40`** | Periodic Dual-Cycle MaT-LoRA ($r=8, \alpha=16$) trên FoNE × Can-Chi 40D (Safe Seam) | $\mathcal{L}_{\text{MASKED\_TARGET}}$ | — | **Anchor xuất phát ban đầu:** Bị lỗi số học gradient overflow. | `REJECT (FAIL)` |
| **`M0-FC40-R1`** | Bản vá số học: Khởi tạo $B = 0$, upcast FP32 CrossEntropy, kích hoạt `GradScaler` | $\mathcal{L}_{\text{MASKED\_TARGET}}$ | — | **Anchor đối chứng chuẩn tắc:** Xác lập đường cơ sở sạch trên biểu diễn đã khóa. | `PLANNED` |
| **`M1-TEMP`** | Cùng cấu hình `M0-FC40-R1` (cùng khởi tạo, cùng số bước) | $\mathcal{L}_{\text{MASKED\_TARGET}} + \lambda \mathcal{L}_{\text{TEMPORAL}}$ ($\lambda = 0{,}1$) | Phân biệt mẫu âm lệch năm trong cùng chuỗi chỉ số huấn luyện | **Ứng viên cải tiến duy nhất:** Kiểm định đóng góp gia tăng của giám sát thời gian đối với độ chính xác và giảm lỗi cú pháp. | `BLOCKED (Chờ M0-FC40-R1)` |
| **`M1-NC`** | Cùng cấu hình `M1-TEMP` (cùng số bước, cùng token exposure) | $\mathcal{L}_{\text{MASKED\_TARGET}} + \lambda \mathcal{L}_{\text{SHUFFLED}}$ ($\lambda = 0{,}1$) | Xáo trộn nhãn năm huấn luyện ngẫu nhiên theo seed | **Đối chứng âm cấu trúc bắt buộc:** Loại trừ giả thuyết cải thiện do hiệu ứng điều hòa ngẫu nhiên hoặc tăng tần suất cập nhật gradient. | `BLOCKED (Chờ M0-FC40-R1)` |

### Định tuyến từ Jev System One API (`jev-1.13.0`):
- Kết quả định tuyến thực tế từ máy chủ TypeSafe AI ([jev_system_one_receipt.json](file:///c:/Users/GF63/Documents/long-range-temporal-alignment/artifacts/audits/jev/jev_system_one_receipt.json)) khẳng định:
  * `is_retention_prerequisite_met`: $0{,}16$ $\implies$ Hoãn toàn bộ nhánh Replay / O-LoRA / EWC sang Phase 2.
  * `is_m0_m1_causally_isolated`: $0{,}68$ $\implies$ Xác nhận thiết kế phân lập nhân quả tốt.
  * `next_optimal_path`: `VERIFY_M0_INTERFACE_AND_SMOKE` (Độ tin cậy $1{,}0$).

---

## 6. Khế Ước Đánh Giá và Tiêu Chuẩn Phán Quyết

1. **Khế ước đánh giá:** Bắt buộc áp dụng `GENERATIVE_STRICT_PARSE_V1`, đo lường qua 3 hạt ngẫu nhiên chuẩn tắc (`SEEDS = [1, 2, 3]`) trên tập dữ liệu kiểm tra 180 ca của `TempUN Small` (`eval_mcqa.jsonl`).
2. **Chỉ số đo lường:** Strict Accuracy ($\%$) làm chỉ số chính; Invalid Syntax Rate ($\%$) làm cổng dừng an toàn (ngưỡng từ chối $> 30{,}0\%$); Valid Accuracy ($\%$) và Backward Transfer (BWT) làm chỉ số chẩn đoán.
3. **Phán quyết khoa học:**
   - So sánh `M1-TEMP` vs `M1-NC`: Phải vượt trội có ý nghĩa thống kê ($p < 0{,}05$) qua kiểm định ghép cặp McNemar trên 180 ca độc lập để chứng minh tính nhân quả của tín hiệu thời gian.
   - So sánh `M1-TEMP` vs `M0-FC40`: Phải đạt $p < 0{,}05$ với $\Delta \text{Strict Acc} > 0$. Nếu $p \ge 0{,}05$, ghi nhận trung thực phán quyết `FAILED_TO_REJECT_H0`.
   - Tuyệt đối không tinh chỉnh tùy tiện $\lambda$ khi kết quả không đạt kỳ vọng để tránh overfitting trên tập đánh giá.
