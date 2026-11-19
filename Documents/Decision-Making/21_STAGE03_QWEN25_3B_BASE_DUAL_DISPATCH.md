---
loại: "quyết-định"
mã: "DECISION-STAGE03-QWEN25-3B-BASE-DUAL-DISPATCH-110"
dự_án: "long-range-temporal-alignment"
trạng_thái: "đang-dùng"
chuẩn_hiện_hành: true
thay_thế: ["[[20_STAGE03_QWEN25_3B_SCALE_TRANSITION]]"]
bị_thay_thế_bởi: []
tham_chiếu: ["[[18_STAGE03_MECHANISM_BEFORE_SCALE]]", "[[19_STAGE03_REPRESENTATION_FIRST_CONTRACT]]"]
được_xác_thực_bởi: []
cập_nhật_lần_cuối: "2026-10-24"
---

# Quyết Định Chuyển Đổi Mô Hình Nền Sang Qwen2.5-3B Base và Phát Động Thí Nghiệm Đối Chứng Kép (Dual Dispatch)

## 1. Bối Cảnh và Căn Cứ Khoa Học

Dựa trên kết quả kiểm toán pháp y toàn diện ngày 23/10/2026 ([`STAGE03_3B_FORENSIC_VERDICT.md`](file:///c:/Users/GF63/Documents/long-range-temporal-alignment/STAGE03_3B_FORENSIC_VERDICT.md) và [`STATUS_FOUR_EXISTING_GAPS.md`](file:///c:/Users/GF63/Documents/long-range-temporal-alignment/Documents/Documenting/Status/STATUS_FOUR_EXISTING_GAPS.md)):
1. **Lệch pha phân phối của mô hình Instruct**: Việc sử dụng `Qwen2.5-3B-Instruct` trên mẫu lời nhắc trần `Question: ...\nAnswer:` gây ra sự không tương thích nghiêm trọng giữa pha căn chỉnh chỉ dẫn (ChatML) và pha đánh giá hoàn thành chuỗi, dẫn đến thiên lệch logit tiên nghiệm và hiện tượng sụp đổ nhãn 'No' ($100\%$).
2. **Khôi phục tính trung thực với bài báo gốc**: Bài báo Ticktack (Han et al., 2025) sử dụng mô hình nền thuần túy **`Qwen/Qwen2.5-3B` (Base Model)**. Mô hình Base không bị ràng buộc bởi các lớp căn chỉnh đối thoại, vận hành tự nhiên trên bài toán hoàn thành câu tiếp theo (Next Token Prediction), loại bỏ hoàn toàn các rào chắn gây nhiễu của thẻ ChatML.
3. **Quyết định chuyển đổi Backbone**: Toàn bộ các thử nghiệm quy mô lớn tiếp theo của Stage 03 chính thức chuyển sang sử dụng mô hình nền **`Qwen/Qwen2.5-3B`** (Hugging Face ID: `Qwen/Qwen2.5-3B`, giấy phép Apache 2.0).

---

## 2. Thiết Kế Hai Cánh Tay Thực Nghiệm Song Song (Dual-Kernel Dispatch)

Dự án phát động đồng thời hai kernel GPU độc lập trên nền tảng Kaggle Cloud (Tesla T4) để thiết lập thế đối chứng ghép cặp chuẩn tắc trên cùng một mô hình nền Base và cùng tập dữ liệu sạch TempUN Full:

### Cánh tay 1: Baseline Ticktack Gốc Đối Chiếu (`qh0205/lrr-ticktack-3b-base-canonical`)
- **Mục tiêu**: Tái lập chính xác cơ chế của bài báo Ticktack trên mô hình `Qwen2.5-3B` Base.
- **Biểu diễn thời gian**: Tọa độ cực xoắn ốc Archimedes $\theta = 6^\circ \times ((y-3)\%60), r = 0.5 + 0.5\theta$, chiếu Descartes $(x, y)$, mã hóa Sinusoidal $TE \in \mathbb{R}^{2048}$, tiêm trực tiếp vào token ngày.
- **Thích ứng**: LoRA trên `q_proj` và `v_proj` ($r=8, \alpha=16.0$, 72 modules, $1{,}843{,}200$ tham số).
- **Hàm mất mát**: $\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{NTP}} + 0.5 \cdot \mathcal{L}_{\text{intra}}$.
- **Đánh giá**: Choice-Logit Scoring trên lời nhắc `Question: ...\nAnswer:`, đo lường cả Canonical Accuracy và 4-fold Cyclic Permutation Consistency.

### Cánh tay 2: Khung Can Thiệp Mục Tiêu Stage 03 (`qh0205/lrr-stage03-3b-base-targeted-alignment`)
- **Mục tiêu**: Triển khai toàn bộ các giải pháp kỹ thuật đã được kiểm chứng của Stage 03 trên `Qwen2.5-3B` Base.
- **Khử suy biến tuần hoàn (Non-Aliased Manifold)**: Tích hợp thành phần đơn điệu thế kỷ vào bán kính $r(y) = 0.5 + 0.5\theta + 0.5 \lfloor\frac{y-1947}{60}\rfloor$, bảo đảm $\|TE(y_1) - TE(y_2)\|_2 > 0$ khi $|y_1 - y_2| \ge 60$.
- **Hàm mất mát đa nhiệm nhân quả (Multi-task Causal Objective)**: Tích hợp $\mathcal{L}_{\text{rel}}$ (giám sát trực tiếp quan hệ thời gian $y_1 < y_2$) kết hợp $\mathcal{L}_{\text{NTP}}$ bảo toàn ngôn ngữ.
- **Đánh giá đa giao thức**: Bổ sung Sequence Log-likelihood Ranking (Protocol V3) và Prior Margin Calibration (Protocol V5) cùng bài kiểm tra phản thực nghiệm 54 cặp Gold-Flip.

---

## 3. Rào Chắn Kiểm Định Tiền Thực Thi (Pre-Dispatch Gate)

1. Cả hai kernel đều sử dụng tập dữ liệu chính thức đã kiểm toán mã băm SHA256 tại `qh0205/tempun-full-clean` ($104{,}130$ mẫu train, $5{,}292$ mẫu eval).
2. Giữ nguyên rào chắn dữ liệu: Không can thiệp đảo thứ tự phương án trong pha huấn luyện (`OnlineOptionPermutation.enabled = False`).
3. Khóa trạng thái `READY_TO_DISPATCH` và thu thập kết quả đo ghép cặp ngay khi cả hai kernel hoàn tất.
