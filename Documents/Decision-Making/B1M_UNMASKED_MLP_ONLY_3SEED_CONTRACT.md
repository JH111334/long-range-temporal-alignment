---
loại: "quyết-định"
mã: "PENDING-B1M-UNMASKED-MLP-ONLY-3SEED-CONTRACT"
dự_án: "long-range-temporal-alignment"
trạng_thái: "đề-xuất"
chuẩn_hiện_hành: false
thay_thế: []
bị_thay_thế_bởi: []
tham_chiếu: []
được_xác_thực_bởi: []
cập_nhật_lần_cuối: "2026-10-01"
---

# Giao Ước Thực Nghiệm B1M: Ô Đối Chứng Bổ Khuyết Ma Trận 2×2 (B1M 3-Seed Validation Contract)

- **Trạng thái**: ĐÓNG BĂNG TRƯỚC KHI THỰC THI (FROZEN BEFORE EXECUTION)
- **Mã giả thuyết**: `B1M_UNMASKED_MLP_ONLY_V1`
- **Mục tiêu khoa học**: Hoàn tất ô còn thiếu duy nhất trong ma trận nhân quả 2×2 giữa Objective (`UNMASKED_FULL_SEQUENCE` và `MASKED_TARGET`) và Trainable Path (`ALL_LINEAR` và `MLP_ONLY`), nhằm phân rã các hiệu ứng chính (Main Effects) và kiểm định hiệu ứng tương tác thống kê (Statistical Interaction Effect) qua 3 hạt giống (seeds 1, 2, 3).
- **Ranh giới tuyên bố**:
  - LoRA là công trình đã công bố (prior art).
  - MLP/FFN-only LoRA là chiến lược thích nghi đã được công bố (prior-art adaptation strategy).
  - Targeted temporal supervision kế thừa ý tưởng từ các công trình trước (ECONET, TempLAMA).
  - `B1M` chỉ là ô đối chứng kiểm soát (controlled missing cell) trong thiết lập Qwen + TempUN; nhóm không tuyên bố MLP-only là phát minh mới và không suy diễn cơ chế nội tại khi chưa có bằng chứng thực nghiệm.

---

## 1. Vị Trí Canonical Trong Ma Trận Factorial 2×2

Ma trận chuẩn hóa giữa Objective và Trainable Path:

| Objective \ Trainable Path | ALL_LINEAR (168 modules, 1.099.776 params) | MLP_ONLY (72 modules, 829.440 params) |
|---|---|---|
| **UNMASKED_FULL_SEQUENCE** | **B1** (đã kiểm định 3 seed) | **B1M** (ô thực nghiệm bổ khuyết) |
| **MASKED_TARGET** | **B2** (đã kiểm định 3 seed) | **B3** (đã kiểm định 3 seed) |

### 1.1. Nguyên Tắc Kiểm Soát Biến Đơn Nhất
- **So với B1**: B1M chỉ khác B1 đúng MỘT yếu tố duy nhất là Trainable Path (`ALL_LINEAR` $\rightarrow$ `MLP_ONLY`), giữ nguyên 100% hàm mục tiêu `UNMASKED_FULL_SEQUENCE`.
- **So với B3**: B1M chỉ khác B3 đúng MỘT yếu tố duy nhất là Objective (`MASKED_TARGET` $\rightarrow$ `UNMASKED_FULL_SEQUENCE`), giữ nguyên 100% vị trí can thiệp `MLP_ONLY`.
- **Yêu cầu adapter Attention**: Tuyệt đối bằng 0 (`ATTENTION_LORA_TARGET_COUNT = 0`). Danh sách module can thiệp được phát hiện động tại runtime (`gate_proj`, `up_proj`, `down_proj` trên 24 lớp Transformer).

---

## 2. Khung Thực Nghiệm Đóng Băng (Frozen Substrate)

Toàn bộ các tham số nền tảng được khóa cố định, đảm bảo tính đẳng cấu với đợt chạy chuẩn hóa B1/B2/B3:

| Thuộc tính | Giá trị đóng băng | Phân loại bằng chứng |
|---|---|---|
| **Mô hình nền tảng** | `Qwen/Qwen2.5-0.5B-Instruct` | DIRECT (Pinned Revision: `7ae557604adf67be50417f59c2c2f167def9a775`) |
| **Bộ mã hóa từ vựng** | `Qwen/Qwen2.5-0.5B-Instruct` | DIRECT (Pinned Revision: `7ae557604adf67be50417f59c2c2f167def9a775`) |
| **Gói dữ liệu** | TempUNs v1 (`Data/shared_preprocessed/v1`) | DIRECT (Manifest SHA-256: `40b8ac841b1aaace92648dc1b12d3c780435704dd4a6a148984223de07f7fbf7`) |
| **Chặng thời gian huấn luyện** | `year:1947`, `year:1984`, `year:2022` (thứ tự tuần tự) | DIRECT |
| **Số lượng mẫu huấn luyện** | 733 (1947), 1.338 (1984), 390 (2022) — Tổng cộng: 2.461 mẫu NWP | DIRECT |
| **Tập đánh giá MCQA** | 180 ca cố định | DIRECT (Case IDs SHA-256: `d11007d08bf43ea760c473ea8a1504a98776204610ca0136f62ae2965e236d10`) |
| **Cấu hình lượng tử** | 4-bit NF4, double quantization, FP16 compute | DIRECT |
| **Cấu hình LoRA** | Rank $r=2$, $\alpha=4$, dropout $0.0$; task_type `CAUSAL_LM` | DIRECT |
| **Số tham số huấn luyện** | 829.440 tham số (khớp chính xác B3; giảm 24,58% so với B1/B2) | DIRECT |
| **Tối ưu hóa** | AdamW, learning rate $2\times 10^{-5}$, weight decay $0.0$ | DIRECT |
| **Batch size** | Microbatch 1, gradient accumulation 4 (effective batch 4) | DIRECT |
| **Độ dài chuỗi tối đa** | 128 tokens; fail-closed nếu vượt quá giới hạn | DIRECT |
| **Giải mã suy luận** | `do_sample=True`, $\text{temp}=0.2$, $\text{top\_p}=0.94$, $\text{max\_new\_tokens}=100$ | DIRECT |
| **Quy tắc hạt giống suy luận** | Deterministic per-example qua `SHA-256(base_seed, example_id)` | DIRECT |
| **Hậu tố prompt đánh giá** | `. Provide only the correct option, without explaination.` | DIRECT |
| **Giao ước đánh giá** | Evaluation Contract Hash: `e6ada0457b0ef93a0eecac2bfd7b5183ee40b3f5e872ddc350babb46c4f657e2` | DIRECT |
| **Tập hạt giống huấn luyện** | Seed 1, Seed 2, Seed 3 | PROPOSED |

---

## 3. Giao Ước Đo Lường Và Phân Rã Hiệu Ứng

### 3.1. Thước Đo Đa Chiều Bắt Buộc
Do hiện tượng ngõ ra không hợp lệ có thể làm sai lệch độ chính xác điều kiện, mọi so sánh bắt buộc báo cáo đồng thời 3 chỉ số:
1. **Valid Accuracy (Chuẩn TempUN)**:
   $$\text{Valid Accuracy} = \frac{C}{C + I + N}$$
2. **Strict Accuracy (Chuẩn khắt khe toàn diện)**:
   $$\text{Strict Accuracy} = \frac{C}{180}$$
   (Mọi kết quả invalid, suy đoán lan man hoặc không trích xuất được đều tính là 0).
3. **Invalid Rate (Tỷ lệ suy giảm tuân thủ định dạng)**:
   $$\text{Invalid Rate} = \frac{\text{Invalid}}{180}$$

### 3.2. Công Thức Phân Rã Hiệu Ứng Chính (Main Effects)
- **Hiệu ứng Objective dưới All-Linear**:
  $$\Delta_{\text{objective, all}} = B2 - B1$$
- **Hiệu ứng Objective dưới MLP-Only**:
  $$\Delta_{\text{objective, mlp}} = B3 - B1M$$
- **Hiệu ứng Path dưới Unmasked Objective**:
  $$\Delta_{\text{path, unmasked}} = B1M - B1$$
- **Hiệu ứng Path dưới Masked Objective**:
  $$\Delta_{\text{path, masked}} = B3 - B2$$

### 3.3. Kiểm Định Hiệu Ứng Tương Tác (Interaction Effect)
Hiệu ứng tương tác được tính toán độc lập qua hai biểu thức toán học tương đương và phải trùng khớp số học:
$$\text{INTERACTION}_1 = (B3 - B1M) - (B2 - B1)$$
$$\text{INTERACTION}_2 = (B3 - B2) - (B1M - B1)$$

- **Quy tắc diễn giải**:
  - $\text{INTERACTION} \approx 0$: Hai yếu tố hoạt động cộng tính độc lập (additive / independent).
  - $\text{INTERACTION} > 0$: Giám sát có mục tiêu (`MASKED_TARGET`) hưởng lợi tương đối vượt trội khi vị trí can thiệp bị giới hạn vào MLP (hoặc ngược lại).
  - $\text{INTERACTION} < 0$: Tồn tại sự dư thừa hoặc triệt tiêu lẫn nhau (antagonistic / redundancy).
  - Tuyên bố khoa học chỉ ghi nhận là: *hiệu ứng tương tác thống kê quan sát được dưới thiết lập đóng băng Qwen–TempUN*; tuyệt đối không khẳng định cơ chế nội tại của não bộ mô hình.

---

## 4. Phân Tích Lát Cắt và Chẩn Đoán Lỗi

- **Lát cắt trùng lặp nguồn**:
  - `OVERLAP_PRONE` (60 ca: `CP` 24 ca, `DB` 36 ca): kiểm tra hiện tượng thuộc lòng mẫu.
  - `NON_EXACT_OVERLAP` (120 ca: `MM` 24 ca, `RB` 24 ca, `TB` 36 ca, `WB` 36 ca): kiểm tra tổng quát hóa ngoài phân phối trùng lặp chính xác.
- **Lát cắt theo họ tác vụ**: Báo cáo chi tiết 6 tác vụ riêng biệt (`CP, DB, MM, RB, TB, WB`).
- **Phân loại taxonomy ngõ ra không hợp lệ**: Tuân thủ bảng phân loại 7 nhóm: `NO_ANSWER`, `MULTIPLE_CHOICES`, `EXTRA_PROSE`, `MALFORMED_OPTION`, `TRUNCATED_GENERATION`, `PARSER_MISMATCH`, `OTHER`.

---

## 5. Tiêu Chuẩn Nghiệm Thu Preflight Trước GPU

Tệp preflight bắt buộc đạt trạng thái `PASS` cho cả 5 nhóm tiêu chuẩn:
1. `G_PREFLIGHT_CODE`: 0 undefined names, compileall PASS, unit tests PASS.
2. `G_DATA`: Manifest hash khớp `40b8ac...`, evaluation case IDs hash khớp `d11007...`.
3. `G_MODEL`: Revision ghim khớp `7ae557...`, cấu hình lượng tử NF4 đạt chuẩn.
4. `G_METHOD`: Objective implementation khớp B1, target modules khớp B3 (`gate_proj`, `up_proj`, `down_proj`), zero attention LoRA, 829.440 trainable params.
5. `G_EVALUATION`: Evaluator code sha, prompt hash, decoding parameters khớp B1/B2/B3.

Nếu phát hiện bất kỳ độ lệch (drift) nào: **`GPU_DISPATCH = BLOCKED`**.
