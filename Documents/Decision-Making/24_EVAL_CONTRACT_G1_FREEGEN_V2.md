---
loại: "quyết-định"
mã: "DECISION-EVAL-CONTRACT-G1-FREEGEN-V2-114"
dự_án: "long-range-temporal-alignment"
trạng_thái: "đang-dùng"
chuẩn_hiện_hành: true
thay_thế: []
bị_thay_thế_bởi: []
tham_chiếu: ["23_GOAL_LADDER_AND_ACCEPTANCE_20261105", "PROPOSED_GATE_REFINEMENTS", "18_STAGE03_MECHANISM_BEFORE_SCALE"]
được_xác_thực_bởi: []
cập_nhật_lần_cuối: "2026-11-07"
---

# Khế Ước Đánh Giá G1 Khắc Phục Tiêm Thời Gian và Sinh Tự Do Không Ràng Buộc (EVAL_CONTRACT_G1_FREEGEN_V2)

## Tóm lược Quyết định Quản trị (Executive Summary)

Văn bản này ban hành **Khế ước Đánh giá Chuẩn hóa V2 (Evaluation Contract V2)** cho Mục tiêu Goal 1 (`G1 = YEAR-only closed-answer temporal QA`) thuộc đề tài `long-range-temporal-alignment`. Khế ước này kế thừa và hoàn thiện các chẩn đoán tại `Reports/6-11Report/DIAGNOSIS.json`, thiết lập quy chuẩn kỹ thuật cho:
1. **Sửa chữa cơ chế tiêm vector thời gian (Fix A - Temporal Injection Repair):** Chuyển từ chỉ mục cứng `token_idx=14` sang mặt nạ điều biến động có căn cứ siêu dữ liệu (`YEAR_SPAN` và `SEQUENCE_VALID`), triệt tiêu hoàn toàn lỗi đệm trái (`left-padding`) và va chạm thực thể.
2. **Đánh giá sinh tự do không ràng buộc (Fix B - Unconstrained Free Generation & Deterministic Parsing):** Tuyệt đối cấm can thiệp giải mã ràng buộc (constrained decoding / logit argmax trên tập nhãn hạn chế). Mô hình thực hiện sinh tự hồi quy tham lam trên toàn bộ từ vựng (full vocabulary), sau đó một bộ phân tích cú pháp xác định (deterministic post-hoc parser) sẽ trích xuất nhãn và phân loại lỗi theo bảng phân loại chuẩn tắc.
3. **Bảo toàn lịch sử khoa học:** Khế ước này không viết đè các báo cáo cũ. Các số liệu $S_0$ và $S_1$ được lưu giữ nguyên trạng như mốc tham chiếu chẩn đoán kế thừa (legacy diagnostic references).

---

## 1. Ranh Giới Phạm Vi và Cấm Đoán Tuyệt Đối (Hard Boundaries)

1. **Cấm giải mã ràng buộc trong đánh giá chính (No Constrained Decoding):**
   Tuyệt đối không áp dụng các kỹ thuật:
   - Giới hạn từ vựng chọn lọc (restricted token vocabularies / logit masking trên A/B/C/D);
   - Hàm lọc tiền tố (`prefix_allowed_tokens_fn`);
   - Ràng buộc ngữ pháp ép buộc (`grammar-constrained decoding` / `force_words_ids`);
   - So sánh xếp hạng khả năng xuất hiện của nhãn (candidate likelihood ranking).
2. **Ranh giới tập dữ liệu:**
   - Tập `data/tempun_full_clean/train_104k_clean.jsonl.gz` ($104.130$ mẫu) là tác vụ sinh số thực mở `NWP` (Next Word Prediction), không chứa câu hỏi trắc nghiệm MCQA.
   - Tập `data/tempun_full_clean/eval_5292_clean.jsonl.gz` ($5.292$ mẫu) là tập phát triển (**Development Benchmark**), không được tuyên bố là tập kiểm tra độc lập nguyên vẹn (Untouched Test).
3. **Không tự suy diễn năng lực:**
   Việc bộ đo cũ gặp lỗi không đồng nghĩa checkpoint 5 epochs chưa học được gì, và việc chuyển sang sinh tự do có thể làm tăng tỷ lệ đầu ra không hợp lệ (Invalid Rate) trên mô hình nền Qwen2.5-3B Base. Đây là kết quả thực nghiệm khách quan cần được ghi nhận trung thực.

---

## 2. Đặc Tả Kỹ Thuật Khắc Phục Tiêm Thời Gian (Fix A)

### 2.1. Công thức toán học tổng quát
Biểu diễn embedding đầu vào sau điều biến thời gian:
$$E'_{b, j, :} = E_{b, j, :} + M_{b, j} \cdot \Delta_b$$

Trong đó:
- $E_{b, j, :} \in \mathbb{R}^D$ là vector embedding từ vựng gốc tại vị trí token $j$ của mẫu $b$;
- $\Delta_b = \text{gate}(\alpha) \cdot \text{LayerNorm}(W v_{40}(t_b) + b) \in \mathbb{R}^D$ là biểu diễn thời gian 40 chiều qua tầng Seam Projection;
- $M \in \mathbb{R}^{B \times S}$ là ma trận mặt nạ nhị phân hoặc phân bổ trọng số ($M_{b, j} = 0$ trên toàn bộ các token đệm PAD hoặc token đặc biệt).

### 2.2. Bốn chính sách điều biến thời gian (Temporal Policies)

| Mã Chính Sách | Cơ Chế Phân Bổ $M_{b, j}$ | Mục Đích Khoa Học |
| :--- | :--- | :--- |
| `LEGACY_FIXED14` | $M_{b, 14} = 1.0$ (nếu $S > 14$ và $\text{mask}_{b, 14} = 1$), ngược lại bằng $0$. | Chỉ dùng để tái hiện lịch sử kiểm toán; tuyệt đối không dùng làm mặc định. |
| `YEAR_SPAN` | $M_{b, j} = \frac{1}{\text{len}(\text{span})}$ trên toàn bộ các subtoken của mốc năm neo; $0$ ở các vị trí khác. | **Chính sách trọng tâm**: Bảo toàn tổng năng lượng Seam ($1.0$), tiêm chính xác vào mốc năm. |
| `SEQUENCE_VALID` | $M_{b, j} = 1.0$ cho mọi token hợp lệ trong câu hỏi ($\text{mask}_{b, j} = 1$). | Kiểm chứng giả thuyết tiêm toàn chuỗi (lấy cảm hứng từ Ticktack Eq. 5). |
| `SEAM_OFF` | $M_{b, j} = 0.0$ cho toàn bộ chuỗi. | Đối chứng loại bỏ Seam (vẫn duy trì ma trận thời gian động MaT-LoRA ở Arm 5). |

### 2.3. Quy tắc định vị token mốc năm (Temporal Locator Contract)
- Bắt buộc sử dụng Fast Tokenizer (`tokenizer.is_fast == True`) với `return_offsets_mapping=True`.
- Tìm kiếm mốc năm neo (`anchor_year`) dựa trên ranh giới từ nguyên vẹn (`\b<YEAR>\b`) trong câu hỏi đã định dạng chuẩn.
- Đối chiếu vị trí ký tự $[c_{\text{start}}, c_{\text{end}})$ sang dải token $[t_{\text{start}}, t_{\text{end}})$.
- **Nguyên tắc fail-closed:** Nếu mốc năm bị thiếu, bị cắt cụt do độ dài hoặc không thể ánh xạ, hệ thống trả về lỗi cấu trúc `StructuredLocatorError`, tuyệt đối không tự ý rơi về vị trí 14 hay thay bằng năm khác.
- Đối với câu hỏi so sánh thời gian (`Rel`) chứa hai mốc năm, siêu dữ liệu chỉ định rõ mốc năm nào là năm neo (`anchor_year`) và ghi nhận mốc còn lại là năm thứ cấp (`secondary_mention`).

---

## 3. Đặc Tả Kỹ Thuật Đánh Giá Sinh Tự Do và Bộ Phân Tích Cú Pháp (Fix B)

### 3.1. Cơ chế sinh tự do không ràng buộc (Unconstrained Generation)
- **Phương thức:** Sinh tự hồi quy tham lam tất định (`do_sample=False`, `num_beams=1`, `temperature=0.0`).
- **Prefill:** Chèn trực tiếp tensor embedding đã điều biến thời gian $E'$ vào lượt chuyển tiếp đầu tiên, khởi tạo bộ nhớ đệm `past_key_values`.
- **Duy trì ngữ cảnh thời gian Arm 5:** Tại mỗi bước sinh tiếp theo, ma trận thời gian $\theta(t)$ và $z(t)$ của MaT-LoRA được duy trì liên tục trong toàn bộ các khối MLP.
- **Dừng sinh:** Khi gặp token `<|endoftext|>` (EOS), chuỗi dừng cấu hình (ví dụ `\n`), hoặc đạt trần `max_new_tokens = 16`.
- **Trích xuất:** Chỉ thu thập các token ID thuộc phần tiếp nối sinh mới (continuation tokens), giải mã độc lập không cắt ghép chuỗi thô.

### 3.2. Bộ phân tích cú pháp xác định (Deterministic Answer Parser)
Parser hoạt động thuần túy bằng logic biểu thức chính quy (pure Python), hoàn toàn không gọi mô hình và **tuyệt đối không được tiếp cận nhãn đúng (gold label)**.

#### Hợp đồng chính: `LABEL_ONLY`
- **MCQA:**
  - Nhận diện ký tự đơn lập: `A`, `B`, `C`, `D`, `(A)`, `[B]`, `A.`, `B:`.
  - Nhận diện tiền tố rõ ràng: `Answer: A`, `Option (B)`, `Choice C`.
  - **Triệt tiêu mạo từ tiếng Anh:** Chuỗi văn xuôi bắt đầu bằng mạo từ `'a'` (ví dụ: `"a significant historical event..."`) bắt buộc phải bị từ chối và gán nhãn `NO_RECOGNIZABLE_LABEL`, tuyệt đối không nhận vơ là phương án A.
- **Rel:**
  - Nhận diện độc lập: `Yes`, `No`, `Answer: Yes`, `Answer is No`.
- **Phân loại lỗi hợp lệ (Taxonomy of Invalid Reasons):**
  - `EMPTY_OUTPUT`: Chuỗi sinh rỗng;
  - `LENGTH_EXHAUSTION`: Hết ngân sách token mà chưa xuất hiện nhãn;
  - `NO_RECOGNIZABLE_LABEL`: Văn bản tự do không chứa ký hiệu phương án hợp lệ;
  - `MULTIPLE_CONTRADICTORY_LABELS`: Chứa nhiều phương án xung đột (ví dụ `"Both A and C"`, `"A or B"`);
  - `HALLUCINATED_OPTION`: Xuất hiện phương án ngoài phạm vi A..D (ví dụ `"Option E"`);
  - `REFUSAL_OR_UNSUPPORTED`: Mô hình từ chối trả lời (ví dụ `"I cannot answer..."`).

#### Hợp đồng chẩn đoán phụ: `UNIQUE_EXACT_NUMERIC_VALUE`
- Chỉ kích hoạt ở kênh chẩn đoán phụ khi các phương án là số thực (ví dụ `(a) 0.84 (b) 0.89`).
- Chuẩn hóa chuỗi số sinh ra bằng thư viện `Decimal` chính xác tuyệt đối.
- Nếu giá trị số khớp duy nhất với một phương án $\rightarrow$ gán nhãn phương án đó. Nếu không khớp hoặc khớp nhiều phương án $\rightarrow$ gán `INVALID`.
- Báo cáo độc lập, tuyệt đối không gộp lẫn hoặc chọn điểm cao hơn để thay thế cho hợp đồng chính `LABEL_ONLY`.

---

## 4. Hệ Thống Thước Đo Nghiệm Thu G1 (Evaluation Metrics)

Mọi đánh giá nghiệm thu bắt buộc phải bảo toàn mẫu số gốc $N$ (không loại bỏ các ca sinh lỗi ra khỏi mẫu số tính tỷ lệ):

1. **Canonical Accuracy:** Độ chính xác trên cấu hình câu hỏi nguyên bản (hoán vị 0).
2. **Debiased Mean Accuracy:** Điểm trung bình cộng độ chính xác qua toàn bộ các cấu hình hoán vị phương án (4 vòng hoán vị cho MCQA, 2 vòng nghịch đảo cho Rel).
3. **Strict Accuracy:** Tỷ lệ câu hỏi trả lời chính xác trên **100% các vòng hoán vị** ($4/4$ với MCQA và $2/2$ với Rel).
4. **Invalid Output Rate:** Tỷ lệ phần trăm các câu trả lời bị parser bác bỏ (`is_valid == False`) trên tổng số lượt sinh.
5. **Task Stratification:** Báo cáo phân rã độc lập cho 6 nhóm tác vụ: `DB`, `TB`, `WB`, `MM`, `RB`, `Rel`.
6. **Khóa định danh tổng hợp:** Mỗi bản ghi đánh giá được định danh duy nhất bằng khóa bộ ba `(global_idx, example_id, rotation_id)`.

---

## 5. Kế Hoạch Chuyển Tiếp và Trạng Thái Cổng Kiểm Định

- **Trạng thái Khế ước:** `APPROVED_ACTIVE_CONTRACT`.
- **Mã định danh gói nguồn:** `src.temporal_alignment.stage03_eval`.
- **Kiểm định mã nguồn:** Toàn bộ logic tiêm thời gian, định vị token và phân tích cú pháp đã được kiểm chứng bằng bộ kiểm thử tự động `tests/test_stage03_g1_freegen_v2.py` đạt trạng thái **22/22 PASS** trên môi trường CPU.
- **Trạng thái thực thi mô hình thực tế:** `CPU_READY_GPU_UNVERIFIED`. Mọi lượt chạy trên GPU A10G với mô hình Qwen2.5-3B và các checkpoint 5 epochs cần được cấp quyền độc lập với trần ngân sách rõ ràng trước khi khởi chạy.

---

## 6. Tu Chính Kỹ Thuật Red-Team (Tu chính ngày 2026-11-08)

Đợt kiểm toán độc lập ngày 2026-11-08 xác nhận và khắc phục các điểm nghẽn kỹ thuật:
1. **Khóa chế độ cổng chiếu thời gian tuyến tính (`gate_mode="linear"`):** Đồng bộ hóa chính xác với công thức tại script huấn luyện gốc `run_stage03_targeted_arms4_5_5epochs_modal.py` ($\alpha \cdot \text{LayerNorm}(W v_{40} + b)$ thay vì mặc định `sigmoid`), triệt tiêu độ lệch biên độ $\sim 76\times$ và bảo toàn dấu âm của $\alpha$ ($\approx -0.0066$).
2. **Kiểm định nghiêm ngặt trọng số checkpoint (`load_checkpoint_weights`):** Bắt buộc đối soát đủ 216 nhóm trọng số LoRA, 4 tensor Seam và 5 tensor MaT-LoRA core (cho Arm 5); chặn đứng hoàn toàn việc báo thành công khi danh sách rỗng hoặc tensor chứa NaN/Inf.
3. **Thực thi CLI và Dry-run hoàn chỉnh:** Lệnh `smoke` thực thi suy luận mock đầy đủ với cấu trúc record/summary chuẩn; lệnh `eval --dry-run` kiểm định toàn diện tham số, dữ liệu và kiểm tra cấu trúc checkpoint mà không tốn chi phí GPU.
4. **Kiểm chứng bộ nhớ đệm Qwen2 (`DynamicCache`):** Thực nghiệm độc lập trên mô hình Qwen2 từ cấu hình nhỏ xác nhận cơ chế sinh tăng dần với `past_key_values` đạt độ khớp chính xác $100\%$ so với phương pháp sinh không đệm trên toàn chuỗi.
