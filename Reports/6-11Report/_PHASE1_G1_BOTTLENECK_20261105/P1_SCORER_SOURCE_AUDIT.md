# Báo Cáo Kiểm Toán Mã Nguồn Bộ Đánh Giá và Cơ Chế Chấm Điểm (P1 Scorer & Source Audit)

**Ngày lập:** 05/11/2026  
**Dự án:** `long-range-temporal-alignment`  
**Chủ thể thẩm quyền khoa học:** Người dùng (Chủ nhiệm đề tài) + ChatGPT (Cố vấn phương pháp luận)  
**Đơn vị thực thi:** Antigravity (Executor / Auditor / Documentation Maintainer)  
**Tệp mã nguồn kiểm toán:** `scripts/run_stage03_targeted_arms4_5_5epochs_modal.py`  
**Mục tiêu trọng tâm:** Định vị nguyên nhân thắt nút cổ chai (Bottleneck Localization) của `GOAL 1`

---

## 1. Tóm Lược Kết Quả Kiểm Toán (Executive Summary)

Kiểm toán pháp y mã nguồn trên tệp thực thi `scripts/run_stage03_targeted_arms4_5_5epochs_modal.py` trên môi trường CPU đã phát hiện **2 lỗi cốt lõi mang tính hệ thống (Critical & High Severity Bugs)** cùng **3 rủi ro phương pháp luận** giải thích hoàn toàn lý do tại sao cơ chế điều hòa thời gian (Temporal Conditioning) không chuyển hóa thành sự cải thiện độ chính xác câu trả lời đóng trên tập kiểm chuẩn:

1. **Lỗi Triệt Tiêu Tín Hiệu Thời Gian do Căn Trái Đệm (Left-Padding Seam Injection Annihilation - Critical)**:
   - Trong quá trình đánh giá, mã nguồn đặt `tokenizer.padding_side = "left"` nhưng lại tiêm vector 40D vào chỉ số tuyệt đối `token_idx=14`.
   - Với các chuỗi ngắn hơn độ dài tối đa của batch, token thứ 14 rơi trực tiếp vào **token đệm (Padding Token)** với `attention_mask = 0`. Causal attention của Transformer hoàn toàn triệt tiêu (zero-out) vector thời gian này.
   - Với các chuỗi có độ dài đệm khác nhau, vị trí tiêm bị dịch chuyển hỗn loạn, khiến bộ đánh giá **hoàn toàn mất tính bất biến theo batch (Batch Non-Invariant)**.
2. **Lỗi Lệch Mẫu Nhắc Giữa Các Dạng Bài (Template Token Index Mismatch - High)**:
   - Mẫu câu hỏi MCQA (DB, TB, WB, MM, RB) có tiền tố `"Question: Which option is correct for the question: In <YEAR>..."` (mốc năm nằm tại token 12..15).
   - Mẫu câu hỏi Quan hệ (Rel) có tiền tố `"Question: In <YEAR>, <ENTITY>..."` (mốc năm nằm tại token 4..7; token 14 là tên thực thể `<ENTITY>`).
   - Kết quả: Đối với 100% tập Rel (852 câu), vector thời gian bị tiêm vào **tên thực thể** thay vì mốc năm.
3. **Lỗi Giả Định Phản Đối Xứng Trong Cặp Phản Thực Nghiệm CF V2 (CF V2 Tie Inversion Bug - High)**:
   - Mã nguồn tự động gán nhãn đảo `gold_flipped = "No" if gold_orig == "yes" else "Yes"`.
   - Khi hai mốc năm có giá trị chỉ số bằng nhau ($V(Y_1) = V(Y_2)$), cả hai câu hỏi xuôi và ngược đều mang giá trị chân lý `No`. Quy tắc đảo nhãn mù quáng dẫn đến việc ép nhãn sai cho các trường hợp hòa (ties).
4. **Lỗi Thiên Vị Chữ Cái do Không Chuẩn Hóa Logit Tiền Tố (Unnormalized Prior Bias - Medium)**:
   - Hàm trích xuất gộp cả 4 mã token (`a`, ` a`, `A`, ` A`) và lấy `max()`. Token 264 (` a`) là mạo từ tiếng Anh có tần suất tiên nghiệm khổng lồ trong pretraining, áp đảo các chữ cái khác khi lấy logit thô không chuẩn hóa.
5. **Ranh Giới Bằng Chứng Lịch Sử (Historical Logit Boundary - Medium)**:
   - Tệp kết quả gốc chỉ lưu trữ `is_correct`, `is_strict`, `debiased_score`, hoàn toàn **không lưu trữ logit thô**. Việc so sánh đối chứng Replay chỉ có thể thực hiện trên mức độ chính xác từng ca, không thể đối chiếu logit lịch sử.

---

## 2. Bảng Phân Loại Chi Tiết Các Khiếm Khuyết Mã Nguồn (Detailed Defect Matrix)

| STT | Vấn đề / Lỗi | Vị trí mã nguồn | Hiện tượng quan sát | Hành vi mong đợi | Mức độ nghiêm trọng | Đề xuất giải pháp | Cần GPU? |
| :---: | :--- | :--- | :--- | :--- | :---: | :--- | :---: |
| **B1** | **Left-Padding Seam Injection Annihilation** | `scripts/run_stage03_targeted_arms4_5_5epochs_modal.py`, dòng 593, 680, 696 | Khi `padding_side="left"`, `token_idx=14` là chỉ số tuyệt đối từ đầu tensor. Nếu `pad_len >= 15`, tiêm vào token đệm (mask=0). Nếu `pad_len < 15`, tiêm vào vị trí chữ ngẫu nhiên. Sai số tensor giữa BS=1 và BS=2 là $1.171292$. | Tiêm tương đối vào đúng token mốc năm dựa trên offset mapping (`token_idx = year_span_start`), hoặc đánh giá ở `batch_size=1` không đệm. | **CRITICAL** | Ban hành giao thức `S1_AUDITED` với vị trí tiêm căn theo mốc năm thực tế. | Không (đã chứng minh bằng CPU) |
| **B2** | **Template Token Index Mismatch (Rel vs MCQA)** | `scripts/run_stage03_targeted_arms4_5_5epochs_modal.py`, dòng 696, 763, 772, 820, 830 | `token_idx=14` cố định cho mọi dạng bài. Ở bài Rel, mốc năm nằm ở token 4..7, token 14 là thực thể (ví dụ `' Norway'`). Ở bài MCQA, mốc năm ở token 12..15. | Vị trí tiêm phải được dò tìm động dựa trên tokenizer offset mapping cho từng chuỗi prompt cụ thể. | **HIGH** | Viết hàm `locate_year_token_indices()` tìm đúng vị trí năm trong chuỗi. | Không |
| **B3** | **CF V2 Anti-Symmetry Tie Assumption** | `scripts/run_stage03_targeted_arms4_5_5epochs_modal.py`, dòng 799 | Gán nhãn `gold_flipped = "No" if gold_orig == "yes" else "Yes"`. Bỏ qua trường hợp $V(Y_1) = V(Y_2)$ (cả hai chiều đều là `No`). | Kiểm tra giá trị cơ sở hoặc chỉ đưa vào CF V2 các cặp có giá trị khác biệt được xác thực ($V(Y_1) \neq V(Y_2)$). | **HIGH** | Lọc bỏ hoặc xử lý đúng logic trường hợp hòa (equality handling). | Không |
| **B4** | **Unnormalized Prior Logit Extraction** | `scripts/run_stage03_targeted_arms4_5_5epochs_modal.py`, dòng 595-618 | Gộp cả 4 token `a`, ` a`, `A`, ` A` và lấy max logit. Token 264 (` a`) có tần suất unigram áp đảo khiến mô hình luôn thiên vị đáp án A (488/624 ca). | Sau prompt kết thúc bằng `\nAnswer:`, token tiếp theo chuẩn là ` A` (id=362), ` B` (id=425), ` C` (id=356), ` D` (id=422). Hoặc chuẩn hóa theo conditional log-likelihood. | **MEDIUM** | Chuẩn hóa danh sách token mục tiêu, loại bỏ mạo từ `' a'`. | Không |
| **B5** | **Thiếu Logit Thô Trong Kết Quả Cũ** | Các tệp `ARM_*_RESULTS.json` | Tệp JSON cũ chỉ lưu giá trị đúng/sai nhị phân, không lưu vector logit thô của từng vòng xoay phương án. | Thừa nhận giới hạn: Không tuyên bố "trùng khớp logit với lịch sử", chỉ so sánh khớp ca (`case_results`). | **MEDIUM** | Giới hạn tiêu chí Replay ở mức `PREDICTION_REPLAY_MATCH`. | Không |

---

## 3. Phân Tích Cơ Chế Gốc (Root-Cause Analysis)

### 3.1. Tại sao cơ chế tiêm Seam bị triệt tiêu ở vòng đánh giá?
Đoạn mã đánh giá chính thức (dòng 593-698):
```python
tokenizer.padding_side = "left"  # dòng 593
...
inputs = tokenizer(formatted, return_tensors="pt", padding=True, truncation=True, max_length=256)
inputs_embeds = inject_seam_prefill_safe(model, seam_projection, inputs["input_ids"], v40_eval, token_idx=14)
transformer_out = model.model(inputs_embeds=inputs_embeds, attention_mask=inputs["attention_mask"])
```
Trong hàm `inject_seam_prefill_safe`:
```python
delta = torch.zeros_like(inputs_embeds)
delta[:, token_idx, :] = inj  # token_idx = 14 cố định
inputs_embeds = inputs_embeds + delta
```
Khi mã hóa một batch gồm câu ngắn (34 tokens) và câu dài (63 tokens):
- Độ dài đệm của câu ngắn: $\text{pad\_len} = 63 - 34 = 29$ tokens.
- Các token tại vị trí $0 \dots 28$ là `<|endoftext|>` với $\text{attention\_mask} = 0$.
- Vị trí $\text{token\_idx} = 14$ nằm hoàn toàn trong vùng đệm này!
- Vì cơ chế self-attention có mặt nạ chú ý:
  $$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}} + M\right)V$$
  Với $M_{ij} = -\infty$ tại các vị trí đệm, mọi giá trị được tiêm vào $\text{inputs\_embeds}[:, 14, :]$ bị nhân với trọng số 0 sau softmax, hoàn toàn không lan truyền được vào các lớp Transformer tiếp theo.
- Đây là bằng chứng toán học và thực nghiệm chứng minh vì sao mô hình học 5 epochs nhưng khi đánh giá theo batch thì tín hiệu thời gian dường như biến mất.

### 3.2. Sự khác biệt giữa Quá trình Huấn luyện và Đánh giá (Train/Eval Discrepancy)
- Trong huấn luyện (`collate_fn`, dòng 408-418): Mã nguồn dùng `tokenizer.padding_side = "right"`, đệm nằm ở cuối chuỗi. Do đó vị trí 14 nằm trong văn bản (dù vẫn bị lệch giữa MCQA và Rel).
- Trong đánh giá (dòng 593): Mã nguồn đổi sang `tokenizer.padding_side = "left"` để lấy logit ở token cuối `[:, -1, :]`. Việc đổi cơ chế đệm mà không đổi cơ chế tiêm đã trực tiếp phá hủy tính toàn vẹn của tín hiệu thời gian.
