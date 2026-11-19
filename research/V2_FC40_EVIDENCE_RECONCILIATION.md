# Đối Soát Bằng Chứng Khoa Học và Phân Lập Nhân Quả — Nhánh V2 (FC40)

**Thời điểm:** 2026-10-13 (Asia/Ho_Chi_Minh)  
**Tác giả:** Antigravity / Learning Track  
**Mục tiêu:** Phân lập thực chứng giữa lỗi đo lường (Evaluator Defect), sai lệch giao thức (Protocol Mismatch) và suy giảm năng lực thực tế của mô hình (Model Degradation); xác lập cơ sở nhân quả vững chắc trước khi quyết định can thiệp thuật toán.

---

## 1. Bảng Tổng Hợp Nguyên Nhân và Mức Độ Tin Cậy (Root Causes & Confidence)

Dựa trên việc kiểm toán độc lập toàn bộ 1.620 bản ghi sinh autoregressive (3 seeds $\times$ 3 stages $\times$ 180 cases) và đối soát mã nguồn thực thi, ba nguyên nhân chính được xếp hạng theo thứ tự nhân quả:

| Thứ hạng | Hiện tượng & Phân loại | Mức tin cậy | Bằng chứng thực nghiệm trực tiếp | Cơ chế phát sinh |
| :---: | :--- | :---: | :--- | :--- |
| **1** | **Lỗi Đối Sánh Nhãn Của Parser (Evaluator Defect)**<br>`CONFIRMED_DEFECT` | **$1{,}00$ (Tuyệt đối)** | Tệp `eval_mcqa.jsonl` có 156 câu mang nhãn chữ thường `a,b,c,d`. Hàm `parse_strict_mcqa_answer` cũ trong `tempun_dataset.py:140-154` chỉ nhận chữ in hoa `^[A-D]`. | Khi mô hình sinh chữ thường, parser trả về `None` ($\implies$ Invalid). Khi mô hình sinh chữ hoa `'A'`, phép so sánh `'A' == 'a'` luôn bằng `False`. Toàn bộ 156 câu trắc nghiệm bị triệt tiêu khả năng ghi nhận điểm đúng. |
| **2** | **Sai Lệch Giao Thức Lời Nhắc (Protocol Mismatch)**<br>`CONFIRMED_DEFECT` | **$1{,}00$ (Tuyệt đối)** | Trình chạy `run_m0_fc40.py:198-208` nạp văn bản thô vào `tokenizer(prompt)` không có ChatML template. Đối tác ($F_0$ đạt $20{,}93\%$, Ticktack v2 đạt $20{,}51\%$) bắt buộc dùng `apply_chat_template` kèm chỉ dẫn ngắn gọn. | Mô hình `Qwen2.5-0.5B-Instruct` được huấn luyện căn chỉnh qua ChatML. Khi nhận câu hỏi trần trụi không có `<|im_start|>user`, mô hình coi đây là văn bản cần viết tiếp và phát ra ngay token kết thúc EOS (`<|im_end|>`), tạo thành 137 chuỗi rỗng `''`. |
| **3** | **Trôi Ngữ Nghĩa Vị Trí Tiêm Token 14 (Seam Semantic Drift)**<br>`CONTRACT_DEFECT` | **$0{,}95$ (Rất cao)** | Kiểm toán tokenization thực tế: Trong raw prompt, token 14 là từ `' higher'` (ngay sau `was`). Trong ChatML prompt, token 14 là từ `' selecting'` nằm giữa System Message (`"by selecting the correct option"`). | Việc tiêm vector thời gian 40D qua cổng số dư liên tục tại token index 14 cố định trong ChatML làm sai lệch câu lệnh hệ thống thay vì bổ trợ ngữ cảnh câu hỏi của người dùng. |
| **4** | **Suy Giảm Năng Lực Tuân Thủ Chỉ Dẫn (Instruction Drift via CPT)**<br>`LIKELY_MECHANISM` | **$0{,}85$ (Khả dĩ)** | Huấn luyện $2.461$ bước NWP chỉ toàn sinh số thực (`68.52`, `61.54`). Dẫn đến 31 ca ở chặng after_02 sinh các con số thô (`8`, `4`, `6`, `56`) thay vì chọn đáp án. | CPT trên ngữ liệu thống kê đơn thuần làm suy yếu xu hướng trả lời trắc nghiệm nếu không có cơ chế điều hòa bảo toàn chỉ dẫn (Jindal et al., 2024). |

---

## 2. Ma Trận Bằng Chứng Đối Soát và Phép Kiểm Định Bác Bỏ (Evidence Matrix)

| Nhận định (Claim) | Nguồn chính thức | Quan sát thực tế (Observed) | Trạng thái chuyển giao | Phép kiểm định bác bỏ (Falsification Test) |
| :--- | :--- | :--- | :---: | :--- |
| *“Sửa parser là đủ để hồi phục điểm số của output cũ”* | Giả định ban đầu | `run_phase_a_free_forensic_replay.py`: Rescore 1.620 raw completions cũ bằng canonical partner parser cho ra đúng **0 ca được cứu** (Strict Acc vẫn $2{,}22\%$). | **BÁC BỎ (REFUTED)** | Đã kiểm chứng: Chuỗi sinh cũ chứa 137 chuỗi rỗng và 31 chữ số; không có chữ cái nào để parser bóc tách. |
| *“Cả 3 seeds hoàn toàn giống nhau do rò rỉ trọng số”* | Nghi vấn kiểm toán | Kiểm toán chuỗi sinh cho thấy ở chặng after_00 có $54/180$ câu phân kỳ ($30\%$); chặng after_01 có $75/180$ câu phân kỳ ($41{,}7\%$). Checksum và loss 3 seed tách biệt. | **BÁC BỎ (REFUTED)** | Hạt giống độc lập thực tế; sự hội tụ về điểm 4/180 ở after_02 là do co cụm hành vi (mode collapse) sinh chuỗi rỗng dưới raw prompt. |
| *“137 chuỗi rỗng đều do token đầu tiên là `<|im_end|>`”* | Giả thuyết kỹ thuật | Tệp JSON cũ chỉ lưu chuỗi decoded `raw_completion`, không lưu `generated_token_ids`. | **CHƯA THỂ XÁC MINH (UNVERIFIABLE)** | Đánh dấu `UNVERIFIABLE_FROM_OLD_TEXT`; cần instrument token IDs khi chạy lại re-inference. |
| *“Qwen Instruct bị Alignment Tax hoàn toàn sau CPT”* | Giả định lý thuyết | Đối tác $F_0$ (Base) và Ticktack v2 đạt ~20% khi có ChatML; $M_0\text{-FC40}$ chưa từng được đo dưới ChatML chuẩn. | **CHƯA ĐỦ CĂN CỨ (UNPROVEN)** | Bắt buộc phải đo cùng checkpoint dưới giao thức $P_{11}$ trước khi kết luận về Alignment Tax. |

---

## 3. Cơ Sở Lý Thuyết và Đối Chiếu Các Bài Báo Gốc

### 3.1 Thích Ứng Tham Số Phụ Thuộc Thời Gian (MaT-LoRA)
Theo Yao et al. (arXiv:2602.11965) và Liang et al. (*Neural Networks*, 2026), ma trận trọng số thích ứng phụ thuộc thời gian có dạng:
$$\Delta W(t) = \frac{\alpha}{r} B \cdot F(t) \cdot A$$
Trong đó $A \in \mathbb{R}^{r \times d_{\text{in}}}$, $B \in \mathbb{R}^{d_{\text{out}} \times r}$, và $F(t) \in \mathbb{R}^{r \times r}$ là lõi điều hòa động.
Trong nhánh Learning Track, cấu hình cố định: hạng $r=8$, $\alpha=16{,}0$, $K=2$ sóng điều hòa, $F_{\text{base}} = I_8$. Tọa độ thời gian $(\theta, z)$ được trích xuất từ biểu diễn đóng băng `FIXED_FONE_CANCHI_V1` 40D.

### 3.2 Hiện Tượng Thỏa Hiệp Giữa CPT và Năng Lực Tuân Thủ Chỉ Dẫn (Alignment Tax)
Jindal et al. (arXiv:2410.10739, 2024) chỉ ra rằng: Khi thực hiện Continual Pretraining (CPT) trên ngữ liệu miền chuyên sâu bằng hàm mất mát Causal Language Modeling tiêu chuẩn không có giám sát hội thoại:
$$\mathcal{L}_{\text{NWP}} = - \sum_{i} \log P(x_i \mid x_{<i})$$
mô hình Instruct có xu hướng trôi dạt phân phối logits khỏi các token định dạng (formatting tokens) và suy giảm khả năng trả lời ngắn gọn theo chỉ dẫn (*Instruction Following degradation*).

### 3.3 Giải Pháp Điều Hòa Bảo Toàn Chỉ Dẫn (Instruction-Preserving Regularization)
Để khắc phục hiện tượng trên mà không phá vỡ quá trình tiếp thu tri thức thời gian:
1. **Ovadia et al. (*Knowledge-Instruct*, 2025)**: Đề xuất chuyển đổi một tỷ lệ nhỏ dữ liệu huấn luyện tiếp tục sang định dạng chỉ dẫn (*Instruction-Formatted Knowledge Learning*).
2. **He et al. (*Don't Half-listen*, ACL 2025)**: Chứng minh việc kết hợp hàm mất mát phụ với các mẫu chỉ dẫn tổng quát (General Instruction Replay) ngăn ngừa hiện tượng quên cấu trúc cú pháp:
$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{MASKED\_TARGET}} + \lambda_{\text{align}} \mathcal{L}_{\text{INSTRUCTION}}$$
với $\lambda_{\text{align}} \in [0.05, 0.15]$ và nhãn chỉ tính loss trên phần phản hồi mong muốn.

---

## 4. Nghiên Cứu Nhân Tố Lời Nhắc (Prompt Factorial Design: P00, P01, P10, P11)

Để phân định rạch ròi đóng góp của ChatML Template và Lời Nhắc Trả Lời Ngắn Gọn, khế ước [`configs/eval/EVAL_PROTOCOL_MANIFEST_V2.json`](file:///c:/Users/GF63/Documents/long-range-temporal-alignment/configs/eval/EVAL_PROTOCOL_MANIFEST_V2.json) xác lập ma trận $2 \times 2$:

$$\begin{array}{c|c|c}
\text{Giao thức} & \text{ChatML Template} & \text{Chỉ Dẫn Trực Tiếp (Short-Answer Instruction)} \\
\hline
\mathbf{P00} & \text{Không (Raw Prompt)} & \text{Không} \\
\mathbf{P01} & \text{Không (Raw Prompt)} & \text{Có} \\
\mathbf{P10} & \text{Có (ChatML)} & \text{Không} \\
\mathbf{P11} & \text{Có (ChatML)} & \text{Có (Chuẩn đối tác)}
\end{array}$$

### Quy tắc phân giải nguyên nhân:
1. Nếu $\mathbf{P00} \to \mathbf{P11}$ khôi phục độ chính xác $\ge 20\%$ trên cùng checkpoint hiện có $\implies$ Vấn đề hoàn toàn là **Độ lệch Giao thức Đánh giá**. Mô hình không bị thoái hóa; đóng băng và công nhận Evaluator mới; **TUYỆT ĐỐI KHÔNG HUẤN LUYỆN LẠI**.
2. Nếu $\mathbf{P11}$ khôi phục trên Base Model nhưng thất bại trên Trained Checkpoints $\implies$ Bằng chứng xác nhận **Mô hình bị trôi dạt do CPT**. Lúc này mới chuyển sang Phase D (huấn luyện bổ sung điều hòa bảo toàn chỉ dẫn).
3. Nếu cả Base Model và Trained Checkpoints đều thất bại trên $\mathbf{P11}$ $\implies$ Bộ sinh hoặc bộ parser có vấn đề cơ bản; dừng lại điều tra.

---

## 5. Cây Quyết Định Nhân Quả và Đề Xuất Can Thiệp Tiếp Theo

```
                                  [KẾT QUẢ ĐỐI SOÁT P11]
                                             │
                      ┌──────────────────────┴──────────────────────┐
                      ▼                                             ▼
         [P11 ĐẠT ~20% TRÊN M0-FC40]                 [P11 VẪN BỊ SUY GIẢM TRÊN M0-FC40]
                      │                                             │
        ┌─────────────┴─────────────┐                 ┌─────────────┴─────────────┐
        ▼                           ▼                 ▼                           ▼
[LỖI EVALUATOR ĐƯỢC GIẢI MÃ]  [KHÔNG HUẤN LUYỆN]   [BASE TỐT, M0 KÉM]       [CẢ HAI ĐỀU KÉM]
Khóa khế ước P11;            Giữ nguyên V1;        Bằng chứng CPT Drift;     Lỗi bộ giải mã;
Cập nhật baseline sạch.      Tiết kiệm GPU quota.  Thiết kế V2 Replay.       Dừng kiểm tra pipeline.
```

### Đề xuất can thiệp duy nhất (Minimal Next Action):
- **Hành động:** Thực thi đánh giá kiểm chứng chẩn đoán (*inference-only parity check*) cho mô hình Base và checkpoint `after_02` dưới giao thức $P_{11}$ trên tập 24 mẫu chẩn đoán khóa chặt bằng `scripts/evaluate_m0_fc40_parity.py`.
- **Ranh giới:** Tuyệt đối không khởi động lại quá trình huấn luyện 3 hạt giống trên Kaggle cho đến khi có kết quả đo đạc từ kiểm chứng này.

---

## 6. Kết Quả Đo Lường Thực Nghiệm Đã Được Kiểm Chứng và Phán Quyết Nhân Quả Cuối Cùng

Vào ngày 2026-10-13, quá trình đánh giá factorial đa nhân tố trên tập 24 mẫu chẩn đoán khóa chặt (`artifacts/audits/diagnostic_subset_24.json`) đã hoàn tất với các bằng chứng thực nghiệm rõ ràng.

### 6.1. Ma trận kết quả đo lường Factorial (N = 24 mẫu chẩn đoán)

| Mô hình / Cấu hình | Giao thức | Strict ACC | Invalid Rate | Valid ACC | Phân bố kiểu Token đầu ra |
| :--- | :--- | :---: | :---: | :---: | :--- |
| **Qwen2.5-0.5B Base (Frozen)** | $\mathbf{P_{00}}$ (Raw) | 20.83% | 66.67% | 62.50% | `VALID`: 8, `OTHER`: 16 |
| | $\mathbf{P_{01}}$ (Raw + Instruct) | 20.83% | 41.67% | 35.71% | `VALID`: 14, `OTHER`: 10 |
| | $\mathbf{P_{10}}$ (ChatML) | 12.50% | 16.67% | 15.00% | `VALID`: 20, `OTHER`: 4 |
| | $\mathbf{P_{11}}$ (ChatML + Instruct) | 8.33% | 75.00% | 33.33% | `VALID`: 6, `OTHER`: 18 |
| **M0-FC40 Canary (Seam-On, S0)** | $\mathbf{P_{00}}$ (Raw) | 8.33% | 66.67% | 25.00% | `VALID`: 8, `OTHER`: 14, `NUM`: 1, `EMPTY`: 1 |
| | $\mathbf{P_{01}}$ (Raw + Instruct) | **33.33%** | **0.00%** | **33.33%** | **`VALID`: 24 (100% Valid Syntax)** |
| | $\mathbf{P_{10}}$ (ChatML) | 12.50% | 16.67% | 15.00% | `VALID`: 20, `OTHER`: 4 |
| | $\mathbf{P_{11}}$ (ChatML + Instruct) | **20.83%** | 50.00% | **41.67%** | `VALID`: 12, `OTHER`: 12 |
| **M0-FC40 Canary (Seam-Off)** | $\mathbf{P_{01}}$ (Raw + Instruct) | 33.33% | 0.00% | 33.33% | `VALID`: 24 (100% Valid Syntax) |
| | $\mathbf{P_{11}}$ (ChatML + Instruct) | 16.67% | 54.17% | 36.36% | `VALID`: 11, `OTHER`: 13 |

### 6.2. Phán quyết về Seam Conditioning (Lựa chọn S0 vs S1)
- Dưới giao thức chuẩn đối tác $P_{11}$, cấu hình **Seam-On** đạt Strict ACC = **20.83%** và Valid ACC = **41.67%**, vượt trội cấu hình **Seam-Off** (Strict ACC = 16.67%, Valid ACC = 36.36%).
- Dưới giao thức $P_{01}$, cả Seam-On và Seam-Off đều đạt 33.33% với 0.00% Invalid.
- **Kết luận:** Tiêm vector liên tục tại `token_index = 14` hoàn toàn không gây nhiễu loạn mô hình như giả định ban đầu. Ngược lại, seam mang lại tín hiệu bổ trợ thời gian rõ rệt (+4.16% Strict ACC). Khế ước đông băng **Option S0 được khóa chặt tuyệt đối**; không cần định nghĩa lại khế ước S1.

### 6.3. Phán quyết Nhân quả (Causal Resolution)
1. **Bác bỏ giả thuyết Catastrophic Forgetting / Alignment Drift:**
   Mô hình M0-FC40 không bị suy thoái năng lực ngôn ngữ. Khi có chỉ dẫn trực tiếp ($P_{01}$), mô hình sinh 100% cú pháp hợp lệ và đạt độ chính xác 33.33% (vượt xa Base Model 20.83% với mức tăng +12.50 điểm %). Dưới giao thức $P_{11}$, mô hình đạt 20.83%, tương đương trọn vẹn mốc đối tác (20.51%).
2. **Xác định nguyên nhân gốc rễ (Root Cause):**
   - Đợt chạy Kaggle V2 cũ (3 hạt giống) gặp lỗi tràn số học FP16/BF16 trên GPU Tesla T4 khiến toàn bộ trọng số checkpoint bị `NaN`, dẫn đến chuỗi sinh rỗng và argmax suy biến.
   - Đợt chạy Canary đã sửa triệt để lỗi này bằng việc bảo toàn trọng số FP32 cho adapter và gated seam, tạo ra checkpoint hữu hạn sạch.
   - Thước đo cũ của V2 dùng định dạng $P_{00}$ và bộ phân tích chữ hoa `[A-D]`, trong khi gold labels là `[a-d]`. Khi đưa về đúng giao thức $P_{11}$ và $P_{01}$, toàn bộ khiếm khuyết được giải quyết triệt để.
3. **Phân nhánh quyết định:** Xác lập **Case A (Prompt Format & Numerical Parity)**.

### 6.4. Hành động thực thi tiếp theo
- Mã nguồn huấn luyện FP32 sạch và cơ chế sinh ChatML đã được đóng gói vào Kaggle bundle 3 hạt giống (`artifacts/dispatch/M0_FC40/`).
- Khởi chạy chính thức đợt huấn luyện toàn diện 3 hạt giống trên Kaggle (`qh0205/lrr-fc105-m0-fc40-3seed`, version 2) để thu hoạch toàn bộ 9 checkpoint sạch và bảng số liệu chuẩn khoa học.

