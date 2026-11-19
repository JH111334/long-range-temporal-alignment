# TỔNG HỢP NGHIÊN CỨU SÂU VÀ KẾ HOẠCH THỰC NGHIỆM ĐÃ KHÓA
## Họ Thực Nghiệm: FCLEARN_105 — Starting Anchor: Periodic Dual-Cycle MaT-LoRA (M0-FC40)
### Thời điểm chốt khoa học: 12/10/2026 18:45 (+07:00)

---

## 1. Ranh Giới Nghiên Cứu Bất Biến (Absolute Invariants)

1. **Kiến trúc xuất phát cố định (Starting Architecture):**
   - **Periodic Dual-Cycle MaT-LoRA** của đối tác (thích ứng trên giao diện 40D độc lập của Learning Track).
   - Bãi bỏ hoàn toàn việc định tuyến `FC0 (Standard MLP-LoRA) vs FC1 (MaT-LoRA)`. Standard MLP-LoRA chỉ giữ vai trò mốc đối chứng lịch sử (Week 1 $B_3 = C_0$) hoặc kiểm soát dự phòng khi người phản biện yêu cầu.
2. **Khóa cứng mô hình nền tảng:**
   - `Qwen/Qwen2.5-0.5B-Instruct` tại commit `7ae557604adf67be50417f59c2c2f167def9a775`. Trọng số $W_0$ được đóng băng hoàn toàn ($100\%$ frozen). Tuyệt đối không mở rộng dung lượng lên 1.5B hoặc 3B.
3. **Khóa cứng biểu diễn đầu vào của Learning Track:**
   - Khế ước `FIXED_FONE_CANCHI_V1`: Ghép nối FoNE số học 32D và Can-Chi 8D thành vector 40 chiều bất biến:
     $$\mathbf{z} = [\mathbf{e}_{\text{FoNE}} \,\|\, \mathbf{e}_{\text{CanChi}}] \in \mathbb{R}^{40}$$
   - Giao diện tiêm: Safe Conditioning Seam tại token prefill thứ 14 (`token_index=14`) qua cổng số dư liên tục với hệ số khởi tạo $\alpha_0 = 0{,}0$ (`Linear(40, 896)` + `LayerNorm(896)`). Virtual tokens: 0.
   - SHA256 Golden Vectors: `6c35fc55cbdfd571c03cd6bf56d3d4b65b367730b8bb3bc7d89c9dc82f78eddc` ([feature_manifest.json](file:///c:/Users/GF63/Documents/long-range-temporal-alignment/artifacts/releases/FCLEARN_105/feature_manifest.json)).
4. **Phân định quyền sở hữu:**
   - Đối tác sở hữu đa tạp hình học thời gian (hình trụ $\mathbb{S}^1 \times \mathbb{R}$, hình cầu $\mathbb{S}^2$).
   - Nhánh Learning Track sở hữu: MaT-LoRA learning/optimization, targeted temporal continual objective, và cơ chế chống quên lãng liên tục.

---

## 2. Ma Trận Nguồn Bằng Chứng Khoa Học và Đối Soát Thực Tế

Mọi quyết định thiết kế đều được đối soát minh bạch giữa các tầng bằng chứng:

| Nguồn tham chiếu | Phân loại | Nội dung lý thuyết & Cơ chế cốt lõi | Ranh giới chuyển giao sang Qwen-0.5B / TempUN | Mức độ tin cậy |
| :--- | :---: | :--- | :--- | :---: |
| **Yao et al. (arXiv:2602.11965, 2026)** | `PREPRINT` | MaT-LoRA: Biến thiên thích ứng low-rank theo thời gian $\Delta W(t) = \frac{\alpha}{r} B F(t) A$. | Cung cấp nền tảng toán học cho lõi động $F(t) \in \mathbb{R}^{r \times r}$. Kiến trúc Dual-Cycle mở rộng thêm vi chu kỳ và kỷ nguyên. | `HIGH` |
| **Liang et al. (Neural Networks 2026, DOI 10.1016/j.neunet.2026.108642)** | `PAPER` | MTLoRA: Tái tham số hóa ma trận tĩnh $\Delta W = \frac{\alpha}{r} B T A$. | Chứng minh ma trận chuyển đổi $T$ hỗ trợ tối ưu gradient mà không tăng hạng biểu diễn ($\operatorname{rank} \le r$). Khác với MaT-LoRA động. | `CANONICAL` |
| **Han, Ren & Peng (EMNLP 2021)** | `PAPER` | ECONET: Continual pre-training với giám sát quan hệ thời gian (Event Order / Indicator Corruption). | Áp dụng trên Encoder (BERT/RoBERTa) cho sự kiện tin tức. Chuyển giao sang Causal LM TempUN đòi hỏi thiết kế giám sát chuỗi giá trị chỉ số thống kê. | `CANONICAL` |
| **Hu et al. (ICLR 2022)** | `PAPER` | LoRA: Thích ứng tham số hạng thấp đóng băng $W_0$, tối ưu $B A$. | Cơ sở toán học nền tảng cho PEFT. | `CANONICAL` |
| **Wang et al. (Findings EMNLP 2023)** | `PAPER` | O-LoRA: Chiếu gradient vào không gian con trực giao để chống quên lãng. | Xếp vào Giai đoạn 2 (Phase 2), chỉ kích hoạt khi có bằng chứng suy giảm BWT thực tế. | `CANONICAL` |
| **Goodfellow et al. (Deep Learning)** | `BOOK_THEORY` | Chương 7 (Regularization tr. 243); Chương 8 (Optimization tr. 289). | Cơ sở thiết kế đối chứng âm $M_1\text{-NC}$ nhằm loại trừ hiệu ứng điều hòa ngẫu nhiên do tăng số bước cập nhật. | `CANONICAL` |
| **Zhang et al. (Dive into Deep Learning)** | `BOOK_THEORY` | Mục 2.5 (Autodiff tr. 99); Mục 3.6/5.1 (Generalization tr. 152); Chương 12 (AdamW tr. 572). | Kiểm soát độ ổn định gradient và hiện tượng quá khớp qua các chặng lịch sử. | `CANONICAL` |
| **Raschka (Build an LLM from Scratch)** | `BOOK_THEORY` | Chương 5 (Pretraining tr. 150); Chương 7 (Instruction Tuning tr. 226); Appendix E (LoRA tr. 273). | Khẳng định tính cốt tử của masking câu trả lời (`MASKED_TARGET`) để bảo toàn cú pháp instruct. | `CANONICAL` |
| **Tunstall et al. (NLP with Transformers)** | `BOOK_THEORY` | Chương 2 (Tokenization tr. 37); Chương 4 (Fine-Tuning tr. 81). | Ràng buộc cấu trúc tokenization và collator, bảo vệ token-0 ChatML khỏi can thiệp thô. | `CANONICAL` |
| **Ticktack v2 Kernel v4 (Partner Run)** | `CANONICAL_RUN` | Strict Acc $20.51\% \pm 3.06\%$, Invalid $26.85\%$ (3 seeds, 180 ca strict). | Kết quả đo đạc chuẩn tắc nhưng mang nhãn `FAVORABLE_BUT_CONFOUNDED` do thay đổi đồng thời cylinder + adapter. | `HIGH` |
| **Rescue 103 (B2 Seam Run)** | `CANONICAL_RUN` | Strict Acc $21.48\%$, Invalid $40.19\%$, McNemar vs Baseline $p \in [0.25, 1.00]$. | Phán quyết: `FAILED_TO_REJECT_H0`. Xác lập cơ chế tiêm prefill an toàn tại token 14 qua cổng số dư $\alpha_0=0.0$. | `CANONICAL` |

---

## 3. Đính Chính Bốn Điểm Kỹ Thuật Trọng Yếu (Technical Disambiguation)

1. **Minh bạch về Anchor xuất phát `M0-FC40` — Không tự nhận identical partner checkpoint:**
   - Module Dual-Cycle MaT-LoRA gốc nhận cặp $(\theta, z)$ từ module trụ riêng của đối tác.
   - Tại Learning Track, kiến trúc Dual-Cycle MaT-LoRA được thích ứng giao diện (Interface Adaptation): Tọa độ pha $\theta$ và kỷ nguyên $z$ được trích xuất trực tiếp từ 8 chiều Can-Chi có sẵn trong vector 40D, nuôi lõi $F(\theta, z)$ trên 72 khối MLP; đồng thời toàn bộ vector 40D được tiêm tại token 14 prefill.
   - Do có sự thay đổi luồng nạp dữ liệu và giao diện Safe Seam, checkpoint cũ của partner **không tương thích nhị phân trực tiếp**. Anchor mới được định danh chính thức là **`M0-FC40`** với trạng thái hiện tại là **`PROPOSED / NOT_TRAINED`**. Bắt buộc phải huấn luyện cơ sở riêng và tuyệt đối không gọi đây là identical partner checkpoint.
2. **Triệt tiêu nguy cơ rò rỉ hình học trong từ vựng và code:**
   - Vector đầu vào của Learning Track là vector số thực 40 chiều cố định. Mọi phép tính pha $\theta$ và kỷ nguyên $z$ bên trong adapter chỉ là các hàm lượng giác điều biến lõi ma trận $F$, không gọi tới bất kỳ hàm đa tạp hay chuyển đổi tọa độ nào của đối tác.
3. **Phân biệt rạch ròi Protocol Freezing và Parameter Freezing:**
   - **Protocol Freezing:** Khóa cứng mô hình nền $W_0$, siêu tham số cấu hình ($r=8, \alpha=16{,}0$, $K=2$), 72 module mục tiêu, tập dữ liệu 3 chặng và khế ước đánh giá `GENERATIVE_STRICT_PARSE_V1`.
   - **Parameter Training:** Trong cả $M_0$ và $M_1$, **các tham số adapter của MaT-LoRA ($A, B$ và lõi $F$) hoàn toàn được huấn luyện (trainable)**. Đính chính triệt để câu chữ cũ: Điểm khác biệt DUY NHẤT giữa $M_0$ và $M_1$ là **hàm mục tiêu huấn luyện** (loss function). Cả hai được huấn luyện từ **cùng một chính sách khởi tạo (identical initialization policy)** với ngân sách token và số bước cập nhật tương đương.
4. **Kiểm toán dữ liệu TempUN và Thiết kế hàm mục tiêu thời gian hợp lệ:**
   - Dữ liệu TempUN bao gồm các bản ghi thống kê: `In {Year}, {Indicator} in {Country} was {Value}`.
   - Bài toán đánh giá MCQA: So sánh tương quan thời gian giữa 2 mốc năm trên cùng chỉ số (`In {Year}, {Indicator} in {Country} was higher than {Year2}? Answer as Yes or No`).
   - Tuyệt đối không bịa đặt "sự kiện lịch sử Việt Nam" không có trong dữ liệu TempUN.
   - Hàm mục tiêu thời gian $\mathcal{L}_{\text{temporal}}$ được thiết kế chuẩn xác: Phân biệt sự nhất quán chuỗi giá trị theo thời gian của chỉ số quốc gia từ các mẫu huấn luyện thuần túy (`train_nwp.jsonl.gz`), tạo mẫu âm bằng cách ghép năm lệch trong cùng chuỗi hoặc hoán vị năm giữa các bản ghi có giá trị chênh lệch rõ rệt. Tuyệt đối không sử dụng các câu hỏi hay thực thể thuộc tập đánh giá `eval_mcqa.jsonl.gz`.

---

## 4. Kết Quả Định Tuyến Thực Tế Từ Jev System One API (`jev-1.13.0`)

Đã thực hiện cuộc gọi API thực tế tới máy chủ TypeSafe AI (`POST https://api.typesafe.ai/v1/systemone`) sử dụng khóa chứng thực được phát hiện an toàn. Kết quả định tuyến kiểu mẫu (typed answers) được ghi nhận chính thức:

- **Mã định danh biên bản:** [artifacts/audits/jev/jev_system_one_receipt.json](file:///c:/Users/GF63/Documents/long-range-temporal-alignment/artifacts/audits/jev/jev_system_one_receipt.json).
- **Mô hình phục vụ:** `jev-1.13.0` (TypeSafe System One).
- **Điểm danh câu hỏi và kết quả phân định:**
  1. `is_retention_prerequisite_met` (`noul`): **`0.16`** (Xác suất $84\%$ khẳng định: **Chưa đủ bằng chứng về quên lãng thảm họa để mở nhánh retention ở Giai đoạn 1**; đồng thuận tuyệt đối với System Two: hoãn Replay/O-LoRA).
  2. `is_m0_m1_causally_isolated` (`noul`): **`0.68`** (Xác nhận thiết kế $M_0$ vs $M_1$ vs $M_1\text{-NC}$ trên biểu diễn 40D đóng băng phân lập tốt tác động nhân quả của hàm mục tiêu).
  3. `next_optimal_path` (`choice`): **`VERIFY_M0_INTERFACE_AND_SMOKE`** (Độ tin cậy tuyệt đối **`1.0`**, xác suất $100\%$ chọn bước xác minh interface và chạy smoke test trên CPU trước khi yêu cầu GPU).
  4. `causal_isolation_grade` (`score`): **`1.83 / 3.0`** (Mức đánh giá kiểm soát chặt chẽ với đối chứng ghép cặp).
- **Chi phí token:** 1.027 input tokens, 144 output tokens.
- **Ranh giới:** Quyết định của Jev hoàn toàn nhất quán với định hướng của Người nghiên cứu; Jev không thay đổi starting architecture hay backbone đã khóa.

---

## 5. Thí Nghiệm Đã Khóa: Ma Trận Đối Chứng Ghép Cặp M0-FC40 vs M1-TEMP vs M1-NC

Thí nghiệm duy nhất được khóa cho giai đoạn tiếp theo:

| Mã nhánh | Kiến trúc Adapter & Biểu diễn | Hàm mục tiêu huấn luyện | Cơ chế kiểm chứng âm | Mục đích khoa học / Câu hỏi nghiên cứu | Trạng thái |
| :---: | :--- | :---: | :---: | :--- | :---: |
| **`M0-FC40`** | Periodic Dual-Cycle MaT-LoRA ($r=8, \alpha=16$) trên FoNE × Can-Chi 40D (Safe Seam token 14) | $\mathcal{L}_{\text{MASKED\_TARGET}}$ | — | **Anchor đối chứng chuẩn tắc:** Xác lập đường cơ sở sạch trên biểu diễn đã khóa. | `PROPOSED` |
| **`M1-TEMP`** | Cùng cấu hình `M0-FC40` (cùng khởi tạo, cùng số bước) | $\mathcal{L}_{\text{MASKED}} + \lambda \mathcal{L}_{\text{temporal}}$ ($\lambda = 0{,}1$) | Phân biệt mốc năm lệch trong cùng chuỗi chỉ số huấn luyện | **Ứng viên cải tiến duy nhất:** Kiểm định đóng góp gia tăng của giám sát thời gian đối với độ chính xác và giảm lỗi cú pháp. | `PROPOSED` |
| **`M1-NC`** | Cùng cấu hình `M1-TEMP` (cùng số bước, cùng token exposure) | $\mathcal{L}_{\text{MASKED}} + \lambda \mathcal{L}_{\text{shuffled}}$ ($\lambda = 0{,}1$) | Xáo trộn nhãn năm huấn luyện ngẫu nhiên theo seed | **Đối chứng âm cấu trúc bắt buộc:** Loại trừ giả thuyết cải thiện do hiệu ứng điều hòa ngẫu nhiên hoặc tăng tần suất cập nhật gradient. | `PROPOSED` |

### Quy tắc tạo mẫu âm cho $\mathcal{L}_{\text{temporal}}$ (Negative Sampling Protocol):
1. Nguồn trích xuất: Độc quyền từ `train_nwp.jsonl.gz`.
2. Với mỗi bản ghi $(E, I, y_{\text{true}}, v_{\text{true}})$, chọn mẫu âm $(E, I, y_{\text{corrupt}}, v_{\text{corrupt}})$ từ cùng chuỗi thời gian của thực thể $E$ và chỉ số $I$ nhưng tại mốc năm khác có giá trị $|v_{\text{corrupt}} - v_{\text{true}}| > \epsilon$.
3. Cấm tuyệt đối: Không sử dụng bất kỳ câu hỏi hoặc chỉ số nào xuất hiện trong 180 mẫu của `eval_mcqa.jsonl.gz`.

---

## 6. Tiêu Chuẩn Phán Quyết và Cổng Dừng (Stop / Rollback Gates)

1. **Khế ước đánh giá:** Bắt buộc áp dụng `GENERATIVE_STRICT_PARSE_V1`, đo lường qua 3 hạt ngẫu nhiên (Seed 42, 1024, 2026) trên 180 ca test của `vietnamese_history_qa_v1`.
2. **Cổng kiểm định nhân quả:**
   - `M1-TEMP` phải vượt trội `M1-NC` với $p < 0{,}05$ (kiểm định ghép cặp Edwards McNemar trên 180 ca cho từng seed). Nếu $p \ge 0{,}05$, kết luận: Lợi ích quan sát được chỉ là hiệu ứng điều hòa ngẫu nhiên; bác bỏ giả thuyết về tín hiệu thời gian.
3. **Cổng kiểm định tiến bộ:**
   - So sánh `M1-TEMP` vs `M0-FC40`: Phải đạt $p < 0{,}05$ và $\Delta \text{Strict Acc} > 0$. Nếu $p \ge 0{,}05$, ghi nhận trung thực phán quyết `FAILED_TO_REJECT_H0`, tuyệt đối không tùy tiện điều chỉnh siêu tham số $\lambda$ để tránh rủi ro overfitting trên tập test.
4. **Cổng an toàn cú pháp:** Tỷ lệ cú pháp không hợp lệ (Invalid Rate) phải giữ $\le 30{,}0\%$.
5. **Hành động kế tiếp duy nhất:** Chạy kiểm thử forward pass và tính ổn định gradient trên CPU (`VERIFY_M0_INTERFACE_AND_SMOKE`). Tuyệt đối không tự ý khởi chạy huấn luyện full GPU trên Kaggle khi chưa có phiếu phê duyệt chính thức.
