# Ma Trận Đánh Giá Chất Lượng Bằng Chứng và Nguồn Gốc Phương Pháp
*Khung kiểm toán toàn diện giai đoạn Internal Temporal Learning (Week 1 -> FoNE -> REP_ABLATION_102 -> TEMPORAL_CONDITIONING_RESCUE_103)*

## 1. Khung Tiêu Chuẩn Đánh Giá Chất Lượng Bằng Chứng (Five-Factor Framework)

Khung đánh giá loại bỏ hoàn toàn các thang điểm chủ quan 0–100, thay thế bằng 4 trạng thái định danh tất định:
- `PASS`: Đạt chuẩn nghiêm ngặt, có bằng chứng bất biến xác thực.
- `PARTIAL`: Đạt một phần, cần thu hẹp phạm vi tuyên bố (narrowed scope).
- `FAIL`: Không đạt chuẩn, dẫn tới đình chỉ tuyên bố (blocked claim).
- `NOT_APPLICABLE`: Chưa có dữ liệu thực nghiệm để đánh giá.

### Năm Chiều Đánh Giá Cốt Lõi:
1. **Accurate (Độ chuẩn xác)**: Dẫn xuất trực tiếp từ các tệp artifact bất biến; mã băm và chuỗi nguồn gốc (provenance) khớp hoàn toàn; công thức tính toán và mẫu số không bị sai lệch; không bị ghi đè bởi báo cáo phái sinh.
2. **Appropriately Precise (Độ chính xác phù hợp)**: Định nghĩa rõ ràng thước đo (phân biệt điểm phần trăm pp với % tương đối); nêu rõ mẫu số; phạm vi seed và giai đoạn rõ ràng; có khoảng tin cậy hoặc độ lệch chuẩn; không làm tròn quá mức cho phép.
3. **Sufficient (Độ đầy đủ)**: Đủ số lượng ca kiểm thử; đủ số seed độc lập; đủ số lượng mẫu bất đồng phục vụ kiểm định thống kê; đủ số giai đoạn thời gian; phân tách rõ quan sát hiện tượng khỏi cơ chế nhân quả.
4. **Representative (Tính đại diện)**: Bao phủ miền của tuyên bố (các giai đoạn, các họ nhiệm vụ, nhóm trùng lặp vs không trùng lặp, phân phối năm phổ biến vs hiếm); không khái quát hóa từ 3 giai đoạn nhỏ sang toàn bộ 76 giai đoạn.
5. **Authoritative (Tính xác thực / Thẩm quyền)**: Đối với công trình bên ngoài: bài báo gốc, kỷ yếu chính thức, mã nguồn công bố; Đối với đề tài: artifact chạy thực nghiệm bất biến, biên nhận nghiệm thu đã thẩm định, mã băm cấu hình.

---

## 2. Ma Trận Nguồn Gốc Phương Pháp Luận (Method Provenance Matrix)

Mọi kỹ thuật thực nghiệm trong đề tài được phân loại nghiêm ngặt theo 3 mức độ thẩm quyền nguồn gốc:
- `PUBLISHED_PRIOR_ART`: Kỹ thuật đã xuất bản trong tài liệu khoa học bình duyệt chính thức.
- `PRIOR_ART_INSPIRED_PROJECT_DESIGN`: Thiết kế của đề tài lấy cảm hứng từ tài liệu tiền đề nhưng có cải biến cấu trúc cho bài toán hiện tại.
- `PROJECT_SPECIFIC_EXPERIMENTAL_CHOICE`: Lựa chọn thực nghiệm độc lập của nhóm tác giả nhằm giải quyết ràng buộc dữ liệu hoặc tài nguyên tính toán.

| Kỹ thuật / Quyết định thiết kế | Phân loại nguồn gốc | Thẩm quyền tiền đề (Prior Art) | Đặc tả kỹ thuật trong đề tài | Ranh giới vận hành và ghi nhận nguồn gốc |
| :--- | :---: | :--- | :--- | :--- |
| **LoRA (Low-Rank Adaptation)** | `PUBLISHED_PRIOR_ART` | Hu et al. (ICLR 2022), arXiv:2106.09685 | $W = W_0 + \frac{\alpha}{r} B A$, $r=16, \alpha=32$, đóng băng toàn bộ $W_0$. Áp dụng trên Qwen2.5-0.5B-Instruct. | Thừa hưởng nguyên bản nguyên lý phân rã ma trận hạng thấp của Hu et al. |
| **MLP-Only Adaptation** | `PRIOR_ART_INSPIRED_PROJECT_DESIGN` | Houlsby et al. (ICML 2019), He et al. (ICLR 2022) | Giới hạn LoRA vào 72 module MLP (`gate_proj`, `up_proj`, `down_proj`), đóng băng toàn bộ module Self-Attention. | Cảm hứng từ việc can thiệp biểu diễn tri thức sự kiện trong FFN; tuy nhiên cấu hình MLP-only trên Qwen2.5 là lựa chọn thực nghiệm riêng của đề tài. |
| **Targeted Temporal Continual Training** | `PRIOR_ART_INSPIRED_PROJECT_DESIGN` | ECONET (Han et al., EMNLP 2021) | Huấn luyện tăng dần tuần tự theo 3 giai đoạn lịch sử (1947 -> 1984 -> 2022) trên tập dữ liệu TempUN. | Cảm hứng từ học tăng dần theo thời gian để giảm quên lãng; phiên bản hiện tại vận hành trên causal decoder Qwen thay vì masked language model. |
| **MASKED_TARGET Objective** | `PROJECT_SPECIFIC_EXPERIMENTAL_CHOICE` | Thiết kế độc lập của đề tài | Chỉ tính hàm mất mát Cross-Entropy trên token phản hồi đích (target tokens), gán nhãn `-100` cho toàn bộ prompt chỉ thị và bối cảnh. | Lựa chọn thực nghiệm nhằm tối ưu hóa học chỉ thị có cấu trúc trong điều kiện ngữ cảnh thời gian dài hạn. |
| **B1 Textual Temporal Conditioning** | `PUBLISHED_PRIOR_ART` | Dhingra et al. (TACL 2022), Cao & Wang (Findings of EMNLP 2022) | Nhúng trực tiếp thông tin thời gian (Dương lịch + Can-Chi) vào tiền tố văn bản của lời nhắc hệ thống. | Nguyên lý điều kiện hóa văn bản kế thừa từ prior art; khuôn mẫu văn bản cụ thể hóa cho lịch sử Việt Nam là thiết kế của đề tài. |
| **B2 Gated Continuous Residual Conditioning** | `PRIOR_ART_INSPIRED_PROJECT_DESIGN` | Li & Liang (ACL 2021 Prefix-Tuning), Zhang et al. (2023 LLaMA-Adapter), Alayrac et al. (NeurIPS 2022 Flamingo) | $h' = h + \alpha \cdot \text{LayerNorm}(W z_{\text{time}})$, khởi tạo cổng $\alpha_0 = 0.0$, Can-Chi 8D, tiêm tại token thứ 14 trong prefill, không chèn virtual token. | Cảm hứng từ cơ chế gating zero-init của LLaMA-Adapter và Flamingo; thiết kế tiêm ẩn vào hidden state tại vị trí xác định là cải biên của đề tài. |
| **FoNE (Fourier Numerical Embeddings)** | `PUBLISHED_PRIOR_ART` | Tancik et al. (NeurIPS 2020 Fourier Features), Lermen et al. (2023) | Biểu diễn năm $y$ thành vector liên tục thông qua chuỗi hàm điều hòa $[\sin(\omega_k y), \cos(\omega_k y)]$. | Kế thừa nguyên lý biểu diễn số học bằng đặc trưng Fourier; phương thức ghép nối vào LLM được kiểm định độc lập trong đề tài. |
| **Can-Chi / TickTack Representation** | `PUBLISHED_PRIOR_ART` | Khung biểu diễn chu kỳ thời gian Đông Á (TickTack Framework) | Mã hóa chu kỳ Thiên Can (10) và Địa Chi (12) thành vector đặc trưng tuần hoàn 8 chiều kết hợp lịch Gregory. | Kế thừa hệ hình biểu diễn chu kỳ Can-Chi; ánh xạ sang vector nhúng thời gian cho mô hình ngôn ngữ là đóng góp của đề tài. |
| **MTLoRA (Matrix Transformation LoRA)** | `PUBLISHED_PRIOR_ART` | Liang et al. (Neural Networks 2026 / arXiv:2403.07440) | $\Delta W = \frac{\alpha}{r} B T A$ với $T \in \mathbb{R}^{r \times r}$. Khi $T = I_r$, thu về LoRA chuẩn. Bổ sung đúng $r^2$ tham số/module (288 params cho 72 khối MLP khi $r=2$). | Kế thừa nguyên lý tối ưu hóa không gian tham số thích ứng; phân định rạch ròi với biến thể MaT-LoRA tích hợp chu kỳ của nhánh Representation. |
| **Continual LoRA (O-LoRA / InfLoRA)** | `PUBLISHED_PRIOR_ART` | Wang et al. (EMNLP 2023 O-LoRA), Liang et al. (CVPR 2024 InfLoRA) | Phạt trực giao $\|A_t A_i^T\|_F^2$ hoặc chiếu cập nhật vào không gian triệt tiêu null space để ngăn chặn xung đột gradient giữa các chặng lịch sử. | Cơ sở khoa học cho cơ chế chống quên lãng khi huấn luyện tăng dần; đề tài ưu tiên kiểm chứng đối chứng bằng Replay Buffer 5% trước khi mở rộng. |
| **ECONET Temporal Loss** | `PUBLISHED_PRIOR_ART` | Han et al. (EMNLP 2021 ECONET), arXiv:2104.08779 | $\mathcal{L} = \mathcal{L}_{\text{target}} + \lambda \mathcal{L}_{\text{temporal}}$ với số hạng phạt phân biệt xáo trộn mốc thời gian (Year-Corruption Discrimination). | Cảm hứng từ hàm mục tiêu thời gian cho LM; đề tài áp dụng trên mô hình tự hồi quy ChatML kết hợp kiểm chứng đối chứng thời gian xáo trộn (Shuffled-Time Control). |
| **Option Scoring (Log-Likelihood MCQA)** | `PUBLISHED_PRIOR_ART` | Brown et al. (NeurIPS 2020 GPT-3), Robinson & Wingate (ACL 2023) | Tính tổng log-likelihood chuẩn hóa trên các token đáp án: $\operatorname{Score}(O_k) = \frac{1}{|O_k|} \sum \log P(w_t)$. Triệt tiêu 100% lỗi cú pháp. | Thước đo chẩn đoán độc lập (Diagnostic Evaluation) nhằm bóc tách lỗi cú pháp khỏi tri thức lịch sử thực chất; không thay thế số liệu sinh tự do lịch sử. |
| **DFA Constrained Decoding** | `PUBLISHED_PRIOR_ART` | Willard & Louf (Outlines 2023), Scholak et al. (PICARD EMNLP 2021) | Áp mặt nạ trên phân phối logits tại mỗi bước giải mã dựa trên Otomata DFA: $z'_{t, v} = -\infty$ nếu $v \notin \mathcal{V}_{\text{valid}}$. | Giải pháp kỹ thuật bảo đảm 100% câu sinh ra tuân thủ đúng định dạng nhãn trắc nghiệm mà không thay đổi phân phối biểu diễn tri thức nội tại. |

---

## 3. Bảng Tổng Hợp Kiểm Toán Các Tuyên Bố Khoa Học (Claims Audit)

| Mã định danh | Tuyên bố khoa học tóm lược | Trạng thái hiện tại | Accurate | Precise | Sufficient | Representative | Authoritative | Quyết định thăng hạng |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `CLAIM_01` | LoRA PEFT duy trì hoặc cải thiện nhẹ hiệu năng so với Zero-Shot | `SUPPORTED_WITHIN_SCOPE` | **PASS** | **PASS** | **PARTIAL** | **PARTIAL** | **PASS** | `NARROWED_TO_TEMPUN_SMALL_3_STAGES` |
| `CLAIM_02` | MASKED_TARGET vượt trội unmasked training (+20.37 pp, 0% invalid) | `SUPPORTED_WITHIN_SCOPE` | **PASS** | **PASS** | **PASS** | **PARTIAL** | **PASS** | `NARROWED_TO_TEMPUN_SMALL` |
| `CLAIM_03` | MLP-only LoRA giảm 57% tham số nhưng bảo toàn hiệu năng unmasked | `SUPPORTED_WITHIN_SCOPE` | **PASS** | **PASS** | **PASS** | **PARTIAL** | **PASS** | `NARROWED_TO_UNMASKED_REGIME` |
| `CLAIM_04` | Tương tác âm giữa MASKED_TARGET và MLP_ONLY (-7.78 pp interaction) | `SUPPORTED_WITHIN_SCOPE` | **PASS** | **PASS** | **PASS** | **PARTIAL** | **PASS** | `NARROWED_TO_CURRENT_ARCHITECTURE` |
| `CLAIM_05` | FoNE-inspired input conditioning cải thiện biểu diễn canonical_year | `REFUTED` | **PASS** | **PASS** | **PASS** | **PARTIAL** | **PASS** | `REFUTED_AND_BOUNDED_TO_TEMPUN_SMALL` |
| `CLAIM_06` | Tiêm vector thô vào token-0 ChatML gây sụp đổ cú pháp phân loại | `SUPPORTED_WITHIN_SCOPE` | **PASS** | **PASS** | **PASS** | **PARTIAL** | **PASS** | `ATTRIBUTION_TO_TOKEN0_INJECTION_BOUNDED` |
| `CLAIM_07` | B1 Textual conditioning phục hồi tuân thủ định dạng mà không cần thêm tham số | `SUPPORTED_WITHIN_SCOPE` | **PASS** | **PASS** | **PASS** | **PARTIAL** | **PASS** | `SUPPORTED_ON_3_STAGES_3_SEEDS` |
| `CLAIM_08` | B2 Gated residual conditioning ổn định giao diện biểu diễn thời gian | `SUPPORTED_WITHIN_SCOPE` | **PASS** | **PASS** | **PASS** | **PARTIAL** | **PASS** | `SUPPORTED_WITH_ZERO_INIT_GATE` |
| `CLAIM_09` | B1 và B2 tương đương thống kê so với Baseline C0, chưa vượt trội về độ chính xác | `SUPPORTED_WITHIN_SCOPE` | **PASS** | **PASS** | **PASS** | **PARTIAL** | **PASS** | `PARITY_ESTABLISHED_NO_SUPERIORITY` |
| `CLAIM_10` | Cổng alpha dương trong B2 phản ánh việc nhận tín hiệu thời gian | `CONSISTENT_WITH_SIGNAL_USE` | **PASS** | **PASS** | **PASS** | **PARTIAL** | **PASS** | `NOT_CAUSALLY_ESTABLISHED_PENDING_SHUFFLE` |

---

## 4. Chi Tiết Bằng Chứng Theo Từng Giai Đoạn

### Giai Đoạn 1: Week 1 Internal Learning Ablation (B0 -> B3)
- **Nguồn gốc minh chứng**: `artifacts/releases/WEEK1TARGET/`
- **Kết quả cốt lõi**:
  - `B0` Zero-Shot: $17.04\%$ Acc, $38.70\%$ Invalid.
  - `B1` Unmasked All-Linear LoRA: $18.52\%$ Acc, $36.67\%$ Invalid.
  - `B1M` Masked Target All-Linear LoRA: $38.89\%$ Acc, $0.00\%$ Invalid.
  - `B2` Unmasked MLP-Only LoRA: $18.33\%$ Acc, $38.52\%$ Invalid.
  - `B3` Masked Target MLP-Only LoRA: $31.11\%$ Acc, $0.00\%$ Invalid.
- **Thẩm định**: Xác lập hiệu lực rõ rệt của `MASKED_TARGET` trong việc loại bỏ lỗi vi phạm cú pháp; xác lập hiện tượng tương tác âm khi kết hợp đồng thời `MASKED_TARGET` và `MLP_ONLY`.

### Giai Đoạn 2: LOGGING 102 (FoNE Controlled Experiment & McNemar Audit)
- **Nguồn gốc minh chứng**: `artifacts/releases/LOGGING102_FONE/` (hoặc `LOGGING102_DATA_PAIR_PARITY.json`)
- **Kết quả cốt lõi**:
  - `F0` (Baseline Standard): $34.76\%$ ACC, $+0.86\%$ BWT.
  - `F1` (FoNE Conditioning): $28.15\%$ ACC, $-2.12\%$ BWT.
  - Delta: $\Delta_{\text{ACC}} = -6.61$ pp, $\Delta_{\text{BWT}} = -2.98$ pp.
  - Kiểm định McNemar (after_02, 180 ca): Seed 1 ($p = 0.0104$), Seed 2 ($p = 0.0101$), Seed 3 ($p = 0.8676$).
- **Thẩm định**: Bác bỏ giả thuyết FoNE additive conditioning giúp tăng cường độ chính xác trong cấu hình Qwen + TempUN; xác nhận tính toàn vẹn ghép cặp dữ liệu (180 case IDs khớp tuyệt đối).

### Giai Đoạn 3: REP_ABLATION_102 (2x2 Factorial Representation Ablation)
- **Nguồn gốc minh chứng**: `artifacts/releases/REP_ABLATION_102/`
- **Thiết kế**: 2x2 Factorial (FoNE $\in \{0, 1\} \times$ Can-Chi $\in \{0, 1\}$) tiêm trực tiếp vào token-0 ChatML (`<|im_start|>`).
- **Kết quả cốt lõi**:
  - `R00` (Baseline no conditioning): Strict Acc $21.48\% \pm 4.63\%$, Invalid Rate $40.19\% \pm 5.28\%$.
  - `R10` (FoNE token-0): Strict Acc $1.30\% \pm 0.37\%$, Invalid Rate $96.85\% \pm 0.93\%$.
  - `R01` (Can-Chi token-0): Strict Acc $0.93\% \pm 0.64\%$, Invalid Rate $97.22\% \pm 1.60\%$.
  - `R11` (FoNE + Can-Chi token-0): Strict Acc $0.74\% \pm 0.37\%$, Invalid Rate $97.78\% \pm 0.93\%$.
- **Thẩm định**: Tiêm additive vector vào token-0 gây sụp đổ cấu trúc phân tích cú pháp của mô hình Instruct. Bằng chứng thực nghiệm quy trách nhiệm lớn cho cơ chế tiêm thô trực tiếp vào token-0 ChatML; mức độ đóng góp nhân quả riêng biệt của vị trí token so với biên độ vector/chuẩn hóa/giao diện vẫn chưa được phân lập hoàn toàn.

### Giai Đoạn 4: TEMPORAL_CONDITIONING_RESCUE_103 (B1 Textual vs B2 Gated Residual)
- **Nguồn gốc minh chứng**: `artifacts/releases/TEMPORAL_CONDITIONING_RESCUE_103/`
- **Thiết kế**: Giải cứu mô hình bằng 2 giao diện biểu diễn thời gian an toàn:
  - `B1`: Điều kiện hóa qua văn bản tự nhiên (Textual Temporal Context).
  - `B2`: Tiêm vector qua cổng số dư có trọng số $\alpha$ khởi tạo bằng 0 tại token 14 trong prefill (không chèn virtual token).
- **Kết quả cốt lõi**:
  - `B1`: Strict Acc $20.93\% \pm 2.51\%$, Valid Acc $26.58\% \pm 2.91\%$, Invalid Rate $21.30\% \pm 3.06\%$ (thấp kỷ lục).
  - `B2`: Strict Acc $21.48\% \pm 4.63\%$, Valid Acc $35.67\% \pm 4.65\%$, Invalid Rate $40.19\% \pm 5.28\%$, Trọng số cổng $\alpha$ hội tụ về $0.0017 - 0.0024$.
- **So sánh thống kê McNemar (180 ca, chặng after_02)**:
  - So với $N_0$ (thất bại token-0): Cả B1 và B2 vượt trội có ý nghĩa thống kê áp đảo ($p < 10^{-7}$ và $p < 10^{-5}$).
  - So với $C_0$ (baseline không điều kiện hóa): Đạt tính tương đương thống kê ($p \in [0.25, 1.00]$ trên mọi seed), loại bỏ nguy cơ suy thoái.
- **Ranh giới nhân quả**:
  - Cổng $\alpha > 0$: Nhất quán với việc mô hình tiếp nhận và sử dụng tín hiệu thời gian (`CONSISTENT_WITH_SIGNAL_USE`), nhưng chưa chứng minh quan hệ nhân quả có lợi cho tác vụ (`NOT_CAUSALLY_ESTABLISHED`) do chưa thực hiện phép thử xáo trộn đặc trưng (feature shuffle / ablation control).
  - B1 và B2 khôi phục hoàn toàn tính toàn vẹn cú pháp nhưng chưa tạo ra bước nhảy vọt thống kê vượt trên baseline $C_0$.

---

## 5. Đối Soát Nguồn Dữ Liệu Ticktack-v2 / MaT-LoRA (Source Reconciliation Audit)

- **Minh chứng đối soát**: `artifacts/audits/SOURCE_RECONCILIATION_AUDIT.json`.
- **Mâu thuẫn đã giải tỏa**:
  - Bản ghi 8/10 (Teacher Report): Continual Final ACC $= 36.93\% \pm 1.14\%$ (Seed 1: $38.24\%$, Seed 2: $36.16\%$, Seed 3: $36.39\%$).
  - Bản ghi 9/10 (`TICKTACK_V2_CONTINUAL_EVAL.json`): Strict ACC $= 20.51\% \pm 3.06\%$ (Seed 1: $17.22\%$, Seed 2: $21.11\%$, Seed 3: $23.33\%$), Invalid Rate $= 26.85\% \pm 4.63\%$, Valid ACC $= 28.01\% \pm 2.79\%$.
- **Nguyên nhân gốc rễ (Root Cause)**:
  - Bản ghi 8/10 đo theo giao thức Continual Evaluation unmasked/lenient của đối tác (không áp phạt 0 điểm lỗi format ChatML).
  - Bản ghi 9/10 được kiểm toán lại theo khế ước sinh tự do nghiêm ngặt `GENERATIVE_STRICT_PARSE_V1`. Do tỷ lệ lỗi format là $26.85\%$ và toàn bộ bị tính điểm 0, Strict Accuracy giảm xuống $20.51\%$.
  - Kiểm định Edwards McNemar ghép cặp trên 180 ca trắc nghiệm giữa Ticktack v2 và F0 Baseline cho thấy $p > 0.15$ trên cả 3 seed, phán quyết là `INCONCLUSIVE`.
  - Kết luận: Không có sai lệch vật lý; mâu thuẫn là hệ quả của việc chuyển dịch khế ước đánh giá (Evaluation Contract Shift).

---

## 6. Đặc Tả Khế Ước Nhánh Học Tập: Anchor M0 và Thực Nghiệm M1

Toàn bộ các thực nghiệm nhánh Learning tuân thủ nghiêm ngặt các nguyên tắc:
1. **Backbone bất biến**: `Qwen/Qwen2.5-0.5B-Instruct` tại commit cố định `7ae557604adf67be50417f59c2c2f167def9a775`. Không tăng dung lượng mô hình.
2. **Anchor xuất phát M0 (Starting Point Anchor)**:
   - Kiến trúc: `Periodic Dual-Cycle MaT-LoRA` của đối tác kết hợp vi chu kỳ Can-Chi $F_{\text{cycle}}(\theta)$ và đại kỷ nguyên $F_{\text{epoch}}(z)$ trên 72 khối MLP ($r=8, \alpha=16.0$).
   - Hàm mục tiêu: `MASKED_TARGET` Cross-Entropy.
   - Trạng thái kiểm chứng: `SELECTED_WITH_LIMITATIONS`. Strict Acc $= 20.51\% \pm 3.06\%$, Invalid Rate $= 26.85\% \pm 4.63\%$. Được chọn làm điểm khởi đầu theo chỉ đạo của GVHD; phạm vi tuyên bố vượt trội thuần adapter được thu hẹp cho đến khi có matched ablation.
3. **Thực nghiệm tiếp theo M1 (Causal Targeted Temporal Objective)**:
   - Giữ nguyên mô hình nền, bộ trọng số MaT-LoRA và tọa độ trụ của $M_0$.
   - Thay đổi biến duy nhất: $\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{MASKED\_TARGET}} + \lambda \mathcal{L}_{\text{TEMPORAL}}$ ($\lambda = 0{,}1$ screen candidate).
   - Hàm mục tiêu phụ: Phân biệt mốc thời gian bị xáo trộn (Year-Corruption Discrimination) trên các sự kiện thuộc tập huấn luyện (train-only facts) với mẫu âm có kiểm chứng từ cùng họ sự kiện, triệt tiêu nguy cơ rò rỉ tập đánh giá.
   - Đối chứng cấu trúc: Nhánh $M_1\text{-SHUFFLE}$ (xáo trộn nhãn năm theo seed cố định) để loại bỏ giả thuyết cải thiện do hiệu ứng điều hòa ngẫu nhiên.
4. **Đối chứng lịch sử (Historical Controls)**:
   - Các biến thể Standard MLP-LoRA ($L_0$, $r=2$) và generic MTLoRA ($L_1$, $\Delta W = BTA$) được lưu giữ như các mốc quy chiếu đối chứng lịch sử, không phải hướng phát triển chính.


