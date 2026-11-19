# Báo Cáo Phán Quyết Pháp Y Thực Nghiệm Qwen2.5-3B và Ticktack Baseline (Stage 03)

**Mã văn bản**: `STAGE03_3B_FORENSIC_VERDICT`  
**Dự án**: `long-range-temporal-alignment` (Learning Track)  
**Thời điểm lập**: 23/10/2026  
**Nhánh hoạt động**: `v2-targeted-continual-alignment`  
**Không gian làm việc đối tác**: `scratch/partner_repo` (STRICT READ-ONLY, commit `c33c576683dc1acb01f6a4e2744babe48b3bdfd0`)  
**Mô hình mục tiêu**: `Qwen/Qwen2.5-3B-Instruct` (Hub snapshot `aa8e72537993ba99e69dfaafa59ed015b17504d1`, license `qwen-research`)  
**Phán quyết tổng quát**: `GO_3B_PROTOCOL_DIAGNOSTIC` / `NO_GO_SCALE`

---

## 1. Phán Quyết Tổng Quát và Rào Chắn Khoa Học

$$\mathbf{PHÁN\ QUYẾT:\ GO\_3B\_PROTOCOL\_DIAGNOSTIC\ \big|\ NO\_GO\_SCALE}$$

1. **Cho phép thực thi chẩn đoán giao thức (GO_3B_PROTOCOL_DIAGNOSTIC)**: Cho phép chạy kiểm tra chẩn đoán đánh giá đa giao thức (V1 đến V5) trên các tập chia kiểm định độc lập, có đối chứng, không can thiệp vào trọng số và không thay đổi cấu trúc dữ liệu.
2. **Khóa dừng toàn diện huấn luyện quy mô lớn (NO_GO_SCALE)**: Tiếp tục duy trì lệnh cấm khởi chạy huấn luyện GPU quy mô lớn, cấm mở rộng tập dữ liệu, cấm nới lỏng rào chắn khi chưa giải quyết triệt để sự sụp đổ nhãn tiên nghiệm (prior logit collapse) và chưa phân lập được đóng góp của biểu diễn thời gian dài hạn.
3. **Bảo tồn bất biến dữ liệu gốc**: Tuyệt đối không đảo thứ tự phương án trong tập huấn luyện (`OnlineOptionPermutation.enabled = False`), không cân bằng lại tập kiểm định, không can thiệp vào file vàng và không điều chỉnh nhãn.

---

## 2. Bóc Tách 10 Sai Lệch Thực Nghiệm (Forensic Discrepancy Matrix)

Quá trình kiểm toán độc lập đã phát hiện, phân loại mức độ nghiêm trọng và xử lý triệt để 10 sai lệch số liệu và khế ước (đăng ký tại [STAGE03_3B_DISCREPANCY_REGISTER.jsonl](file:///c:/Users/GF63/Documents/long-range-temporal-alignment/Receipts/STAGE%203/10_QWEN25_3B_FORENSIC_AUDIT/STAGE03_3B_DISCREPANCY_REGISTER.jsonl)):

| Mã ID | Vấn đề phát hiện | Mức độ nghiêm trọng | Thực trạng bằng chứng kiểm toán | Biện pháp điều hòa / Phán quyết |
|---|---|---|---|---|
| **DISC_3B_001** | Lỗi hiển thị `Base_3B_Ref` strict accuracy là $25.10\%$ | `P0_REPORTING_ERROR` | Dữ liệu thô từng ca (`case_results` trong `base_3b_benchmark_results.json`) cho thấy độ chính xác nghiêm ngặt thực tế là $6.24\%$ ($330/5{,}292$), tương đương Ticktack 3B ($6.22\%$, $329/5{,}292$). Con số $25.10\%$ thực chất là hằng số đoán ngẫu nhiên $1/4$ được ghi đè do nhầm lẫn hiển thị. | Đã sửa cục bộ các bảng tổng hợp về con số thực chứng $6.24\%$. |
| **DISC_3B_002** | Xung đột điểm số nhiệm vụ Rel của `Base_3B_Ref` ($37.91\%$ vs $23.12\%$) | `P0_CROSS_ARTIFACT_CONTRADICTION` | File `base_3b_benchmark_results.json` ghi nhận Rel canonical accuracy đạt $37.91\%$ ($323/852$). Trong khi đó, phần nhúng `Base_3B_Ref` trong `ticktack_3b_benchmark_results.json` lại ghi Rel $= 23.12\%$. | Cách ly khối `Base_3B_Ref` nhúng tĩnh cũ; chỉ sử dụng `case_results` thô độc lập từ file chuẩn của Base 3B. |
| **DISC_3B_003** | Mô hình đạt $\approx 37.91\%$ trên bài toán so sánh thời gian Rel | `P0_METHODOLOGICAL_COLLAPSE` | Tập Rel có 852 câu ($529$ Yes, $323$ No). Cả Base 3B và Ticktack 3B dự đoán 'No' cho $852/852$ và $851/852$ câu. Độ chính xác $37.91\%$ phản ánh đúng tỷ lệ mẫu 'No' có sẵn trong tập dữ liệu ($323/852$). Độ chính xác trên các câu có đáp án 'Yes' thực chất là $0.0\%$. | Xác nhận sụp đổ nhãn toàn bộ về 'No' do logit tiên nghiệm lấn át; tiền đăng ký chẩn đoán hiệu chuẩn biên (Protocol V5). |
| **DISC_3B_004** | Nhầm lẫn mẫu số trắc nghiệm 4 lựa chọn với toàn bộ tập đánh giá | `P0_REPRESENTATION_BOTTLENECK` | Trên 4.440 câu trắc nghiệm 4 lựa chọn, có $86.55\%$ ($3{,}843/4{,}440$) đạt điểm debiased đúng $0.25$ (chọn cố định 1 ký tự qua 4 vòng quay). Tỷ lệ $72.62\%$ ($3{,}843/5{,}292$) là do tính gộp cả 852 câu Yes/No. Tính nhất quán ngữ nghĩa nghiêm ngặt trên 4 lựa chọn chỉ đạt $0.16\%$ ($7/4{,}440$). | Tách bạch mẫu số từng dạng nhiệm vụ trong báo cáo khoa học. |
| **DISC_3B_005** | Mô hình Instruct bị đánh giá bằng prompt completion thô | `P1_EVALUATOR_PROMPT_MISMATCH` | `eval_debiasing.py` ghép chuỗi dạng `Question: ...\nAnswer:` thay vì định dạng ChatML chuẩn `<|im_start|>user...<|im_start|>assistant\n` của họ Qwen2.5. | Thiết lập giao thức ChatML chuẩn tắc (Protocol V2) để chẩn đoán đối chứng không can thiệp checkpoint. |
| **DISC_3B_006** | Phân kỳ thuật toán mặt nạ ngày (Date Mask Fallback) | `P1_ARCHITECTURAL_DIVERGENCE` | `date_mask.py` có cơ chế fallback `[1:min(5, S)]` khi không tìm thấy chuỗi năm, trong khi `build_kaggle_notebook_3b_full.py` gán toàn bộ False. Trên TempUN, nhờ tokenizer tách số rời rạc, cả hai đều đạt $100\%$ độ phủ ($5{,}292/5{,}292$ câu eval và $1{,}000/1{,}000$ câu train), nhưng fallback tiềm ẩn rủi ro tiêm sai vị trí trên dataset khác. | Chuẩn hóa sang `find_date_token_mask_strict` kèm bộ đếm cảnh báo. |
| **DISC_3B_007** | Hiện tượng tuần hoàn đa thế kỷ (TE Cyclical Aliasing mod 60) | `P1_THEORETICAL_LIMITATION` | Trong `models.py`, tọa độ cực xoắn ốc Archimedes có góc $\theta = 6^\circ \times ((y-3)\%60)$ và bán kính $r = 0.5 + 0.5\theta$. Do cả $r$ và $\theta$ đều tuần hoàn theo chu kỳ 60 năm, chuẩn $\|TE(y) - TE(y+60)\|_2 = 0.0$ với mọi năm $y$. | Chứng minh toán học giới hạn của baseline Ticktack đối tác; ghi nhận sự cần thiết của trục thời gian thực trong Stage 03. |
| **DISC_3B_008** | Rò rỉ nhãn tuyệt đối tại CTM-Bench v2 (Oracle Leakage) | `P0_CRITICAL_ORACLE_LEAKAGE` | Dòng 106 trong `SourceCode/ticktack_v2/eval/run_ctm_eval.py` hàm `predict_ctm_item_v2_cylindrical` trả về trực tiếp `item['answer']`. Kết quả $100\%$ ($8{,}750/8{,}750$) là do rò rỉ chân lý từ file dữ liệu, hoàn toàn không phản ánh năng lực LLM. | Cách ly toàn diện; chặn tuyệt đối việc sử dụng con số $100\%$ CTM làm bằng chứng khoa học; áp dụng AST scanner kiểm toán. |
| **DISC_3B_009** | So sánh khập khiễng giữa kết quả Local $27.85\%$ và Bài báo gốc $64.96\%$ | `P1_COMPARATIVE_MISALIGNMENT` | Bài báo Ticktack (Han et al., arXiv:2503.04150v3 Bảng 1) huấn luyện 10 epochs trên cụm $8 \times \text{A100}$ với đầy đủ hàm mất mát liên chu kỳ/EWC, $lr=10^{-4}$. Bản chạy của partner chỉ huấn luyện 1 epoch trên 1 GPU Tesla T4, $lr=2\times 10^{-4}$, LoRA $q/v$ rút gọn và chỉ có intra-loss đơn độc. | Lập ma trận đối chiếu giao thức chi tiết; làm rõ ranh giới khác biệt điều kiện thực nghiệm. |
| **DISC_3B_010** | Tuyên bố kết quả phân rã nhiệm vụ có ý nghĩa thống kê | `P1_STATISTICAL_OVERREACH` | Các giá trị $p$ riêng lẻ trên nhiệm vụ DB ($p=0.0184$) và RB ($p=0.0449$) không vượt qua hiệu chỉnh Bonferroni cho 6 giả thuyết thử nghiệm đồng thời ($\alpha_{\text{bonf}} = 0.05 / 6 = 0.00833$). Kiểm định McNemar tổng thể toàn diện ghi nhận $\chi^2 = 1.3703, p = 0.2418 > 0.05$. | Đưa toàn bộ nhận định phân rã nhiệm vụ về trạng thái thăm dò (`EXPLORATORY / NON_SIGNIFICANT`). |

---

## 3. Phân Tích Chuyên Sâu Hai Điểm Nghẽn Cốt Lõi

### 3.1. Sự sụp đổ nhãn toàn bộ trên bài toán quan hệ thời gian Rel (Yes/No Collapse)

Nhiệm vụ Rel trong TempUN Full bao gồm 852 câu hỏi nhị phân so sánh mốc thời gian giữa hai sự kiện. Kiểm toán dữ liệu thô ghi nhận:
- Tập vàng chứa 529 câu đáp án `Yes` ($62.09\%$) và 323 câu đáp án `No` ($37.91\%$).
- Phân phối dự đoán của mô hình:
  - Base 3B: Dự đoán `No` cho $852/852$ câu ($100.0\%$), dự đoán `Yes` cho $0/852$ câu ($0.0\%$).
  - Ticktack 3B: Dự đoán `No` cho $851/852$ câu ($99.88\%$), dự đoán `Yes` cho $1/852$ câu ($0.12\%$).
- Độ chính xác được báo cáo $37.91\%$ ($323/852$) trùng khớp hoàn toàn với tỷ lệ các câu mang nhãn `No` trong tập dữ liệu. Trên tập các câu mang nhãn `Yes`, độ chính xác của cả hai mô hình là **$0.0\%$**.

**Nguyên nhân gốc rễ**: Hiện tượng này bắt nguồn từ thiên lệch logit tiên nghiệm trong không gian từ vựng của mô hình ngôn ngữ khi đối mặt với mẫu câu hỏi đóng. Do chưa được hiệu chuẩn biên (Marginal Prior Calibration), logit của token `'no'` chiếm ưu thế tuyệt đối trước logit của token `'yes'`. Việc huấn luyện LoRA ở quy mô nhỏ với hàm mất mát tự hồi quy thông thường không đủ lực để khắc phục thiên lệch tiên nghiệm này, dẫn đến suy biến thành bộ đoán một nhãn duy nhất.

### 3.2. Hiện tượng sụp đổ vị trí ký tự trên trắc nghiệm 4 lựa chọn (MCQA Position Bias)

Để đo lường độ bền vững trước thiên lệch vị trí, giao thức đánh giá áp dụng 4 vòng xoay hoán vị vòng tròn các phương án trắc nghiệm (Debiasing Cyclic Rotation). Kết quả kiểm toán trên 4.440 câu trắc nghiệm 4 lựa chọn bộc lộ:
- **$3{,}843 / 4{,}440$ câu ($86.55\%$) đạt điểm debiased chính xác là $0.25$**. Điểm số $0.25$ xuất hiện khi và chỉ khi mô hình kiên định chọn đúng một ký tự phương án (ví dụ luôn chọn `A` hoặc luôn chọn `C`) qua cả 4 lần đảo vị trí, bất kể nội dung ngữ nghĩa của phương án đó là gì.
- Số lượng câu đạt điểm tuyệt đối $1.0$ (chọn đúng ngữ nghĩa bất biến qua 4 vòng quay) chỉ đạt **$7 / 4{,}440$ câu ($0.16\%$)** trên Base 3B và $6 / 4{,}440$ câu ($0.14\%$) trên Ticktack 3B.
- Tỷ lệ $72.62\%$ ($3{,}843/5{,}292$) được trích dẫn trước đây là do tính gộp cả 852 câu Yes/No (vốn không có điểm $0.25$ vì chỉ có 2 phương án).

Bảng phân phối điểm debiased trên 4.440 câu trắc nghiệm:

| Mức điểm Debiased | Base 3B (Số lượng / Tỷ lệ) | Ticktack 3B (Số lượng / Tỷ lệ) | Ý nghĩa thực nghiệm |
|---|---|---|---|
| **$0.00$** | $275$ ($6.19\%$) | $276$ ($6.22\%$) | Sai toàn bộ 4 vòng quay |
| **$0.25$** | $\mathbf{3{,}843\ (86.55\%)}$ | $\mathbf{3{,}844\ (86.58\%)}$ | **Sụp đổ vị trí: chọn cố định 1 ký tự** |
| **$0.50$** | $285$ ($6.42\%$) | $284$ ($6.40\%$) | Đúng 2/4 vòng quay |
| **$0.75$** | $30$ ($0.68\%$) | $29$ ($0.65\%$) | Đúng 3/4 vòng quay |
| **$1.00$** | $\mathbf{7\ (0.16\%)}$ | $\mathbf{6\ (0.14\%)}$ | **Đạt tính nhất quán ngữ nghĩa thực sự** |
| **Tổng số** | $4{,}440$ ($100.0\%$) | $4{,}440$ ($100.0\%$) | Phân phối tương đồng tuyệt đối |

---

## 4. Kiểm Định Thống Kê Ghép Cặp McNemar

Để xác định liệu việc tích hợp Ticktack LoRA có mang lại sự cải thiện có ý nghĩa thống kê so với mô hình Base 3B hay không, kiểm định McNemar ghép cặp được thực thi trên toàn bộ $N = 5{,}292$ trường hợp kiểm định:

- Số trường hợp Base đúng, Ticktack đúng ($a$): $1{,}222$
- Số trường hợp Base đúng, Ticktack sai ($b$): $280$
- Số trường hợp Base sai, Ticktack đúng ($c$): $252$
- Số trường hợp Base sai, Ticktack sai ($d$): $3{,}538$

Giá trị thống kê kiểm định:
$$\chi^2 = \frac{(|b - c| - 1)^2}{b + c} = \frac{(|280 - 252| - 1)^2}{280 + 252} = \frac{27^2}{532} = \frac{729}{532} \approx 1.3703$$

Với bậc tự do $df = 1$, giá trị $p$ tương ứng là:
$$p = 0.2418 > 0.05$$

**Kết luận kiểm định**: Không có sự khác biệt mang ý nghĩa thống kê giữa Qwen2.5-3B Base và Ticktack 3B trên tập dữ liệu TempUN Full ($p = 0.2418$). Cả hai mô hình thể hiện hành vi suy luận tương đương nhau, chi phối bởi thiên lệch vị trí và sụp đổ nhãn tiên nghiệm.

---

## 5. Rà Soát Kiến Trúc và Tiêm Nhúng Thời Gian

1. **Kiểm tra độ phủ mặt nạ thời gian (Date Mask Coverage)**:
   - Thuật toán tìm kiếm chuỗi con trên biểu diễn token của Qwen2.5-3B đạt độ phủ chính xác **$100.0\%$** trên cả tập kiểm định ($5{,}292/5{,}292$ câu) và tập con huấn luyện ($1{,}000/1{,}000$ câu).
   - Không có trường hợp nào bị bỏ sót tiêm nhúng mốc thời gian trên tập TempUN.
2. **Kiểm kê tham số LoRA**:
   - Kiến trúc LoRA can thiệp vào 2 ma trận chiếu `q_proj` ($2048 \times 2048$) và `v_proj` ($2048 \times 256$) trên toàn bộ $36$ tầng biến đổi.
   - Với hạng thích ứng $r = 8$ và hệ số tỷ lệ $\alpha = 16$:
     - Mỗi tầng: $(2048 \times 8 + 8 \times 2048) + (2048 \times 8 + 8 \times 256) = 32{,}768 + 18{,}432 = 51{,}200$ tham số.
     - Toàn bộ 36 tầng: $51{,}200 \times 36 = 1{,}843{,}200$ tham số thích ứng (chiếm $0.0597\%$ tổng số tham số của mô hình 3B).
3. **Hiện tượng chu kỳ lặp mod 60 của biểu diễn cực**:
   - Biểu diễn thời gian của Ticktack sử dụng tọa độ cực $\theta = 6^\circ \times ((y-3)\%60)$, khiến các mốc thời gian cách nhau bội số của 60 năm (ví dụ năm 1960 và 2020) có vector nhúng đồng nhất hoàn toàn ($\|TE(y) - TE(y+60)\|_2 = 0.0$). Đây là giới hạn lý thuyết đã được chứng minh toán học.

---

## 6. Khung Điều Hòa 5 Giao Thức Đánh Giá (Protocol Harmonization Suite)

Để loại trừ các yếu tố gây nhiễu do định dạng lời nhắc và phương thức giải mã, module `eval_diagnostic_protocols.py` đã thiết lập 5 giao thức đánh giá chuẩn hóa:

1. **Protocol V1 (`PROTOCOL_LEGACY_CHOICE_LOGIT_V1`)**: Giao thức hoàn trả hiện trạng (Legacy Baseline) dùng lời nhắc trần `Question: ...\nAnswer: ` và so sánh logit trực tiếp của các ký tự nhãn.
2. **Protocol V2 (`PROTOCOL_CHATML_SINGLE_VERBALIZER_V2`)**: Đánh giá theo cấu trúc ChatML chuẩn của Qwen2.5 (`<|im_start|>user...<|im_start|>assistant\n`), so sánh logit các ký tự phương án tại vị trí sinh đầu tiên.
3. **Protocol V3 (`PROTOCOL_CHATML_SEQUENCE_LOGPROB_V3`)**: Tính tổng log-probability của toàn bộ chuỗi văn bản phương án trả lời dưới định dạng ChatML, loại bỏ hoàn toàn thiên lệch chữ cái định danh.
4. **Protocol V4 (`PROTOCOL_CHATML_GENERATIVE_STRICT_PARSE_V4`)**: Giải mã sinh tự do với `max_new_tokens=16`, kết hợp bộ phân tích cú pháp nghiêm ngặt để kiểm tra tính tuân thủ định dạng sinh văn bản thực tế.
5. **Protocol V5 (`PROTOCOL_MARGINAL_CALIBRATED_DIAGNOSTIC_V5`)**: Áp dụng hiệu chuẩn logit tiên nghiệm không điều kiện (Prior Margin Normalization) trên tập dev độc lập để khắc phục hiện tượng sụp đổ nhãn 'No'.

Toàn bộ 5 giao thức đã vượt qua 10 bài kiểm thử độc lập tại `tests/test_stage03_3b_forensic_and_protocols.py`.

---

## 7. Rào Chắn Preflight và Điều Kiện Cấp Phép Kế Tiếp

| Cổng kiểm định | Nội dung rào chắn | Trạng thái |
|---|---|---|
| **P0: Khóa tham chiếu và dữ liệu** | Hash tập eval 5,292 và train 104k khớp nguyên vẹn; partner repo READ-ONLY tuyệt đối ($0$ byte ghi). | **PASS** |
| **P1: Cách ly và điều hòa sai lệch** | Đăng ký 10 mục sai lệch chi tiết; sửa lỗi hiển thị $25.10\% \to 6.24\%$; cách ly bản nhúng tĩnh cũ. | **PASS** |
| **P2: Xác minh kiến trúc & tiêm nhúng** | Độ phủ mặt nạ ngày đạt $100.0\%$; chứng minh chu kỳ lặp mod 60; khớp $1{,}843{,}200$ tham số LoRA. | **PASS** |
| **P3: Chống rò rỉ & hiệu chỉnh mẫu số** | Cách ly tuyệt đối $100\%$ CTM oracle leakage; bóc tách mẫu số 4.440 câu 4 lựa chọn vs 852 câu Rel. | **PASS** |
| **P4: Điều hòa giao thức đánh giá** | Xây dựng hoàn chỉnh 5 giao thức V1–V5; bộ test $10/10$ PASS ($281/281$ toàn repo PASS). | **READY_FOR_GPU** |
| **P5: Khả thi phần cứng & bản quyền** | Xác nhận tương thích Tesla T4 Kaggle Cloud (bfloat16 + gradient checkpointing); tuân thủ qwen-research. | **PASS** |
| **P6: Quản trị Stage 03** | Duy trì lệnh cấm mở rộng dữ liệu và cấm retrain GPU khi chưa có phê duyệt; khóa dừng `NO_GO_SCALE`. | **PASS** |

**Điều kiện chuyển tiếp sang chẩn đoán thực nghiệm**:
- Chỉ thực thi chẩn đoán đánh giá thuần túy ($0$ optimizer step) theo 5 giao thức V1–V5 trên môi trường GPU được chỉ định.
- Báo cáo kết quả so sánh song song giữa Protocol V1 và Protocol V2/V3 để định lượng chính xác tác động của mẫu lời nhắc ChatML và thiên lệch chữ cái trước khi đề xuất bất kỳ can thiệp huấn luyện nào.
