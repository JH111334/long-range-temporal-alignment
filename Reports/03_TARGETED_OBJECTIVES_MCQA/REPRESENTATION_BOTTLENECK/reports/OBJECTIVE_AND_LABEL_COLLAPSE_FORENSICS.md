# Báo Cáo Pháp Y Hàm Mục Tiêu Huấn Luyện và Cơ Chế Sụp Đổ Nhãn (Phase C Report)

**Dự án**: `long-range-temporal-alignment`  
**Nhánh thực hiện**: `v2-targeted-continual-alignment`  
**Thời điểm khóa sổ**: `2026-10-20`  
**Đối tượng phân tích**: Checkpoint [`R52_B_NTP_REL_MCQA.pt`](file:///c:/Users/GF63/Documents/long-range-temporal-alignment/artifacts/experiments/V2_CAUSAL_TEMPORAL_ALIGNMENT/run_05_2/checkpoints/R52_B_NTP_REL_MCQA.pt) và quy trình huấn luyện V2 Run 05.2  
**Mục tiêu**: Bóc tách nguyên nhân gốc rễ (root cause) gây ra hiện tượng sụp đổ nhãn 'All-No' trên tác vụ Yes/No và co cụm lựa chọn trên tác vụ MCQA.

---

## 1. Hiện Tượng Sụp Đổ Nhãn Trên Checkpoint Thẩm Định

Qua quá trình đối soát dữ liệu thô 432 bản ghi tại [`RAW_PREDICTIONS.jsonl`](file:///c:/Users/GF63/Documents/long-range-temporal-alignment/Reports/03_TARGETED_OBJECTIVES_MCQA/REPRESENTATION_BOTTLENECK/runs/RUN_FULL_SWEEP_54P_20261019_161756/RAW_PREDICTIONS.jsonl) và bảng điểm lịch sử [`SCOREBOARD_A_B_C_BY_TASK_AND_PROTOCOL.json`](file:///c:/Users/GF63/Documents/long-range-temporal-alignment/artifacts/experiments/V2_CAUSAL_TEMPORAL_ALIGNMENT/run_05_2/SCOREBOARD_A_B_C_BY_TASK_AND_PROTOCOL.json), chúng tôi ghi nhận hai hiện tượng bất thường mang tính hệ thống:

1. **Sụp đổ nhãn tuyệt đối về 'No' trên tác vụ nhị phân (Yes/No Collapse)**:
   - Trên tập kiểm toán phản thực nghiệm 108 câu hỏi Yes/No (`COUNTERFACTUAL_GOLD_AUDIT.jsonl`), cánh tay can thiệp chính `ARM_B_FULL_TRUE_YEAR` dự đoán **108/108 nhãn là "No"** (100.0% No, 0% Yes). Entropy phân bố dự đoán bằng $0.0000$ bits.
   - Trên 24 câu hỏi Yes/No của tập `tempun180`, cả Arm A (`R52_A_NTP_REL`) và Arm B (`R52_B_NTP_REL_MCQA`) đều dự đoán chính xác **24/24 câu là "No"**.
   - Kiểm tra ablation: Khi vô hiệu hóa LoRA (`C_LORA_OFF` hoặc `C_BOTH_OFF`), mô hình quay trở lại phân phối của mô hình nền (Base Model), dự đoán **24/24 câu là "Yes"** trên `tempun180` và **103/108 câu là "Yes"** trên 54 cặp phản thực nghiệm.
   - *Kết luận cục bộ*: Hiện tượng sụp đổ nhãn sang "No" hoàn toàn do các trọng số thích ứng LoRA được cập nhật trong quá trình huấn luyện tạo ra, không phải thuộc tính vốn có của mô hình nền.

2. **Bóc tách tính ổn định giả tạo (Pseudo-Stability on Gold-Same)**:
   - Trước đây, chỉ số "18/18 Gold-Same Stability" được coi là bằng chứng về khả năng duy trì tính nhất quán khi mốc thời gian không làm thay đổi chân lý.
   - Tuy nhiên, dữ liệu thô khẳng định rằng tính ổn định này là **hệ quả thứ cấp của sự sụp đổ nhãn**: vì mô hình luôn luôn sinh ra "No" trên 100% câu hỏi, hiển nhiên dự đoán ở cả hai mốc thời gian $t_1$ và $t_2$ đều là "No" và trùng khớp nhau. Mô hình hoàn toàn không thể hiện bất kỳ sự phân biệt ngữ nghĩa hay tính bất biến thời gian có chọn lọc nào.

---

## 2. Giải Phẫu Hàm Mục Tiêu và Động Lực Học Gradient (Objective & Loss Forensics)

Để xác định nguyên nhân tại sao LoRA bị kéo lệch hoàn toàn về "No", chúng tôi phân tích mã nguồn huấn luyện tại [`src/temporal_alignment/training_run05_2.py`](file:///c:/Users/GF63/Documents/long-range-temporal-alignment/src/temporal_alignment/training_run05_2.py):

### 2.1 Cấu Trúc Hàm Mất Mát Của Arm B
Hàm mục tiêu của Arm B được định nghĩa:
$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{NTP}} + 0.5 \cdot \mathcal{L}_{\text{rel}} + 0.5 \cdot \mathcal{L}_{\text{MCQA}}$$

Trong đó:
- $\mathcal{L}_{\text{NTP}}$: Next-Token Prediction trên văn bản tự nhiên, không chứa cấu trúc giám sát Yes/No.
- $\mathcal{L}_{\text{MCQA}}$: Cross-Entropy có giám sát mục tiêu trên các phương án trắc nghiệm A/B/C/D.
- $\mathcal{L}_{\text{rel}}$: Cross-Entropy có giám sát mục tiêu trên các cặp quan hệ thời gian (`train_relation_pairs.jsonl`).

### 2.2 Kiểm Toán Dữ Liệu Huấn Luyện và Bất Đối Xứng Gradient
1. **Kiểm toán dữ liệu huấn luyện**: Tập `train_relation_pairs.jsonl` có phân bố nhãn cân bằng hoàn hảo gồm **963 mẫu "No" và 963 mẫu "Yes"** (tỷ lệ 50/50). Trong 40 bước huấn luyện thực tế với bộ chọn mẫu phân tầng ngẫu nhiên (`seed=42`), mô hình tiếp xúc chính xác **20 mẫu "No" và 20 mẫu "Yes"**. Dữ liệu huấn luyện hoàn toàn không bị mất cân bằng nhãn.
2. **Cơ chế gây sụp đổ: Xung đột giữa thiên kiến tiên nghiệm của mô hình nền và hàm mất mát Cross-Entropy**:
   - Mô hình nền `Qwen2.5-0.5B-Instruct` chưa thích ứng có thiên kiến mạnh nghiêng về "Yes" khi trả lời câu hỏi xác nhận (95.37% Yes trên tập kiểm thử). Do đó, đối với bất kỳ câu hỏi nào, xác suất ban đầu luôn thỏa mãn $P(\text{Yes}) \gg P(\text{No})$.
   - Khi huấn luyện với Cross-Entropy:
     - Với mẫu có nhãn mục tiêu là **"Yes"**: Mô hình đã gán xác suất cao ($P(\text{Yes}) \approx 0.95$), dẫn đến tổn thất rất nhỏ:
       $$\mathcal{L}_{\text{rel}}(\text{Yes}) = -\log P(\text{Yes}) \approx 0.05$$
       Gradient lan truyền ngược xấp xỉ bằng $0$.
     - Với mẫu có nhãn mục tiêu là **"No"**: Mô hình gán xác suất rất thấp ($P(\text{No}) \approx 0.05$), dẫn đến tổn thất khổng lồ:
       $$\mathcal{L}_{\text{rel}}(\text{No}) = -\log P(\text{No}) \approx 3.0$$
       Gradient lan truyền ngược có độ lớn cực đại, tập trung toàn bộ lực kéo trọng số về hướng triệt tiêu logit của "Yes" và đẩy mạnh logit của "No".
3. **Hiện tượng Gradient Overshooting trong chế độ Few-Step**:
   - Với tốc độ học $\eta = 2 \times 10^{-4}$ và ngân sách chỉ 40 bước lặp, gradient dốc từ các mẫu "No" liên tục đẩy dời siêu phẳng quyết định (decision boundary) mà không có lực đối trọng từ các mẫu "Yes" (vốn đã có gradient triệt tiêu).
   - Kết quả là chỉ sau 40 bước, LoRA bị kéo quá đà (overshoot), dịch chuyển toàn bộ phân phối logit sang phía "No" với biên độ vượt quá $1.75$ đơn vị logit, dẫn tới sự sụp đổ nhãn hoàn toàn.

---

## 3. Tương Tác Giữa Sụp Đổ Nhãn và Tín Hiệu Can Thiệp Thời Gian

Sự sụp đổ nhãn này giải thích triệt để nghịch lý quan sát được trong các thí nghiệm can thiệp:
1. **Biên độ can thiệp thời gian**: Qua đo đạc cơ chế, việc thay đổi vector thời gian $z(t_1) \to z(t_2)$ qua Gated Seam và MaT-LoRA tạo ra độ dịch chuyển logit trung bình khoảng **$0.1186$ đơn vị**.
2. **Biên độ thiên kiến sụp đổ**: Sự sụp đổ nhãn do LoRA tạo ra áp đặt một độ chênh lệch logit lên tới **$> 1.75$ đơn vị** nghiêng về "No".
3. **Hệ quả vật lý**: Độ nhạy $0.1186$ của vector thời gian nhỏ hơn một bậc độ lớn so với biên độ thiên kiến $1.75$. Do đó, dù tín hiệu thời gian thực sự đến được các tầng ẩn và làm thay đổi nhẹ giá trị logit, nó hoàn toàn bất lực trong việc đưa logit vượt qua ngưỡng kích hoạt để đảo ngược nhãn từ "No" sang "Yes". Mọi cặp phản thực nghiệm Gold-Flip (vốn có 100% chiều Yes -> No) đều có câu hỏi $t_1$ bị đoán sai thành "No", dẫn đến chỉ số `BothCorrect = 0/36 (0.0%)`.

---

## 4. Định Hướng Khắc Phục Thuật Toán

Dựa trên phân tích pháp y, nguyên nhân gốc rễ không nằm ở kiến trúc biểu diễn FoNE hay Gated Seam, mà nằm ở **hàm mục tiêu tối ưu hóa thiếu cân bằng gradient**. Để khắc phục triệt để hiện tượng này trong các pha nghiên cứu tiếp theo, cần áp dụng các biện pháp sau:

1. **Thay thế Cross-Entropy thuần bằng Balanced / Focal Loss trên tác vụ nhị phân**: Áp dụng trọng số tổn thất nghịch đảo động theo độ chênh lệch logit ban đầu hoặc Focal Loss để giảm thiểu hiện tượng gradient dốc từ các mẫu thiểu số.
2. **Bổ sung ràng buộc bảo toàn Entropy phân bố (Distributional Entropy Regularization)**: Phạt trực tiếp khi entropy phân bố dự đoán trên batch rơi xuống dưới ngưỡng $0.5$ bits.
3. **Phân định ranh giới can thiệp**: Không đồng nhất giữa bài toán sụp đổ nhãn nhị phân (Yes/No) và bài toán thiên kiến vị trí trắc nghiệm (MCQA Option Bias). Mỗi bài toán đòi hỏi một cơ chế can thiệp độc lập.
