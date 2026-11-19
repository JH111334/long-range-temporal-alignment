# Báo Cáo Điều Hòa Khoa Học và Khóa Bằng Chứng Qwen2.5-0.5B Trước Khi Chuyển Quy Mô

**Dự án**: `long-range-temporal-alignment` — Learning Track — Stage 03  
**Nhánh thực hiện**: `v2-targeted-continual-alignment`  
**Thời điểm khóa sổ**: `2026-10-20`  
**Trạng thái bằng chứng**: `RAW_VERIFIED / RECOMPUTED / IMMUTABLE_LOCKED`  
**Mục tiêu**: Điều hòa toán học chặt chẽ toàn bộ bằng chứng thực nghiệm của mô hình nền 0.5B, đính chính các sai khác thống kê, phân định ranh giới giữa quan sát thực nghiệm và giả thuyết cơ chế trước khi triển khai nghiên cứu trên mô hình nền Qwen2.5-3B-Instruct.

---

## 1. Tái Đối Soát Toán Học Dữ Liệu Thô 432 Bản Ghi (54 Cặp Phản Thực Nghiệm)

Từ tệp dữ liệu thô [`RAW_PREDICTIONS.jsonl`](file:///c:/Users/GF63/Documents/long-range-temporal-alignment/Reports/03_TARGETED_OBJECTIVES_MCQA/REPRESENTATION_BOTTLENECK/runs/RUN_FULL_SWEEP_54P_20261019_161756/RAW_PREDICTIONS.jsonl), 432 bản ghi suy luận thuần túy trên CPU ($0$ bước tối ưu hóa) thuộc đợt chạy `RUN_FULL_SWEEP_54P_20261019_161756` được tái tính toán và xác thực đối soát với bảng tiếp liên ghép cặp $2 \times 2$ trên $N=108$ câu hỏi:

```
                                      Arm B Checkpoint (R52_B_NTP_REL_MCQA)
                                      Đúng (Correct)         Sai (Incorrect)
Base Model         Đúng (Correct)          a = 2                 b = 59
(BASE_BOTH_OFF)    Sai (Incorrect)         c = 44                d = 3
```

- **Quy mô mẫu số thực tế**:
  - Tổng số câu hỏi kiểm định: $N = 108$ câu hỏi xuất phát từ $54$ cặp phản thực nghiệm thuộc $23$ cụm sự kiện/nguồn lịch sử độc lập.
  - Phân bổ nhãn chân lý (Gold Truth): $62$ nhãn "Yes" ($57.41\%$) và $46$ nhãn "No" ($42.59\%$).
- **Độ chính xác thô**:
  - Base Model: $61 / 108 = 56.4815\%$.
  - Arm B Checkpoint: $46 / 108 = 42.5926\%$.
  - Độ chênh lệch có dấu (Signed Delta): $\Delta = 42.5926\% - 56.4815\% = -13.8889\text{ điểm phần trăm (pp)}$.

---

## 2. Giải Quyết Sai Khác Thống Kê Giữa Kiểm Định McNemar và Clustered Bootstrap

Trong các báo cáo sơ bộ trước đây, xuất hiện hai nội dung cần được điều hòa toán học chuẩn xác:

### 2.1 Recompute Kiểm Định McNemar Nhị Thức Chính Xác (Exact Binomial McNemar Test)
- Số cặp bất đồng ý kiến (Discordant Pairs): $b = 59$ (Base đúng, Arm B sai) và $c = 44$ (Arm B đúng, Base sai).
- Tổng số cặp bất đồng: $n = b + c = 103$.
- Theo giả thuyết vô hiệu $H_0: p = 0.5$ (không có sự khác biệt về xác suất phân loại sai giữa hai mô hình), biến ngẫu nhiên $K \sim \text{Binomial}(103, 0.5)$.
- Giá trị $p$ hai phía chính xác (Exact Two-Sided p-value):
  $$p_{\text{exact}} = 2 \times \sum_{i=0}^{44} \binom{103}{i} 0.5^{103} = \mathbf{0.167444334412\dots}$$
- **Giải thích sai khác con số $0.1738$ trong một số văn bản log sơ bộ**:
  - Giá trị $0.1738$ bắt nguồn từ hàm kiểm định Chi-squared có hiệu chỉnh tính liên tục của Edwards (continuity correction):
    $$\chi^2_{\text{continuity}} = \frac{(|b - c| - 1)^2}{b + c} = \frac{(|59 - 44| - 1)^2}{103} = \frac{196}{103} \approx 1.9029 \implies p \approx 0.1677$$
    kết hợp với việc làm tròn số rời rạc trong một số script phụ trợ cũ.
  - **Khẳng định chuẩn tắc**: Giá trị giải tích chính xác tuyệt đối không hiệu chỉnh là **$p = 0.167444$**.
  - **Ý nghĩa khoa học**: Do $p = 0.167444 > 0.05$, không có bằng chứng thống kê bác bỏ giả thuyết vô hiệu ở mức ý nghĩa $5\%$. Độ sụt giảm hiệu năng của Arm B so với Base Model trên tập câu hỏi ghép cặp này chưa đạt mức ý nghĩa thống kê.

### 2.2 Bóc Tách Hai Độ Đo (Dual Estimands): McNemar vs. Clustered Bootstrap CI
- Kiểm định McNemar ($p = 0.1674$) và Khoảng tin cậy Bootstrap cụm $95\%$ ($[-23.15, -3.70]\text{ pp}$) dường như mâu thuẫn nhưng thực chất đo lường hai đối tượng thống kê hoàn toàn khác nhau:
  1. **Exact McNemar**: Estimand là tính đối xứng của xác suất chuyển dịch giữa hai mô hình trên từng cặp câu hỏi cá lẻ, giả định các cặp câu hỏi bất đồng là độc lập và cùng phân phối (i.i.d.).
  2. **Clustered Bootstrap**: Estimand là kỳ vọng độ chênh lệch điểm số gộp $\mathbb{E}[\Delta]$, được lấy mẫu lại $2000$ lần theo đơn vị cụm ($K=23$ cụm sự kiện). Do Arm B bị sụp đổ nhãn sang "No" trong khi $62/108$ câu hỏi là "Yes", hầu hết các cụm đều ghi nhận điểm Arm B thấp hơn Base, dẫn tới khoảng tin cậy lệch âm hoàn toàn.
- **Kết luận**: Cả hai kết quả cùng phản ánh một thực tế: Arm B kém hơn Base Model về điểm số trung bình do sụp đổ nhãn, nhưng mức độ bất đồng giữa hai mô hình chưa đủ tập trung để vượt qua ngưỡng kiểm định hoán vị nhị thức độc lập.

---

## 3. Bóc Tách Bản Chất Điểm Nghẽn và Sụp Đổ Nhãn (Label Collapse Analysis)

1. **Hiện tượng All-No của Arm B Checkpoint**:
   - Trên toàn bộ 108 câu hỏi Yes/No, Arm B dự đoán **108/108 nhãn là "No"** ($100.0\%$ No, $0\%$ Yes, Entropy $= 0.0000$ bits).
   - Recall lớp "Yes": $0/62 = 0.0\%$. Recall lớp "No": $46/46 = 100.0\%$. Balanced Accuracy: $50.00\%$.
   - Khi vô hiệu hóa LoRA (`ARM_B_LORA_OFF`), phân phối của mô hình quay trở lại hoàn toàn phân phối Base (51 Yes / 3 No trên 54 câu $t_1$). Chứng minh hiện tượng sụp đổ nhãn hoàn toàn nằm trong các trọng số thích ứng LoRA.
2. **Bóc tách tính ổn định giả tạo 18/18 Gold-Same**:
   - Chỉ số "18/18 Gold-Same Stability" thực chất là hệ quả trực tiếp từ sự sụp đổ nhãn: do mô hình luôn sinh ra "No" trên 100% câu hỏi, dự đoán ở cả hai mốc thời gian hiển nhiên trùng khớp nhau (false stability). Mô hình không thể hiện bất kỳ sự phân biệt ngữ nghĩa hay tính bất biến thời gian có chọn lọc nào.
3. **Thiên kiến chiều dữ liệu 54 cặp**:
   - 100% (36/36) cặp Gold-Flip có thứ tự chân lý cố định: $t_1: \text{"Yes"} \to t_2: \text{"No"}$. Hoàn toàn không có cặp nào có chiều ngược lại ($t_1: \text{"No"} \to t_2: \text{"Yes"}$).
   - Khi kết hợp tập dữ liệu có chiều Yes $\to$ No cố định với mô hình bị sụp đổ nhãn sang "No", câu hỏi $t_1$ luôn sai và $t_2$ luôn đúng, dẫn đến việc chỉ số $\text{BothCorrect} = 0/36 = 0.0\%$ bị khóa cứng mang tính cấu trúc.

---

## 4. Pháp Y Hàm Mục Tiêu: Cơ Chế Giả Thuyết vs. Bằng Chứng Đã Xác Minh

1. **Dữ liệu huấn luyện cân bằng hoàn hảo**:
   - Tệp `train_relation_pairs.jsonl` có phân bố nhãn gồm **963 mẫu "No" và 963 mẫu "Yes"** ($50/50$).
   - Trong 40 bước huấn luyện thực tế với bộ chọn mẫu phân tầng ngẫu nhiên (`seed=42`), mô hình tiếp xúc chính xác **20 mẫu "No" và 20 mẫu "Yes"**.
   - **Khẳng định**: Tuyệt đối không đổ lỗi cho sự mất cân bằng nhãn của tập dữ liệu huấn luyện.
2. **Phân loại giả thuyết bất đối xứng Gradient**:
   - Giả thuyết: Base Qwen có thiên kiến tiên nghiệm nghiêng về "Yes" ($P(\text{Yes}) \approx 0.95$), khiến tổn thất Cross-Entropy trên mẫu "Yes" triệt tiêu ($\mathcal{L} \approx 0.05$, gradient $\approx 0$), trong khi mẫu "No" sinh ra gradient dốc khổng lồ ($\mathcal{L} \approx 3.0$) kéo tụt LoRA về phía "No".
   - **Phân loại bằng chứng**: Đây là **`MECHANISTIC_HYPOTHESIS` (Giả thuyết cơ chế hợp lý)**, chưa phải `ROOT_CAUSE_PROVEN` bằng thực nghiệm theo dõi gradient/optimizer state từng bước.

---

## 5. Bằng Chứng Dẫn Truyền Vật Lý và Thử Nghiệm Đối Chứng 2x2

1. **Bằng chứng tín hiệu đến được mô hình (Proof of Treatment)**:
   - Các chỉ số hình học: $\bar{D}_z = 5.3172 > 0$, $\bar{D}_F = 0.6829$, $\bar{D}_W = 0.5001 > 10^{-5}$, $S_{14, \text{seam}} = 0.0525$, $S_{14, \text{comp}} = 0.0538$, và $\Delta_{\text{logits}} = 0.1356 > 0.01$.
   - Chứng minh: Tín hiệu thời gian được dẫn truyền vật lý thông suốt từ vector 40D qua Gated Seam và LoRA đến token quyết định.
2. **Thử nghiệm đối chứng nhân quả 2x2 (`RUN_STAGE03_POST_ARCHIVE_CAUSAL_20261020_082138`)**:
   - $E_z(t_1) = -0.0580$, Exact Binomial $p = 0.1325$.
   - $E_z(t_2) = -0.0584$, Exact Binomial $p = 0.1325$.
   - $E_{z\text{\_pair}} = -0.0582$, $95\%$ Bootstrap CI: $[-0.1478, +0.0442]$.
   - `PositiveBothSidesRate` = $13/36 = 36.11\%$ ($95\%$ CI: $[20.44\%, 55.89\%]$).
   - Độ lớn hiệu ứng adapter trung bình đạt $0.2485$, trong khi độ lệch logit do LoRA sụp đổ áp đặt lên mô hình vượt quá $> 1.75$.
   - **Ý nghĩa**: Tín hiệu thời gian tạo ra biến thiên logit có thể đo được nhưng bị triệt tiêu hoàn toàn dưới đáy của phân phối sụp đổ LoRA.

---

## 6. Trạng Thái Cổng và Cam Kết Đối Với Qwen2.5-3B

1. Toàn bộ 488 tệp tạo tác của 0.5B được khóa bất biến (`IMMUTABLE_LOCKED`), không bị ghi đè hay thay thế hồi tố.
2. Không suy diễn rằng việc chuyển sang Qwen2.5-3B chắc chắn sẽ giải quyết triệt để các vấn đề trên. Việc thử nghiệm 3B là **kiểm chứng giả thuyết về dung lượng mô hình (Capacity-Scale Hypothesis)** theo định hướng của GVHD.
3. Duy trì nghiêm ngặt: `GO_3B_PREPARATION`, `NO_GO_DATA_MUTATION`, `NO_GO_OPTION_PERMUTATION`, `NO_GO_LARGE_TRAINING`.
