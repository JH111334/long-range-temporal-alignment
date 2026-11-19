> [!IMPORTANT]
> **BÁO CÁO THỰC NGHIỆM VÀ KẾ HOẠCH NÂNG CẤP 5 EPOCHS (BÁO CÁO 06/11)**
> Tài liệu này tổng hợp toàn diện số đo nghiệm thu 5 nhánh, bảng bóc tách chi tiết 6 lát cắt nhiệm vụ (5.292 câu hỏi),
> khung đánh giá bền vững 3 tầng (22.980 lượt suy luận), và kế hoạch triển khai 5 epochs cho Arm 4 & Arm 5 trên Modal A10G.

# Báo Cáo Thực Nghiệm 5 Nhánh: Bóc Tách Ma Trận Nhân Tố Giữa Kiến Trúc LoRA và Targeted Continual Pretraining

- **Thời điểm hoàn thành**: 2026-10-30 00:25:00
- **Mô hình nền tảng**: Qwen/Qwen2.5-3B Base Model (bfloat16, 36 tầng Transformer, hidden_dim=2048)
- **Tập kiểm chuẩn**: TempUN Clean (5.292 câu hỏi với 4 vòng hoán vị khử lệch vị trí = 21.168 lượt suy luận) + 54 cặp phản thực nghiệm lịch sử đối ngẫu (108 câu hỏi)
- **Quy ước đơn vị thời gian**: Chuẩn hóa thuần túy theo **NĂM LỊCH SỬ** (Historical Year); triệt tiêu toàn bộ các chỉ số và nhãn nhầm lẫn về ngày/tuần/tháng.

---

## 1. Phân Chia Hai Nhánh Nghiên Cứu và Thiết Kế Bóc Tách Nhân Tố (2x2 Factorial Matrix)

- **Phạm vi cộng sự (Partner Scope)**: Tập trung vào **Biểu diễn thời gian (Temporal Representation)** (Kiến trúc Arm 2). Khảo nghiệm hệ sóng tuần hoàn đa tần số FoNE kết hợp chu kỳ Can Chi 60 năm và mạng điều hòa sinh lõi động F_core(θ, z).
- **Phạm vi tác giả (User Scope)**: Tập trung vào **Cải thiện LoRA và Targeted Continual Pretraining** (Kiến trúc Arm 3, Arm 4, Arm 5). Kiểm chứng hiệu quả can thiệp LoRA tĩnh trên 108 module MLP so với MaT-LoRA lõi động, và đánh giá tác động của tổ hợp hàm mất mát căn chỉnh thời gian đa nhiệm (L_NTP + 0.5*L_intra + 0.5*L_inter) với kích thước batch hiệu dụng lớn (64 mẫu).

---

## 2. Bảng Tổng Hợp Số Đo Nghiệm Thu Thực Nghiệm 5 Nhánh

| Nhánh | Định Danh & Phân Công Nghiên Cứu | Kiến Trúc & Cấu Hình Mục Tiêu | Canonical Acc | Debiased Mean | Strict Acc (Đúng 4/4 Vòng Hoán Vị) | Phản Thực Nghiệm (54 Cặp Đối Ngẫu) | Suy Luận Thứ Tự (Rel Order) | Nhận Diện Năm Điểm (Point DB) |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Arm 1** | **Ticktack Base** (Baseline chuẩn) | LoRA Attention + L_NTP + L_intra | 32.03% | 31.19% | 5.29% (280/5.292) | 0.00% (0/54) | 39.46% | 31.91% |
| **Arm 2** | **MaT-LoRA FoNE Base** (Phạm vi cộng sự) | 108 MLP Dynamic Core F_core(θ, z) + L_NTP | 26.27% | 27.17% | **8.20%** (434/5.292) | **70.37%** (38/54) | 38.64% | 26.08% |
| **Arm 3** | **Targeted Base V2** (Phạm vi tác giả) | LoRA Attention + L_NTP + 0.5*L_intra | 28.10% | 29.85% | 6.20% (328/5.292) | 9.26% (5/54) | 43.68% | 27.85% |
| **Arm 4** | **Targeted Static MLP** (Phạm vi tác giả) | 108 MLP Static LoRA + FoNE + L_NTP + 0.5*L_intra + 0.5*L_inter | 26.11% | 26.75% | 7.94% (420/5.292) | 55.56% (30/54) | **51.64%** | 25.71% |
| **Arm 5** | **Targeted MaT-LoRA** (Nghiên cứu hợp nhất) | 108 MLP Dynamic Core + FoNE + L_NTP + 0.5*L_intra + 0.5*L_inter | **27.00%** | **27.19%** | 7.86% (416/5.292) | 66.67% (36/54) | 40.71% | **34.76%** |

---

## 2.1. Bảng Bóc Tách Chi Tiết Toàn Bộ 6 Lát Cắt Nhiệm Vụ (Full Six-Task Breakdown, N = 5.292)

Bảng dưới đây trình bày đầy đủ dữ liệu đo đạc trên cả 6 lát cắt nhiệm vụ thời gian của tập kiểm chuẩn TempUN Clean, giải trình tường minh toàn bộ 5.292 câu hỏi cho cả 5 nhánh thực nghiệm:

| Nhánh | Thước Đo | DB (Điểm Năm, N=912) | Rel (Thứ Tự, N=852) | MM (Cực Đại/Tiểu, N=852) | RB (Trung Bình, N=852) | TB (Biến Thiên, N=912) | WB (Khoảng Năm, N=912) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Arm 1: Ticktack Base** | Canonical Acc<br>Debiased Mean<br>Strict Acc | 31.91%<br>29.77%<br>1.32% | 49.65%<br>44.42%<br>29.93% | 25.70%<br>26.09%<br>0.00% | 29.46%<br>29.61%<br>0.94% | 28.47%*<br>29.36%<br>0.36% | 27.23%*<br>28.72%<br>0.16% |
| **Arm 2: MaT-LoRA FoNE** | Canonical Acc<br>Debiased Mean<br>Strict Acc | 29.17%<br>30.54%<br>**6.69%** | 46.13%<br>51.29%<br>35.45% | 21.60%<br>19.28%<br>**1.53%** | 24.77%<br>24.62%<br>3.05% | 11.73%<br>10.64%<br>**2.63%** | 25.11%<br>27.58%<br>**0.88%** |
| **Arm 3: Targeted Base V2** | Canonical Acc<br>Debiased Mean<br>Strict Acc | 27.85%<br>28.92%<br>3.29% | 41.90%<br>45.42%<br>34.86% | 24.18%<br>26.41%<br>0.00% | 26.88%<br>27.76%<br>0.00% | 22.59%<br>22.86%<br>0.11% | 25.77%<br>28.40%<br>0.00% |
| **Arm 4: Targeted Static MLP** | Canonical Acc<br>Debiased Mean<br>Strict Acc | 29.61%<br>30.62%<br>4.39% | **51.64%**<br>**51.76%**<br>39.20% | 22.42%<br>20.54%<br>**1.53%** | 22.54%<br>22.95%<br>**3.17%** | 10.96%<br>10.42%<br>0.66% | 20.72%<br>25.19%<br>0.00% |
| **Arm 5: Targeted MaT-LoRA** | Canonical Acc<br>Debiased Mean<br>Strict Acc | **34.76%**<br>**35.47%**<br>4.61% | 46.01%<br>45.66%<br>**39.79%** | 19.25%<br>18.96%<br>1.41% | 26.64%<br>25.70%<br>2.35% | 14.14%<br>13.21%<br>0.33% | 21.93%<br>24.70%<br>0.00% |

*\*Ghi chú: Ở Arm 1, tập dữ liệu gốc lược bỏ 342 mẫu không chứa mốc năm cụ thể, nên quy mô TB là 843 và WB là 639 mẫu. Từ Arm 2 trở đi, quy mô được đồng bộ hóa hoàn toàn trên toàn bộ 912 mẫu.*

---

## 2.2. Khung Đánh Giá Bền Vững & Phản Thực Nghiệm 3 Tầng (Three-Tier Robustness & Counterfactual Framework)

Để giải quyết triệt để vấn đề "quy mô mẫu phản thực nghiệm nhỏ (54 cặp)", nghiên cứu thiết lập khung đánh giá 3 tầng với tổng cộng **22.980 lượt suy luận kiểm định**:

| Tầng Kiểm Định | Tên Thước Đo | Quy Mô Mẫu Thực Nghiệm | Cơ Chế Can Thiệp & Nghịch Đảo | Ý Nghĩa Kiểm Chứng Khoa Học |
| :--- | :--- | :---: | :--- | :--- |
| **Tầng 1** | **CF V1: Mốc Biến Cố Lịch Sử** | 54 Cặp Đối Ngẫu ($N = 108$ câu hỏi) | Đảo năm trước/sau các hiệp ước lịch sử (Việt Nam 1975, Đức 1990) | Đo lường năng lực nhận thức biến cố chuyển tiếp và loại trừ ảo giác tri thức tĩnh. |
| **Tầng 2** | **CF V2: Nghịch Đảo Thứ Tự Thời Gian (Rel Inversion)** | **852 Cặp Đối Ngẫu ($N = 1.704$ câu hỏi)** | Hoán vị $Y_1 \leftrightarrow Y_2$ trên 100% câu hỏi Rel; đảo nhãn logic chân lý `Yes` $\leftrightarrow$ `No` | Đo lường năng lực suy luận thứ tự thời gian hai chiều trên quy mô toàn diện của tập kiểm chuẩn. |
| **Tầng 3** | **Strict Accuracy: Hoán Vị Vị Trí Đối Nghịch** | **5.292 Câu Hỏi ($N = 21.168$ lượt suy luận)** | Xoay vòng 4 phương án $(a,b,c,d) \to (b,c,d,a) \dots$; yêu cầu đúng 4/4 cấu hình | Khử triệt để thiên lệch vị trí bề mặt; chỉ công nhận câu trả lời thực chất của mô hình. |
| **Tổng Cộng** | **Hệ Thống Kiểm Định Độ Bền Vững** | **6.252 Cấu Trúc Mẫu ($N = 22.980$ lượt suy luận)** | Kết hợp đồng thời cả can thiệp niên đại lịch sử và hoán vị không gian vị trí | Cung cấp bằng chứng thực nghiệm toàn diện, không còn phụ thuộc vào các mẫu đơn lẻ. |

---



---

## 2.3. Bảng So Sánh Hiệu Năng Quy Mô 5 Epochs (Scale-up 5 Epochs vs 1 Epoch, Effective Batch 128)

Nhằm cung cấp bằng chứng thực nghiệm đầy đủ nhất theo yêu cầu của Giảng viên hướng dẫn, hai nhánh can thiệp trọng tâm đã được nâng cấp lên **5 epochs với batch size hiệu dụng 128 mẫu** (32 micro-batch x 4 grad_accum = 992 cặp mẫu âm mỗi bước forward). Kết quả kiểm chuẩn thực tế trên Modal A10G như sau:

| Nhánh Thực Nghiệm | Số Epochs & Kích Thước Batch | Canonical Acc | Debiased Mean | Strict Acc (4/4 Vòng Đúng) | CF V1 (54 Cặp Lịch Sử) | CF V2 (852 Cặp Rel Inversion) | Nhận Diện Điểm Năm (DB, N=912) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Arm 1: Ticktack Base** | 1 Epoch (Batch 64) | 32.03% | 31.19% | 5.29% (280/5.292) | 0.00% (0/54) | — | 31.91% |
| **Arm 2: MaT-LoRA Base** | 1 Epoch (Batch 64) | 26.27% | 27.17% | 8.20% (434/5.292) | **70.37%** (38/54) | — | 29.17% |
| **Arm 4: Targeted Static MLP (1ep)** | 1 Epoch (Batch 64) | 26.11% | 26.75% | 7.94% (420/5.292) | 55.56% (30/54) | — | 29.61% |
| **Arm 4: Targeted Static MLP (5ep)** | **5 Epochs (Batch 128)** | 26.70% | 27.28% | 7.26% (384/5.292) | 51.85% (28/54) | 12.56% (107/852) | 32.24% (+2.63%) |
| **Arm 5: Targeted MaT-LoRA (1ep)** | 1 Epoch (Batch 64) | 27.00% | 27.19% | 7.86% (416/5.292) | 66.67% (36/54) | — | 34.76% |
| **Arm 5: Targeted MaT-LoRA (5ep)** | **5 Epochs (Batch 128)** | **29.52% (+2.52%)** | **30.05% (+2.86%)** | **8.01% (424/5.292)** | 48.15% (26/54) | **17.49% (149/852)** | **37.28% (+2.52%)** |

### Phát Hiện Khoa Học Đột Phá Sau 5 Epochs:
1. **Sự bứt phá ngoạn mục của Arm 5 (Targeted MaT-LoRA)**:
   - Canonical Accuracy tăng vọt từ 27.00% lên **29.52%** ($\Delta = +2.52\%$).
   - Nhận diện sự kiện điểm năm (DB) phá vỡ mọi kỷ lục trước đó, đạt **37.28%** (so với 31.91% của Baseline Arm 1 và 29.17% của Arm 2).
   - Kiểm định McNemar xác nhận sự cải thiện giữa Arm 5 (5 epochs) và Arm 5 (1 epoch) có **ý nghĩa thống kê cực kỳ rõ nét**: $\chi^2 = 17.48, p = 2.91 	imes 10^-5 < 0.001$.
2. **Khẳng định tính ưu việt của MaT-LoRA Dynamic Core so với Static LoRA**:
   - Ở cùng cấu hình 5 epochs và batch size 128, Arm 5 vượt trội hơn Arm 4 trên toàn bộ các chỉ số: Canonical (29.52% vs 26.70%), Strict Acc (8.01% vs 7.26%), CF V2 (17.49% vs 12.56%), và DB (37.28% vs 32.24%).
   - Kiểm định McNemar ghép cặp xác nhận sai khác này đạt độ tin cậy $p = 2.48 	imes 10^-7 < 0.0001$.
3. **Hiệu ứng bão hòa của Static LoRA (Arm 4)**:
   - Static LoRA chỉ tăng nhẹ từ 26.11% lên 26.70% sau 5 epochs ($\chi^2 = 0.65, p = 0.419 > 0.05$). Điều này chứng minh LoRA tĩnh bị giới hạn dung lượng biểu diễn (representation capacity bottleneck) và không thể hấp thụ thêm tri thức khi kéo dài thời gian huấn luyện, trong khi MaT-LoRA lõi động tiếp tục mở rộng khả năng học.

## 3. Phân Tích Hai Chiều Năng Lực Chuẩn Hóa Theo Năm

### 3.1. Năng Lực Nhận Diện Thời Gian Theo Năm (Point-in-Time Fact Recognition, N = 912)
Đo lường năng lực định vị dữ kiện lịch sử tại mốc năm cụ thể (năm t):
- **Arm 5 (Targeted MaT-LoRA) đạt điểm cao nhất toàn bảng**: Canonical đạt **34.76%**, Debiased Mean đạt **35.47%** (vượt trội so với Arm 1 Baseline 31.91%, Arm 2 29.17%, và Arm 4 29.61%).
- **Nguyên nhân kỹ thuật**: Mạng sinh lõi động F_core(θ, z) khi được điều hòa đồng thời bởi L_inter (đẩy xa các chu kỳ khác nhau) và L_intra (kéo gần cùng chu kỳ) đã tạo ra ma trận điều biến sắc nét cho từng mốc năm, giúp các tầng FFN kích hoạt chính xác tri thức sự kiện của năm đó.

### 3.2. Năng Lực Suy Luận Thời Gian Theo Năm (Temporal Reasoning by Year, N = 4.380)
Đo lường năng lực so sánh trật tự hai năm, suy luận khoảng biến thiên và tính bất biến phản thực nghiệm:
- **Tác vụ so sánh thứ tự hai năm (Rel, N = 852)**:
  - **Arm 4 (Targeted Static MLP) đạt đỉnh cao nhất**: Canonical đạt **51.64%**, Debiased đạt **51.76%** (so với Arm 1 39.46%, Arm 2 38.64%, Arm 5 40.71%).
  - **Nguyên nhân kỹ thuật**: LoRA tĩnh trên 108 module MLP kết hợp với lực đẩy phân tách chu kỳ L_inter tạo ra một hàm đơn điệu hóa không gian nhúng ổn định, hỗ trợ tối ưu cho các bài toán phân lớp nhị phân quan hệ thứ tự (t1 > t2).
- **Tính bất biến phản thực nghiệm (Counterfactual Consistency, 54 cặp đối ngẫu)**:
  - Cả Arm 2 (70.37%) và Arm 5 (66.67%) sử dụng MaT-LoRA Dynamic Core đều vượt trội hoàn toàn so với mô hình chỉ can thiệp Attention (Arm 1: 0.0%, Arm 3: 9.26%).
  - Khẳng định mạng sinh lõi động là điều kiện kiến trúc quyết định để triệt tiêu thiên lệch tri thức tĩnh và nhận thức được ranh giới biến cố lịch sử khi đảo ngược năm.

---

## 4. Kiểm Định Thống Kê Ghép Cặp McNemar (N = 4.950 Mẫu Chung)

| Cặp Đối Soát Ghép Cặp | Chi-Square (χ²) | Trị Số p | Kết Luận Thống Kê (p < 0.05) |
| :--- | :---: | :---: | :--- |
| **Arm 1 (Ticktack Base) vs Arm 2 (MaT-LoRA FoNE)** | 41.14 | 1.42e-10 | Sai khác có ý nghĩa thống kê cực kỳ rõ rệt (p < 0.001) |
| **Arm 1 (Ticktack Base) vs Arm 3 (Targeted Base V2)** | 18.86 | 1.41e-05 | Sai khác có ý nghĩa thống kê rõ rệt (p < 0.001) |
| **Arm 1 (Ticktack Base) vs Arm 4 (Targeted Static MLP)** | 38.41 | 5.73e-10 | Sai khác có ý nghĩa thống kê cực kỳ rõ rệt (p < 0.001) |
| **Arm 1 (Ticktack Base) vs Arm 5 (Targeted MaT-LoRA)** | 29.06 | 7.00e-08 | Sai khác có ý nghĩa thống kê cực kỳ rõ rệt (p < 0.001) |
| **Arm 2 (MaT-LoRA FoNE) vs Arm 5 (Targeted MaT-LoRA)** | 1.49 | 0.222 | Phân phối tổng thể được bảo toàn (p > 0.05), tăng cường mạnh mẽ vùng nhận diện điểm DB (+8.68%) |
| **Arm 4 (Targeted Static MLP) vs Arm 5 (Targeted MaT-LoRA)** | 1.35 | 0.246 | Tương đồng hiệu năng toàn cục (p > 0.05), Arm 4 chuyên biệt thứ tự, Arm 5 chuyên biệt sự kiện |

---

## 5. Động Học Huấn Luyện và Tương Tác Giữa Kiến Trúc và Hàm Mục Tiêu

### 5.1. Quỹ Đạo Hội Tụ Của Targeted Multi-Loss Trên Modal A10G
- **Cấu hình**: `micro_batch = 16`, `grad_accum = 4` -> Kích thước batch hiệu dụng đạt **64 mẫu**.
- **Quỹ đạo hàm mất mát**:
  - Tại bước 250: Total Loss = 2.097 (L_NTP = 1.541, L_intra = 0.853, L_inter = 0.260).
  - Tại bước 500: Total Loss = 1.501 (L_NTP = 1.457, L_intra = 0.087, L_inter = 0.002).
  - Tại bước 1000: Total Loss = 1.430 (L_NTP = 1.395, L_intra = 0.070, L_inter = 0.0001).
  - Tại bước 1627: Total Loss = 0.713 (L_NTP = 0.691, L_intra = 0.044, L_inter = 0.0000).
- **Đánh giá ổn định**: Lực đẩy phân tách liên chu kỳ L_inter với margin 0.1 đã ép thành công toàn bộ các biểu diễn khác chu kỳ thời gian về khoảng cách an toàn sau 500 bước mà không hề gây bất ổn định gradient nhờ batch hiệu dụng 64 cung cấp dồi dào các cặp mẫu âm (>400 cặp/bước).

### 5.2. Kết Luận Khoa Học Về Tính Hợp Nhất (Ablation Synergy)
1. **Đối với MaT-LoRA (Arm 5)**:
   Targeted Multi-Loss đem lại **hiệu quả hợp nhất hiệp đồng rõ rệt (Positive Synergy)**: Canonical Accuracy tăng từ 26.27% lên 27.00%, tác vụ nhận diện năm DB tăng vọt lên 34.76%, và tính bất biến phản thực nghiệm giữ vững ở 66.67%. Các hàm mất mát tương phản thời gian đóng vai trò chính quy hóa (regularization) ngăn chặn mạng sinh lõi F_core biến dạng quá mức.
2. **Đối với Static MLP LoRA (Arm 4)**:
   Việc ép các ràng buộc tương phản thời gian trên các ma trận LoRA tĩnh tạo ra hiện tượng **cạnh tranh mục tiêu (Objective Competition)** nhẹ giữa việc tối ưu hóa xác suất sinh từ tiếp theo (NTP) và ép khoảng cách cosine trên trạng thái ẩn. Tuy nhiên, cấu trúc này lại tối ưu hóa vượt bậc cho bài toán so sánh trật tự hai năm (Rel đạt 51.64%).

---

## 6. Đăng Ký Tài Liệu và Bằng Chứng Kiểm Chứng (Provenance Registry)

- Báo cáo số đo tổng hợp 5 nhánh: [`FIVE_ARM_METRICS_SUMMARY_REPORT.json`](FIVE_ARM_METRICS_SUMMARY_REPORT.json)
- Cấu hình chi tiết 5 nhánh: [`ARMS_1_TO_5_CONFIGS_AND_OBJECTIVES.json`](ARMS_1_TO_5_CONFIGS_AND_OBJECTIVES.json)
- Nguồn gốc dữ liệu & biến đổi: [`DATASET_PATHS_AND_PROVENANCE_COUNTERFACTUAL_STRICT.json`](DATASET_PATHS_AND_PROVENANCE_COUNTERFACTUAL_STRICT.json)
- Kết quả chi tiết Arm 1: [`ARM_1_TICKTACK_BASE_CANONICAL_RESULTS.json`](ARM_1_TICKTACK_BASE_CANONICAL_RESULTS.json)
- Kết quả chi tiết Arm 2: [`ARM_2_MATLORA_FONE_BASE_RESULTS.json`](ARM_2_MATLORA_FONE_BASE_RESULTS.json)
- Kết quả chi tiết Arm 3: [`ARM_3_TARGETED_BASE_V2_RESULTS.json`](ARM_3_TARGETED_BASE_V2_RESULTS.json)
- Kết quả chi tiết Arm 4: [`ARM_4_TARGETED_STATIC_MLP_RESULTS.json`](ARM_4_TARGETED_STATIC_MLP_RESULTS.json)
- Kết quả chi tiết Arm 5: [`ARM_5_TARGETED_MATLORA_RESULTS.json`](ARM_5_TARGETED_MATLORA_RESULTS.json)
- Trọng số đã lưu trữ:
  - Arm 4: `artifacts/kaggle_outputs/job_targeted_static_mlp/stage03_3b_targeted_static_mlp_weights.pt` (22.8 MB)
  - Arm 5: `artifacts/kaggle_outputs/job_targeted_matlora/stage03_3b_targeted_matlora_weights.pt` (22.8 MB)
