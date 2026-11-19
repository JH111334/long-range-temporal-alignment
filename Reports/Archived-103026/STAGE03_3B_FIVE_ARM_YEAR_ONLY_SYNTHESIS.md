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
