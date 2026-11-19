# Danh Mục Bằng Chứng & Kết Quả Thực Nghiệm Stage 03 Qwen2.5-3B (Chuẩn Năm Lịch Sử 5 Nhánh)

- **Ngày cập nhật**: 2026-10-30 00:25:00
- **Mô hình nền**: `Qwen/Qwen2.5-3B` Base Model
- **Tập kiểm chuẩn**: TempUN Full Clean (5.292 MCQA) + 54 Historical Counterfactual Pairs
- **Quy ước chuẩn**: Toàn bộ chỉ số được phân loại thuần túy theo **NĂM LỊCH SỬ**, triệt tiêu toàn bộ ranh giới ngày/tuần/tháng.

## 1. Tệp Biên Nhận Chuẩn Hóa và Báo Cáo Đối Soát

| Tên Tệp | Định Dạng | Mô Tả | Đường Dẫn |
| :--- | :--- | :--- | :--- |
| **`FIVE_ARM_YEAR_ONLY_RECEIPT.json`** | JSON | Toàn bộ biên nhận 5 nhánh phân rã ma trận bóc tách nhân tố và kiểm định McNemar | [`FIVE_ARM_YEAR_ONLY_RECEIPT.json`](FIVE_ARM_YEAR_ONLY_RECEIPT.json) |
| **`RECEIPT_STAGE03_3B_YEAR_ONLY_CLOSURE.yaml`** | YAML | Biên nhận nghiệm thu khoa học khép góc chuẩn năm lịch sử phục vụ báo cáo giảng viên | [`RECEIPT_STAGE03_3B_YEAR_ONLY_CLOSURE.yaml`](RECEIPT_STAGE03_3B_YEAR_ONLY_CLOSURE.yaml) |
| **`STAGE03_3B_FIVE_ARM_YEAR_ONLY_SYNTHESIS.md`** | Markdown | Báo cáo học thuật chi tiết 5 nhánh đối soát kiến trúc LoRA, Targeted Loss và suy luận phản thực nghiệm | [`STAGE03_3B_FIVE_ARM_YEAR_ONLY_SYNTHESIS.md`](../../../../Reports/03_TARGETED_OBJECTIVES_MCQA/STAGE03_3B_FIVE_ARM_YEAR_ONLY_SYNTHESIS.md) |
| **`ticktack_3b_base_canonical_results.json`** | JSON | Kết quả dự đoán chi tiết từng câu của Arm 1 (Ticktack Base) | [`ticktack_3b_base_canonical_results.json`](ticktack_3b_base_canonical_results.json) |
| **`stage03_3b_matlora_fone_results.json`** | JSON | Kết quả dự đoán chi tiết từng câu của Arm 2 (MaT-LoRA FoNE Base) | [`stage03_3b_matlora_fone_results.json`](stage03_3b_matlora_fone_results.json) |
| **`stage03_3b_base_targeted_results.json`** | JSON | Kết quả dự đoán chi tiết từng câu của Arm 3 (Targeted Base V2) | [`stage03_3b_base_targeted_results.json`](stage03_3b_base_targeted_results.json) |
| **`stage03_3b_targeted_static_mlp_results.json`** | JSON | Kết quả dự đoán chi tiết từng câu của Arm 4 (Targeted Static MLP LoRA) | [`stage03_3b_targeted_static_mlp_results.json`](stage03_3b_targeted_static_mlp_results.json) |
| **`stage03_3b_targeted_matlora_results.json`** | JSON | Kết quả dự đoán chi tiết từng câu của Arm 5 (Targeted MaT-LoRA) | [`stage03_3b_targeted_matlora_results.json`](stage03_3b_targeted_matlora_results.json) |

## 2. Bảng Tổng Hợp Số Đo Nghiệm Thu 5 Nhánh

| Nhánh | Định Danh & Phân Công Nghiên Cứu | Kiến Trúc & Cấu Hình Mục Tiêu | Canonical Acc | Debiased Mean | Strict Acc (Đúng 4/4 Vòng Hoán Vị) | Phản Thực Nghiệm (54 Cặp Đối Ngẫu) | Suy Luận Thứ Tự (Rel Order) | Nhận Diện Năm Điểm (Point DB) |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Arm 1** | **Ticktack Base** (Baseline chuẩn) | LoRA Attention + L_NTP + L_intra | 32.03% | 31.19% | 5.29% (280/5.292) | 0.00% (0/54) | 39.46% | 31.91% |
| **Arm 2** | **MaT-LoRA FoNE Base** (Phạm vi cộng sự) | 108 MLP Dynamic Core F_core(θ, z) + L_NTP | 26.27% | 27.17% | **8.20%** (434/5.292) | **70.37%** (38/54) | 38.64% | 26.08% |
| **Arm 3** | **Targeted Base V2** (Phạm vi tác giả) | LoRA Attention + L_NTP + 0.5*L_intra | 28.10% | 29.85% | 6.20% (328/5.292) | 9.26% (5/54) | 43.68% | 27.85% |
| **Arm 4** | **Targeted Static MLP** (Phạm vi tác giả) | 108 MLP Static LoRA + FoNE + L_NTP + 0.5*L_intra + 0.5*L_inter | 26.11% | 26.75% | 7.94% (420/5.292) | 55.56% (30/54) | **51.64%** | 25.71% |
| **Arm 5** | **Targeted MaT-LoRA** (Nghiên cứu hợp nhất) | 108 MLP Dynamic Core + FoNE + L_NTP + 0.5*L_intra + 0.5*L_inter | **27.00%** | **27.19%** | 7.86% (416/5.292) | 66.67% (36/54) | 40.71% | **34.76%** |
