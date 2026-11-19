# Thư Mục Lưu Trữ Baseline Bài Báo Gốc TickTack (arXiv:2503.04150v3)

> **Căn cứ tài liệu:** Xue Han et al., *"Temporal Alignment of LLMs through Cycle Encoding for Long-Range Time Representations"*, arXiv:2503.04150v3, sửa đổi ngày 21/10/2025.  
> **Ngày lập hồ sơ:** 14/10/2026.  
> **Cơ quan quản trị:** Learning Track — Long-Range Temporal Alignment.

---

## 1. Mục Đích và Phân Định Ranh Giới Bằng Chứng

Thư mục này lưu trữ độc lập và có cấu trúc toàn bộ các số liệu và đặc tả kỹ thuật **có thể xác thực được** từ bài báo TickTack gốc, nhằm phục vụ công tác đối chuẩn (benchmark) khoa học trung thực, loại bỏ hoàn toàn các nhận định so sánh thiên lệch.

Bắt buộc phân biệt ba tầng bằng chứng:
1. **`PAPER_REPORTED`**: Các con số do tác giả bài báo công bố (Table 1, Table 2, Table 3, Eq 1-10); sao chép nguyên văn với tọa độ ô và liên kết kiểm chứng. **Tuyệt đối không phải kết quả đo trên máy nhóm.**
2. **`REPRODUCED`**: Các con số do nhóm trực tiếp tái lập từ mã nguồn và dữ liệu công bố (hiện trạng: **`BLOCKED_MISSING_ARTIFACTS`** do tác giả chưa phát hành mã nguồn, checkpoint và tập kiểm thử TempLS).
3. **`INTERNAL_MATCHED`**: Các kết quả nội bộ được đo đạc trên **cùng tập câu hỏi đóng băng, cùng giao thức phân tích cú pháp và cùng mô hình nền**.

---

## 2. Cấu Trúc Thư Mục

```text
data/Output/TICKTACK_ORIGINAL_2503_04150_V3/
├── README.md                                 # Tài liệu định hướng và ranh giới khoa học
├── manifest.json                             # Bản kê toàn bộ các tệp và mã băm SHA256
├── sources/
│   ├── paper_source.json                     # Thông tin nguồn arXiv, tác giả, phiên bản
│   ├── asset_discovery.json                  # Rà soát mã nguồn, checkpoints, bộ dữ liệu
│   └── ambiguities_and_missing_info.json     # Các điểm nhập nhằng kỹ thuật trong bài báo
├── paper_reported/
│   ├── table_1_all_datasets.json             # Toàn bộ Bảng 1 (TempLS, TempLAMA, TempUN)
│   ├── table_1_tempun.json                   # Cột TempUN Bảng 1 phục vụ đối chiếu
│   ├── table_2_encoding_ablation.json        # Bảng 2 cắt bỏ mã hóa chu kỳ trên TempLS
│   ├── additional_tables_and_figures_index.json # Mục lục 9 hình vẽ và Bảng 3 mẫu dữ liệu
│   └── equations_and_protocol.json           # Phương trình (1)-(10) và siêu tham số huấn luyện
├── reproducibility/
│   ├── feasibility_matrix.json               # Ma trận khả thi 6 kịch bản tái lập
│   ├── tempun_dataset_parity.json            # Đối soát TempUN gốc (~9M) vs TempUN Small (180)
│   ├── blockers.json                         # Rào cản tái lập (Mã nguồn, phần cứng, split)
│   ├── proposed_reproduction_plan.md         # Đề xuất tái lập thích ứng (Chờ duyệt)
│   └── status.json                           # Trạng thái tái lập tổng thể
├── internal_matched/
│   ├── comparability_matrix.json             # Ma trận so sánh ghép cặp trung thực
│   ├── historical_results_index.json         # Danh mục các run lịch sử của nhóm và đối tác
│   ├── matched_run_plan.json                 # Kế hoạch thực nghiệm ghép cặp chuẩn tắc
│   └── matched_metrics.json                  # Số liệu đo đạc thực tế trên tập 24 mẫu P12
└── audit/
    ├── validation_report.json                # Báo cáo kiểm định tính toàn vẹn hồ sơ
    ├── sha256_manifest.json                  # Bảng mã băm SHA256 kiểm tra tính bất biến
    └── claims_register.json                  # Đăng ký và phân loại các tuyên bố khoa học
```

---

## 3. Ranh Giới So Sánh Khoa Học Bắt Buộc (Scientific Invariants)

1. **Tuyệt đối KHÔNG tuyên bố $37.50\% >$ TickTack gốc:**
   - Mốc $37.50\% \pm 4.17\%$ là kết quả khảo sát trên tập chẩn đoán sơ bộ $n=24$ mẫu dưới giao thức $P_{12}$, không đại diện cho toàn bộ benchmark.
   - Bài báo gốc công bố kết quả trên các mô hình lớn (Qwen2.5-3B: $64.96\%$, Qwen2.5-7B: $74.29\%$), hoàn toàn **không công bố kết quả cho Qwen2.5-0.5B**.
   - Trên cùng tập 24 mẫu $P_{12}$, mô hình nền nguyên bản chưa thích ứng (`BASE_MODEL_FROZEN`) đạt **$41.67\%$** (cao hơn M0-FC40 $37.50\%$). Do đó, dữ liệu hiện tại **chưa chứng minh được độ lợi (gain) của M0-FC40 so với Base model**.
2. **Khác biệt bản chất giữa TickTack gốc và Dual-Cycle MaT-LoRA:**
   - *TickTack gốc (Han et al.):* Tọa độ cực Lục Thập Hoa Giáp $	o$ Descartes $	o$ cộng trực tiếp vào Input Embedding ($h' = h + TE(x) + TE(y)$) + standard LoRA.
   - *Periodic Dual-Cycle MaT-LoRA (Nhóm đối tác):* Nhân ma trận thích ứng động $\Delta W = rac{lpha}{r} B F(	heta, z) A$ trên 72 module MLP, không phải phát minh của Han et al.
3. **Quy định vận hành:** Không kích hoạt GPU hay huấn luyện lại nếu chưa có phê duyệt chính thức.
