# QUY TRÌNH THỰC THI THỰC NGHIỆM: NHÁNH B2 (GATED CONTINUOUS RESIDUAL)

Tài liệu này đặc tả quy trình kỹ thuật tuần tự để chạy thực nghiệm nhánh **B2** (Điều kiện hóa biểu diễn thời gian qua cổng số dư khởi tạo bằng 0 - Gated Residual Conditioning) trong đề tài `Long-Range-Representation`.

---

## 1. Tổng Quan Kiến Trúc Nhánh B2

Nhánh B2 được thiết kế nhằm tiếp nhận tín hiệu thời gian Can-Chi dạng vector liên tục mà không làm phá vỡ cấu trúc ngôn ngữ của mô hình chỉ thị (`Qwen2.5-0.5B-Instruct`):

1. **Biểu diễn đầu vào (Input Representation):** Năm lịch sử $y$ được ánh xạ thành vector hình học Can-Chi 8 chiều $z_{\text{time}} \in \mathbb{R}^8$ thông qua chu kỳ Lục Thập Hoa Giáp và tọa độ cực TickTack:
   $$z_{\text{time}} = [x, y, \cos(\theta), \sin(\theta), r, \text{stem\_norm}, \text{branch\_norm}, \text{cycle\_norm}]$$
2. **Module chiếu và cổng số dư (Gated Residual Projection):**
   $$h' = h + \alpha \cdot \text{LayerNorm}(W z_{\text{time}} + b)$$
   - Ma trận chiếu $W \in \mathbb{R}^{896 \times 8}$, độ lệch $b \in \mathbb{R}^{896}$.
   - Chuẩn hóa lớp $\text{LayerNorm}$ với 896 tham số trọng số và bias.
   - Trọng số cổng vô hướng $\alpha \in \mathbb{R}$, **khởi tạo tuyệt đối bằng $0{,}0$**.
   - Tổng tham số bổ sung của module: $8 \times 896 + 896 + 896 \times 2 + 1 = 8.961$ tham số (trong đó trainable gồm $W, b$ và $\alpha$).
3. **Vị trí tiêm (Injection Interface):** Vector thời gian được cộng trực tiếp vào trạng thái ẩn (hidden state) tại **vị trí token thứ 14** (token phân cách câu hỏi trong cấu trúc prompt) trong **pha tính toán ngữ cảnh ban đầu (prefill phase)**. Phương pháp này **không chèn thêm virtual token vào chuỗi đầu vào**, giúp giữ nguyên độ dài và cấu trúc chú ý của câu.
4. **Hàm mục tiêu và cấu hình học:** Huấn luyện tăng dần tuần tự qua 3 chặng lịch sử (1947 $\rightarrow$ 1984 $\rightarrow$ 2022) kết hợp MLP-only LoRA ($r=2, \alpha=4$, 72 module) với hàm mục tiêu `MASKED_TARGET` (chỉ tính loss trên target response tokens).

---

## 2. Bản Đồ Mã Nguồn và Thư Mục Liên Quan

| Thành phần kỹ thuật | Đường dẫn tệp mã nguồn | Vai trò trong quy trình |
| :--- | :--- | :--- |
| **Mã hóa Can-Chi 8D** | `SourceCode/long_range_representation/representation/ticktack_canchi.py` | Hàm `encode_canchi_geometry(year, t0=1900)` tính toán vector 8 chiều. |
| **Dữ liệu huấn luyện** | `Data/experiment_views/logging102/TEMPUN_STANDARD_V1/` | Dữ liệu `TempUN_Small` chuẩn hóa với các trường `canonical_year` và `stage_id`. |
| **Pipeline thực thi chính** | `SourceCode/scripts/run_rescue_branches.py` | Lớp `GatedResidualProjection`, hook tiêm prefill, vòng lặp huấn luyện 3 chặng và đánh giá. |
| **Đóng gói Kaggle (Tùy chọn)** | `SourceCode/scripts/create_rescue_kaggle_bundle.py` | Tự động hóa đóng gói notebook, mã nguồn và dữ liệu gửi lên Kaggle GPU. |
| **Tổng hợp số liệu & McNemar** | `SourceCode/scripts/aggregate_rescue_branches.py` | Gom kết quả từ các seed, tính `mean ± SD` và kiểm định ghép cặp McNemar. |
| **Module kiểm định McNemar** | `SourceCode/long_range_representation/evaluation/mcnemar.py` | Triển khai Exact Binomial và Edwards Asymptotic với hiệu chỉnh Holm-Bonferroni. |

---

## 3. Quy Trình Chạy Tuần Tự (Step-by-Step Execution)

### Bước 1: Thiết lập môi trường và biến hệ thống
Từ thư mục gốc của repository `Long-Range-Representation`, kích hoạt môi trường Python 3.11/3.12 và gán `PYTHONPATH`:

```powershell
# Trên Windows PowerShell:
$env:PYTHONPATH = "SourceCode;src"

# Kiểm tra tính tương thích của mô-đun
python -c "import torch; from long_range_representation.representation.ticktack_canchi import encode_canchi_geometry; print('Môi trường sẵn sàng:', encode_canchi_geometry(1947))"
```

---

### Bước 2: Kiểm tra tính sẵn sàng của dữ liệu (Data Preflight)
Xác nhận dữ liệu `TempUN_Small` đã được chuẩn hóa và có đầy đủ bản ghi trắc nghiệm cho cả 3 chặng lịch sử:

```powershell
python -c "
from pathlib import Path
import json

data_root = Path('Data/experiment_views/logging102/TEMPUN_STANDARD_V1')
manifest = json.loads((data_root / 'MANIFEST.json').read_text(encoding='utf-8'))
print('Tập dữ liệu:', manifest.get('dataset_name'))
print('Mã băm dữ liệu:', manifest.get('checksum_sha256')[:12])
"
```

---

### Bước 3: Chạy bộ kiểm thử đơn vị bảo đảm toàn vẹn (Unit Test Gate)
Trước khi khởi chạy huấn luyện trên GPU, bắt buộc phải chạy bộ kiểm thử để đảm bảo module `ticktack_canchi` và logic của runner `test_rescue_branches` hoạt động chính xác:

```powershell
pytest SourceCode/tests/unit/test_rescue_branches.py SourceCode/tests/unit/test_ticktack_canchi.py SourceCode/tests/unit/test_mcnemar.py -v
```
*Yêu cầu:* Toàn bộ các ca kiểm thử phải đạt trạng thái `PASSED`.

---

### Bước 4: Khởi chạy thực nghiệm B2 trên máy cục bộ hoặc GPU (Local / GPU Run)
Tệp `SourceCode/scripts/run_rescue_branches.py` là bộ điều phối toàn diện cho việc huấn luyện tăng dần và đánh giá. Chạy thực nghiệm B2 độc lập trên 3 seed ngẫu nhiên (1, 2, 3):

```powershell
# Chạy toàn diện nhánh B2 cho 3 seed:
python SourceCode/scripts/run_rescue_branches.py `
  --branch B2 `
  --seeds 1 2 3 `
  --model_id "Qwen/Qwen2.5-0.5B-Instruct" `
  --data_root "Data/experiment_views/logging102/TEMPUN_STANDARD_V1" `
  --output_dir "artifacts/releases/TEMPORAL_CONDITIONING_RESCUE_103/runs/B2" `
  --device cuda `
  --lr 2e-5 `
  --batch_size 1 `
  --grad_accum 4 `
  --eval_batch_size 1
```

#### Tiến trình tự động diễn ra bên trong `run_rescue_branches.py`:
1. **Khởi tạo Backbone:** Tải mô hình `Qwen2.5-0.5B-Instruct` ở định dạng `bfloat16`, cấu hình LoRA trên 72 khối MLP (`gate_proj`, `up_proj`, `down_proj`) với $r=2, \alpha=4$.
2. **Khởi tạo Adapter B2:** Khởi tạo `GatedResidualProjection(in_features=8, out_features=896)` với `alpha=0.0`. Gắn `forward_pre_hook` vào mô hình để cộng vector thời gian vào hidden state tại token thứ 14 trong prefill.
3. **Huấn luyện tăng dần Chặng 0 (1947):** Huấn luyện trên các sự kiện tiền hiện đại với loss `MASKED_TARGET`. Sau chặng, chạy suy luận greedy trên 180 mẫu đánh giá và lưu kết quả tại `predictions/after_00/predictions.jsonl`.
4. **Huấn luyện tăng dần Chặng 1 (1984):** Huấn luyện tiếp trên các sự kiện hiện đại. Đánh giá và lưu tại `predictions/after_01/predictions.jsonl`.
5. **Huấn luyện tăng dần Chặng 2 (2022):** Huấn luyện tiếp trên các sự kiện đương đại. Đánh giá và lưu tại `predictions/after_02/predictions.jsonl`.
6. **Lưu biên nhận (Run Receipts):** Xuất tệp `run_receipt.json`, trọng số projection `b2_projection.pt`, trọng số LoRA và log GPU.

---

### Bước 5 (Tùy chọn): Đóng gói và điều phối trên Kaggle GPU (Cloud Dispatch)
Nếu thực hiện huấn luyện trên hạ tầng Kaggle GPU Tesla T4/P100:

```powershell
# Đóng gói bundle mã nguồn, config và dữ liệu:
python SourceCode/scripts/create_rescue_kaggle_bundle.py --branch B2 --output_dir artifacts/dispatch/rescue_b2

# Đẩy kernel lên Kaggle bằng Kaggle API:
kaggle kernels push -p artifacts/dispatch/rescue_b2/kernel
```
Sau khi kernel hoàn thành, tải kết quả về thư mục `artifacts/releases/TEMPORAL_CONDITIONING_RESCUE_103/runs/B2/`.

---

### Bước 6: Tổng hợp số liệu và thực hiện kiểm định ghép cặp McNemar
Sau khi 3 seed của B2 đã có đầy đủ kết quả tại thư mục runs, thực hiện tổng hợp kết quả và so sánh thống kê ghép cặp với mô hình cơ sở $C_0$ (tái sử dụng từ $R_{00}$):

```powershell
python SourceCode/scripts/aggregate_rescue_branches.py `
  --runs_root "artifacts/releases/TEMPORAL_CONDITIONING_RESCUE_103/runs" `
  --baseline_c0_root "artifacts/releases/REP_ABLATION_102/runs/R00" `
  --negative_n0_root "artifacts/releases/REP_ABLATION_102/runs/R11" `
  --output_dir "artifacts/releases/TEMPORAL_CONDITIONING_RESCUE_103/aggregate"
```

#### Đầu ra sinh ra sau Bước 6:
- `final_metrics.json`: Báo cáo `Strict Accuracy`, `Valid Accuracy`, `Invalid Rate` (mean ± SD) của B2 và các nhánh đối chứng.
- `mcnemar.json`: Bảng liên hợp $2 \times 2$, số mẫu bất đồng $n_{\text{disc}}$, trị số $p$ gốc và mức $\alpha$ hiệu chỉnh Holm-Bonferroni cho từng cặp so sánh ($C_0$ vs $B_2$, $N_0$ vs $B_2$).
- `screen_summary.json`: Tóm tắt hiển thị nhanh kết quả nghiệm thu.

---

### Bước 7: Cập nhật tệp minh chứng bàn giao (Research Evidence Handoff)
Sau khi kết quả được nghiệm thu, chuyển các tệp kết quả chuẩn vào thư mục bàn giao `Reports/01_Research_Evidence/`:
- Cập nhật số liệu vào `branch_results.json`.
- Cập nhật các bảng liên hợp vào `mcnemar_verified.json`.
- Trích xuất 180 bản ghi kết quả của B2 vào `paired_case_outcomes.jsonl`.
- Cập nhật lại mã băm SHA256 tại `MANIFEST.sha256`.

---

## 4. Bảng Kiểm Tra Định Kỳ Khi Chạy (Execution Checklist)

- [ ] Trọng số cổng $\alpha$ bắt đầu huấn luyện ở giá trị đúng $0{,}0000$.
- [ ] Trọng số cổng $\alpha$ sau 3 chặng hội tụ về khoảng dương hẹp: $\alpha \in [0{,}0017, 0{,}0024]$.
- [ ] Tỷ lệ lỗi định dạng (`Invalid Rate`) của B2 dao động quanh mức $35\% - 45\%$ (loại bỏ hoàn toàn sự cố sụp đổ $97{,}78\%$ của $N_0$).
- [ ] Độ chính xác nghiêm ngặt (`Strict Accuracy`) đạt khoảng $21{,}48\% \pm 4{,}63\%$.
- [ ] Kiểm định McNemar so với $C_0$ cho kết quả $p > 0{,}05$ trên toàn bộ 3 seed (xác lập tính tương đương thống kê).
- [ ] Kiểm định McNemar so với $N_0$ cho kết quả $p < 10^{-5}$ trên toàn bộ 3 seed (chứng minh giải cứu thành công).
