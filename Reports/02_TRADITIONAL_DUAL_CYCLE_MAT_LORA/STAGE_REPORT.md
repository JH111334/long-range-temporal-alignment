# STAGE 02 — Dual-Cycle MaT-LoRA Truyền Thống và Mốc Xuất Phát M0-FC40

**Thời gian thực hiện:** 08/10/2026 – 13/10/2026 (Kiểm toán bổ sung 15/10/2026)  
**Chủ trì:** Nhánh Cơ chế học tập (Learning Track)  
**Mô hình cơ sở:** `Qwen/Qwen2.5-0.5B-Instruct` (Revision `7ae557604adf67be50417f59c2c2f167def9a775`)  
**Trạng thái tri thức:** `MEASURED_VERIFIED / N180_P12_CONFIRMED` (Đã phân giải dứt điểm N24 vs N180)

---

## 1. Bối Cảnh và Cơ Sở Kiến Trúc

Sau khi Stage 01 xác lập cơ chế tiêm an toàn tại token index 14 prefill (Gated Seam) và phạm vi PEFT trên 72 module MLP, Stage 02 tập trung khảo sát: **Việc tích hợp chu kỳ lịch sử kép (Dual-Cycle: vi chu kỳ Lục Thập Hoa Giáp 60 năm $\theta$ và đại kỷ nguyên thế kỷ $z$) vào lõi thích ứng động $\Delta W(\theta, z)$ trên nền biểu diễn đóng băng FoNE-CanChi 40D có giúp mô hình nhỏ 0.5B định vị mốc thời gian dài hạn hay không?**

### Cấu trúc kỹ thuật của đề tài:
1. *Matrix Time-LoRA (MaT-LoRA, Yao et al., arXiv:2602.11965 / Liang et al., 2026):*
   Biến đổi tham số thích ứng: $\Delta W(t) = \frac{\alpha}{r} B \cdot F(t) \cdot A$ với $r=8, \alpha=16.0$ trên 72 module MLP (`gate_proj`, `up_proj`, `down_proj`).
2. *Lõi Dual-Cycle đề xuất của Đề tài:*
   $$F(\theta, z) = I_r + \sum_{k=1}^2 \left[ \cos(k\theta) C_k + \sin(k\theta) S_k \right] + z E_{\text{lin}} + z^2 E_{\text{quad}}$$
   *Ranh giới tiền đề:* Nhóm nghiên cứu nhấn mạnh rằng cấu trúc Dual-Cycle kết hợp Lục Thập Hoa Giáp và thế kỷ trên tọa độ trụ $\mathbb{S}^1 \times \mathbb{R}$ là thiết kế riêng của đề tài, **không phải là bản tái lập y nguyên bài báo gốc TickTack (arXiv:2503.04150v3)**.

---

## 2. Khế Ước Biểu Diễn Đóng Băng FIXED_FONE_CANCHI_V1 (Decision 105)

Theo quyết định `DECISION-FIXED-FONE-CANCHI-LEARNING-TRACK-105` (12/10/2026):
- Khóa cứng vector đặc trưng 40 chiều: FoNE 32 chiều ghép nối Can-Chi 8 chiều ($\mathbf{z}_{\text{time}} \in \mathbb{R}^{40}$, SHA256: `6c35fc55...`).
- Khóa mốc xuất phát chuẩn tắc (Starting Anchor M0-FC40): Vận hành Dual-Cycle MaT-LoRA trên khế ước 40D kết hợp Safe Seam token 14, huấn luyện 3 chặng học liên tục tăng dần (1947 $\to$ 1984 $\to$ 2022) qua 3 hạt giống độc lập (Seed 1, 2, 3). Ba hạt giống này được huấn luyện song song độc lập từ cùng mô hình nền Qwen-0.5B đóng băng, không kế thừa tuần tự từ nhau.

---

## 3. Lịch Sử Khắc Phục Số Học và Kiểm Toán Pháp Y Bộ Nạp Trọng Số

### 3.1. Sự cố sụp đổ số học và bản vá Canary (12/10/2026)
- Lượt chạy đầu tiên bị tràn số `NaN loss` do khởi tạo ma trận LoRA $B \ne 0$ trên 72 module và hàm mất mát CrossEntropy FP16 trên $151,936$ logits.
- **Canary khắc phục:** Khởi tạo $B = 0$, upcast logits sang FP32, vận hành adapter ở FP32, loại bỏ `GradScaler`. Kết quả: $45/45$ bước hữu hạn $100\%$, $0$ NaN/Inf (`NUMERICAL_PASS`), triệt tiêu hoàn toàn chuỗi lặp `"!!!!!!!!"`.

### 3.2. Bóc tách lỗi bộ nạp trọng số (Loader Bypass Defect)
- Sau khi huấn luyện full 3 seed ($7,383$ bước), script đánh giá cũ `scripts/evaluate_m0_fc40_180_p12.py` đã tìm sai tên khóa trong checkpoint (`adapted_modules` thay vì `adapters_state`, và `gated_proj` thay vì `gated_proj_state`).
- **Hệ quả pháp y:** Script cũ hoàn toàn không nạp trọng số adapter đã học, dẫn đến việc cả 3 seed đều cho kết quả $32.22\%$ ($58/180$) trùng khớp tuyệt đối với mô hình nền (Base Qwen).
- **Khắc phục:** Bộ nạp chuẩn tắc `load_m0_fc40_for_inference` (`src/temporal_alignment/models/m0_fc40_loader.py`) đã được ban hành với khế ước fail-closed, kiểm tra bắt buộc sự tồn tại của `adapters_state` và `gated_proj_state`.

---

## 4. Phân Giải Dứt Điểm Sai Lệch Mẫu Số: N24 (Diagnostic Pilot) vs N180 (Full Benchmark)

Kiểm toán pháp y đã làm rõ sự nhầm lẫn giữa tập chẩn đoán mini-batch ($n=24$) và tập đánh giá đầy đủ ($n=180$):

### 4.1. Bản chất con số 37.50% ± 4.17% (Tập chẩn đoán N=24)
- Tệp `artifacts/audits/M0_FC40_FACTORIAL_ATTRIBUTION.md` và `Receipts/M0_FC40_FULL_VOCAB_FORENSIC_VERDICT.yaml` ghi nhận:
  - Cấu trúc tập kiểm tra: **$16$ câu MCQA + $8$ câu Yes/No = $24$ câu**.
  - Seed 1: $10/24 = 41.67\%$
  - Seed 2: $9/24 = 37.50\%$
  - Seed 3: $8/24 = 33.33\%$
  - Trung bình: $\mathbf{37.50\% \pm 4.17\%}$.
- *Khẳng định kiểm toán:* Đây là kết quả trên tập chẩn đoán sơ bộ **$n=24$ mẫu (`DIAGNOSTIC_PILOT_N24`)**. Do $37.50\%$ của $180$ là $67.5$ câu (số lẻ, không thể là số nguyên câu đúng), việc báo cáo sơ thảo trước đây gọi con số này là kết quả trên full 180 câu là **HOÀN TOÀN SAI LỆCH VÀ BỊ THAY THẾ (SUPERSEDED)**.

### 4.2. Kết quả đo lường thực tế trên Full TempUN Small 180 câu (N=180)
Kết quả đánh giá suy luận thực tế (inference-only, $0$ optimizer steps) với bộ nạp chuẩn tắc trên toàn bộ 180 câu TempUN Small (156 MCQA + 24 Yes/No) dưới giao thức $P_{12}$ (`RAW_GREEDY_V1`) được bảo tồn bất biến tại [`artifacts/audits/M0_FC40_CAUSAL_INFERENCE_RECOVERY_20261015/09_statistics_and_slices.json`](file:///c:/Users/GF63/Documents/long-range-temporal-alignment/artifacts/audits/M0_FC40_CAUSAL_INFERENCE_RECOVERY_20261015/09_statistics_and_slices.json):

| Mô hình / Hạt giống | Số câu đúng ($/180$) | Strict Accuracy | MCQA Acc ($/156$) | YN Acc ($/24$) | Invalid Rate | $\Delta$ vs Base | McNemar $p$-value |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Base Qwen2.5-0.5B** | **$58 / 180$** | **$32.22\%$** | $26.28\%$ ($41/156$) | $70.83\%$ ($17/24$) | **$0.00\%$** | — | — |
| **M0-FC40 Seed 1** | $49 / 180$ | $27.22\%$ | $20.51\%$ ($32/156$) | $70.83\%$ ($17/24$) | $0.00\%$ | $-5.00\text{ pp}$ | $p = 0.1763$ |
| **M0-FC40 Seed 2** | $59 / 180$ | $32.78\%$ | $28.85\%$ ($45/156$) | $58.33\%$ ($14/24$) | $0.00\%$ | $+0.56\text{ pp}$ | $p = 1.0000$ |
| **M0-FC40 Seed 3** | $53 / 180$ | $29.44\%$ | $23.08\%$ ($36/156$) | $70.83\%$ ($17/24$) | $0.00\%$ | $-2.78\text{ pp}$ | $p = 0.5419$ |
| **M0-FC40 Trung bình 3 seed** | **$53.7 / 180$** | **$29.81\% \pm 2.80\%$** | **$24.15\%$** | **$66.66\%$** | **$0.00\%$** | **$-2.41\text{ pp}$** | **$p > 0.05$ (Inconclusive)** |

---

## 5. Phân Tích Lát Cắt và Ranh Giới Khoa Học Đối Chứng Bài Báo Gốc

### 5.1. Phân tích lát cắt nguồn (Source Overlap vs Source Non-overlap)
- Dữ liệu TempUN Small chứa 60 câu hỏi có mã `source_record_id` xuất hiện trong tập huấn luyện `train_nwp.jsonl` (`seen_source`).
- 120 câu hỏi còn lại không chứa mã nguồn trùng lặp (`source_nonoverlap`). Nhóm nghiên cứu chỉ gắn nhãn `source_nonoverlap` đối với các mã định danh đã kiểm tra, **không tuyên bố đây là tập blind test tuyệt đối**.

### 5.2. Ranh giới đối chứng bài báo gốc TickTack (arXiv:2503.04150v3)
- Bài báo gốc sử dụng mô hình lớn `Qwen2.5-7B-Base`, huấn luyện đồng thời (joint all-at-once) trên toàn bộ 6,000 câu TempUN, tiêm vector tại token từ chỉ năm trong câu hỏi.
- Thử nghiệm Stage 02 sử dụng mô hình nhỏ `Qwen2.5-0.5B-Instruct`, học liên tục 3 chặng, tiêm tại Safe Seam token 14, và sử dụng lõi thích ứng Dual-Cycle riêng của đề tài.
- **Kết luận:** Hai thiết lập khác biệt căn bản về quy mô mô hình, phương thức tiêm và kiến trúc lõi. Việc so sánh giữa hai bên chỉ có ý nghĩa tham khảo định hướng bối cảnh, tuyệt đối không tuyên bố là bản tái lập trực tiếp (direct replication).

---

## 6. Phán Quyết Khoa Học Giai Đoạn 02

1. **Trạng thái mô hình:** `NO_RETRAIN_AUTHORISED`. Quá trình huấn luyện M0-FC40 đã kết thúc; việc tái đánh giá suy luận (eval-only) đã được hoàn thành đầy đủ trên 180 ca.
2. **Hiệu năng thực tế:** M0-FC40 đạt **$29.81\% \pm 2.80\%$** Strict Accuracy trên 180 câu TempUN Small dưới giao thức $P_{12}$.
3. **Phán quyết thống kê:** Chênh lệch $\Delta = -2.41\text{ pp}$ so với Base Model ($32.22\%$) không có ý nghĩa thống kê ($p > 0.05$ trên cả 3 seed ghép cặp). M0-FC40 đạt chuẩn ổn định số học (`NUMERICAL_PASS`), nhưng chưa chứng minh được độ lợi suy luận thời gian vượt trội (`TEMPORAL_REASONING_GAIN: UNPROVEN`). Mốc này đóng vai trò xuất phát điểm (Anchor) chuyển giao cho Stage 03.
