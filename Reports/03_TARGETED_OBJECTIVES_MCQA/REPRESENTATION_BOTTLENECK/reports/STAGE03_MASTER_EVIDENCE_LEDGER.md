# Sổ Cái Bằng Chứng Chủ Lực Stage 03 (Stage 03 Master Evidence Ledger)

**Dự án**: `long-range-temporal-alignment`  
**Giai đoạn**: `STAGE 03 — TARGETED OBJECTIVES & MCQA TRAINING`  
**Nhánh hoạt động**: `v2-targeted-continual-alignment`  
**Cập nhật lần cuối**: `2026-10-20`  
**Mục đích**: Tổng hợp toàn diện các bằng chứng thực nghiệm đã được thẩm định pháp y độc lập qua các chu kỳ can thiệp của Stage 03.

---

## 1. Danh Mục Các Mốc Bằng Chứng Chính Của Stage 03

| ID Bằng Chứng | Tên Thực Nghiệm / Can Thiệp | Mã Run ID / Checkpoint SHA-256 | Cỡ Mẫu ($N$) & Giao Thức | Kết Quả Đo Lường Chính | Trạng Thái Thẩm Định |
|---|---|---|---|---|---|
| **EV_S03_01** | Bảng điểm Run 05.2 đa nhiệm | `R52_B_NTP_REL_MCQA.pt`<br>`1af285df...` | $N=180$ TempUN<br>$N=482$ Relation<br>$N=120$ MCQA | DEV Relation: $62.03\%$, DEV MCQA: $50.83\%$, TempUN Small: $24.44\%$ | `MEASURED_VERIFIED` |
| **EV_S03_02** | Rào chắn Gatekeeper Run 05.2 | P0–P8 | 9 cổng logic máy đọc | Backbone đóng băng $100\%$, Syntax valid $= 1.0$, Parity diff $= 0.0$ | `MEASURED_VERIFIED` |
| **EV_S03_03** | Kiểm toán pháp y số liệu Run 05.2 | `audit/RUN052_RECOMPUTED_AUDIT.json` | $N=482, 120, 180$ | Bóc tách rò rỉ $60$ câu nguồn; McNemar LoRA vs Base: $p = 0.1654$ | `MEASURED_VERIFIED` |
| **EV_S03_05** | Can thiệp mốc năm sai (Proof of Treatment) | Checkpoint Arm B sau sửa caller | $n=36$ date-based | $\Delta_{\text{logits}} = 0.118655$, $4/36$ ca lật đáp án | `TREATMENT_VERIFIED` |
| **EV_S03_06** | Hoán vị vòng tròn 4 lựa chọn (Rotated MCQA) | Checkpoint Arm B, 4 rotations | $156 \times 4 = 624$ | Co cụm nhãn `b` $78.2\%$, Semantic consistency $= 0/156$ ($0.00\%$) | `LABEL_COLLAPSE_CONFIRMED` |
| **EV_S03_07** | Phản thực nghiệm ngữ nghĩa sơ bộ | 5 cặp lịch sử | $n=5$ pairs | 100% đoán 'No', $0/5$ flips | `INSUFFICIENT_POWER` |
| **EV_S03_11** | Nghiệm thu hạ tầng nền tảng Week 1 | Evaluator 21 trường, Tracer, Clustered McNemar | Khế ước chuẩn tắc | 5/5 unit tests PASS $100\%$ | `VALIDATED` |
| **EV_S03_12** | Kiểm toán tập phản thực nghiệm 54 cặp | `COUNTERFACTUAL_GOLD_AUDIT.jsonl`<br>`49eea7fe...` | 54 cặp (36 flip, 18 same) | $100\%$ cặp vượt qua kiểm tra hình thức logic AST | `VALIDATED` |
| **EV_S03_13** | Quét điểm nghẽn pilot 6 cặp | `RUN_BOTTLENECK_20261019_150638` | $N=6$ pairs (12 câu) | Delivery PASS, Sensitivity PASS, Selectivity PASS (margin $0.0398$) | `SUPERSEDED_BY_FULL_SWEEP` |
| **EV_S03_14** | Hiệu chỉnh khoa học pilot 6 cặp | `PILOT_CORRECTION_RECEIPT.json` | $N=6$ pairs | Exact Binomial $p = 0.03125$, hạ cấp telemetry `OTEL_LOCAL_ONLY` | `VALIDATED` |
| **EV_S03_15** | Quét nhân quả toàn diện 54 cặp | `RUN_FULL_SWEEP_54P_20261019_161756` | 54 cặp (108 câu, 23 cụm) | Delivery PASS, Sensitivity PASS, Selectivity INCONCLUSIVE, BothCorrect $= 0.0\%$ | `MEASURED_VERIFIED` |
| **EV_S03_16** | Tái tính toán dữ liệu thô & kiểm toán hướng dữ liệu | `POST_ARCHIVE_RAW_RECOMPUTATION.md` | 432 bản ghi, 54 cặp | Bóc tách All-No collapse (108/108 No), 100% Yes->No order bias, McNemar $p=0.1738$ | `AUDITED_CONFIRMED` |
| **EV_S03_17** | Thử nghiệm đối chứng nhân quả 2x2 Text x Adapter | `RUN_STAGE03_POST_ARCHIVE_CAUSAL_20261020_082138` | 54 cặp x 4 ô = 216 lượt suy luận | $E_z(t_1) = -0.0580$, $E_z(t_2) = -0.0584$, PositiveBothSides $= 13/36$ ($36.11\%$), Text vs Adapter dominance | `MEASURED_VERIFIED` |
| **EV_S03_18** | Pháp y hàm mục tiêu & cơ chế gradient overshooting | `OBJECTIVE_AND_LABEL_COLLAPSE_FORENSICS.md` | Mã nguồn Run 05.2 & data 50/50 | Xung đột tiên nghiệm Base (95% Yes) và loss CE mẫu No gây gradient quá đà | `ANALYZED_CONFIRMED` |
| **EV_S03_19** | Khóa ranh giới module Option Permutation & Quản trị | `NEXT_SINGLE_VARIABLE_DECISION.md` | OnlineOptionPermutation | Ranh giới MCQA chỉ chữa vị trí chữ cái; giữ vững NO_GO_SCALE | `GOVERNANCE_LOCKED` |
| **EV_S03_20** | Thẩm định tiền triển khai Qwen2.5-3B-Instruct & ma trận rào chắn 15 tiêu chí | Hub aa8e7253... / `Reports/03_TARGETED_OBJECTIVES_MCQA/QWEN25_3B_TRANSITION/` | 36 layers, 108 LoRA modules, $d=2048$ | 14/15 gates PASS (G11 NOT_VERIFIED chờ tải weights). Tokenizer parities 100%. Khóa khế ước giữ nguyên dataset gốc, OnlineOptionPermutation.enabled = False, Zero Cross-Scale loading. | `PRE_3B_READY (NO_GO_LARGE_TRAINING)` |
| **EV_S03_21** | Tiếp nhận gói Baselines chất lượng cao & TempUN Clean (Small & Full) từ đối tác | Hub commit 694a9b6 / `baselines/`, `data/tempun_clean/`, `data/tempun_full_clean/` | 271/271 unit tests PASS | Tiếp nhận thành công package `baselines/` (Ticktack 0.5B/3B, FoNE, Discrete, losses, date_mask, eval_debiasing), Kaggle 3B bundle, dữ liệu sạch 180 Small eval (0 null-year) + 5,292 Full eval (76 năm) và 104,130 NWP train, 0 rò rỉ. | `VERIFIED_ADMITTED_CANONICAL` |
| **EV_S03_22** | Kiểm toán thực nghiệm Qwen2.5-3B Full Baseline & Ticktack (76 năm, 5,292 eval, 104k train) | Commit e076fe3 / `Receipts/STAGE 3/09_QWEN25_3B_FULL_BENCHMARK/`, `artifacts/audits/MCNEMAR_3B_BASE_VS_TICKTACK.json` | 5,292 eval, 104,130 train NWP | Quá trình chạy hoàn tất không lỗi (train 8.12h, eval 2.63h, loss 2.8919, invalid rate 0%). Zero data leakage. Hiệu chỉnh lỗi hiển thị strict_acc 25.1% -> 6.24%. Kiểm định McNemar toàn diện: chi2=1.3703, p=0.2418 (không có ý nghĩa thống kê so với Base 3B). Phát hiện 100% No collapse trên bài toán Rel và 72.6% option letter bias trên trắc nghiệm 4 lựa chọn. | `AUDITED_CONFIRMED_NON_SIGNIFICANT` |

---

## 2. Bóc Tách Bản Chất Điểm Nghẽn Khoa Học

Dựa trên toàn bộ sổ cái bằng chứng, điểm nghẽn của Temporal-Qwen tại Stage 03 được xác định rành mạch:
1. **Tín hiệu thời gian được dẫn truyền vật lý thông suốt**: Các chỉ số $D_z = 5.3172 > 0$, $D_W = 0.5001 > 10^{-5}$, $S_{14, \text{seam}} = 0.0525$, $S_{14, \text{comp}} = 0.0538$, và $\Delta_{\text{logits}} = 0.1356 > 0.01$ chứng minh không có sự tắc nghẽn về mặt kiến trúc từ vector $z(t)$ qua Gated Seam và LoRA adapters đến token quyết định.
2. **Hiện tượng sụp đổ nhãn kép (Dual Label Collapse)**:
   - Trên nhiệm vụ MCQA 4 lựa chọn: Mô hình sụp đổ nhãn co cụm về chữ cái `b` ($78.2\%$).
   - Trên nhiệm vụ Yes/No: Mô hình sụp đổ nhãn co cụm về chữ `'No'` ($108/108 = 100\%$).
3. **Ý nghĩa của sự sụp đổ nhãn**: Hiện tượng sụp đổ nhãn này làm tê liệt hoàn toàn năng lực suy luận nhân quả: tỷ lệ đúng cả hai mốc thời gian $\text{BothCorrect} = 0.0\%$, và biên độ định hướng chân lý $\Delta G$ âm ở $63.89\%$ số trường hợp.

---

## 3. Phán Quyết Quản Trị và Khóa Dừng

$$\mathbf{PHÁN\ QUYẾT:\ NO_GO_SCALE}$$

Tuyệt đối không mở rộng tập dữ liệu huấn luyện hoặc thực hiện huấn luyện GPU quy mô lớn. Dự án duy trì chiến lược **Representation-First**, yêu cầu phân nhánh đối tác tinh chỉnh bộ mã hóa thời gian để bảo đảm tính phân tách ngữ nghĩa trước khi tiếp tục các bước can thiệp tiếp theo.
