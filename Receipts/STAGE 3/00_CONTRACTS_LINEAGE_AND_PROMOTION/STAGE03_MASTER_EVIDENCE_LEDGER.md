# Sổ Cái Bằng Chứng Chủ Lực Stage 03 (Stage 03 Master Evidence Ledger)

**Dự án**: `long-range-temporal-alignment`  
**Giai đoạn**: `STAGE 03 — TARGETED OBJECTIVES & MCQA TRAINING`  
**Nhánh hoạt động**: `v2-targeted-continual-alignment`  
**Cập nhật lần cuối**: `2026-10-23`  
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
| **EV_S03_21** | Benchmark toàn diện Qwen2.5-3B và Ticktack 3B trên TempUN Full | `Receipts/STAGE 3/09_QWEN25_3B_FULL_BENCHMARK/` | $N=5{,}292$ TempUN Full (DB, MM, RB, TB, Rel, WB) | Base Canonical Acc: $28.38\%$ (Strict $6.24\%$), Ticktack: $27.85\%$ (Strict $6.22\%$). Kiểm định McNemar ghép cặp: $N=5292, b=280, c=252, \chi^2=1.3703, p=0.2418 > 0.05$ (không có sự khác biệt có ý nghĩa thống kê). | `MEASURED_VERIFIED` |
| **EV_S03_22** | Kiểm toán pháp y số liệu 3B, bóc tách rò rỉ CTM-Bench và sụp đổ vị trí | `STAGE03_3B_DISCREPANCY_REGISTER.jsonl` | 10 mục sai lệch chi tiết | Sửa lỗi hiển thị $25.10\% \to 6.24\%$; bóc tách $100\%$ CTM là oracle leakage; chứng minh toán học TE aliasing mod 60; xác nhận $86.55\%$ câu 4 lựa chọn sụp đổ điểm $0.25$ và Rel $100\%$ chọn 'No'. | `AUDITED_CONFIRMED` |
| **EV_S03_23** | Khung điều hòa giao thức đánh giá V1-V5 và niêm phong rào chắn P0-P6 | `Receipts/STAGE 3/10_QWEN25_3B_FORENSIC_AUDIT/` | 5 giao thức đánh giá, 10 unit tests độc lập | Đạt $10/10$ unit tests PASS ($281/281$ toàn repo); phê chuẩn rào chắn P0–P6 (`GO_3B_PROTOCOL_DIAGNOSTIC` / `NO_GO_SCALE`). | `DIAGNOSTIC_SUITE_VERIFIED` |
| **EV_S03_24** | Phát động đối chứng kép trên Qwen2.5-3B Base (Job 1 Ticktack Canonical vs Job 2 Stage 03 Targeted) | `Receipts/STAGE 3/11_QWEN25_3B_BASE_DUAL_DISPATCH/` | $104{,}130$ train NWP, $5{,}292$ eval MCQA, $54$ cặp phản thực nghiệm | Chuyển đổi chính thức sang `Qwen/Qwen2.5-3B` Base; đẩy thành công 2 GPU kernels lên Kaggle (`qh0205/lrr-ticktack-3b-base-canonical-baseline` và `qh0205/lrr-stage03-3b-base-targeted-alignment`), trạng thái RUNNING. | `DISPATCHED_RUNNING` |
| **EV_S03_25** | Nghiệm thu thực nghiệm Job 1 (Ticktack 3B Base Canonical Baseline) | `artifacts/kaggle_outputs/job1_ticktack_base/ticktack_3b_base_canonical_results.json` | $N=5{,}292$ TempUN Full (DB, MM, RB, Rel, TB, WB) | Canonical Acc: $32.03\%$, Debiased: $31.19\%$, Strict: $5.29\%$. Đặc biệt: Rel Canonical đạt $49.65\%$ (tăng từ $2.93\%$ trên Instruct), phá vỡ sụp đổ nhãn 'No'. | `MEASURED_VERIFIED` |
| **EV_S03_26** | Khái quát hóa toàn diện MaT-LoRA 108 khối MLP, FoNE/Can-Chi 40D và Safe Seam trên Qwen2.5-3B (Job 3) | `qh0205/lrr-stage03-3b-matlora-fone-full` (ID 140644073) | $104{,}130$ train NWP, $5{,}292$ eval MCQA, $54$ cặp phản thực nghiệm | Khái quát hóa Dual-Cycle MaT-LoRA trên 108 module MLP, tiêm FoNE/Can-Chi 40D qua Safe Seam (token index 14), vectorized batch core, phát động thành công lên Kaggle GPU. | `DISPATCHED_RUNNING` |

---

## 2. Bóc Tách Bản Chất Điểm Nghẽn Khoa Học

Dựa trên toàn bộ sổ cái bằng chứng, điểm nghẽn của Temporal-Qwen tại Stage 03 được xác định rành mạch:
1. **Tín hiệu thời gian được dẫn truyền vật lý thông suốt**: Các chỉ số $D_z = 5.3172 > 0$, $D_W = 0.5001 > 10^{-5}$, $S_{14, \text{seam}} = 0.0525$, $S_{14, \text{comp}} = 0.0538$, và $\Delta_{\text{logits}} = 0.1356 > 0.01$ chứng minh không có sự tắc nghẽn về mặt kiến trúc từ vector $z(t)$ qua Gated Seam và LoRA adapters đến token quyết định.
2. **Hiện tượng sụp đổ nhãn kép (Dual Label Collapse)**:
   - Trên nhiệm vụ MCQA 4 lựa chọn: Mô hình sụp đổ nhãn co cụm về chữ cái cố định ($86.55\%$ câu nhận điểm $0.25$ dưới 4 vòng xoay hoán vị, độ nhất quán ngữ nghĩa nghiêm ngặt chỉ $0.16\%$).
   - Trên nhiệm vụ Yes/No (Rel): Mô hình sụp đổ nhãn co cụm về chữ `'No'` ($852/852 = 100\%$ ở Base và $851/852 = 99.88\%$ ở Ticktack).
3. **Ý nghĩa của sự sụp đổ nhãn**: Hiện tượng sụp đổ nhãn này làm tê liệt hoàn toàn năng lực suy luận nhân quả: tỷ lệ đúng cả hai mốc thời gian $\text{BothCorrect} = 0.0\%$, và biên độ định hướng chân lý $\Delta G$ âm ở $63.89\%$ số trường hợp. Kiểm định McNemar trên $5{,}292$ ca xác nhận Ticktack 3B không mang lại độ lợi có ý nghĩa thống kê so với Base 3B ($p = 0.2418$).

---

## 3. Phán Quyết Quản Trị và Khóa Dừng

$$\mathbf{PHÁN\ QUYẾT:\ GO\_3B\_PROTOCOL\_DIAGNOSTIC\ \big|\ NO\_GO\_SCALE}$$

Tuyệt đối không mở rộng tập dữ liệu huấn luyện hoặc thực hiện huấn luyện GPU quy mô lớn. Dự án duy trì chiến lược **Representation-First**, yêu cầu phân nhánh đối tác tinh chỉnh bộ mã hóa thời gian để bảo đảm tính phân tách ngữ nghĩa trước khi tiếp tục các bước can thiệp tiếp theo. Giai đoạn tiếp theo chỉ cấp phép thực thi chẩn đoán đánh giá thuần túy ($0$ optimizer step) theo 5 giao thức V1–V5.
