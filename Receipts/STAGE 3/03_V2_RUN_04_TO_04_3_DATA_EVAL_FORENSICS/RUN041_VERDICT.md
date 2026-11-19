# Báo Cáo Phán Quyết Khoa Học Run 04.1: Tái Lập Thực Nghiệm, Khắc Phục Logic Thực Thi và Pháp Y Năng Lực Mô Hình

**Mã kiểm toán:** `V2_RUN04_1_SCIENTIFIC_REPRODUCIBILITY_AND_FORENSICS`  
**Ngày hoàn thành:** 2026-10-15  
**Tác giả kiểm toán:** Antigravity (Principal Research Engineer & Scientific Auditor)  
**Nhánh Git thực nghiệm:** `experiment/v2-run04-1-audit-revamp` (commit cơ sở `c4302f4`)  
**Backbone:** `Qwen/Qwen2.5-0.5B-Instruct` (đóng băng $100\%$, 0 gradient)  
**Biểu diễn thời gian:** `FIXED_FONE_CANCHI_V1` 40D (FoNE 32D + Can-Chi 8D, SHA256 `6c35fc55...`)  
**Khế ước giải mã:** `RAW_GREEDY_V1` (giải mã tự do toàn từ vựng 151.936 tokens, 0 option forcing, 0 logit masking) qua giao thức ChatML `P12_TASK_AWARE_CHATML`.

---

## 1. Tóm Tắt Phán Quyết Khoa Học

Đợt kiểm toán Run 04.1 đã thực hiện tái lập toàn diện, bóc tách nguyên nhân gốc rễ và đo đạc lại các checkpoint thực tế của Run 04 trên 4 bảng điểm độc lập và 6 can thiệp điều kiện suy luận có kiểm soát. Phán quyết khoa học chính thức:

1. **Khắc phục triệt để lỗi logic thực thi (F01 & F02)**: Mã nguồn Run 04 cũ khi đặt `use_adapter=False` chỉ bỏ qua việc gán vector thời gian, nhưng module `DualCycleMaTLoRALinear` vẫn cộng `lora_out` vào `base_out` với tọa độ zero. Bằng việc bổ sung công tắc `self.enabled = False` tường minh, Run 04.1 đã chứng minh tính tương đương tuyệt đối giữa `RAW_QWEN05B` và `PATCHED_BASE_EQUIVALENT` với `max_abs_diff = 0.0000000000`, `cosine_similarity = 1.000000`, `top1_flip = False` trên toàn bộ 300 ca suy luận thực tế (Gate G1 PASS).
2. **Xác nhận trạng thái R4_D là `REPLAY_NOT_TESTED` (F03 & F08)**: Toàn bộ 508 tensor khả huấn của checkpoint $R4\_D$ hoàn toàn giống $R4\_B$ về mặt trọng số (`max_abs_diff = 0.0`, cùng tensor fingerprint `3f4fa790...`). Sự khác biệt về file SHA256 chỉ do khóa siêu dữ liệu `"arm_name"` trong từ điển pickle. Mã nguồn Kaggle của $R4\_D$ lấy mẫu từ cùng tập dữ liệu như $R4\_B$, không hề có mẫu hay hàm mất mát replay. Do đó, $R4\_D$ không phải là thử nghiệm replay hợp lệ.
3. **Pháp y 4 góc phần tư năng lực mô hình (K / G / R / F)**:
   - **K (Factual Knowledge Recall)**: Mô hình nền có tri thức sự thật hạn chế về các chuỗi số liệu đóng (Base MCQA đạt $40{,}0\%$, Retention đạt $31{,}67\%$).
   - **G (Temporal Grounding)**: Được xác lập rõ rệt trên nhiệm vụ Quan hệ thời gian (S2 Dev Relation: $R4\_B$ đạt $86{,}67\%$ so với Shuffled $R4\_C$ đạt $70{,}00\%$ và Base đạt $48{,}33\%$). Tuy nhiên, trên nhiệm vụ Trắc nghiệm sạch (S1 Clean MCQA), can thiệp nội bộ trên cùng checkpoint $R4\_B$ chứng minh độ chính xác **không phụ thuộc** vào vector thời gian: `ADAPTED_TRUE` ($65{,}83\%$), `ADAPTED_WRONG` ($65{,}83\%$), `ADAPTED_ZERO` ($65{,}00\%$) và `ADAPTED_SEAM_OFF` ($66{,}67\%$) cho kết quả tương đương nhau.
   - **R (Relational / Comparative Reasoning)**: Trên bảng điểm chẩn đoán Oracle Evidence (S3), khi số liệu thật được cung cấp trực tiếp trong đề bài, mô hình đạt $61{,}67\%$ ($R4\_B$) và $58{,}33\%$ (Base), chứng minh năng lực so sánh số học được bảo toàn.
   - **F (Task / Format Compliance)**: Adapter MaT-LoRA huấn luyện thành công việc thích ứng định dạng trắc nghiệm sạch (nâng độ chính xác từ Base $40{,}0\%$ lên $65{,}83\%$, triệt tiêu lỗi cú pháp về $0{,}0\%$ Invalid). Khi ngắt LoRA (`ADAPTED_LORA_OFF` hoặc `ADAPTED_BOTH_OFF`), độ chính xác sụp đổ về mức Base ($38{,}33\%$ và $40{,}00\%$).

---

## 2. Bảng Điểm Đối Soát Chuẩn Tắc (Reconciled Scoreboards)

Toàn bộ 7 biến thể mô hình được đánh giá đồng nhất trên 4 bảng điểm độc lập ($n=300$ ca):
- **S1: Clean MCQA** ($n=120$, cân bằng $50/50$ hướng, $25\%$ mỗi ký tự nhãn, $0\%$ rò rỉ).
- **S2: Dev Relation** ($n=60$, held-out dev pairs của 2 thực thể `BCXZ` và `GVRO`).
- **S3: Oracle Diagnostic** ($n=60$, cung cấp số liệu thực chứng trong đề bài).
- **S4: Retention Slice** ($n=60$, mẫu ngẫu nhiên cố định từ benchmark TempUN Small 180 câu).

| Biến thể mô hình | S1 Clean MCQA (120) | S2 Dev Rel (60) | S3 Oracle Diag (60) | S4 Retention (60) | Trạng thái kiểm toán |
| :--- | :---: | :---: | :---: | :---: | :--- |
| `W0_RAW_BASE` | $40{,}00\%$ ($48/120$) | $48{,}33\%$ ($29/60$) | $58{,}33\%$ ($35/60$) | $31{,}67\%$ ($19/60$) | Base Qwen2.5-0.5B thuần |
| `PATCHED_BASE_EQUIVALENT` | $40{,}00\%$ ($48/120$) | $48{,}33\%$ ($29/60$) | $58{,}33\%$ ($35/60$) | $31{,}67\%$ ($19/60$) | Khớp bit-wise với RAW Base |
| `RUN02_C1` | $21{,}67\%$ ($26/120$) | $75{,}00\%$ ($45/60$) | $25{,}00\%$ ($15/60$) | $38{,}33\%$ ($23/60$) | Chuyên biệt hóa quan hệ |
| `R4_A_CLEAN_MCQA_ONLY` | $59{,}17\%$ ($71/120$) | $48{,}33\%$ ($29/60$) | $61{,}67\%$ ($37/60$) | $33{,}33\%$ ($20/60$) | Chỉ học MCQA |
| `R4_B_CLEAN_JOINT_TRUE` | **$65{,}83\%$** ($79/120$) | **$86{,}67\%$** ($52/60$) | $61{,}67\%$ ($37/60$) | $28{,}33\%$ ($17/60$) | Nhánh chính V2 |
| `R4_C_CLEAN_JOINT_SHUFFLED` | $64{,}17\%$ ($77/120$) | $70{,}00\%$ ($42/60$) | $60{,}00\%$ ($36/60$) | $33{,}33\%$ ($20/60$) | Đối chứng xáo trộn năm train |
| `R4_D_CLEAN_JOINT_REPLAY` | $65{,}83\%$ ($79/120$) | $86{,}67\%$ ($52/60$) | $61{,}67\%$ ($37/60$) | $28{,}33\%$ ($17/60$) | `REPLAY_NOT_TESTED` ($= R4\_B$) |

---

## 3. Can Thiệp Điều Kiện Suy Luận Trên Cùng Một Checkpoint (R4_B)

Để bóc tách nhân quả mà không bị nhiễu bởi quá trình tối ưu hóa khác nhau, 6 can thiệp suy luận được thực thi trực tiếp trên cùng một tập trọng số $R4\_B$:

| Can thiệp | Cấu hình Seam & LoRA | S1 Clean MCQA Acc | Higher Acc ($n=60$) | Lower Acc ($n=60$) | Nhận xét pháp y |
| :--- | :--- | :---: | :---: | :---: | :--- |
| `ADAPTED_TRUE` | Năm đúng ($y_{\text{eval}}$), Seam ON, LoRA ON | $65{,}83\%$ ($79/120$) | $73{,}3\%$ | $58{,}3\%$ | Cấu hình chuẩn V2 |
| `ADAPTED_WRONG` | Năm sai ($y_{\text{base}}$), Seam ON, LoRA ON | $65{,}83\%$ ($79/120$) | $71{,}7\%$ | $60{,}0\%$ | Điểm không đổi so với True |
| `ADAPTED_ZERO` | Tọa độ zero ($0_{40\text{D}}$), Seam ON, LoRA ON | $65{,}00\%$ ($78/120$) | $73{,}3\%$ | $56{,}7\%$ | Chênh lệch $1/120$ ca (không đáng kể) |
| `ADAPTED_SEAM_OFF` | Seam ngắt hoàn toàn ($\alpha=0$), LoRA ON | $66{,}67\%$ ($80/120$) | $73{,}3\%$ | $60{,}0\%$ | Ngắt Seam không làm giảm điểm MCQA |
| `ADAPTED_LORA_OFF` | Seam ON, LoRA ngắt hoàn toàn | $38{,}33\%$ ($46/120$) | $40{,}0\%$ | $36{,}7\%$ | Điểm sụp đổ về mức Base |
| `ADAPTED_BOTH_OFF` | Seam ngắt, LoRA ngắt (Base thuần) | $40{,}00\%$ ($48/120$) | $43{,}3\%$ | $36{,}7\%$ | Trùng khớp Base model |

**Kết luận nhân quả:** Trọng số adapter LoRA chịu trách nhiệm cho $+25{,}83\text{ pp}$ độ lợi so với Base model nhờ việc học cú pháp câu hỏi trắc nghiệm sạch. Vector thời gian tại Seam token 14 không điều khiển việc chọn nhãn MCQA đóng.

---

## 4. Thống Kê Ghép Cặp McNemar và Giới Hạn Cụm Thực Thể

Kiểm định Edwards-McNemar ghép cặp trên 120 ca Clean MCQA:
- **$R4\_B$ vs `RAW_QWEN05B`**: $\chi^2 = 24{,}32$, $p = 1{,}23 \times 10^{-7}$ (Significant, $b=34, c=3$).
- **$R4\_B$ vs $R4\_C$ (Shuffled)**: $\chi^2 = 0{,}25$, $p = 0{,}625$ (Not Significant, $b=3, c=1$).
- **$R4\_B$ vs $R4\_D$ (Replay)**: $\chi^2 = 0{,}00$, $p = 1{,}00$ (Discordant pairs = 0).

**Giới hạn cụm thực thể (Cluster Limitations):** Tập kiểm định DEV chỉ bao gồm $2$ chuỗi thực thể (`BCXZ` có 107 câu, `GVRO` có 13 câu). Mặc dù $p$-value có ý nghĩa thống kê ở cấp độ từng ca (case-level), các tuyên bố khái quát hóa ở cấp độ quần thể (population-level) cần được diễn giải thận trọng do số lượng cụm độc lập còn hạn chế ($n=2$).

---

## 5. Tổng Kết Kiểm Toán 12 Giả Thuyết Pháp Y (F01–F12)

| Mã | Giả thuyết pháp y | Trạng thái | Bằng chứng kiểm chứng được |
| :---: | :--- | :---: | :--- |
| **F01** | `W0_BASE` bị nhiễm trọng số adapter | **CONFIRMED** | `use_adapter=False` cũ vẫn chạy forward LoRA. Đã sửa bằng `self.enabled = False`. |
| **F02** | Rò rỉ trạng thái tọa độ thời gian | **CONFIRMED** | Biến module lưu tọa độ cũ; hook thiếu handle. Đã thêm `try/finally` context manager. |
| **F03** | $R4\_D$ chưa thực thi cơ chế replay | **CONFIRMED** | Mã nguồn thiếu replay pool; 508 tensor trùng $100\%$ $R4\_B$ (`diff = 0.0`). |
| **F04** | Tính tương đương đại số của hàm mất mát | **CONFIRMED** | `loss / 2` tương đương tuyệt đối $0{,}5 \mathcal{L}_{\text{rel}} + 0{,}5 \mathcal{L}_{\text{mcqa}}$. |
| **F05** | Bất đồng mẫu số giữa Kaggle và Local | **CONFIRMED** | Kaggle chạy full 180 retention, local chạy 60-slice. Đã đối soát và tách rõ ràng. |
| **F06** | $R4\_C$ là so sánh huấn luyện, không phải can thiệp | **CONFIRMED** | $R4\_C$ nhận năm đúng lúc eval. Đã bổ sung 6 can thiệp suy luận nội bộ trên $R4\_B$. |
| **F07** | `strict=False` có thể che giấu thiếu khóa | **CONFIRMED** | Đã ban hành bộ nạp kiểm tra cấu trúc nghiêm ngặt đủ 72 module, 508 tensor. |
| **F08** | Lệch mã băm tệp .pt do metadata | **CONFIRMED** | File SHA khác nhau do `"arm_name"`; tensor fingerprint hoàn toàn trùng khớp. |
| **F09** | Báo cáo McNemar thiếu điều chỉnh cụm | **CONFIRMED** | Báo cáo cũ in $p=0.0$; đã cập nhật dạng khoa học và ghi chú giới hạn $n=2$ thực thể. |
| **F10** | Giá trị fallback `or 1984` che giấu thiếu dữ liệu | **CONFIRMED** | Đã loại bỏ fallback; dữ liệu thiếu metadata bắt buộc fail-closed. |
| **F11** | Giới hạn 30 bước trên 1 hạt giống | **CONFIRMED** | Ghi nhận rõ số bước và hạt giống; cách ly tuyên bố khái quát hóa đa hạt giống. |
| **F12** | Oracle Evidence đo so sánh số học, không phải tri thức | **CONFIRMED** | Phân rã rạch ròi 4 góc phần tư năng lực K/G/R/F. |

---

## 6. Trạng Thái Các Cổng Kiểm Định Chất Lượng (Quality Gates G0–G7)

- **G0 (Problem & Architecture Lock)**: **PASS** (Cố định Qwen2.5-0.5B, FoNE-CanChi 40D, 72 Dual-Cycle MaT-LoRA).
- **G1 (Raw Base Parity)**: **PASS** (`max_abs_diff = 0.0000000000`, $100\%$ không lật nhãn).
- **G2 (Data Contract & Anti-Leakage)**: **PASS** ($100\%$ đơn nhất chân lý, $0\%$ rò rỉ benchmark).
- **G3 (Model State Verification)**: **PASS** (Cô lập ngữ cảnh an toàn, thu hồi hook nghiêm ngặt).
- **G4 (Training Semantics Audit)**: **PASS_WITH_REPLAY_QUARANTINED** ($R4\_D$ được phân loại đúng là `REPLAY_NOT_TESTED`).
- **G5 (Causal Intervention Probe)**: **PASS** (Hoàn tất đo đạc 6 can thiệp suy luận trên cùng checkpoint).
- **G6 (Reproducibility Contract)**: **PASS** (Khóa môi trường và lệnh tái lập độc lập).
- **G7 (Knowledge Writeback)**: **PASS** (Đồng bộ cập nhật vault tri thức chuẩn mực).
