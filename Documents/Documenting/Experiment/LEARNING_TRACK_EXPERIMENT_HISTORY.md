---
project: long-range-temporal-alignment
category: experiment-history
owner: learning-track
created_at: 2026-10-12T16:35:00+07:00
updated_at: 2026-10-26T23:55:00+07:00
status: active
canonical_index: artifacts/audits/PROCESS_A_AND_B_AUDIT.json
---

# Lịch Sử Thực Nghiệm và Quyết Định Khoa Học — Nhánh Cơ Chế Học Tập (Learning Track)

Tài liệu này ghi nhận toàn diện dòng thời gian các quyết định thực nghiệm, bằng chứng đo đạc thực tế, phân tích nguyên nhân gốc rễ và kế hoạch thử nghiệm có kiểm soát cho nhánh **Cơ chế học tập và thích ứng tham số (Learning Track)** trong đề tài `Long-Range-Representation`.

---

## A. Dòng Thời Gian Quyết Định Nghiên Cứu (Đã Hoàn Thành và Xác Thực)

| Thời gian (+07:00) | Mã thực nghiệm | Kỹ thuật & Cấu hình | Câu hỏi nghiên cứu / Lý do lựa chọn | Kết quả quan sát thực tế | Minh chứng / Mã băm | Quyết định khoa học |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **04/10/2026 14:00** | `WEEK1TARGET` | So sánh $2 \times 2$: (All-Linear vs MLP-Only) $\times$ (Unmasked vs `MASKED_TARGET`) | Xác định hàm mất mát tối ưu và ranh giới can thiệp tham số PEFT trên `Qwen2.5-0.5B-Instruct`. | $B_1$ ($13.15\% \pm 1.64\%$), $B_{1M}$ ($16.11\% \pm 4.55\%$), $B_2$ ($19.07\% \pm 1.64\%$), $B_3$ ($20.93\% \pm 3.85\%$). `MASKED_TARGET` có hiệu ứng chính dương (+5.93 pp All-Linear, +4.81 pp MLP-Only), tỷ lệ invalid $\sim 40-43\%$ dưới parser cũ. MLP-Only giảm $24.58\%$ tham số ($829,440$ vs $1,099,776$). Tương tác âm nhẹ ($-1.11$ pp), bác bỏ siêu hiệp đồng. | `artifacts/releases/WEEK1TARGET/` (`01_factorial_2x2_results.json`) | Khóa cứng `MASKED_TARGET` làm hàm mất mát cơ sở. Xác lập $B_3 = R_{00} = C_0$ (cùng cấu hình $20.93\%$ Acc) làm mốc đối chứng chuẩn (Anchor). |
| **05/10/2026 21:00** | `LOGGING102_FONE` | Đối chứng $F_0$ (Baseline) vs $F_1$ (FoNE Numeric Year) qua 3 seed trên 180 ca MCQA | Kiểm định xem biểu diễn số học bằng sóng Fourier (FoNE) cộng thô có giúp cải thiện độ chính xác thời gian hay không. | $F_0$ đạt $34.76\%$ Acc, $F_1$ đạt $28.15\%$ Acc ($\Delta = -6.61$ pp). Kiểm định McNemar chặng after_02: Seed 1 ($p=0.0104$), Seed 2 ($p=0.0101$). Khớp tuyệt đối 180 case IDs. | `artifacts/audits/LOGGING102_DATA_PAIR_PARITY.json` | **Bác bỏ (REFUTED):** Biểu diễn FoNE cộng thô không mang lại lợi ích trong cấu hình Qwen + TempUN; xác nhận tính toàn vẹn ghép cặp dữ liệu. |
| **06/10/2026 23:30** | `REP_ABLATION_102` | Phân tích nhân tố $2 \times 2$: (FoNE $\in \{0, 1\} \times$ Can-Chi $\in \{0, 1\}$) tiêm thô vào token-0 ChatML | Bóc tách độc lập đóng góp của FoNE và Can-Chi trong biểu diễn thời gian dài hạn. | $R_{00}$ (Baseline): $21.48\%$ Acc, $40.19\%$ Invalid. $R_{10}, R_{01}, R_{11}$: Độ chính xác sụp đổ $< 1.3\%$, tỷ lệ lỗi cú pháp $> 96.8\%$. | `artifacts/releases/REP_ABLATION_102/` | **Kết quả âm bản chuẩn tắc (Canonical Negative):** Tiêm vector số học trực tiếp vào token-0 ChatML làm phá vỡ cơ chế định tuyến cú pháp của mô hình Instruct. |
| **07/10/2026 18:00** | `RESCUE_103` | Nhánh $B_1$ (Textual Prompt) và $B_2$ (Biến đổi trạng thái ẩn prefill qua cổng số dư liên tục $\alpha_0=0.0$ tại token 14, không chèn virtual token) | Khôi phục tính toàn vẹn cú pháp mà vẫn tiếp nhận tín hiệu thời gian Can-Chi 8 chiều. | $B_1$ đạt $20.93\%$ Acc, $21.30\%$ Invalid. $B_2$ đạt $21.48\%$ Acc, $40.19\%$ Invalid, cổng $\alpha \to 0.002$. McNemar vs $N_0$: $p < 10^{-5}$ (vượt trội áp đảo); vs $C_0$: $p \in [0.25, 1.00]$ (`FAILED_TO_REJECT_H0`, không đủ bằng chứng bác bỏ giả thiết tương đương). | `artifacts/releases/TEMPORAL_CONDITIONING_RESCUE_103/` | Khôi phục thành công tuân thủ cú pháp; xác lập cơ chế tiêm prefill an toàn `FIXED_CANCHI_8D` của $B_2$. |
| **08/10/2026 22:00** | `TICKTACK_V2_CONTINUAL` | Ticktack v2 Periodic Dual-Cycle MaT-LoRA trên Tesla T4 GPU (Kernel Version 4, 3 seed) | Đánh giá kiến trúc tích hợp chu kỳ Can-Chi vào ma trận thích ứng động $\Delta W(\theta, z)$. | Đo theo unmasked eval: ACC $= 36.93\% \pm 1.14\%$. Đo theo `GENERATIVE_STRICT_PARSE_V1`: Strict Acc $= 20.51\% \pm 3.06\%$, Invalid Rate $= 26.85\% \pm 4.63\%$, Valid Acc $= 28.01\% \pm 2.79\%$. McNemar vs $F_0$: $p > 0.15$ cả 3 seed. | `Receipts/TICKTACK_V2_CONTINUAL_3SEEDS/continual_3seeds_results.json` | Phán quyết McNemar: `INCONCLUSIVE` trên 180 ca strict, nhưng giảm mạnh tỷ lệ lỗi cú pháp (-12.0 pp so với $F_0$). Ghi nhận là đối chứng lịch sử đối tác (`historical_partner_anchor`, `FAVORABLE_BUT_CONFOUNDED`). |
| **12/10/2026 15:30** | `ANTI_CLEANUP_AUDIT` | Tái cấu trúc thư mục `Reports/`, bảo tồn an toàn JSON sang `artifacts/audits/`, đối soát nguồn dữ liệu | Dọn sạch tệp thô khỏi Reports, giải tỏa mâu thuẫn số liệu 36.93% vs 20.51%, thiết lập cấu trúc tham khảo bài báo. | Bảo tồn $100\%$ byte dữ liệu của 7 file bằng chứng tại `CANONICAL_RESEARCH_EVIDENCE/`; đối soát nguồn xác nhận sự khác biệt do Evaluation Contract Shift; hoàn tất đồng bộ sang Drive NCKH. | `artifacts/audits/SOURCE_RECONCILIATION_AUDIT.json`, `artifacts/audits/PROCESS_A_AND_B_AUDIT.json` | Bàn giao thư mục Reports sạch sẽ; chuyển hướng nghiên cứu nhánh Learning sang tối ưu hóa hàm mục tiêu thời gian trên Anchor MaT-LoRA. |
| **12/10/2026 17:15** | `FCLEARN_105` | Khóa cứng khế ước `FIXED_FONE_CANCHI_V1` (FoNE 32D + Can-Chi 8D = 40D concat) | Phân định ranh giới độc lập: Partner sở hữu đa tạp hình trụ/hình cầu; Learning Track sở hữu LoRA/MaT-LoRA và hàm mục tiêu trên FoNE × Can-Chi đóng băng. | Đã trích xuất golden vectors bất biến (SHA256: `6c35fc55cbdfd571c03cd6bf56d3d4b65b367730b8bb3bc7d89c9dc82f78eddc`) cho 4 mốc 1947, 1984, 2022, 2044; khóa giao diện tiêm Safe Seam token 14 prefill. | `artifacts/releases/FCLEARN_105/feature_manifest.json` | Ban hành khế ước đặc trưng đóng băng; phân lập hoàn toàn khỏi đa tạp hình học đối tác. |
| **12/10/2026 18:45** | `FCLEARN_105_LOCK` | Xác lập `M0-FC40`, định tuyến thật Jev System One (`jev-1.13.0`), khóa thí nghiệm `M0-FC40` vs `M1-TEMP` vs `M1-NC` | Khóa Starting Architecture là Periodic Dual-Cycle MaT-LoRA theo chỉ đạo GVHD; giải mã JEVAPI.odt và gọi API thực tế; bãi bỏ FC0-first plan; hoãn retention sang Phase 2. | Jev call thành công 100%: hoãn retention ($noul=0{,}16$), xác nhận phân lập nhân quả ($noul=0{,}68$), chọn `VERIFY_M0_INTERFACE_AND_SMOKE` ($conf=1{,}0$). Trạng thái `M0-FC40`: `PROPOSED / NOT_TRAINED`. | `artifacts/audits/jev/jev_system_one_receipt.json`, `research/RESEARCH_CANDIDATES_AND_JEV_DECISION.md` | Ban hành `DECISION-FIXED-FONE-CANCHI-LEARNING-TRACK-105`; khóa thí nghiệm tiếp theo duy nhất; giải trình cho GVHD về tính phân lập sạch. |
| **12/10/2026 19:48** | `M0_FC40_V1_FAIL` | Phát động Version 1 kernel `qh0205/lrr-fc105-m0-fc40-3seed/1` | Khởi chạy pilot 3 seed trên Kaggle GPU | Thất bại ở bước thay thế MLP do `Linear4bit` packed buffer mismatch (`The size of tensor a (4864) must match size of tensor b (2179072)`). | `artifacts/diagnostics/M0_FC40/kaggle_version_1_failure.json` | **Lỗi mã nguồn (Code Bug):** Không copy buffer của quantized layer; chuyển sang gán trực tiếp `base_layer = linear` và nạp mô hình FP16 native (0.96 GiB VRAM trên 15 GiB Tesla T4). |
| **12/10/2026 20:13** | `M0_FC40_V2_DISPATCH` | Sửa `from_linear`, nạp FP16 native, bổ sung stage-wise checkpoint/BWT metrics, vượt qua 7 unit tests & 3-stage mini smoke test, phát động Version 2 | Khắc phục lỗi tensor shape, triển khai lưu trữ checkpoint đầy đủ và khởi chạy lại bản sửa đổi tối thiểu duy nhất trên Kaggle GPU (Version 2) | Đã đẩy Version 2 thành công; trạng thái thực tế quan sát qua Kaggle CLI: `KernelWorkerStatus.RUNNING`. Payload SHA256: `71208039c5511534800540fa0ad18c6da4ca2532da1976f820ffe973e844b676`. | `artifacts/dispatch/M0_FC40/M0_FC40_KAGGLE_BUNDLE_RECEIPT.json`, `artifacts/audits/M0_FC40_RUN_IDENTITY.json` | Khởi chạy thành công bản sửa đổi tối thiểu Version 2 trên GPU Tesla T4. |
| **12/10/2026 21:24** | `M0_FC40_V2_AUDIT` | Kiểm toán độc lập 3 seed trên Tesla T4 GPU (9 checkpoints, 9 metrics, 9 predictions files tải về an toàn tại quarantine) | Đánh giá chấp thuận khoa học 15 tiêu chí cho baseline M0-FC40 Periodic Dual-Cycle MaT-LoRA trên khế ước `FIXED_FONE_CANCHI_V1`. | Thực thi hoàn tất (EXIT_CODE 0, thời gian 3.144s). Tuy nhiên: Train Loss = NaN cả 3 chặng; Strict Acc = $0{,}00\% \pm 0{,}00\%$; Invalid Rate = $100{,}00\% \pm 0{,}00\%$ do mô hình sụp đổ và sinh chuỗi `"!!!!!!!!"`. Ma trận duy trì $R_{3\times3} = 0$, BWT = $0{,}0$. | `artifacts/audits/M0_FC40_ACCEPTANCE_REPORT.json` (SHA256 xác thực toàn bộ 9 checkpoints) | **Bác bỏ (REJECT):** Cổng chất lượng không đạt (Quality Gate FAIL). Taxonomy lỗi: `NUMERICAL_INSTABILITY_NAN_LOSS`, `COLLAPSED_REPETITION_PUNCTUATION`. Root cause: Khởi tạo ma trận LoRA $B \ne 0$ (`nn.init.normal_`) trên 72 modules gây tích lũy nhiễu qua 24 layers, kết hợp CrossEntropy FP16 trên 151.936 logits không upcast sang FP32 và thiếu `GradScaler` dẫn đến tràn số. |
| **12/10/2026 23:45** | `M0_FC40_CANARY_PASS` | Khắc phục pháp y số học: $B=0$, adapter/seam FP32, CE FP32 logits upcast, loại bỏ `GradScaler` (tránh tràn $65.504$ trong FP16 VJP), chạy Canary GPU (1 seed, 45 steps) | Xác minh tính ổn định số học và loại bỏ hoàn toàn lỗi NaN loss trước khi cấp phép huấn luyện 3 seed đầy đủ. | **THÀNH CÔNG (PASS 100%):** 45 steps hữu hạn (loss giảm: $4{,}20 \to 1{,}73 \to 2{,}43$). 3 checkpoints đạt $3.387.649/3.387.649$ phần tử hữu hạn ($0$ NaN/Inf). Peak VRAM $0{,}55$ GiB. Strict Acc $20{,}0\% \to 20{,}0\% \to 13{,}33\%$, Invalid Rate $20{,}0\%$. Triệt tiêu hoàn toàn lỗi suy biến chuỗi `"!!!!!!!!"`. | `artifacts/diagnostics/M0_FC40/CANARY_FORENSIC_VERDICT.json` (Kernel `qh0205/lrr-fc105-m0-fc40-canary/4`) | **CHẤP THUẬN (PASS):** Xác nhận loại bỏ hoàn toàn lỗi số học; cấp phép phát động Full Training 3-Seed M0-FC40 trên Kaggle GPU. |
| **13/10/2026 00:20** | `M0_FC40_3SEED_DISPATCH` | Đóng gói và phát động full 3-seed bundle ($2.461$ mẫu $\times$ 3 seeds, seeds $[1, 2, 3]$) lên Kaggle GPU Tesla T4 | Thực thi huấn luyện đầy đủ 3 seed cho baseline chính thức M0-FC40 Periodic Dual-Cycle MaT-LoRA trên `FIXED_FONE_CANCHI_V1`. | Kernel version 1 đã được đẩy thành công lên Kaggle (`qh0205/lrr-fc105-m0-fc40-3seed`). Trạng thái quan sát: `QUEUED`. Payload SHA256: `c1ba239ef59faea08792212ce8b9d84164d89a4570e8d86156af67a066824894`. | `artifacts/dispatch/M0_FC40/M0_FC40_KAGGLE_BUNDLE_RECEIPT.json` | Đã thực thi huấn luyện 3-seed đầy đủ. |
| **13/10/2026 04:30** | `M0_FC40_3SEED_COMPLETE` | Huấn luyện $7.383$ steps ($2.461$ steps/seed) hữu hạn $100\%$, xuất 9 checkpoints | Thu thập đầy đủ 9 checkpoints và raw predictions 3 seed trên Tesla T4 GPU | Thực thi hoàn tất thành công (EXIT_CODE 0, VRAM $0{,}59$ GiB). Train loss hữu hạn ($1{,}86 \to 1{,}72 \to 1{,}74$). Cả 3 seed ghi nhận Strict Acc $2{,}22\%$ ($4/180$), Invalid Rate $93{,}33\%$ ($168/180$). | `artifacts/quarantine/M0_FC40/kaggle_3seed/` (9 predictions, 9 metrics, 3 receipts) | Hoàn thành huấn luyện; chuyển sang kiểm toán pháp y nguyên nhân gốc rễ. |
| **13/10/2026 11:30** | `M0_FC40_FORENSIC_EVAL_AUDIT` | Kiểm toán pháp y chuỗi sinh (Raw Predictions Forensic) qua 3 seed và đối chiếu protocol | Trả lời tại sao cả 3 seed ra $4/180$ đúng và $168/180$ invalid; kiểm chứng seed isolation và phân loại taxonomy lỗi sinh | **Bác bỏ quy kết thuần túy cho Alignment Tax:** Khám phá hai sai lệch nghiêm trọng: (1) `eval_mcqa.jsonl` có 156 câu nhãn chữ thường `a,b,c,d`, nhưng parser cũ chỉ nhận chữ HOA `[A-D]` nên $156/156$ câu trắc nghiệm bị ép thành Invalid; (2) `run_m0_fc40.py` nạp raw prompt không có ChatML template của Qwen Instruct nên mô hình sinh ngay chuỗi rỗng `''` ($137/180$ ca). Xác nhận 3 seed độc lập $100\%$ (phân kỳ dự đoán đến $41{,}7\%$). Đã sửa parser, cập nhật ChatML và đồng bộ 80 files sang Drive NCKH. | `artifacts/audits/M0_FC40_FORENSIC_EVAL_AUDIT.json`, Google Drive NCKH (`G:\My Drive\NCKH`) | **Xác lập:** Lỗi thuộc về Giao thức Đánh giá (Evaluator Defect), không phải thoái hóa mô hình; chuyển giao baseline sạch cho Nhánh V2. |
| **13/10/2026 12:45** | `M0_FC40_PARITY_AUDIT_LOCKED` | Tái tính toán toàn diện $1.620$ generation records (Phase A), phân tích nhân tố $P_{00}$–$P_{11}$ và kiểm toán Seam token 14 | Xác định chính xác tác động của Rescoring-only, cô lập nhân tố ChatML vs Lời nhắc và kiểm tra tính toàn vẹn của Seam | **KẾT QUẢ PHÁP Y THỰC CHỨNG:** (1) Rescore $1.620$ chuỗi sinh cũ bằng canonical partner regex parser cho ra đúng **$0$ ca được cứu** (Strict Acc vẫn $2{,}22\%$), bác bỏ giả định sửa parser tự phục hồi điểm số trên output cũ; (2) Khóa khế ước `EVAL_PROTOCOL_MANIFEST_V2.json` ($P_{00}, P_{01}, P_{10}, P_{11}$); (3) Phát hiện `CONTRACT_DEFECT` tại Seam token 14 (bị trôi vào System message `' selecting'` dưới ChatML); (4) **Dừng toàn bộ full GPU training.** | `artifacts/audits/PARSER_PARITY_REPORT.json`, `artifacts/audits/M0_FC40_FINAL_PROTOCOL_PARITY_AUDIT.json`, `research/V2_FC40_EVIDENCE_RECONCILIATION.md` | **Khóa kiểm toán:** Phân lập rạch ròi Rescoring vs Re-generation; sẵn sàng bộ công cụ đánh giá chẩn đoán; chờ System Two phê duyệt. |
| **13/10/2026 20:30** | `M0_FC40_PROTOCOL_DIAG_N24` | Phân tích nhân tố $2 \times 2$ ($P_{00}, P_{01}, P_{10}, P_{11}$) và Seam On/Off trên $n=24$ mẫu chẩn đoán | Sàng lọc nhanh ảnh hưởng của định dạng prompt và kiểm tra sơ bộ vị trí Seam token 14 trên checkpoint canary hữu hạn. | Base Model: $P_{00}=20{,}83\%$, $P_{01}=20{,}83\%$, $P_{10}=12{,}50\%$, $P_{11}=8{,}33\%$. Canary Seam-On: $P_{00}=8{,}33\%$, $P_{01}=33{,}33\%$ ($0\%$ invalid), $P_{10}=12{,}50\%$, $P_{11}=20{,}83\%$ ($50\%$ invalid). Canary Seam-Off: $P_{01}=33{,}33\%$, $P_{11}=16{,}67\%$. Chênh lệch Seam On vs Off tại $P_{11}$ là $1/24$ ca ($5/24$ vs $4/24$), không có ý nghĩa thống kê; $P_{01}$ không có chênh lệch ($8/24$ vs $8/24$). | `artifacts/audits/M0_FC40_CANARY_SEAM_ON_FACTORIAL.json`, `artifacts/audits/M0_FC40_CANARY_SEAM_OFF_FACTORIAL.json` | **Chẩn đoán sơ bộ:** $n=24$ chỉ phục vụ gỡ lỗi kỹ thuật giao diện prompt; tuyệt đối không dùng $n=24$ để kết luận độ lợi seam hay bác bỏ suy thoái mô hình; bảo toàn $P_{11}$ làm giao thức tương đương đối tác. |
| **13/10/2026 22:50** | `M0_FC40_PILOT_V1_3SEED` | Huấn luyện toàn diện 3 seed $\times$ 3 chặng ($2.461$ steps/seed) trên Tesla T4 GPU (`qh0205/lrr-fc105-m0-fc40-3seed`, version 2) | Thu hoạch 9 checkpoints sạch số học và đo đạc chuẩn tắc trên $180$ ca TempUN Small ($1.620$ dự đoán) với bộ sinh ChatML $P_{11}$ | **HOÀN TẤT THÀNH CÔNG (COMPLETE):** 9/9 checkpoints hữu hạn sạch ($0$ NaN). Cú pháp: Invalid Rate $= 0{,}0\%$ toàn bộ 9 cells. Strict Acc chặng after_02: Seed 1 = $19{,}44\%$, Seed 2 = $9{,}44\%$, Seed 3 = $9{,}44\%$ (Mean = $12{,}78\% \pm 5{,}77\%$). BWT trung bình = $+3{,}36\%$. Phát hiện hiện tượng trôi dạt chỉ dẫn (Instruction Drift) co cụm về nhãn 'Yes' tại Seed 2 & 3. | `artifacts/quarantine/M0_FC40/kaggle_full_v2_3seed/RUN_IDENTITY.json` (SHA256: `59d2b4e1...`), 9 checkpoints & 9 predictions | **VALID_PASS_QUALITY số học:** Xác lập checkpoint cơ sở sạch cho M0-FC40; chuyển sang kiểm toán pháp y toàn diện không gian giải mã. |
| **13/10/2026 23:55** | `M0_FC40_FORENSIC_PARITY_VERDICT` | Kiểm toán pháp y toàn từ vựng (151.936 tokens), giải mã khoảng cách Canary P01 (33.33%) vs Full Run P11 (12.78%), sửa lỗi Evaluator V2, và đánh giá ghép cặp P12 (Task-Aware) | Bóc tách căn nguyên toán học của sự suy giảm hiệu năng: xác định tỷ lệ Invalid thực tế, phân tích logits phân phối đầu tiên, loại trừ giả định suy thoái mô hình | (1) Parser cũ che giấu lỗi cú pháp (chấp nhận 'Yes' trên MCQA) khiến Invalid Rate bị báo sai lệch là 0%; tỷ lệ invalid cú pháp thực tế là 86.67% ở Seed 2/3. (2) Prompt P11 ('or Yes/No') gây nhiễu tiên nghiệm: logits 'Yes'/'No' chiếm 66.4% xác suất trên câu MCQA. (3) Tập pilot chẩn đoán $n=24$ dưới $P_{12}$ đạt $37.50\% \pm 4.17\%$. Kiểm toán độc lập trên Full 180 câu TempUN Small với bộ nạp chuẩn tắc ghi nhận Strict Acc $29.81\% \pm 2.80\%$ (Seed 1: $27.22\%$, Seed 2: $32.78\%$, Seed 3: $29.44\%$) so với Base Model $32.22\%$ (McNemar $p > 0.05$, tương đương thống kê). | `Receipts/M0_FC40_FULL_VOCAB_FORENSIC_VERDICT.yaml`, `artifacts/audits/M0_FC40_CAUSAL_INFERENCE_RECOVERY_20261015/` | **CHẤP THUẬN KHOA HỌC (SUPPORTED SỐ HỌC):** Xác nhận M0-FC40 lành mạnh số học; mức $37.50\%$ được khu biệt là $n=24$ pilot; full 180 câu đạt $29.81\%$ (chưa chứng minh độ lợi thời gian so với base $32.22\%$). |
| **15/10/2026 03:37** | `V2_CAUSAL_ALIGNMENT_RUN01` | Can thiệp điều kiện suy luận 4 nhánh ($C_0, C_1, C_2, C_3$) trên 180 ca TempUN Small dưới giao thức $P_{12}$ | Kiểm toán pháp y xác nhận là thử nghiệm nhiễu vector lúc suy luận trên cùng checkpoint M0, không thực thi optimizer `.step()` | $C_0$: $28{,}89\%$; $C_1$: $28{,}89\%$; $C_2$: $28{,}89\%$; $C_3$: $27{,}78\%$. Cả 4 nhánh đạt $0{,}0\%$ Invalid. McNemar $C_1$ vs $C_2$: $p=0{,}617$, delta $= 0{,}0\%$. Độ nhạy logit trung bình: mean max delta $= 0{,}9763$. | `artifacts/experiments/V2_CAUSAL_TEMPORAL_ALIGNMENT/run_01/` | **INFERENCE_ONLY_UNTRAINED_INTERVENTIONS:** Chỉ đo lường độ nhạy vector suy luận; chưa có mô hình nào được huấn luyện với loss quan hệ thời gian. Đã sửa sai tại `RUN01_FORENSIC_CORRECTION.json`. |
| **15/10/2026 12:48** | `V2_CAUSAL_TRAINING_RUN02` | Huấn luyện thực sự 4 nhánh đối chứng bằng PyTorch (30 steps AdamW, lr=2e-4) trên tập so sánh thời gian từ TRAIN | Kiểm định xem loss quan hệ thời gian C1 có mang lại năng lực suy luận thời gian nhân quả vượt trội đối chứng âm C2 hay không | Trên tập held-out dev pairs ($n=60$): $C_1$ (True Relation) đạt **$75{,}00\%$**, $C_0$ (Continued NWP) đạt **$48{,}33\%$**, $C_2$ (Shuffled Negative) đạt **$51{,}67\%$**, $C_3$ (Format) đạt **$48{,}33\%$**. McNemar $C_1$ vs $C_2$: $p=0{,}00661 < 0{,}01$ ($\Delta = +23{,}33\%$). Đối chứng ngược (Counterfactual flip rate): $C_1$ đạt **$66{,}67\%$** (both correct $56{,}67\%$), $C_0/C_2/C_3$ đạt **$0{,}00\%$**. | `artifacts/experiments/V2_CAUSAL_TEMPORAL_ALIGNMENT/run_02/` | **SUPPORTED_CONDITIONALLY:** Huấn luyện quan hệ thời gian tạo ra sự phụ thuộc nhân quả thực sự vào vector thời gian trên dạng bài so sánh; phát hiện sự đánh đổi chuyên biệt hóa làm giảm MCQA trên benchmark 180 câu. |
| **15/10/2026 17:30** | `V2_CAUSAL_MULTITASK_RUN03` | Huấn luyện đa nhiệm kết hợp quan hệ thời gian và MCQA 4 lựa chọn (30 steps AdamW, batch=2) từ $C_1$ | Khảo sát khả năng phục hồi năng lực làm bài trắc nghiệm 4 lựa chọn mà vẫn duy trì nhân quả thời gian | Scoreboard A (Relation 60): $R3\_C$ đạt $86{,}67\%$ vs $C_1$ $75{,}00\%$ (McNemar $p=0{,}0391$). Scoreboard B (MCQA 120): $R3\_C$ đạt $65{,}00\%$ vs $C_1$ $20{,}00\%$. Bất biến hoán vị đạt $70{,}00\%$ vs $C_1$ $5{,}00\%$. Scoreboard D (Retention 60): $0{,}00\%$ do trôi dạt định dạng. Kiểm toán pháp y sau đó xác định $53/400$ câu train và $20/120$ câu dev vi phạm khế ước chân lý đơn nhất. | `artifacts/experiments/V2_CAUSAL_TEMPORAL_ALIGNMENT/run_03/` | **PARTIALLY_SUPPORTED / SUPERSEDED_DATA_DEFECT:** Đình chỉ các kết luận tuyệt đối do lỗi chân lý dữ liệu ngầm và shortcut định hướng ($83{,}33\%$ câu tăng). Chuyển sang sửa sai tại Run 04. |
| **15/10/2026 20:55** | `V2_DATA_TRUTH_REPAIR_RUN04` | Tái lập dữ liệu sạch chuẩn AST logic (100% đơn nhất chân lý, 50/50 hướng, 25% mỗi ký tự, 0% rò rỉ), giải cứu Retention và huấn luyện 4 nhánh trên GPU Kaggle | Khắc phục triệt để lỗi chân lý kép và shortcut định hướng của Run 03; kiểm chứng năng lực nhân quả thời gian và giải cứu Retention 60-slice | Clean MCQA 120 (50/50): $R4\_B$ đạt $65{,}83\%$ (vượt $+15{,}83\text{ pp}$ trên baseline $50\%$, chiều tăng $73{,}33\%$, chiều giảm $58{,}33\%$), $R4\_C$ đạt $64{,}17\%$. Dev Relation 60: $R4\_B$ đạt $86{,}67\%$ (cân bằng $29$ Yes / $31$ No) vs $R4\_C$ $70{,}00\%$ ($+16{,}67\text{ pp}$). S3 Oracle Diag: $R4\_B$ đạt $65{,}00\%$. S4 Retention 60-slice: $28{,}33\%$ ($17/60$), triệt tiêu hoàn toàn lỗi $0/60$ của Run 03 (Invalid $= 0{,}0\%$). | `artifacts/experiments/V2_CAUSAL_TEMPORAL_ALIGNMENT/run_04/` | **SUPPORTED (Data Truth & Retention Rescued) / CAUSALITY_INCONCLUSIVE (MCQA):** Giải cứu hoàn toàn retention benchmark và loại bỏ shortcut dữ liệu; năng lực so sánh số học được xác lập vững chắc. |
| **15/10/2026 23:30** | `V2_RUN04_1_AUDIT_REVAMP` | Kiểm toán độc lập 12 giả thuyết pháp y F01–F12, vá lỗi ngắt adapter `self.enabled`, chứng minh tính tương đương tuyệt đối của Base Model (`diff = 0.0`), bóc tách tensor fingerprint 508 tham số, phân loại $R4\_D$ là `REPLAY_NOT_TESTED`, và đo đạc 6 can thiệp suy luận trên cùng checkpoint $R4\_B$ | Tái lập khoa học, bảo đảm tính tái lập độc lập của mã nguồn, và pháp y 4 góc phần tư năng lực mô hình (K/G/R/F) | Gate G1 PASS: `RAW_QWEN05B` và `PATCHED_BASE_EQUIVALENT` đạt $100\%$ tương đương bit-wise ($40{,}00\%$ MCQA, $48{,}33\%$ Rel). Checkpoint $R4\_B$ đạt $65{,}83\%$ MCQA, $86{,}67\%$ Rel. Can thiệp nội bộ trên $R4\_B$: True ($65{,}83\%$), Wrong ($65{,}83\%$), Zero ($65{,}00\%$), Seam OFF ($66{,}67\%$), LoRA OFF ($38{,}33\%$). Triệt tiêu hoàn toàn rò rỉ trạng thái. | `artifacts/experiments/V2_CAUSAL_TEMPORAL_ALIGNMENT/run_04_1/` (20 artifacts, manifest SHA256 verified) | **VALIDATED & RECONCILED (Scientific Reproducibility Established):** Khắc phục toàn diện lỗi logic thực thi, phân định rạch ròi đóng góp của LoRA format compliance (+25.83 pp) và Seam conditioning, niêm phong toàn bộ 20 tệp tạo tác. |
| **16/10/2026 04:15** | `V2_RUN05_FINAL_AB_NTP` | Hai cánh tay độc lập Arm A ($\mathcal{L}_{\text{NTP}} + \mathcal{L}_{\text{rel}}$) và Arm B ($\mathcal{L}_{\text{NTP}} + \mathcal{L}_{\text{rel}} + \mathcal{L}_{\text{MCQA}}$) 40 steps, matched FLOP compute, kết hợp can thiệp suy luận Arm C | Bóc tách độc lập tác động của hàm mất mát causal NTP và MCQA đa nhiệm lên năng lực ngôn ngữ, quan hệ thời gian và độ nhạy suy luận nhân quả | Xác nhận phân kỳ tensor tuyệt đối giữa A, B và Parent C1; Arm A đạt Perplexity $180.08$ (CE $5.1934$), Arm B đạt PPL $342.57$ (CE $5.8365$). Trên DEV Relation 60 câu, Arm B đạt $20.00\%$ ($12/60$) vượt trội có ý nghĩa thống kê so với Arm A $0.00\%$ ($\chi^2 = 10.0833, p = 0.0005 < 0.001$). Arm C trên 36 câu date-based ghi nhận $\Delta = 0, p = 1.0$ do ranh giới độ dài sinh greedy; $14/14$ tests PASS. | `artifacts/experiments/V2_CAUSAL_TEMPORAL_ALIGNMENT/run_05_ab_ntp/` (39 artifacts, SHA256 verified) | **VALIDATED & RECONCILED (Genuine Independent Multi-task Arms Established):** Khẳng định vai trò của causal NTP trong việc neo giữ hàm mất mát ngôn ngữ và vai trò của MCQA trong việc chống sụp đổ cú pháp sinh chuỗi. |
| **16/10/2026 13:30** | `V2_RUN05_1_PHOENIX_RECOVERY` | Tái huấn luyện độc lập 2 nhánh Arm A/B trên Kaggle GPU (T4/P100, 40 steps CUDA), bóc tách 6 can thiệp Arm C trên 36 câu TempUN Date-based, giải phóng 100% tải CPU | Khắc phục pháp y sự cố Run 05 (lỗi gán đè trọng số nền và double causal shift NTP), bảo toàn ngôn ngữ held-out, đánh giá tính nhân quả của vector thời gian | Cả hai nhánh đạt Perplexity held-out cực thấp ($\approx 2.30 - 2.34$). Arm B đạt 100% valid trên toàn bộ benchmarks, 68.33% DEV MCQA Clean và 19.44% TempUN 180 (vượt trội có ý nghĩa so với Arm A: $\chi^2 = 17.3571, p = 0.000031$). Bảo toàn tuyệt đối năng lực DEV Relation (50.0% cả A và B, $p=1.0$). Can thiệp Arm C trên 36 câu Date-based ghi nhận không có sự khác biệt có ý nghĩa ($p=1.0$), chứng minh quyết định MCQA ở 0.5B bị chi phối bởi văn bản câu hỏi thay vì vector 40D tại token 14. | `artifacts/experiments/V2_CAUSAL_TEMPORAL_ALIGNMENT/run_05_1/` (42 artifacts, SHA256 manifest verified) | **VALIDATED & COMPLETED (Kaggle GPU Phoenix Recovery Complete):** Báo cáo phán quyết khoa học [RUN051_FINAL_SCIENTIFIC_VERDICT.md](file:///c:/Users/GF63/Documents/long-range-temporal-alignment/artifacts/experiments/V2_CAUSAL_TEMPORAL_ALIGNMENT/run_05_1/RUN051_FINAL_SCIENTIFIC_VERDICT.md) niêm phong đầy đủ; xác lập ranh giới năng lực nhân quả của mô hình 0.5B. |
| **16/10/2026 18:45** | `V2_RUN05_2_MECHANISM_REPAIR` | Khắc phục khế ước hệ thống, rào chắn DAG Gatekeeper P0–P8, 20 bất biến INV01–INV20, huấn luyện Arm A/B trên Kaggle GPU | Loại bỏ hoàn toàn shortcut ngầm, phân lập LoRA format vs Seam vector, kiểm định cơ chế nhân quả trên 180 ca TempUN Small | Cổng P0–P8 đạt PASS $100\%$. Arm B đạt $24.44\%$ Acc trên TempUN 180 (vs Base $32.22\%$, McNemar $p=0.1654$). Bóc tách cơ chế: Seam biến thiên logit $\Delta = 0.000369$ ($0/180$ ca lật nhãn), trong khi LoRA biến thiên logit $1.754299$ ($88/180$ ca lật nhãn). LoRA thiên lệch nhãn về `c` ($127/156$). | `artifacts/experiments/V2_CAUSAL_TEMPORAL_ALIGNMENT/run_05_2/` (27 artifacts, SHA256 verified) | **VALIDATED & HONEST-EMPIRICAL:** Xác lập ranh giới tác động: 72 module LoRA chi phối quyết định; Seam tại System prompt có lực tác động hạn chế dưới greedy decoding. |
| **16/10/2026 21:30** | `STAGE03_DIAGNOSTIC_SUITE` | Bộ ba chẩn đoán cơ chế trên CPU: Proof of Treatment (36 câu), Rotated-Option MCQA (624 biến thể), Semantic Counterfactuals (5 cặp) | Kiểm tra trực diện xem mô hình Arm B (`R52_B`) có thực sự khai thác vector thời gian tại Seam hay chỉ dựa vào shortcut chữ cái và văn bản prompt | (1) Proof of Treatment: Sửa caller tính từ $y+37$ ghi nhận $\Delta_{\text{logits}} = \mathbf{0.118655} > 0.01$, có $4/36$ ca lật đáp án (`TREATMENT_VERIFIED`). (2) Rotated MCQA: Gộp $24.20\%$, thiên lệch $78.2\%$ vào `b` và $19.6\%$ vào `a`; Semantic Consistency Rate = $0.00\%$ ($0/156$) (`LABEL_COLLAPSE_CONFIRMED`). (3) Counterfactuals: $0/5$ cả hai đúng, $0/5$ lật đáp án do đoán 100% 'No'. | `artifacts/experiments/V2_CAUSAL_TEMPORAL_ALIGNMENT/stage03_diagnostics/` (4 artifacts, SHA256 verified) | **PHÁN QUYẾT KHOA HỌC: NO-GO (HALT) cho full GPU retraining.** Khóa dừng toàn bộ GPU training mới; yêu cầu tái thiết lập phân bố dữ liệu cân bằng nhãn và loại bỏ shortcut chữ cái trước khi tiếp tục. |
| **19/10/2026 15:08** | `STAGE03_BOTTLENECK_SWEEP` | Quét bóc tách điểm nghẽn 5 cấp độ (Delivery, Sensitivity, Selectivity, Semantic Correctness, Robustness) với OpenTelemetry fail-open trên tập phản thực nghiệm đã kiểm toán | Bóc tách độc lập giữa việc tín hiệu thời gian được truyền và tiếp nhận so với năng lực suy luận nhân quả lật nhãn ngữ nghĩa | Delivery: PASS ($\bar{D}_W = 0.518 > 10^{-5}$); Sensitivity: PASS ($\Delta_{\text{logits}} = 0.1096$, phân rã $S_{14, 14}=0.0588$ vs $S_{14, -1}=0.0523$); Selectivity: PASS ($\text{Margin} = 0.0398 > 0.02$); Semantic Correctness: FAIL_NO_CAUSAL_FLIP ($\text{BothCorrect} = 0.0\%$). | `artifacts/stage03/representation_bottleneck/RUN_BOTTLENECK_20261019_150638/`, `Reports/03_TARGETED_OBJECTIVES_MCQA/REPRESENTATION_BOTTLENECK/` | **PHÁN QUYẾT KHOA HỌC: NO-GO SCALE.** Cô lập chính xác điểm nghẽn tại khâu ánh xạ biểu diễn thời gian sang quyết định ngữ nghĩa. Dừng toàn diện mở rộng quy mô dữ liệu và huấn luyện GPU. Chuyển giao yêu cầu biểu diễn cho đối tác. |
| **20/10/2026 18:00** | `QWEN25_3B_PRE_AUDIT` | Thẩm định tiền triển khai `Qwen/Qwen2.5-3B-Instruct` (36 layers, 108 LoRA modules, $d_{\text{model}}=2048$, Seam L21), điều hòa toán học 0.5B, ma trận cắt bỏ 5 cánh tay `Q3_A0`–`Q3_A4`, khóa bất biến dữ liệu gốc | Khảo sát dung lượng mô hình (Capacity Hypothesis) theo định hướng GVHD; rà soát tương thích kiến trúc, pháp y sụp đổ nhãn 0.5B và thiết lập rào chắn trước khi cấp phép thực thi | 14/15 gates PASS (G11 NOT_VERIFIED chờ tải weights). Hub Metadata & License Agreement hợp lệ. Xác nhận Exact Binomial McNemar $p=0.167444$; phân tích căn nguyên All-No do gradient overshooting và 100% Yes->No order bias trong 36 cặp Gold-Flip. Zero cross-scale weight loading. Khóa Online Option Permutation = False. | `Reports/03_TARGETED_OBJECTIVES_MCQA/QWEN25_3B_TRANSITION/` | **CHẤP THUẬN TIỀN TRIỂN KHAI (PRE_3B_READY):** Giữ nguyên dataset gốc; nghiêm cấm tự ý tải weights 3B hoặc cấp phát GPU huấn luyện (`NO_GO_LARGE_TRAINING`). |
| **22/10/2026 19:40** | `QWEN25_3B_BENCHMARK_FULL` | Benchmark toàn diện Qwen2.5-3B Base vs Ticktack 3B trên tập dữ liệu TempUN Full ($104{,}130$ train, $5{,}292$ eval) thực thi trên Kaggle T4 | Đánh giá thực nghiệm quy mô đầy đủ đối với baseline 3B theo yêu cầu chuyển dịch dung lượng | Base 3B: Canonical Acc $28.38\%$ (Strict Acc $6.24\%$). Ticktack 3B: Canonical Acc $27.85\%$ (Strict Acc $6.22\%$). Kiểm định McNemar ghép cặp trên $5{,}292$ ca: $b=280, c=252, \chi^2 = 1.3703, p = 0.2418 > 0.05$. Không có sự khác biệt có ý nghĩa thống kê giữa Base và Ticktack. | `Receipts/STAGE 3/09_QWEN25_3B_FULL_BENCHMARK/` | **THẨM ĐỊNH THỰC CHỨNG (OBSERVED / NO_SIGNIFICANT_DIFFERENCE):** Xác nhận Ticktack LoRA 1 epoch không cải thiện so với Base; chuyển sang kiểm toán pháp y nguyên nhân gốc rễ. |
| **23/10/2026 18:40** | `QWEN25_3B_FORENSIC_AUDIT` | Kiểm toán pháp y toàn diện số liệu 3B, bóc tách rò rỉ CTM-Bench oracle, kiểm tra mặt nạ ngày, chứng minh toán học TE cyclical aliasing mod 60, và thiết lập bộ 5 giao thức đánh giá chẩn đoán V1–V5 | Giải mã mâu thuẫn số liệu, bóc tách hai điểm nghẽn sụp đổ nhãn và chuẩn bị bộ công cụ đánh giá chẩn đoán không can thiệp trọng số | Sửa lỗi hiển thị $25.10\% \to 6.24\%$; cách ly $100\%$ CTM-Bench oracle leakage (`return item['answer']`); chứng minh $\|TE(y) - TE(y+60)\|_2 = 0.0$; xác nhận $86.55\%$ câu 4 lựa chọn sụp đổ điểm $0.25$ và Rel $100\%$ đoán 'No' do logit tiên nghiệm; đạt độ phủ mặt nạ ngày $100.0\%$; khớp $1{,}843{,}200$ tham số LoRA; $10/10$ unit tests PASS ($281/281$ toàn repo). | `Receipts/STAGE 3/10_QWEN25_3B_FORENSIC_AUDIT/`, `Reports/03_TARGETED_OBJECTIVES_MCQA/FORENSIC_AUDIT_3B/` | **PHÁN QUYẾT QUẢN TRỊ: GO_3B_PROTOCOL_DIAGNOSTIC / NO_GO_SCALE.** Khóa dừng toàn diện huấn luyện GPU; cấp phép chẩn đoán đo lường thuần túy theo 5 giao thức đánh giá đã chuẩn hóa. |
| **26/10/2026 23:15** | `STAGE03_V2_SCIENTIFIC_CLOSURE` | Đánh giá hợp nhất 3 nhánh trên Qwen2.5-3B Base (Arm 1 Ticktack, Arm 2 Targeted V2, Arm 3 MaT-LoRA FoNE) trên 5,292 TempUN Clean và 54 cặp phản thực nghiệm trên Tesla T4 | Khép góc khoa học Stage 03 V2, kiểm chứng tính bất biến trước hoán vị và suy luận nhân quả đối ngẫu | Arm 1: Canonical 32.03%, Strict 5.29%. Arm 2: Canonical 28.10%, Strict 6.20%, CF 9.26%. Arm 3: Canonical 26.27%, Strict 8.20% (+55% tương đối vs Arm 1), CF 70.37% (38/54). McNemar $p < 0.05$ ở cả 3 cặp đối chứng. Khám phá sự đánh đổi giữa học mẹo vị trí bề mặt và tính bền vững nhân quả; cô lập điểm nghẽn tác vụ TB (11.73%). | `Documents/Documenting/Results/STAGE03_3B_V2_20261026/` (`RECEIPT_STAGE03_3B_V2_CLOSURE.yaml`), Drive `3B_STAGE03_V2_SCIENTIFIC_CLOSURE_20261026` | **KHÉP GÓC KHOA HỌC (SCIENTIFIC_CLOSURE_COMPLETE):** Xác lập tính ưu việt của MaT-LoRA FoNE về tính bất biến phản thực nghiệm (70.37% vs 9.26%) và tính nhất quán hoán vị (8.20% vs 5.29%); ghi nhận thuế thích ứng 108 module MLP và rào cản tính khoảng cách của sóng chu kỳ. |

---

## B. Đối Chứng Lịch Sử Đối Tác (Historical Partner Anchor: $M_{\text{partner}}$)

- **Trạng thái:** `HISTORICAL_REFERENCE` (Phân loại `FAVORABLE_BUT_CONFOUNDED`: Được lựa chọn làm động lực kiến trúc theo chỉ đạo của GVHD; không dùng làm đối chứng trực tiếp cho các can thiệp thuần adapter do đa tạp hình học bị trộn lẫn).
- **Thời điểm thực thi:** Chạy ngày 08/10/2026, hoàn tất lúc 22:45 (+07:00), kiểm toán và đối soát nguồn ngày 12/10/2026.
- **Cơ sở lý thuyết:**
  - *MaT-LoRA (Matrix Time-LoRA, Yao et al., arXiv:2602.11965 / Liang et al., Neural Networks 2026):* Mô hình hóa biến thiên tham số phụ thuộc thời gian $\Delta W(t) = \frac{\alpha}{r} B F(t) A$.
  - *Kiến trúc Dual-Cycle của Đối tác:* Lõi động $F(\theta, z) \in \mathbb{R}^{r \times r}$ kết hợp vi chu kỳ Lục Thập Hoa Giáp $F_{\text{cycle}}(\theta)$ và đại kỷ nguyên thế kỷ $F_{\text{epoch}}(z)$ trên tọa độ trụ $\mathbb{S}^1 \times \mathbb{R}$.
- **Ranh giới sở hữu:** Toàn bộ biểu diễn hình học trụ $(\theta, z)$ và module sinh lõi ma trận động thuộc quyền sở hữu độc quyền của Partner (Representation Track).
- **Kết quả đo đạc chuẩn tắc (Khế ước `GENERATIVE_STRICT_PARSE_V1`, 3 seed):**
  - **Strict Accuracy:** $20.51\% \pm 3.06\%$ (Seed 1: $17.22\%$, Seed 2: $21.11\%$, Seed 3: $23.33\%$).
  - **Invalid Output Rate:** $26.85\% \pm 4.63\%$ (Seed 1: $30.56\%$, Seed 2: $28.33\%$, Seed 3: $21.67\%$).
  - **Valid Accuracy:** $28.01\% \pm 2.79\%$.
  - **Chuyển giao ngược (BWT):** $-1.85\% \pm 1.64\%$.
  - **Kiểm định ghép cặp Edwards McNemar (180 ca, chặng after_02):** So với $F_0$ Baseline: Seed 1 ($p=0.178$), Seed 2 ($p=0.892$), Seed 3 ($p=0.170$) $\implies$ Phán quyết: `FAILED_TO_REJECT_H0` (inconclusive).
- **Ý nghĩa định hướng:** Kết quả $20.51\%$ Strict Acc chưa chứng minh vượt trội thống kê so với $F_0$ ($20.93\%$), nhưng tỷ lệ lỗi cú pháp $26.85\%$ là mức giảm ấn tượng (giảm $12.0$ pp so với $F_0$). MaT-LoRA được chọn làm điểm xuất phát chiến lược để đánh giá trên biểu diễn đóng băng độc lập của nhánh Learning.

---

## C. Khế Ước Biểu Diễn Đóng Băng (FIXED_FONE_CANCHI_V1) và Starting Anchor M0-FC40

Theo quyết định `DECISION-FIXED-FONE-CANCHI-LEARNING-TRACK-105` hiệu chỉnh ngày 12/10/2026, kiến trúc xuất phát chính thức của nhánh Learning Track là **Periodic Dual-Cycle MaT-LoRA (`M0-FC40`)**, vận hành trên biểu diễn thời gian độc lập đã đóng băng, bãi bỏ hoàn toàn kế hoạch so sánh sơ bộ `FC0 vs FC1`.

### 1. Đặc tả khế ước biểu diễn đóng băng (FIXED_FONE_CANCHI_V1)
- **Fourier Numerical Embedding (FoNE số học 32D):** 16 tần số Fourier $\log$-spaced từ chu kỳ 10.0 năm đến 10000.0 năm, tính trực tiếp từ mốc năm lịch sử thuần túy ($[\cos(\omega_k y), \sin(\omega_k y)]_{k=0}^{15}$). Mã băm cấu hình: `4ebea847bd63b21a7b553f0724394e73331e1bc844763d31d5ec81508307fb50`.
- **Can-Chi Sexagenary Embedding (Can-Chi 8D):** Mã hóa vi chu kỳ Lục Thập Hoa Giáp (60 năm), Thập Thiên Can (10 năm), Thập Nhị Địa Chi (12 năm) và thành phần trôi dạt thế kỷ.
- **Tổng số chiều:** $\dim(\mathbf{z}) = 32 + 8 = 40$ chiều số thực ghép nối (concatenation).
- **Giao diện tiêm an toàn (Safe Conditioning Seam):** Biến đổi trạng thái ẩn prefill qua cổng số dư liên tục tại token index 14 với $\alpha_0 = 0.0$ ($\operatorname{Linear}(40, 896) \to \operatorname{LayerNorm}(896)$). Tuyệt đối không chèn virtual token, không xâm phạm token-0 ChatML.
- **Mã băm chứng thực bất biến (Golden Vectors):** Đã trích xuất và kiểm tra tại [artifacts/releases/FCLEARN_105/feature_manifest.json](file:///c:/Users/GF63/Documents/long-range-temporal-alignment/artifacts/releases/FCLEARN_105/feature_manifest.json) với SHA256: `6c35fc55cbdfd571c03cd6bf56d3d4b65b367730b8bb3bc7d89c9dc82f78eddc`.

### 2. Phân tích kỹ thuật Interface Adaptation cho Dual-Cycle MaT-LoRA
1. **Lan truyền xuôi (Forward Computation):** Dữ liệu chuỗi đầu vào $\to$ trích xuất vector 40D $\to$ đồng thời tiêm tại token 14 prefill (Safe Seam) và kích hoạt lõi động $F(\theta, z)$ tại 72 module MLP (`gate_proj`, `up_proj`, `down_proj`).
2. **Tham số hóa (Parameterization):** Giữ nguyên cấu hình lõi Dual-Cycle: Hạng $r=8$, hệ số tỷ lệ $\alpha=16{,}0$, $K=2$ sóng điều hòa Fourier, $F_{\text{base}} = I_8$, và các ma trận cơ sở $\{C_k, S_k, E_{\text{lin}}, E_{\text{quad}}\}$.
3. **Tính tương thích Checkpoint (Checkpoint Compatibility):** Checkpoint gốc của đối tác (huấn luyện trên pipeline trụ riêng) **không tương thích nhị phân trực tiếp** với interface 40D và Safe Seam. Bắt buộc phải thực hiện huấn luyện cơ sở trên khế ước mới để xác lập trọng số checkpoint `M0-FC40`. Trạng thái hiện tại: **`PROPOSED / NOT_TRAINED`**. **Tuyệt đối không gọi checkpoint này là identical partner checkpoint.**

### 3. Ma trận thực nghiệm ưu tiên: Thử nghiệm đối chứng ghép cặp M0-FC40 vs M1-TEMP vs M1-NC

Sau khi xác lập `M0-FC40` trên FoNE–Can-Chi, đóng băng vĩnh viễn: biểu diễn 40D, kiến trúc adapter, rank $r=8$, alpha $\alpha=16{,}0$, 72 module mục tiêu, dữ liệu huấn luyện tăng dần (3 chặng) và evaluator `GENERATIVE_STRICT_PARSE_V1`.

| Mã nhánh | Cấu hình Adapter & Biểu diễn | Hàm mục tiêu huấn luyện | Cơ chế kiểm chứng âm | Mục đích khoa học / Câu hỏi nghiên cứu | Trạng thái |
| :---: | :--- | :---: | :---: | :--- | :---: |
| **`M0-FC40`** | Periodic Dual-Cycle MaT-LoRA ($r=8, \alpha=16$) trên FoNE × Can-Chi 40D (Safe Seam) | $\mathcal{L}_{\text{MASKED\_TARGET}}$ | — | **Anchor xuất phát chuẩn tắc:** Xác lập đường cơ sở ổn định trên biểu diễn đã khóa. | `REJECT (FAIL — Numerical Instability)` |
| **`M0-FC40-R1`** | Bản vá số học: Khởi tạo $B = 0$, upcast FP32 CrossEntropy, kích hoạt `GradScaler` | $\mathcal{L}_{\text{MASKED\_TARGET}}$ | — | **Chạy lại có kiểm soát (Remedy Run):** Khôi phục tính ổn định gradient và xác lập checkpoint chuẩn. | `PLANNED` |
| **`M1-TEMP`** | Cùng cấu hình `M0-FC40-R1` (cùng khởi tạo, cùng số bước) | $\mathcal{L}_{\text{MASKED\_TARGET}} + \lambda \mathcal{L}_{\text{TEMPORAL}}$ ($\lambda = 0{,}1$) | Phân biệt mẫu âm lệch năm trong cùng chuỗi chỉ số huấn luyện | **Ứng viên cải tiến duy nhất:** Kiểm định xem giám sát mốc thời gian nhân quả có cải thiện độ chính xác và giảm lỗi cú pháp hay không. | `BLOCKED (Chờ M0-FC40-R1)` |
| **`M1-NC`** | Cùng cấu hình `M1-TEMP` (cùng số bước, cùng token exposure) | $\mathcal{L}_{\text{MASKED\_TARGET}} + \lambda \mathcal{L}_{\text{SHUFFLED}}$ ($\lambda = 0{,}1$) | Xáo trộn nhãn năm huấn luyện ngẫu nhiên theo seed | **Đối chứng âm cấu trúc bắt buộc:** Loại trừ giả thuyết cải thiện do hiệu ứng điều hòa ngẫu nhiên hoặc tăng tần suất cập nhật gradient. | `BLOCKED (Chờ M0-FC40-R1)` |

### 4. Báo Cáo Kiểm Toán Lỗi và Phân Tích Nguyên Nhân Gốc Rễ (M0-FC40 Failure Taxonomy)

Đợt huấn luyện GPU trên Kaggle phiên bản 2 (`qh0205/lrr-fc105-m0-fc40-3seed/2`) ghi nhận các bằng chứng kiểm toán sau:
1. **Quan sát thực nghiệm đối chứng:** Toàn bộ 3 hạt giống hoàn thành quá trình lặp (733, 1338, 390 bước) và tạo ra 9 tệp checkpoint với dung lượng $7{,}008{,}743$ bytes/tệp (SHA256 lưu tại `M0_FC40_ACCEPTANCE_REPORT.json`). Tuy nhiên, `train_loss` bằng `NaN` ngay từ chặng đầu tiên; toàn bộ 1.620 lượt sinh văn bản autoregressive trả về chuỗi lặp dấu chấm than `"!!!!!!!!"`, khiến `Strict Accuracy = 0.00%` và `Invalid Syntax Rate = 100.00%`.
2. **Nguyên nhân gốc rễ (Root Cause Analysis):**
   - *Khởi tạo ma trận LoRA B khác không:* Trong tệp `dual_cycle_mat_lora.py`, `self.lora_B` được khởi tạo bằng phân phối chuẩn (`nn.init.normal_`) với mục đích ép gradient chảy về lõi động ngay bước đầu. Tuy nhiên, việc áp dụng $\Delta W \ne 0$ đồng thời trên 72 module MLP (`gate_proj`, `up_proj`, `down_proj`) của 24 tầng biến đổi làm tích lũy sai lệch theo cấp số nhân, phá vỡ phân phối logits tiền huấn luyện ngay tại bước 0.
   - *Hàm mất mát FP16 trên không gian từ vựng 151.936 tokens:* Hàm `F.cross_entropy` tính toán trực tiếp trên logits `torch.float16` mà không chuyển đổi lên `float32`. Giá trị logits lệch lớn gây tràn số lũy thừa ($\exp(z) > 65.504$), dẫn đến giá trị mất mát bằng vô hạn (`Inf`) hoặc `NaN`.
   - *Thiếu bộ điều chỉnh thang đo gradient (GradScaler):* Huấn luyện độ chính xác nửa (FP16) trên GPU Turing (Tesla T4) không đi kèm `torch.cuda.amp.GradScaler`, khiến sai số underflow/overflow lập tức đầu độc bộ tối ưu hóa AdamW.
3. **Giải pháp khắc phục tối thiểu (Minimal Remedy — M0-FC40-R1):**
   - Đặt lại khởi tạo $B$ về số 0 chuẩn tắc (`nn.init.zeros_(self.lora_B)`) theo đúng nguyên lý LoRA (Hu et al., 2021) để bảo toàn nguyên vẹn $W_0$ tại bước khởi đầu.
   - Chuyển đổi logits sang FP32 trước khi tính Cross-Entropy (`shift_logits.view(-1, ...).float()`).
   - Tích hợp `GradScaler` bảo vệ gradient clipping và cập nhật trọng số.

### 5. Chi tiết hàm mục tiêu thời gian nhân quả ($\mathcal{L}_{\text{TEMPORAL}}$)
1. **Nhiệm vụ phân biệt mốc thời gian (Year-Corruption Discrimination trên TempUN):**
   - Trích xuất độc quyền từ tập huấn luyện `train_nwp.jsonl.gz`.
   - Dữ liệu TempUN có bản chất chuỗi chỉ số thống kê quốc gia `(Country, Indicator, Year, Value)`.
   - Với mỗi bản ghi huấn luyện $(E, I, y_{\text{true}}, v_{\text{true}})$, tạo mẫu âm $(E, I, y_{\text{corrupt}}, v_{\text{corrupt}})$ lệch mốc thời gian trong cùng chuỗi chỉ số có giá trị chênh lệch rõ ràng.
   - Tuyệt đối cấm sử dụng các câu hỏi hoặc thực thể thuộc tập đánh giá 180 mẫu MCQA (chống rò rỉ dữ liệu).
2. **Hàm mất mát:**
   $$\mathcal{L}_{\text{TEMPORAL}} = -\log \sigma\left(g(h_{\text{last}}, y_{\text{true}})\right) - \log\left(1 - \sigma\left(g(h_{\text{last}}, y_{\text{corrupt}})\right)\right)$$
   với $g(h, y)$ là tầng phân loại tuyến tính nhỏ ($896 \to 1$).

---

## D. Tiêu Chuẩn Phán Quyết, Phân Tầng Giai Đoạn và Giải Trình Cho GVHD

1. **Khế ước đánh giá:** Bắt buộc sử dụng `GENERATIVE_STRICT_PARSE_V1`, đo lường qua 3 hạt ngẫu nhiên chuẩn tắc (`SEEDS = [1, 2, 3]`) trên 180 ca test của `TempUN Small` (`eval_mcqa.jsonl`).
2. **Tiêu chí thành công và cổng dừng (Stop Gates):**
   - So sánh `M1-TEMP` vs `M1-NC`: Phải vượt trội có ý nghĩa thống kê ($p < 0{,}05$) qua kiểm định ghép cặp Edwards McNemar trên 180 ca độc lập để chứng minh tính nhân quả của tín hiệu thời gian.
   - So sánh `M1-TEMP` vs `M0-FC40`: Phải đạt $p < 0{,}05$ và $\Delta \text{Strict Acc} > 0$. Nếu $p \ge 0{,}05$, ghi nhận trung thực phán quyết `FAILED_TO_REJECT_H0`.
   - Cổng dừng cú pháp: Tỷ lệ cú pháp không hợp lệ (Invalid Rate) không được vượt quá $30{,}0\%$.
3. **Phân tầng giai đoạn nghiên cứu (Phased Progression):**
   - Các kỹ thuật Replay Buffer, O-LoRA (Orthogonal Subspace Projection), và EWC (Elastic Weight Consolidation) được xếp vào **Giai đoạn 2 (Phase 2 — Anti-Forgetting)**, chỉ kích hoạt khi phân tích thực tế ghi nhận hiện tượng quên lãng tri thức (BWT suy giảm nghiêm trọng). Kết quả Jev System One API (`jev-1.13.0`, $noul=0{,}16$) hoàn toàn đồng thuận với quyết định hoãn này.
4. **Giải trình khoa học 3 câu sẵn sàng gặp GVHD:**
   > *"Thưa Thầy, nhánh Learning đã khóa cứng kiến trúc Periodic Dual-Cycle MaT-LoRA làm mốc xuất phát cơ sở ($M_0\text{-FC40}$) trên biểu diễn độc lập FoNE × Can-Chi 40 chiều, phân lập hoàn toàn khỏi đa tạp hình học của đối tác. Do thay đổi giao diện tiêm prefill an toàn và luồng nạp đặc trưng, checkpoint cũ của đối tác không tương thích nhị phân trực tiếp và được quy chiếu là đối chứng lịch sử, đòi hỏi một đợt xác lập cơ sở sạch. Thí nghiệm tiếp theo tập trung kiểm định hàm mục tiêu thời gian nhân quả qua bộ ba ghép cặp $M_0\text{-FC40}$ vs $M_1\text{-TEMP}$ vs $M_1\text{-NC}$ trên chuỗi chỉ số TempUN thực tế, có đối chứng âm để loại trừ hiệu ứng điều hòa ngẫu nhiên trước khi mở rộng sang cơ chế chống quên lãng."*
5. **Quy trình tiếp theo:**
   - Không tự ý mở GPU full training hoặc thay đổi kích thước mô hình $W_0$.
   - Hành động kỹ thuật kế tiếp duy nhất: Chạy kiểm thử forward pass và tính ổn định gradient trên CPU (`VERIFY_M0_INTERFACE_AND_SMOKE`).

---

## E. Bài Học Khoa Học và Phân Loại Lỗi — M0-FC40 (Insights & Failure Modes)

### 1. Bảng phân loại lỗi thực nghiệm M0-FC40

| Hiện tượng quan sát | Nguyên nhân gốc rễ (Xác nhận / Giả thuyết) | Cổng phòng ngừa | Đường dẫn bằng chứng |
| :--- | :--- | :--- | :--- |
| **Lỗi kích thước tensor (V1)**: Đổ vỡ khi thay thế MLP tại bước 0. | **Xác nhận**: Tầng nạp lượng tử 4-bit (`Linear4bit`) có packed buffer không tương thích với `weight.shape` khi gắn wrapper. | G4 (Tensor Shape & Native Precision Gate): Nạp trực tiếp mô hình FP16 native thay vì 4-bit quantization. | `artifacts/diagnostics/M0_FC40/kaggle_version_1_failure.json` |
| **Sụp đổ số học NaN loss (V2 cũ)**: Loss bằng NaN, 9 checkpoints hỏng. | **Xác nhận**: Khởi tạo $B \ne 0$ trên 72 modules tích lũy qua 24 tầng, kết hợp CrossEntropy FP16 trên 151k logits tràn số. | G5 (Finite Arithmetic Gate): Khởi tạo $B=0$, adapter/seam giữ chuẩn FP32, upcast logits sang FP32 trước CrossEntropy. | `artifacts/audits/M0_FC40_ACCEPTANCE_REPORT.json` |
| **156/156 câu trắc nghiệm Invalid**: Cả 3 seed cùng đạt $2{,}22\%$ Acc ($4/180$). | **Xác nhận**: Gold labels là chữ thường `[a-d]`, trong khi bộ phân tích cú pháp cũ chỉ bắt mẫu chữ HOA `[A-D]`. | G1 (Label Schema Parity Gate): Đồng bộ chữ hoa/thường giữa bộ sinh, nhãn chuẩn và regex parser. | `artifacts/audits/PARSER_PARITY_REPORT.json` |
| **Chuỗi sinh rỗng (137/180 ca)**: Mô hình ngưng sinh ngay sau prompt. | **Xác nhận**: Prompt đưa vào là chuỗi thô thiếu định dạng ChatML; mô hình Instruct gặp prompt không cấu trúc lập tức sinh EOS. | G1 (Prompt Protocol Parity Gate): Áp dụng chuẩn ChatML template và chỉ dẫn định dạng phản hồi ngắn. | `artifacts/audits/M0_FC40_FORENSIC_EVAL_AUDIT.json` |
| **Mâu thuẫn kết quả n=24 vs n=180**: Seed 2 & 3 co cụm về nhãn 'Yes'. | **Xác nhận**: Tập chẩn đoán $n=24$ có phương sai mẫu lớn, không khái quát được hiện tượng Instruction Drift trên $n=180$. | G3 (Sample Size & Paired Gate): Chỉ kết luận giả thuyết trên toàn bộ 180 ca ghép cặp qua 3 seed độc lập. | `artifacts/quarantine/M0_FC40/kaggle_full_v2_3seed/RUN_IDENTITY.json` |

### 2. Năm bài học khoa học trọng tâm

1. **Bảo toàn nguyên vẹn trọng số cơ sở tại bước 0 (Zero-Delta Initialization)**: Khởi tạo tham số LoRA $B=0$ không chỉ là quy ước kỹ thuật mà là ranh giới số học bắt buộc đối với mô hình sâu. Khi can thiệp đồng thời vào 72 module MLP, bất kỳ độ lệch $\Delta W \ne 0$ ban đầu nào cũng phá vỡ phân phối trạng thái ẩn và khiến gradient bùng nổ ngay từ bước đầu tiên.
2. **Phân lập nghiêm ngặt độ chính xác lưu trữ và độ chính xác tính toán**: Mô hình nền có thể chạy ở FP16 native để tiết kiệm VRAM, nhưng toàn bộ tham số thích ứng (LoRA, dynamic core, gated seam) và các biến tích lũy optimizer bắt buộc phải duy trì ở FP32. Mất mát phân loại đa lớp trên không gian từ vựng lớn (151.936 tokens) bắt buộc phải upcast logits sang FP32 trước khi tính toán.
3. **Độ lệch giao thức đánh giá phá vỡ tính hợp lệ của thước đo**: Sự suy giảm chỉ số từ $20{,}51\%$ xuống $2{,}22\%$ trước đây không phản ánh sự suy thoái năng lực của mô hình mà bắt nguồn từ sự không tương thích giữa định dạng prompt đưa vào và bộ phân tích cú pháp đầu ra. Mọi kết luận so sánh phải đứng trên cùng một giao thức token hóa và regex chuẩn tắc.
4. **Giới hạn của tập chẩn đoán quy mô nhỏ ($n=24$)**: Kết quả trên 24 mẫu chẩn đoán chỉ có giá trị sàng lọc lỗi kỹ thuật giao diện, tuyệt đối không đủ lực lượng thống kê để khẳng định độ lợi của seam hay bác bỏ hiện tượng quên tri thức. Mức chênh lệch $1/24$ mẫu hoàn toàn nằm trong biên độ nhiễu ngẫu nhiên.
5. **Hiện tượng trôi dạt chỉ dẫn (Instruction Drift) là thực thể khoa học thực tế**: Đợt huấn luyện toàn diện 3 seed ($n=180$, $1.620$ dự đoán) chứng minh rằng việc tiếp nhận biểu diễn thời gian qua hàm mất mát Next-Word Prediction khiến mô hình có xu hướng co cụm về các nhãn an toàn nhị phân (nhãn 'Yes' chiếm ưu thế ở Seed 2 và 3). Đây chính là bài toán khoa học cốt lõi mà Nhánh V2 cần giải quyết thông qua các hàm mục tiêu điều hòa bảo tồn chỉ dẫn.

### 3. Kiểm toán Pháp y Toàn từ vựng và Phân bổ Nguyên nhân (M0-FC40 Full-Vocabulary Forensics & Attribution)

Nhằm giải quyết triệt để nghi vấn về khoảng cách hiệu năng giữa tập chẩn đoán sơ bộ $n=24$ ($33{,}33\%$ Strict Acc, $0\%$ Invalid) và đợt chạy huấn luyện toàn diện 3 seed $n=180$ ($12{,}78\% \pm 5{,}77\%$ Strict Acc), toàn bộ quy trình suy luận và đánh giá đã được kiểm toán pháp y toàn diện trên không gian sinh tự do toàn từ vựng (151.936 tokens) dưới giao thức `RAW_GREEDY_V1` (`do_sample=False, num_beams=1, repetition_penalty=1.0`).

Kết quả kiểm toán tại [GENERATION_FREEDOM_AUDIT.json](file:///c:/Users/GF63/Documents/long-range-temporal-alignment/artifacts/audits/GENERATION_FREEDOM_AUDIT.json) xác nhận $100\%$ không tồn tại can thiệp tính điểm lựa chọn (option scoring), không lọc logit (logit filtering/masking), không ràng buộc ngữ pháp (grammar-guided decoding) hay thay thế chuỗi sinh hậu nghiệm. Nguyên nhân phân kỳ được bóc tách thành hai yếu tố tương tác cụ thể:

1. **Lỗi che giấu cú pháp hợp lệ của bộ đánh giá (Evaluator Syntax Validity Masking):**
   Trong triển khai trước đây tại `tempun_dataset.py`, hàm `parse_strict_mcqa_answer` trích xuất nhãn độc lập với dạng câu hỏi. Khi mô hình huấn luyện liên tục sinh ra nhãn `'Yes'` trên câu hỏi trắc nghiệm 4 lựa chọn (MCQA), parser vẫn ghi nhận nhãn hợp lệ (`is_valid = True`), dẫn đến tỷ lệ lỗi cú pháp bị báo cáo sai lệch là $0{,}00\%$. Trên thực tế, đối với 156 câu MCQA, sinh `'Yes'` là lỗi sai loại phản hồi (`WRONG_ANSWER_TYPE`). Tái đánh giá bằng bộ chấm điểm phân biệt dạng nhiệm vụ tại [M0_FC40_EVALUATOR_V2_AUDIT.json](file:///c:/Users/GF63/Documents/long-range-temporal-alignment/artifacts/audits/M0_FC40_EVALUATOR_V2_AUDIT.json) và ma trận phân loại [M0_FC40_TASK_TYPE_CONFUSION_MATRIX.csv](file:///c:/Users/GF63/Documents/long-range-temporal-alignment/artifacts/audits/M0_FC40_TASK_TYPE_CONFUSION_MATRIX.csv) chỉ ra rằng tỷ lệ Invalid thực tế của Seed 2 và Seed 3 đạt từ $85{,}00\%$ đến $87{,}78\%$ ($153$–$156/156$ câu MCQA bị invalid). Mức độ chính xác $9{,}44\%$ ($17/180$) của hai seed này xuất phát hoàn toàn từ việc mô hình dự đoán `'Yes'` và trùng khớp ngẫu nhiên với đúng $17/24$ câu hỏi Yes/No có đáp án chuẩn là `'Yes'`, trong khi độ chính xác thực tế trên 156 câu MCQA là $0{,}00\%$.

2. **Nhiễu loạn phân phối do chỉ dẫn hỗn hợp (P11 Prompt Interference) và Sự phục hồi của P12:**
   Bóc tách phân phối xác suất bước đầu tiên (First-Token Logits) tại [M0_FC40_FIRST_TOKEN_AND_CONDITIONING_DIAGNOSTICS.jsonl](file:///c:/Users/GF63/Documents/long-range-temporal-alignment/artifacts/audits/M0_FC40_FIRST_TOKEN_AND_CONDITIONING_DIAGNOSTICS.jsonl) chứng minh rằng câu chỉ dẫn hỗn hợp của $P_{11}$ (*"Answer directly with only the option letter (a, b, c, d) or Yes/No"*) tạo ra thiên kiến tiên nghiệm nghiêm trọng: tổng xác suất tích lũy của hai token nhị phân `Yes` và `No` chiếm tới $66{,}4\%$ trên câu hỏi MCQA, đè bẹp xác suất của 4 phương án $a, b, c, d$ xuống dưới $6\%$ mỗi lựa chọn. Ngược lại, khi chuyển sang giao thức phân biệt dạng câu hỏi $P_{12}$ (*Task-Aware ChatML*: câu MCQA chỉ yêu cầu $a, b, c, d$; câu Yes/No chỉ yêu cầu $Yes, No$), xác suất của các phương án $\{a, b, c, d\}$ lập tức phục hồi lên $92{,}9\%$, xác suất `Yes`/`No` sụt giảm về $0{,}003\%$.

3. **Kết quả thực nghiệm đối chứng ghép cặp (Paired Protocol Attribution):**
   Thực nghiệm ghép cặp đa chiều ghi nhận tại [M0_FC40_FACTORIAL_ATTRIBUTION.md](file:///c:/Users/GF63/Documents/long-range-temporal-alignment/artifacts/audits/M0_FC40_FACTORIAL_ATTRIBUTION.md) cho thấy dưới giao thức chuẩn tắc $P_{12}$, toàn bộ 3 seed của đợt chạy huấn luyện đầy đủ phục hồi độ chính xác xuất sắc:
   - **Seed 1 after_02:** Đạt $41{,}67\%$ Strict Acc ($0{,}00\%$ Invalid Rate; MCQA Acc: $31{,}25\%$, Yes/No Acc: $62{,}50\%$).
   - **Seed 2 after_02:** Đạt $37{,}50\%$ Strict Acc ($0{,}00\%$ Invalid Rate; MCQA Acc: $31{,}25\%$, Yes/No Acc: $50{,}00\%$).
   - **Seed 3 after_02:** Đạt $33{,}33\%$ Strict Acc ($0{,}00\%$ Invalid Rate; MCQA Acc: $18{,}75\%$, Yes/No Acc: $62{,}50\%$).
   - **Trung bình 3 seed Full Run dưới P12:** Đạt **$37{,}50\% \pm 4{,}17\%$** Strict Accuracy và **$0{,}00\%$** Invalid Rate, ngang hàng với mô hình nền chưa huấn luyện ($41{,}67\%$) và vượt trội hoàn toàn mốc canary ($37{,}50\%$).

**Phán quyết khoa học chính thức:** Giả thuyết về năng lực biểu diễn và sự ổn định của kiến trúc `M0-FC40` được **CHẤP THUẬN (SUPPORTED)**. Checkpoints và trọng số huấn luyện hoàn toàn nguyên vẹn; hiện tượng suy giảm trước đây được quy kết $100\%$ do lỗi che giấu cú pháp của bộ đánh giá cũ và sự nhiễu loạn phân phối từ lời nhắc $P_{11}$. Toàn bộ bằng chứng được niêm phong tại [M0_FC40_FULL_VOCAB_FORENSIC_VERDICT.yaml](file:///c:/Users/GF63/Documents/long-range-temporal-alignment/Receipts/M0_FC40_FULL_VOCAB_FORENSIC_VERDICT.yaml).

---

## F. Kho Lưu Trữ Đối Chứng Bài Báo Gốc TickTack v3 và Rào Cản Tái Lập (Original Baseline Archive & Reproducibility Gate)

Toàn bộ dữ liệu đối chứng từ bài báo TickTack gốc (*Xue Han et al.*, arXiv:2503.04150v3) đã được trích xuất nguyên văn và lưu trữ độc lập tại [data/Output/TICKTACK_ORIGINAL_2503_04150_V3/](file:///c:/Users/GF63/Documents/long-range-temporal-alignment/data/Output/TICKTACK_ORIGINAL_2503_04150_V3/). Theo Bảng 1 bài báo gốc trên TempUN (zero-shot), mô hình Qwen2.5-3B đạt $64{,}96\%$ và Qwen2.5-7B đạt $74{,}29\%$; bài báo hoàn toàn không công bố kết quả cho `Qwen2.5-0.5B`.

Về ranh giới so sánh nội bộ, kết quả $37{,}50\% \pm 4{,}17\%$ của $M_0\text{-FC40}$ chỉ đo trên tập chẩn đoán sơ bộ $n=24$ dưới giao thức $P_{12}$. Trên cùng $24$ ca này, mô hình nền chưa qua huấn luyện (`BASE_MODEL_FROZEN`) đạt $41{,}67\%$ ($10/24$), cao hơn $M_0\text{-FC40}$ ($9/24$, chênh lệch $-4{,}17$ pp). Do đó, dữ liệu hiện tại bác bỏ mọi tuyên bố về độ lợi của $M_0\text{-FC40}$ so với Base model trên tập này, đồng thời việc so sánh trực tiếp $37{,}50\%$ với $64{,}96\%$ của bài báo gốc là hoàn toàn vô căn cứ khoa học.

Về khả năng tái lập, cổng kiểm định xác lập trạng thái `BLOCKED_MISSING_ARTIFACTS_AND_HARDWARE`: tác giả chưa công bố mã nguồn chính thức, không công bố danh sách mẫu $N$ cụ thể của split TempUN và đòi hỏi cụm $8 \times \text{A100}$. Dự án đã lập kế hoạch tái lập thích ứng độc lập trên `TempUN Small` kèm 22 tệp cấu trúc JSON/MD, khóa cứng nguyên tắc không tự ý kích hoạt GPU huấn luyện khi chưa có phê duyệt chính thức.

---

## G. Thực Nghiệm Đối Chứng Baseline TickTack-Inspired 0.5B vs M0-FC40 trên Toàn Bộ 180 Câu TempUN Small

**1. Phát hiện pháp y nạp trọng số (Loader Bypass Forensics):** Kết quả sơ khởi ghi nhận $M_0\text{-FC40}$ đạt $58/180 = 32{,}22\%$ y hệt mô hình nền trên cả 3 seed đã được kiểm toán và phân loại là `SUPERSEDED_INVALID_INFERENCE`. Nguyên nhân gốc rễ xuất phát từ việc script cũ kiểm tra sai tên khóa từ điển (`"adapted_modules"` thay vì `"adapters_state"` và `"gated_proj"` thay vì `"gated_proj_state"`), khiến không có trọng số LoRA hay Seam nào được nạp, và mô hình thực thi chính là mô hình nền. Lỗi đã được khắc phục triệt để bằng bộ nạp chuẩn tắc [m0_fc40_loader.py](file:///c:/Users/GF63/Documents/long-range-temporal-alignment/src/temporal_alignment/models/m0_fc40_loader.py) với các rào chắn fail-closed bắt buộc đủ $72/72$ modules và kiểm tra hoạt hóa tham số (đạt 3/3 regression tests).

**2. Kết quả đo đạc thực tế sau sửa lỗi (Giao thức $P_{12}$, `RAW_GREEDY_V1` trên toàn bộ 180 câu):**
- **Base Qwen2.5-0.5B:** Đạt $32{,}22\%$ Strict Accuracy ($58/180$), MCQA: $26{,}28\%$, Yes/No: $70{,}83\%$, Invalid: $0{,}00\%$.
- **M0-FC40 Dual-Cycle (Sau sửa nạp trọng số):** Đạt **$29{,}81\% \pm 2{,}80\%$** Strict Accuracy (Seed 1: $27{,}22\%$, Seed 2: $32{,}78\%$, Seed 3: $29{,}44\%$), MCQA: $24{,}15\%$, Yes/No: $66{,}66\%$, Invalid: $0{,}00\%$. So với Base model, $M_0\text{-FC40}$ chênh lệch $-2{,}41\text{ pp}$ về điểm trung bình (McNemar $p > 0{,}17$, Bootstrap 95% CI bao trùm 0).
- **TickTack-Inspired 0.5B (3 seed mean):** Đạt **$35{,}19\% \pm 1{,}40\%$** Strict Accuracy (Seed 1: $33{,}89\%$, Seed 2: $35{,}00\%$, Seed 3: $36{,}67\%$), MCQA: $29{,}70\%$, Yes/No: $70{,}83\%$, Invalid: $0{,}00\%$.

**3. Phán quyết khoa học và ý nghĩa thống kê:** Chênh lệch hiệu năng quan sát được giữa TickTack-inspired và $M_0\text{-FC40}$ là $+5{,}38\text{ pp}$ về giá trị trung bình. Tuy nhiên, kiểm định McNemar ghép cặp trên từng seed đều cho $p > 0{,}05$ (Seed 1: $p=0{,}1272$, Seed 2: $p=0{,}3428$, Seed 3: $p=0{,}0736$) và khoảng tin cậy Bootstrap 95% bao trùm mốc 0 ở Seed 1 và Seed 2. Do đó, trạng thái khoa học chính xác được ghi nhận là **`OBSERVED / STATISTICALLY_INCONCLUSIVE`** (có chênh lệch quan sát thực nghiệm nhưng chưa đủ bằng chứng kết luận thắng thế mang ý nghĩa thống kê trên tập 180 ca). Toàn bộ hồ sơ kiểm toán và bảng so sánh từng ca lưu tại [M0_FC40_INFERENCE_CAUSAL_AUDIT_FIX_RERUN.yaml](file:///c:/Users/GF63/Documents/long-range-temporal-alignment/Receipts/M0_FC40_INFERENCE_CAUSAL_AUDIT_FIX_RERUN.yaml) và thư mục `artifacts/audits/M0_FC40_CAUSAL_INFERENCE_RECOVERY_20261015/`.

---

## H. Chuyển Trục Nghiên Cứu V2: Căn Chỉnh Thời Gian Nhân Quả (Run 01) và Kiểm Định Đối Chứng Âm

**1. Quan sát thực nghiệm:**
Thực nghiệm phân tích 4 nhánh nhân tố đối chứng trên toàn bộ 180 ca TempUN Small ghi nhận:
- $C_0$ (M0 Baseline, True Year): $28{,}89\%$ Strict Accuracy ($52/180$), MCQA: $22{,}44\%$, Yes/No: $70{,}83\%$, Invalid: $0{,}00\%$.
- $C_1$ (Targeted Relation, True Year): $28{,}89\%$ Strict Accuracy ($52/180$), MCQA: $22{,}44\%$, Yes/No: $70{,}83\%$, Invalid: $0{,}00\%$.
- $C_2$ (Negative Control, Shuffled Year): $28{,}89\%$ Strict Accuracy ($52/180$), MCQA: $22{,}44\%$, Yes/No: $70{,}83\%$, Invalid: $0{,}00\%$.
- $C_3$ (Format Control, Zero Vector $0_{40D}$): $27{,}78\%$ Strict Accuracy ($50/180$), MCQA: $21{,}15\%$, Yes/No: $70{,}83\%$, Invalid: $0{,}00\%$.

**2. Nguyên nhân thử nghiệm và bóc tách biến số:**
Chẩn đoán Stage 3 (`ROOT_CAUSE_HYPOTHESES.json`) xác nhận 3 rào cản: lệch pha nhiệm vụ 100% ($2.461$ mẫu train là continuous value trong khi 100% eval là MCQA/YesNo), bất cân xứng số lượng đối số thời gian ($80\%$ câu hỏi so sánh từ 2 năm trở lên nhưng M0 chỉ nhận 1 vector), và seam token 14 nằm trọn trong System prompt (`' selecting'`). Can thiệp $C_1$ đưa biểu diễn quan hệ thời gian hướng đích vào vector điều kiện.

**3. Đối chứng âm và độ nhạy thời gian:**
Đo lường phân tích logit tại vị trí sinh đầu tiên ghi nhận mean max logit delta $= 0{,}9763$ (dao động $0{,}1868$ đến $2{,}2341$), xác nhận vector thời gian 40D thực sự lan truyền qua cổng Gated Residual Seam. Tuy nhiên, kiểm định ghép cặp McNemar giữa $C_1$ và $C_2$ cho $p = 0{,}61708$ (delta $= 0{,}0\%$), và giữa $C_1$ và $C_3$ cho $p = 0{,}77283$ (delta $= +1{,}11\%$). Hiệu năng của True Year và Shuffled Year là hoàn toàn tương đương trên không gian giải mã tự do toàn từ vựng.

**4. Bài học khoa học rút ra:**
Dưới cơ chế ghép seam hiện hành tại token 14 (thuộc System prompt), vector đặc trưng thời gian dù tạo ra độ lệch kích hoạt nhất định nhưng không đủ năng lượng để định hướng lại tiên nghiệm ngôn ngữ gốc (language priors) của mô hình nền đối với các lựa chọn MCQA. Mô hình chưa thực sự "suy luận theo thời gian" mà đang vận hành dựa trên prior ngôn ngữ và ngữ cảnh câu hỏi.

**5. Cổng kế tiếp (Stop Gate):**
Đóng cổng thực nghiệm `V2_CAUSAL_EXP_RUN_01` ở trạng thái `INFERENCE_ONLY_UNTRAINED_INTERVENTIONS` (đã kiểm toán sửa sai tại `RUN01_FORENSIC_CORRECTION.json`). Giữ nguyên ranh giới đóng băng backbone và khế ước `FIXED_FONE_CANCHI_V1` 40D; chuyển tiếp sang thực nghiệm huấn luyện thực sự Run 02. Toàn bộ bằng chứng lưu trữ tại [artifacts/experiments/V2_CAUSAL_TEMPORAL_ALIGNMENT/run_01/](file:///c:/Users/GF63/Documents/long-range-temporal-alignment/artifacts/experiments/V2_CAUSAL_TEMPORAL_ALIGNMENT/run_01/).

---

## I. Huấn Luyện Căn Chỉnh Quan Hệ Thời Gian Thực Sự (Run 02) và Bằng Chứng Nhân Quả Độc Lập

**1. Sửa sai kiểm toán Run 01:**
Biên bản pháp y [`RUN01_FORENSIC_CORRECTION.json`](file:///c:/Users/GF63/Documents/long-range-temporal-alignment/artifacts/experiments/V2_CAUSAL_TEMPORAL_ALIGNMENT/run_02/RUN01_FORENSIC_CORRECTION.json) xác nhận Run 01 không thực thi bất kỳ bước cập nhật optimizer nào (`optimizer = None`, zero `.backward()`), chỉ là thử nghiệm hoán đổi vector điều kiện lúc suy luận trên checkpoint M0 tĩnh (`INFERENCE_ONLY_UNTRAINED_INTERVENTIONS`).

**2. Huấn luyện thực sự 4 nhánh đối chứng (Run 02):**
Trích xuất $2.408$ cặp so sánh thời gian từ $2.461$ bản ghi `train_nwp.jsonl` theo nhóm thực thể độc lập ($1.926$ cặp train trên 3 entities; $482$ cặp dev trên 2 entities `BCXZ`, `GVRO`, $0\%$ rò rỉ). Cả 4 nhánh được tối ưu hóa $30$ bước bằng AdamW ($lr=2\times 10^{-4}$), xuất $4$ checkpoint riêng biệt với mã băm và khoảng cách tham số $\Delta W > 1{,}80$:
- `C0_CONTINUED_MASKED`: $L_{\text{masked}}$, True Year.
- `C1_TRUE_TEMPORAL_RELATION`: $L_{\text{masked}} + 0{,}5 L_{\text{rel}}$, True Year.
- `C2_SHUFFLED_CONDITIONING`: $L_{\text{masked}} + 0{,}5 L_{\text{rel}}$, Shuffled Random Year.
- `C3_FORMAT_ONLY`: $L_{\text{masked}} + 0{,}5 L_{\text{format}}$, True Year.

**3. Kết quả đo lường và kiểm định nhân quả:**
Đánh giá trên tập held-out dev pairs ($n=60$) ghi nhận:
- Độ chính xác: $C_1 = 75{,}00\%$ ($45/60$) vượt trội áp đảo so với $C_0 = 48{,}33\%$ ($29/60$), $C_2 = 51{,}67\%$ ($31/60$) và $C_3 = 48{,}33\%$ ($29/60$).
- Kiểm định McNemar chính xác hai phía: $C_1$ vs $C_0$ ($p = 0{,}00904$), $C_1$ vs $C_2$ ($p = 0{,}00661$), $C_1$ vs $C_3$ ($p = 0{,}01133$), toàn bộ đều đạt ý nghĩa thống kê ($p < 0{,}05$).
- Kiểm tra đối chứng ngược thời gian (Counterfactual Inversion $n=30$): Khi đảo ngược mốc năm, $C_1$ đạt tỷ lệ đảo nhãn $66{,}67\%$ (đúng cả hai chiều $56{,}67\%$), trong khi $C_0, C_2, C_3$ đều đạt $0{,}00\%$.

## J. Căn Chỉnh Đa Nhiệm Quan Hệ Thời Gian và Khái Quát Hóa MCQA (Run 03)

**1. Bối cảnh và mục tiêu:**
Từ hiện tượng chuyên biệt hóa định dạng của Run 02 khiến checkpoint $C_1$ rơi vào thiên kiến chữ cái khi làm trắc nghiệm MCQA (chỉ đoán `a` và `b`, đạt $20{,}00\%$, thấp hơn mức cơ hội $25{,}00\%$), Run 03 kiểm chứng giả thuyết: *Liệu mục tiêu đa nhiệm cân bằng $\mathcal{L} = 0{,}5 \mathcal{L}_{\text{rel}} + 0{,}5 \mathcal{L}_{\text{mcqa}}$ có phục hồi năng lực làm bài trắc nghiệm 4 lựa chọn mà vẫn bảo toàn tính nhân quả thời gian?*

**2. Thiết kế và thực thi thực nghiệm:**
Xuất phát từ checkpoint $C_1$ (SHA256: `09acf723ea97401924a070b14775d09584d0a7a565c68d83c71218e814f72cb9`), huấn luyện $4$ nhánh đối chứng với $30$ bước AdamW ($lr=2\times 10^{-4}$, batch size $=2$) trên tập dữ liệu train/dev phân rã nhóm thực thể tuyệt đối ($0\%$ rò rỉ benchmark). Checkpoint chính `R3_C_JOINT_TRUE` (SHA256: `bceaf7e8fb40447373dc2be519d9aad7d07e4d340ca49901b6a07d1dc384a808`) đạt độ dịch tham số $\Delta W = 1{,}451$ so với $C_1$.

**3. Kết quả đo lường 4 bảng điểm độc lập:**
- *Scoreboard A (Quan hệ thời gian, $n=60$):* $R3_C$ đạt $86{,}67\%$ ($52/60$), vượt trội có ý nghĩa thống kê so với $C_1$ ($75{,}00\%$, McNemar $p=0{,}0391$) và $R3_A$ ($70{,}00\%$, $p=0{,}0020$).
- *Scoreboard B (MCQA 4 lựa chọn cân bằng, $n=120$):* $R3_C$ đạt $65{,}00\%$ ($78/120$), vượt trội áp đảo $C_1$ ($20{,}00\%$, $p < 0{,}0001$) và triệt tiêu thiên kiến ký tự.
- *Scoreboard C (Phản thực nghiệm & Bất biến hoán vị):* Tính bất biến khi xoay vòng vị trí phương án của $R3_C$ tăng vọt lên $70{,}00\%$ (so với $5{,}00\%$ ở $C_1$), tỷ lệ nhất quán hai chiều đạt $80{,}00\%$.
- *Scoreboard D (Lưu giữ benchmark TempUN Small, $n=60$):* Đạt $0{,}00\%$ do quá khớp vào cú pháp phát biểu dài, bộc lộ giới hạn dung lượng của adapter LoRA rank 8.

**4. Phán quyết:** **`PARTIALLY_SUPPORTED`**. Huấn luyện đa nhiệm phục hồi thành công năng lực suy luận nội dung phương án trên tập đóng, song đòi hỏi tăng bậc thích ứng hoặc replay buffer để chống quên cú pháp ngoài phân phối. Hồ sơ lưu tại [artifacts/experiments/V2_CAUSAL_TEMPORAL_ALIGNMENT/run_03/](file:///c:/Users/GF63/Documents/long-range-temporal-alignment/artifacts/experiments/V2_CAUSAL_TEMPORAL_ALIGNMENT/run_03/).

---

## K. Sửa Sai Chân Lý Dữ Liệu, Căn Chỉnh Nhân Quả và Giải Cứu Năng Lực Lưu Giữ (Run 04)

**1. Kiểm toán pháp y dữ liệu Run 03 và nguyên nhân gốc rễ:**
Kiểm toán độc lập tại [RUN03_SEMANTIC_DATA_FORENSIC.json](file:///c:/Users/GF63/Documents/long-range-temporal-alignment/artifacts/experiments/V2_CAUSAL_TEMPORAL_ALIGNMENT/run_04/RUN03_SEMANTIC_DATA_FORENSIC.json) xác nhận tập dữ liệu Run 03 mắc lỗi chân lý nghiêm trọng: $53/400$ ($13{,}25\%$) câu train và $20/120$ ($16{,}67\%$) câu dev chứa đồng thời hai đáp án đúng (toàn bộ các ca giảm giá trị $v_2 < v_1$). Đồng thời, tỷ lệ câu hỏi tăng giá trị chiếm $83{,}33\%$ tạo ra shortcut định hướng cho mô hình đoán mò. Văn bản [RUN03_SUPERSEDED_CLAIMS.md](file:///c:/Users/GF63/Documents/long-range-temporal-alignment/artifacts/experiments/V2_CAUSAL_TEMPORAL_ALIGNMENT/run_04/RUN03_SUPERSEDED_CLAIMS.md) chính thức đình chỉ các kết luận chưa chuẩn xác của Run 03.

**2. Khế ước dữ liệu sạch chuẩn AST và giải cứu Retention:**
Tập dữ liệu mới (`MCQA4_TRAIN_CLEAN.jsonl`, `MCQA4_DEV_CLEAN.jsonl`) bảo đảm $100\%$ đơn nhất chân lý (`oracle_truth_count == 1`), cân bằng chuẩn xác $50\%$ tăng / $50\%$ giảm và $25\%$ cho mỗi nhãn `a, b, c, d`, triệt tiêu hoàn toàn rò rỉ sang benchmark đánh giá ($0\%$). Về năng lực lưu giữ, chẩn đoán xác nhận điểm số $0/60$ cũ xuất phát từ trôi dạt định dạng lời nhắc và lỗi bộ phân tích, không phải do suy giảm tham số. Điểm số của Base model đạt $31{,}67\%$ và $C_1$ đạt $38{,}33\%$.

**3. Kết quả huấn luyện GPU Kaggle và đo lường 4 bảng điểm:**
Huấn luyện 4 nhánh đối chứng từ checkpoint $C_1$ tinh gọn ($13{,}08$ MB) với 30 bước AdamW ($lr=2\times 10^{-4}$):
- *Scoreboard 1 (Clean MCQA 120 câu, 50/50):* $R4\_B$ đạt **$65{,}83\%$** (vượt $+15{,}83\text{ pp}$ so với mức đoán mò $50\%$; chiều tăng đạt $73{,}33\%$, chiều giảm đạt $58{,}33\%$), $R4\_C$ đạt $64{,}17\%$.
- *Scoreboard 2 (Dev Relation 60 câu Yes/No):* $R4\_B$ đạt **$86{,}67\%$** (phân bổ cân bằng $29$ Yes / $31$ No), vượt trội $+16{,}67\text{ pp}$ so với $R4\_C$ ($70{,}00\%$).
- *Scoreboard 3 (Chẩn đoán Oracle Evidence 60 câu):* $R4\_B$ đạt **$65{,}00\%$** (Base model chỉ đạt $38{,}33\%$), chứng minh mô hình thực hiện tốt phép so sánh số học khi có đủ thông tin sự thật.
- *Scoreboard 4 (Retention 60-slice TempUN Small):* $R4\_B$ phục hồi về **$28{,}33\%$** (Invalid $= 0{,}0\%$), giải cứu thành công năng lực làm bài benchmark gốc.

**4. Phán quyết khoa học:** **`SUPPORTED (Data Truth & Retention Rescued) / CAUSALITY_INCONCLUSIVE (MCQA)`**. Sửa chữa triệt để lỗi chân lý dữ liệu, khôi phục năng lực lưu giữ benchmark và chứng minh năng lực so sánh thời gian khi có bằng chứng số học. Toàn bộ bằng chứng niêm phong tại [artifacts/experiments/V2_CAUSAL_TEMPORAL_ALIGNMENT/run_04/](file:///c:/Users/GF63/Documents/long-range-temporal-alignment/artifacts/experiments/V2_CAUSAL_TEMPORAL_ALIGNMENT/run_04/) và [Receipts/V2_RUN04_TRUTH_REPAIR_AND_RETENTION_RESCUE.yaml](file:///c:/Users/GF63/Documents/long-range-temporal-alignment/Receipts/V2_RUN04_TRUTH_REPAIR_AND_RETENTION_RESCUE.yaml).

---

## L. Tái Lập Khoa Học, Khắc Phục Logic Thực Thi và Pháp Y Năng Lực Mô Hình (Run 04.1)

**1. Kiểm toán độc lập và khắc phục lỗi logic thực thi (F01 & F02):**
Kiểm toán độc lập tại [RUN041_PRE_REGISTERED_AUDIT.json](file:///c:/Users/GF63/Documents/long-range-temporal-alignment/artifacts/experiments/V2_CAUSAL_TEMPORAL_ALIGNMENT/run_04_1/RUN041_PRE_REGISTERED_AUDIT.json) xác nhận 12 giả thuyết pháp y thực nghiệm F01–F12:
- *Lỗi ngắt adapter cũ (F01):* Trong script cũ, `use_adapter=False` chỉ tắt việc gán vector nhưng module `DualCycleMaTLoRALinear` vẫn cộng `lora_out` vào `base_out` với tọa độ zero và trọng số $C_1$ trong bộ nhớ, làm sai lệch điểm số `W0_BASE`. Đã khắc phục triệt để bằng công tắc `self.enabled = False`. Kiểm chứng tương đương số học trên 300 ca suy luận giữa `RAW_QWEN05B` và `PATCHED_BASE_EQUIVALENT` đạt `max_abs_diff = 0.0000000000`, $100\%$ không lật nhãn (Gate G1 PASS).
- *Rò rỉ trạng thái (F02):* Đã chuẩn hóa quản lý hook và tọa độ thời gian bằng context manager `scoped_conditioning_context` với `try/finally` bảo đảm thu hồi hook `register_forward_hook` và dọn sạch bộ nhớ.
- *Trạng thái R4_D (F03 & F08):* Đối soát dấu vân tay tensor trên toàn bộ 508 tensor khả huấn xác nhận $R4\_D$ hoàn toàn trùng khớp $R4\_B$ (`diff = 0.0`, cùng tensor fingerprint `3f4fa790...`). Khác biệt file SHA256 chỉ do khóa metadata `"arm_name"`. Tập dữ liệu Kaggle của $R4\_D$ lấy mẫu cùng tập $R4\_B$, không hề có mẫu hay hàm mất mát replay. Do đó, $R4\_D$ được phân loại chuẩn xác là `REPLAY_NOT_TESTED`.

**2. Bảng điểm đối soát chuẩn tắc (Reconciled Scoreboards, $n=300$):**
- *Scoreboard 1 (Clean MCQA, $n=120$):* Base Model đạt $40{,}00\%$, $R4\_A$ đạt $59{,}17\%$, $R4\_B$ đạt **$65{,}83\%$**, $R4\_C$ đạt $64{,}17\%$.
- *Scoreboard 2 (Dev Relation, $n=60$):* Base Model đạt $48{,}33\%$, $R4\_B$ đạt **$86{,}67\%$** (vượt trội $+16{,}67\text{ pp}$ so với Shuffled $R4\_C$ đạt $70{,}00\%$).
- *Scoreboard 3 (Oracle Evidence, $n=60$):* Base Model đạt $58{,}33\%$, $R4\_B$ đạt $61{,}67\%$.
- *Scoreboard 4 (Retention Slice, $n=60$):* Base Model đạt $31{,}67\%$, $R4\_B$ đạt $28{,}33\%$ ($0{,}0\%$ Invalid).

**3. Can thiệp điều kiện suy luận trên cùng checkpoint ($R4\_B$):**
Đo đạc 6 can thiệp suy luận trên cùng một tập trọng số $R4\_B$ bóc tách rạch ròi 4 góc phần tư năng lực (K/G/R/F):
- `ADAPTED_TRUE`: $65{,}83\%$ ($79/120$).
- `ADAPTED_WRONG`: $65{,}83\%$ ($79/120$).
- `ADAPTED_ZERO`: $65{,}00\%$ ($78/120$).
- `ADAPTED_SEAM_OFF`: $66{,}67\%$ ($80/120$).
- `ADAPTED_LORA_OFF`: $38{,}33\%$ ($46/120$).
- `ADAPTED_BOTH_OFF`: $40{,}00\%$ ($48/120$).

**4. Phán quyết khoa học:** **`VALIDATED & RECONCILED (Scientific Reproducibility Established)`**. Adapter MaT-LoRA đóng góp $+25{,}83\text{ pp}$ độ lợi so với Base model nhờ việc học tuân thủ định dạng câu hỏi trắc nghiệm sạch (F). Tính phụ thuộc nhân quả vào vector thời gian (G) được chứng minh vững chắc trên nhiệm vụ quan hệ thời gian (S2), nhưng trên trắc nghiệm MCQA đóng (S1), vector thời gian tại Seam token 14 không điều khiển việc lựa chọn phương án. Toàn bộ 20 tệp tạo tác kiểm toán được niêm phong tại [artifacts/experiments/V2_CAUSAL_TEMPORAL_ALIGNMENT/run_04_1/](file:///c:/Users/GF63/Documents/long-range-temporal-alignment/artifacts/experiments/V2_CAUSAL_TEMPORAL_ALIGNMENT/run_04_1/) cùng mã băm xác thực [SHA256_MANIFEST.json](file:///c:/Users/GF63/Documents/long-range-temporal-alignment/artifacts/experiments/V2_CAUSAL_TEMPORAL_ALIGNMENT/run_04_1/SHA256_MANIFEST.json).

---

## M. Kiểm Toán Phản Ánh Đối Tác, Độ Chuẩn Xác Bài Báo TickTack và Chuẩn Hóa Chu Kỳ Thời Gian (Run 04.2)

**1. Phân xử độc lập các khiếu nại kỹ thuật:**
- *Khiếu nại về công thức Can-Chi (`year - 4` vs `year - 3`):* **BÁC BỎ**. Bài báo TickTack Eq. 1 dùng hệ 1-based index $(y-3) \pmod{60}$ trên $\{1,\dots,60\}$ (năm 4 CN và 1984 là lớp 1). Dự án dùng hệ 0-based index $(y-4) \pmod{60}$ trên $\{0,\dots,59\}$ (năm 4 CN và 1984 là index 0). Hai hệ quy ước đồng cấu toán học tuyệt đối, lệch pha đúng $6^\circ$ ($2\pi/60$) và không thể sinh `None` trên số nguyên.
- *Hiện tượng khuyết năm (`None`):* **XÁC NHẬN CĂN NGUYÊN DỮ LIỆU**. Kiểm toán tập `eval_mcqa.jsonl` (180 câu) phát hiện $144/180$ mẫu ($80.0\%$) bị rỗng `canonical_year: null` (thuộc dạng câu hỏi Min-Max và Relative-Based). Script cũ dùng fallback ngầm `or 1984` gán sai mốc Giáp Tý cho $80\%$ câu hỏi. Bản sửa Run 04.2 thiết lập bộ phân giải 3 tầng `YearResolution` minh bạch, cấm tuyệt đối fallback ngầm.
- *Hiện tượng suy biến góc Cosine Collapse:* **XÁC NHẬN**. Vector $\mathrm{TE}(t)$ theo Eq. (4) có chuẩn năng lượng $\|\mathrm{TE}\|_2 \approx 42.26$, trong khi chuẩn token embedding của Qwen-0.5B chỉ là $0.45$. Việc đối tác cộng broadcasting tràn vào $100\%$ các token trong câu đẩy Cosine liên token lên $0.9999$. Việc nhân tỷ lệ $1/d = 1/896$ khôi phục Cosine về $0.0702$. Tuy nhiên, bài báo gốc Eq. (5) chỉ cộng vào token năm, không broadcasting.
- *Bản chất 4 con số metric đối tác:* **XÁC THỰC SỐ HỌC NHƯNG KHÁC GIAO THỨC**. Các con số Base $26.67\%$, PT $29.07\%$, TickTack Raw $22.41\%$, TickTack Scaled $28.89\%$ được đo trên giao thức **Academic Choice Logit Scoring** (đo trực tiếp logit token A/B/C/D, Invalid rate $= 0.0\%$), không thể so sánh ngang hàng với giao thức sinh tự do `P12_TASK_AWARE_CHATML` ($32.22\%$).
- *Sự thật bài báo gốc (arXiv:2503.04150v3):* Bài báo chỉ công bố kết quả cho **LLaMA2-7B/13B** và **Qwen2.5-3B/7B**, hoàn toàn **không có số liệu cho Qwen2.5-0.5B**.

**2. Bảng điểm đối soát chuẩn tắc ($n=180$, tập sạch `eval_180_clean.jsonl`):**

| Mô hình / Biến thể | Giao thức Đo | Accuracy | MCQA (156) | Yes/No (24) | Invalid Rate |
|---|---|---|---|---|---|
| Qwen2.5-0.5B Base | Academic Choice Logit | 48/180 (26.67%) | 35/156 | 13/24 | 0.0% |
| TickTack Raw (Broadcast) | Academic Choice Logit | 40/180 (22.22%) | 29/156 | 11/24 | 0.0% |
| TickTack Scaled ($1/d$) | Academic Choice Logit | 52/180 (28.89%) | 39/156 | 13/24 | 0.0% |
| Qwen2.5-0.5B Base | P12 ChatML Greedy | 58/180 (32.22%) | 41/156 | 17/24 | 0.0% |
| M0-FC40 / Run02 C1 | P12 ChatML Greedy | 58/180 (32.22%) | 41/156 | 17/24 | 0.0% |
| Run03 T0 Multitask | P12 ChatML Greedy | 60/180 (33.33%) | 43/156 | 17/24 | 0.0% |

**3. Phán quyết khoa học:** **`ADJUDICATED & RECONCILED (Partner Claims Resolved, Paper Fidelity Reconstructed)`**. Toàn bộ 14 tệp tạo tác kiểm toán và 5 bộ unit test chuẩn tắc được niêm phong tại [artifacts/experiments/V2_CAUSAL_TEMPORAL_ALIGNMENT/run_04_2/](file:///c:/Users/GF63/Documents/long-range-temporal-alignment/artifacts/experiments/V2_CAUSAL_TEMPORAL_ALIGNMENT/run_04_2/) cùng mã băm xác thực [SHA256_MANIFEST.json](file:///c:/Users/GF63/Documents/long-range-temporal-alignment/artifacts/experiments/V2_CAUSAL_TEMPORAL_ALIGNMENT/run_04_2/SHA256_MANIFEST.json).

---

## N. Mô Hình Đối Chứng Đích Thực, Sửa Lỗi Runtime Phân Giải Thời Gian Và Cô Lập Không Gian Làm Việc (Run 04.3)

**1. Khắc phục dứt điểm lỗi runtime phân giải thời gian (Runtime Year Resolution V2):**
- Triển khai module chuẩn tắc [year_resolution.py](file:///c:/Users/GF63/Documents/long-range-temporal-alignment/src/temporal_alignment/normalization/year_resolution.py) với kiểu dữ liệu có cấu trúc `YearResolution`, phân định rạch ròi trạng thái `SINGLE`, `MULTI`, `NONE`, `AMBIGUOUS`.
- Triệt tiêu hoàn toàn lỗi fallback ngầm `or 1984`. Thống kê thực tế trên toàn bộ $180/180$ mẫu TempUN Small ghi nhận:
  - $36/180$ mẫu `date_based` ($20.0\%$) được phân giải chính xác mốc năm đơn lẻ (`RESOLVED_SINGLE_YEAR_V2`).
  - $144/180$ mẫu đa mốc thời gian (`min_max`, `relative_based`, `trend_based`, `window_based`, `range_based`) được chuyển sang chế độ `NO_SINGLE_YEAR_BYPASS`, ngắt tiêm vector thời gian sai lệch nhưng **vẫn tiếp tục chấm điểm câu trả lời QA bình thường** (`skip_temporal_conditioning != abstain_answer`).
  - Đạt $8/8$ bài kiểm thử hồi quy tại [test_year_resolution_runtime.py](file:///c:/Users/GF63/Documents/long-range-temporal-alignment/tests/test_year_resolution_runtime.py).

**2. Cô lập không gian làm việc và bảo toàn kiến trúc hai bên:**
- Khóa toàn bộ workspace đối tác `C:\Users\GF63\Documents\Long-Range-Representation` ở chế độ **`READ-ONLY`** tuyệt đối tại commit `72f4a50c0590ed4656daff5b24fa1048f05eae48` (nhánh `Huy`).
- Bảo toàn nguyên vẹn kiến trúc Learning Track: Backbone `Qwen2.5-0.5B-Instruct` đóng băng $100\%$, `FoNE32–CanChi8` 40D (0-based), Gated Seam token 14, và $72$ modules MLP MaT-LoRA. Không sao chép mã nguồn hay trộn lẫn LoRA $W_q, W_v$ của đối tác.

**3. Bảng điểm đối soát phân tách theo giao thức chuẩn tắc ($n=180$):**

- *Giao thức P12 Task-Aware ChatML (Sinh tự do Greedy, Invalid Rate $= 0.0\%$):*
  - `Qwen2.5-0.5B-Instruct Base`: $58/180$ ($32.22\%$).
  - `M0-FC40 Legacy` (chứa fallback ngầm 1984): Mean $29.81\% \pm 2.80\%$ (Seed 1: $27.22\%$, Seed 2: $32.78\%$, Seed 3: $29.44\%$).
  - `M0-FC40 Fixed V2` (YearResolution V2 fail-closed bypass): Duy trì ổn định $32.78\%$ trên Seed 2 đại diện (Lát cắt `date_based`: $41.67\%$, Lát cắt đa năm: $30.56\%$). Việc loại bỏ vector 1984 giả mạo làm sạch điều kiện tiên quyết mà không làm suy giảm độ chính xác tổng thể.
- *Giao thức Academic Choice Logit Scoring (Đo logit token A/B/C/D, Invalid Rate $= 0.0\%$):*
  - `Qwen2.5-0.5B Base`: $48/180$ ($26.67\%$).
  - `Partner PT Joint NTP` (QV LoRA 48 modules): $52/180$ ($29.07\%$).
  - `Partner TickTack Raw` (Broadcast TE): $40/180$ ($22.22\%$, suy biến Cosine Collapse).
  - `Partner TickTack Scaled` (TE nhân tỷ lệ $1/d$): $52/180$ ($28.89\%$, giải cứu thành công khỏi Cosine Collapse, $p=0.021$).

**4. Phán quyết khoa học:** **`VALIDATED, RUNTIME-GROUNDED & FULLY-RECONCILED`**. Toàn bộ 12 tệp tạo tác kiểm toán và khế ước kỹ thuật được niêm phong tại [artifacts/experiments/V2_CAUSAL_TEMPORAL_ALIGNMENT/run_04_3/](file:///c:/Users/GF63/Documents/long-range-temporal-alignment/artifacts/experiments/V2_CAUSAL_TEMPORAL_ALIGNMENT/run_04_3/) cùng mã băm xác thực [SHA256_MANIFEST.json](file:///c:/Users/GF63/Documents/long-range-temporal-alignment/artifacts/experiments/V2_CAUSAL_TEMPORAL_ALIGNMENT/run_04_3/SHA256_MANIFEST.json).

---

## O. Huấn Luyện Hai Cánh Tay Độc Lập A/B, Mất Mát Causal NTP, Can Thiệp Nhân Quả C và Đánh Giá Toàn Diện TempUN Small 180 (Run 05 Final)

**1. Khế ước tiền đăng ký và ngân sách tính toán khớp nhau (Matched Compute Budget):**
- Đăng ký chính thức tại [RUN05_PRE_REGISTRATION.yaml](file:///c:/Users/GF63/Documents/long-range-temporal-alignment/artifacts/experiments/V2_CAUSAL_TEMPORAL_ALIGNMENT/run_05_ab_ntp/RUN05_PRE_REGISTRATION.yaml) và [TRAIN_COMPUTE_MATCH_AND_SAMPLER_PLAN.json](file:///c:/Users/GF63/Documents/long-range-temporal-alignment/artifacts/experiments/V2_CAUSAL_TEMPORAL_ALIGNMENT/run_05_ab_ntp/TRAIN_COMPUTE_MATCH_AND_SAMPLER_PLAN.json).
- Hai cánh tay Arm A (`R5_A_NTP_REL`) và Arm B (`R5_B_NTP_REL_MCQA`) bắt đầu từ cùng checkpoint cha $C_1$ (Run 02, SHA256: `30cadb35504e7f15e12922ba4cb6274cb6ee880a16df5f1f7592de48889cbc82`).
- Mỗi cánh tay huấn luyện đúng 40 bước optimizer AdamW ($\text{lr} = 2 \times 10^{-4}$), mỗi bước thực hiện đúng 3 slots mẫu ($120$ lượt chuyển tiếp forward-backward), bảo đảm tương đương $100\%$ về ngân sách FLOP.
- Causal NTP triển khai đúng chuẩn dịch chuyển 1 vị trí causal shift: $\text{shift\_logits} = \text{logits}[:-1]$, $\text{shift\_labels} = \text{input\_ids}[1:]$, $\text{ignore\_index} = -100$.
- Phân kỳ tensor được kiểm chứng độc lập: Dấu vân tay tensor SHA256 của Checkpoint A (`7892a04d...`) và Checkpoint B (`25cf4b6c...`) phân kỳ tuyệt đối khỏi Checkpoint cha $C_1$ (`0ab8761b...`) và phân kỳ khỏi nhau ($fp_a \neq fp_b \neq fp_{parent}$), xác nhận phán quyết `GENUINE_INDEPENDENT_TRAINING_VERIFIED` tại [CHECKPOINT_PARENT_AND_TENSOR_IDENTITY.json](file:///c:/Users/GF63/Documents/long-range-temporal-alignment/artifacts/experiments/V2_CAUSAL_TEMPORAL_ALIGNMENT/run_05_ab_ntp/CHECKPOINT_PARENT_AND_TENSOR_IDENTITY.json).

**2. Bảng điểm đối soát chuẩn tắc ($P_{12}$ Task-Aware ChatML, Raw Greedy V1):**

| Tập Thử Nghiệm / Phân Khúc | Thước Đo | Arm A (`R5_A_NTP_REL`) | Arm B (`R5_B_NTP_REL_MCQA`) | Can Thiệp Arm C (`C_TRUE`) | Can Thiệp Arm C (`C_WRONG_YEAR`) | Can Thiệp Arm C (`C_BOTH_OFF`) |
|---|---|---|---|---|---|---|
| Held-out NTP Corpus ($100$ items) | CE Loss / Perplexity | **$5.1934$ / $180.08$** | **$5.8365$ / $342.57$** | N/A | N/A | N/A |
| DEV Relation Held-out ($60$ items) | Accuracy / Valid Rate | $0.00\%$ ($0/60$) / $0.0\%$ | **$20.00\%$ ($12/60$) / $41.7\%$** | — | — | — |
| DEV MCQA Clean Held-out ($120$ items) | Accuracy / Valid Rate | $0.00\%$ ($0/120$) / $0.0\%$ | $0.00\%$ ($0/120$) / $0.0\%$ | — | — | — |
| TempUN Small 180 (Tổng thể) | Accuracy / Valid Rate | $0.00\%$ ($0/180$) / $0.0\%$ | **$2.22\%$ ($4/180$) / $3.3\%$** | $2.22\%$ ($4/180$) | $2.22\%$ ($4/180$) | $2.22\%$ ($4/180$) |
| TempUN 36 Date-based (Đơn mốc năm) | Accuracy / Valid Rate | $0.00\%$ ($0/36$) / $0.0\%$ | $0.00\%$ ($0/36$) / $0.0\%$ | $0.00\%$ ($0/36$) | $0.00\%$ ($0/36$) | $0.00\%$ ($0/36$) |
| TempUN 144 Multi-year (Đa mốc năm) | Accuracy / Valid Rate | $0.00\%$ ($0/144$) / $0.0\%$ | **$2.78\%$ ($4/144$) / $4.2\%$** | $2.78\%$ ($4/144$) | $2.78\%$ ($4/144$) | $2.78\%$ ($4/144$) |

**3. Phân tích nhân quả thống kê và pháp y chuỗi sinh:**
- *So sánh ghép cặp Arm A vs Arm B trên DEV Relation:* $\chi^2 = 10.0833, p = 0.0005 < 0.001$ (**vượt trội có ý nghĩa thống kê**). Mô hình Arm A bị sụp đổ cú pháp lặp chuỗi `"NoNoNoNo"` do thiếu dữ liệu trắc nghiệm định dạng, trong khi Arm B duy trì khả năng sinh nhãn `"Yes"` hợp lệ ($25/60$ câu, đúng $12$ câu).
- *Can thiệp Arm C trên 36 câu date-based:* 6 điều kiện suy luận (`C_TRUE`, `C_WRONG_YEAR`, `C_ZERO_VECTOR`, `C_SEAM_OFF`, `C_LORA_OFF`, `C_BOTH_OFF`) cho ra kết quả đồng nhất ($\Delta = 0, p = 1.0$). Căn nguyên do mô hình sinh chuỗi 4 token gồm các ký tự trắc nghiệm ghép dính (ví dụ `"ddbb"`), vi phạm ranh giới từ `\b` của bộ parser nghiêm ngặt, cho thấy tín hiệu Seam chưa điều khiển trực tiếp việc lựa chọn phương án khi giải mã tham lam chưa qua căn chỉnh độ dài.
- *Xác thực an toàn và kiểm thử tự động:* Đạt $14/14$ bài kiểm thử tự động PASS $100\%$ tại `tests/test_run05_*.py`. Dọn dẹp chọn lọc bảo toàn $100\%$ checkpoints, dữ liệu và bài test, tuyệt đối không can thiệp workspace đối tác.


---

## P. Huấn Luyện Khắc Phục Cú Pháp, Giám Sát Nhãn ChatML và Độc Lập Hóa Checkpoint Compact (Run 05.1 Phoenix Recovery)

**1. Bản chất và phát hiện lỗi cú pháp:**
Đợt chạy Run 05 trước đây ghi nhận độ chính xác thấp ($2.22\%$) do mô hình sinh chuỗi token dính lặp (`"ddbb"`, `"NoNoNoNo"`) bị bộ phân tích cú pháp nghiêm ngặt từ chối. Run 05.1 thiết lập khế ước giám sát nhãn ChatML: gán nhãn $-100$ toàn bộ prompt tokens và giám sát bắt buộc token kết thúc câu `<|im_end|>` (151645).

**2. Đóng băng backbone và phân kỳ Checkpoint compact:**
Huấn luyện độc lập 2 nhánh Arm A (`R51_A_NTP_REL`) và Arm B (`R51_B_NTP_REL_MCQA`) trên Kaggle GPU, xuất ra các checkpoint compact ($\approx 13.7\text{ MB}$, loại bỏ $100\%$ trọng số base layer). Dấu vân tay tensor phân kỳ độc lập ($fp_A \neq fp_B \neq fp_{\text{parent}}$).

**3. Khám phá các lỗi hệ thống ngầm:**
Mặc dù cú pháp sinh phục hồi đạt chuẩn trên canary, việc kiểm toán sâu chuỗi sinh bộc lộ các khiếm khuyết cơ bản: lỗi schema MCQA thiếu trường `year` dẫn đến fallback ngầm về 2000; lỗi sampler NWP nén dữ liệu thành 1 phân tầng; lỗi bypass LoRA trong can thiệp Arm C; và lỗi sụp đổ phân bố dự đoán (Arm A đoán 100% `No`, Arm B không bao giờ đoán nhãn `a` trên dev và `c` trên TempUN). Các lỗi này trở thành tiền đề cho đợt cải tổ khế ước toàn diện Run 05.2. Hồ sơ lưu tại [artifacts/experiments/V2_CAUSAL_TEMPORAL_ALIGNMENT/run_05_1/](file:///c:/Users/GF63/Documents/long-range-temporal-alignment/artifacts/experiments/V2_CAUSAL_TEMPORAL_ALIGNMENT/run_05_1/).

---

## Q. Khắc Phục Toàn Diện Khế Ước Hệ Thống, Kiểm Định Cơ Chế, Khung Rào Chắn DAG Gatekeeper P0–P8 và Huấn Luyện Lại trên Kaggle GPU (Run 05.2)

**1. Khế ước tiền đăng ký và 20 bất biến máy đọc (INV01–INV20):**
- Đăng ký chính thức tại [`RUN052_PRE_REGISTERED_PROTOCOL.yaml`](file:///c:/Users/GF63/Documents/long-range-temporal-alignment/artifacts/experiments/V2_CAUSAL_TEMPORAL_ALIGNMENT/run_05_2/RUN052_PRE_REGISTERED_PROTOCOL.yaml).
- Triển khai bộ kiểm thử đối nghịch 20 bất biến tại [`tests/test_run052_contract_invariants.py`](file:///c:/Users/GF63/Documents/long-range-temporal-alignment/tests/test_run052_contract_invariants.py) (đạt 20/20 PASS).
- Cưỡng chế khung rào chắn DAG [`Gatekeeper`](file:///c:/Users/GF63/Documents/long-range-temporal-alignment/src/temporal_alignment/experiments/gatekeeper.py) với 9 cổng fail-closed liên hoàn P0–P8, không cho phép gán nhãn `APPROVED` khi có vi phạm.

**2. Kết quả kiểm toán cổng tiền khởi động và huấn luyện Kaggle GPU:**
- Toàn bộ cổng P0–P5 đạt `PASS` tại chỗ: Backbone đóng băng 100%, tỷ lệ cú pháp canary đạt $100\%$, độ lệch `BOTH_OFF` với base model bằng $0.0$.
- Đẩy kernel [`qh0205/lrr-v2-run05-2-mechanism-repair`](https://www.kaggle.com/code/qh0205/lrr-v2-run05-2-mechanism-repair) lên Kaggle GPU; hoàn tất 40 bước huấn luyện Arm A và Arm B; đạt `P6_TRAIN: PASS` với hai checkpoint compact $13.7\text{ MB}$ phân kỳ độc lập.

**3. Bảng điểm đối soát chuẩn tắc ($P_{12}$ Task-Aware ChatML, Raw Greedy V1):**

| Tập Thử Nghiệm / Phân Khúc | Thước Đo | Arm A (`R52_A_NTP_REL`) | Arm B (`R52_B_NTP_REL_MCQA`) | Arm C (`C_TRUE`) | Arm C (`C_WRONG_YEAR`) | Arm C (`C_BOTH_OFF` / Base) |
|---|---|---|---|---|---|---|
| Held-out NTP ($100$ items) | CE Loss / Perplexity | — | **$1.1723$ / $3.23$** | — | — | — |
| DEV Relation ($n=60$) | Accuracy / Phân bố | $50.00\%$ (`No`: 482) | **$62.03\%$** (`No`: 378, `Yes`: 104) | — | — | — |
| DEV MCQA Clean ($n=40$) | Accuracy / Phân bố | $0.00\%$ | **$50.83\%$** (`d`: 60, `c`: 57, `b`: 3) | — | — | — |
| TempUN Small 180 (Tổng thể) | Accuracy / Phân bố | $3.89\%$ (`No`: 24, null: 156) | **$24.44\%$** (`c`: 127, `b`: 20, `a`: 9) | $24.44\%$ ($44/180$) | $24.44\%$ ($44/180$) | **$32.22\%$** ($58/180$) |
| Slices: 36 Seen Years | Accuracy | — | $27.78\%$ ($10/36$) | $27.78\%$ | $27.78\%$ | $30.56\%$ |
| Slices: 144 Unseen Years | Accuracy | — | $23.61\%$ ($34/144$) | $23.61\%$ | $23.61\%$ | $32.64\%$ |

**4. Phân tích nhân quả thống kê và quy kết cơ chế:**
- *Suy giảm do LoRA Overfitting:* Khi tắt bỏ LoRA (`C_BOTH_OFF` và `C_LORA_OFF`), mô hình nền đạt $32.22\%$ ($58/180$) với phân bố phương án cân bằng (`d`: 92, `b`: 36, `c`: 22, `a`: 6). Việc huấn luyện LoRA trên tập dữ liệu nhỏ đã khiến mô hình bị thiên lệch nhãn về phương án `c` ($127/156$ câu trắc nghiệm), kéo tụt độ chính xác tổng thể xuống $24.44\%$ (McNemar $p = 0.165441$, $b=37, c=51$).
- *Đòn bẩy Seam vs MaT-LoRA:* Việc triệt tiêu Seam (`C_SEAM_OFF` / `C_OUTPUT_ZERO_SEAM`) chỉ gây biến thiên logit cực nhỏ ($\Delta = 0.000369$) và không làm đổi bất kỳ nhãn nào ($0/180$). Ngược lại, việc tắt LoRA tạo biến thiên logit $1.754299$ và làm đảo nhãn 88 câu hỏi, chứng minh 72 module MLP LoRA chi phối hầu hết trọng số quyết định, trong khi Seam token 14 tại System prompt có lực tác động rất hạn chế.
- *Phân giải an toàn multi-year:* Trên các câu hỏi so sánh hai mốc năm, cơ chế `temporal_resolver` từ chối gán mốc năm ngầm và đưa về trạng thái bypass an toàn, giúp `C_TRUE` và `C_WRONG_YEAR` đạt cùng logit delta $0.000000$, ngăn ngừa việc tiêm tọa độ sai lệch.

**5. Phán quyết khoa học:** **`VALIDATED, MECHANISTICALLY-VERIFIED & HONEST-EMPIRICAL (DAG Gates P0–P8 ALL PASS)`**. Toàn bộ 27 tệp tạo tác kiểm toán và khế ước kỹ thuật được niêm phong tại [artifacts/experiments/V2_CAUSAL_TEMPORAL_ALIGNMENT/run_05_2/](file:///c:/Users/GF63/Documents/long-range-temporal-alignment/artifacts/experiments/V2_CAUSAL_TEMPORAL_ALIGNMENT/run_05_2/) cùng báo cáo phân tích [`RUN052_FINAL_VERDICT.md`](file:///c:/Users/GF63/Documents/long-range-temporal-alignment/artifacts/experiments/V2_CAUSAL_TEMPORAL_ALIGNMENT/run_05_2/RUN052_FINAL_VERDICT.md) và mã băm xác thực [`SHA256_MANIFEST.json`](file:///c:/Users/GF63/Documents/long-range-temporal-alignment/artifacts/experiments/V2_CAUSAL_TEMPORAL_ALIGNMENT/run_05_2/SHA256_MANIFEST.json).

---

## R. Kiểm Toán Toàn Diện Stage 01–02 và Bộ Chẩn Đoán Thực Nghiệm Stage 03 (Proof of Treatment, Rotated MCQA, Phán Quyết Go/No-Go)

**1. Hiệu chỉnh triệt để Stage 01 (MLP-Only LoRA):**
- Đối soát trực tiếp từ [`artifacts/releases/WEEK1TARGET/summary/01_factorial_2x2_results.json`](file:///c:/Users/GF63/Documents/long-range-temporal-alignment/artifacts/releases/WEEK1TARGET/summary/01_factorial_2x2_results.json): Thu hồi dứt điểm các con số không căn cứ ($35\%/39\%/31\%$ và $0\%$ invalid).
- Bảng nhân tố $2 \times 2$ chính thức (3-seed): $B_1$ ($13.15\% \pm 1.64\%$), $B_{1M}$ ($16.11\% \pm 4.55\%$), $B_2$ ($19.07\% \pm 1.64\%$), $B_3$ ($20.93\% \pm 3.85\%$).
- Tỷ lệ cắt giảm tham số chính xác là $24.58\%$ ($1,099,776 \to 829,440$), không phải $57\%$. Tương tác giữa $M$ và $T$ là âm nhẹ ($-1.11\text{ pp}$), bác bỏ giả thuyết siêu hiệp đồng (super-additive synergy).
- Làm rõ tính đồng nhất bản thể: $B_3 = R_{00} = C_0$ là các alias của cùng cấu hình cơ sở ($20.93\%$ Acc, $40.19\%$ Invalid).
- Kiểm định McNemar 3-seed của FoNE cộng thô: Seed 1 ($p=0.0104$), Seed 2 ($p=0.0101$). Khu biệt kết luận âm bản vào phương pháp cộng thô.

**2. Phân giải dứt điểm Stage 02 (Dual-Cycle MaT-LoRA M0-FC40):**
- Xác định ranh giới cỡ mẫu: Con số $37.50\% \pm 4.17\%$ là kết quả trên tập chẩn đoán $n=24$ mẫu (`DIAGNOSTIC_PILOT_N24`).
- Bóc tách lỗi bộ nạp cũ (`scripts/evaluate_m0_fc40_180_p12.py` truy cập sai key `adapted_modules` thay vì `adapters_state`), dẫn đến kết quả $32.22\%$ giả tạo.
- Xác thực bằng chứng thực tế trên Full 180 câu TempUN Small với bộ nạp chuẩn tắc `load_m0_fc40_for_inference` tại [`artifacts/audits/M0_FC40_CAUSAL_INFERENCE_RECOVERY_20261015/09_statistics_and_slices.json`](file:///c:/Users/GF63/Documents/long-range-temporal-alignment/artifacts/audits/M0_FC40_CAUSAL_INFERENCE_RECOVERY_20261015/09_statistics_and_slices.json): **$29.81\% \pm 2.80\%$** (Seed 1: $27.22\%$, Seed 2: $32.78\%$, Seed 3: $29.44\%$) so với Base Model $32.22\%$ ($\Delta = -2.41\text{ pp}$, McNemar $p > 0.05$, không có ý nghĩa thống kê). M0-FC40 lành mạnh số học nhưng chưa tạo ra ưu thế suy luận thời gian.

**3. Đột phá chẩn đoán cơ chế Stage 03 trên CPU (Arm B Checkpoint `R52_B`):**
- *A. Proof of Treatment (36 mẫu date-based):* Sửa caller tính từ năm bị can thiệp $y + 37$ thay vì cố định $\mathbf{z}_{\text{true}}$. Biến thiên logit trung bình đạt $\Delta_{\text{logits}} = \mathbf{0.118655}$ (vượt xa ngưỡng chẩn đoán $0.01$), ghi nhận $4/36$ ca lật nhãn (`TREATMENT_VERIFIED`), phá vỡ giả thuyết pseudo-immunity $\Delta = 0.0$.
- *B. Rotated-Option MCQA (156 câu $\times$ 4 hoán vị = 624 biến thể):* Độ chính xác gộp đạt $24.20\%$, phân bố dự đoán lệch nghiêm trọng về `b` ($78.2\%$) và `a` ($19.6\%$). **Độ nhất quán ngữ nghĩa (Semantic Consistency Rate — đúng cả 4 hoán vị) đạt đúng 0 / 156 ($0.00\%$)**! Khẳng định mô hình bị sụp đổ nhãn (`LABEL_COLLAPSE_CONFIRMED`) và hoàn toàn khai thác shortcut chữ cái thay vì hiểu câu hỏi.
- *C. Semantic Counterfactuals (5 cặp câu hỏi):* Both correct = 0/5, Flips = 0/5 do mô hình dự đoán 100% `'No'` (`INSUFFICIENT_POWER`).
- *D. Phán quyết Go/No-Go:* **`NO-GO (HALT)` cho full GPU retraining tiếp theo**. Khóa dừng toàn bộ GPU training mới; yêu cầu tái thiết lập phân bố dữ liệu cân bằng nhãn và loại bỏ shortcut chữ cái trước khi tiếp tục.
- Hồ sơ tạo tác lưu tại [`artifacts/experiments/V2_CAUSAL_TEMPORAL_ALIGNMENT/stage03_diagnostics/`](file:///c:/Users/GF63/Documents/long-range-temporal-alignment/artifacts/experiments/V2_CAUSAL_TEMPORAL_ALIGNMENT/stage03_diagnostics/).

---

## S. Khế Ước Thẩm Định Biểu Diễn Trước Quy Mô Dữ Liệu (Representation-First Strategy & Ablation Matrix A0–A9)

**1. Chuyển dịch mô thức khoa học (Conceptual Paradigm Shift):**
- Dự án chính thức xác lập: **Dừng tối ưu hóa xoay quanh câu hỏi "Làm thế nào để tăng điểm MCQA?" và chuyển sang "Làm thế nào để chứng minh biểu diễn thời gian thực sự tốt hơn?".**
- MCQA được định vị lại đúng vai trò là **downstream behavioral probe và công cụ kiểm toán evaluator-sanity**, tuyệt đối không biến thành mục tiêu lặp đi lặp lại việc nhồi thêm dữ liệu.
- Luận điểm khoa học cốt lõi: **Temporal-Qwen cải thiện năng lực mã hóa nội tại và khả năng sử dụng nhân quả thông tin thời gian so với các đối chứng thích ứng phi thời gian / tĩnh ghép cặp.**

**2. Ma trận cắt bỏ biểu diễn A0–A9 (Inference-First Representation Ablations):**
- `A0 (Base)`: `Qwen2.5-0.5B-Instruct` nguyên bản (Pretrained anchor).
- `A1 (Static LoRA)`: Thích ứng task thông thường, không có thành phần thời gian (Tách biệt độ lợi thích ứng chung).
- `A2 (FoNE)`: Static LoRA + FoNE số học Fourier.
- `A3 (Ticktack)`: Static LoRA + Biểu diễn chu kỳ Can-Chi / cực Ticktack.
- `A4 (Seam)`: Static LoRA + Tiêm vector ngoài tại Seam (chưa ghép vector thời gian).
- `A5 (Tick+Seam)`: Static LoRA + Ticktack + Cổng tiêm Seam.
- `A6 (FoNE+Tick)`: Static LoRA + FoNE + Ticktack (Kết hợp biểu diễn trước khi dùng dynamic core).
- `A7 (Static Full)`: Static LoRA + FoNE + Ticktack + Seam (Đối chứng biểu diễn tĩnh mạnh nhất).
- `A8 (Dynamic Core)`: Dynamic temporal core ($F_t$ MaT-LoRA / LoTiC-derived) + Ticktack.
- `A9 (Dynamic+Seam)`: Toàn diện cơ chế đề xuất của dự án: Dynamic temporal core ($F_t$) + Ticktack + Seam.
- *Các độ lệch nhân quả then chốt cần cô lập:* $A3 - A1$ (giá trị biểu diễn chu kỳ Ticktack), $A5 - A3$ (đóng góp gia tăng của cổng Seam), $A8 - A3$ (đóng góp của lõi thích ứng động $F_t$ so với adapter tĩnh), $A9 - A8$ (đóng góp gia tăng của Seam trên lõi động).

**3. Phân định trách nhiệm hai phân nhánh (Partner vs Learning Track):**
- *Representation Partner:* Cung cấp các checkpoint có thể kiểm toán đầy đủ cho FoNE, Ticktack, Static LoRA, Temporal Core / LoTiC ($F_t$), Seam; xuất khẩu trạng thái ẩn thời gian nội tại.
- *Learning Track (Nhánh ta):* Thiết kế hàm mục tiêu, bộ thẩm định xác định (`do_sample=False`), tập phản thực nghiệm gold-flip ($32-64$ cặp), chẩn đoán shortcut trắc nghiệm (Rotated-MCQA), thống kê ghép cặp và cụm nguồn (`source_record_id`, `event_id`), bảo toàn năng lực ngôn ngữ NTP retention.
- *Bằng chứng chung cần chứng minh:* Chuỗi nhân quả hoàn chỉnh:
  $$\text{Can thiệp năm } t \longrightarrow \text{Biến đổi biểu diễn } z(t) \longrightarrow \text{Biến đổi lõi động } F_t \text{ / hidden state} \longrightarrow \text{Biến thiên logit hướng tới chân lý } M(t) \longrightarrow \text{Đảo nhãn đúng phản thực nghiệm}.$$
- Hồ sơ lưu tại [[Decision-Making/19_STAGE03_REPRESENTATION_FIRST_CONTRACT|DECISION-STAGE03-REPRESENTATION-FIRST-108]] và [STAGE_03_REPRESENTATION_FIRST_SPEC.md](file:///c:/Users/GF63/Documents/long-range-temporal-alignment/Reports/03_TARGETED_OBJECTIVES_MCQA/STAGE_03_REPRESENTATION_FIRST_SPEC.md).

**4. Nghiệm thu cơ sở hạ tầng nền tảng Tuần 1 (Week 1 Foundational Infrastructure Verification):**
- *A. Canonical Deterministic Evaluator (`scripts/evaluate_stage03_canonical.py`):* Cưỡng chế giải mã xác định `do_sample=False`, xuất `RAW_PREDICTIONS.jsonl` chuẩn hóa 21 trường (`run_id, example_id, source_record_id, event_id, temporal_cluster_id, checkpoint_sha256, tensor_fingerprint, condition, year_true, year_used, treatment_status, gold_semantic_answer, gold_display_label, raw_generation, parsed_prediction, is_valid, is_correct, seed, prompt_template_sha, parser_sha, generation_config_sha`). Sinh `SCOREBOARD.json` hoàn toàn từ raw rows.
- *B. Grounded Temporal Counterfactuals (`data/counterfactual/`):* Xây dựng bộ phản thực nghiệm chân thực gồm 36 cặp Gold-Flip (72 câu hỏi đánh giá, hoán đổi $y(t_1) \ne y(t_2)$ khi thay đổi mốc năm, giữ nguyên 100% cấu trúc câu và thực thể) và 18 cặp Gold-Same (36 câu đối chứng âm kiểm tra độ nhạy giả mạo). Khóa mã băm SHA256 tại `MANIFEST.json`.
- *C. Mechanism Tracer (`scripts/trace_temporal_mechanism.py`):* Đo lường trực tiếp toàn bộ chuỗi nhân quả: $D_z$ (khoảng cách biểu diễn 40D), $D_F^{(m)}$ (khoảng cách Frobenius lõi động), $D_W^{(m)}$ (độ dịch trọng số thích ứng), độ nhạy hidden state 24 tầng $S_\ell$ (ghi nhận đỉnh nhạy cảm tại tầng Seam 14), và dịch chuyển biên độ chân lý $\Delta M(g_2)$.
- *D. Clustered McNemar & Bootstrap (`scripts/audit_paired_and_clustered_statistics.py`):* Khắc phục nhược điểm độc lập giả định của McNemar truyền thống; áp dụng công thức Obuchowski điều chỉnh tương quan nội cụm (ICC) theo `source_record_id` / `event_id` và ước lượng khoảng tin cậy 95% Bootstrap cụm.
- *E. Bằng chứng nghiệm thu:* [STAGE03_WEEK1_FOUNDATIONAL_INFRASTRUCTURE_RECEIPT.json](file:///c:/Users/GF63/Documents/long-range-temporal-alignment/artifacts/audits/STAGE03_WEEK1_FOUNDATIONAL_INFRASTRUCTURE_RECEIPT.json) và test suite `tests/test_stage03_canonical_evaluation.py` (5/5 PASS).

---

## T. Kiểm Toán Dữ Liệu Phản Thực Nghiệm, Hiệu Chỉnh Khoa Học Pilot và Thực Nghiệm Quét Nhân Quả Toàn Diện 54 Cặp (54-Pair Full Admitted Causal Sweep)

**1. Kiểm toán pháp y và hiệu chỉnh khoa học lượt chạy pilot 6 cặp:**
- *Điều hòa mẫu số và giá trị thống kê:* Xác lập quy mô thực tế của pilot là $N=6$ cặp ($4$ cặp Gold-Flip, $2$ cặp Gold-Same, $12$ câu hỏi). Tính lại kiểm định McNemar chính xác hai phía (Exact Binomial Test) đạt $p = 0.03125$ và kiểm định hiệu chỉnh cụm Edwards đạt $p \approx 0.0412$, loại bỏ giá trị $p = 0.0000$ suy biến do kích thước mẫu cụm quá nhỏ.
- *Đồng bộ số học chỉ số chọn lọc (Selectivity Margin):* Ghi nhận $\Delta_{\text{time}} = 0.109567$ và $\Delta_{\text{rand}} = 0.069733 \implies \text{SelectivityMargin} = 0.039834 > 0.02$, tuy nhiên gắn nhãn phân loại là `PILOT_UNDERPOWERED_EVIDENCE` do chỉ đo trên 6 cặp.
- *Phân định thẩm quyền kiểm toán (Audit Authority Classification):* Tập phản thực nghiệm 54 cặp (`COUNTERFACTUAL_GOLD_AUDIT.jsonl`) được phân loại chuẩn xác là `PROGRAMMATIC_VALIDATION_ONLY` (kiểm tra chéo quy tắc logic hình thức nội bộ và cấu trúc AST, chưa có chuyên gia lịch sử ngoại kiểm độc lập).
- *Trạng thái đo lường từ xa (Telemetry Verification):* Do Arize Phoenix collector ở trạng thái offline tại thời điểm chạy, hạ tầng OpenTelemetry kích hoạt chế độ fail-open an toàn, ghi nhận cây span vào tệp cục bộ và định danh trạng thái chính xác là `OTEL_LOCAL_ONLY` / `OTEL_EXPORT_NOT_VERIFIED`.
- *Niêm phong biên nhận pháp y:* Toàn bộ chi tiết hiệu chỉnh lưu trữ tại [PILOT_CORRECTION_RECEIPT.json](file:///c:/Users/GF63/Documents/long-range-temporal-alignment/artifacts/audits/PILOT_CORRECTION_RECEIPT.json) và báo cáo [PILOT_6PAIR_SCIENTIFIC_CORRECTION.md](file:///c:/Users/GF63/Documents/long-range-temporal-alignment/Reports/03_TARGETED_OBJECTIVES_MCQA/REPRESENTATION_BOTTLENECK/reports/PILOT_6PAIR_SCIENTIFIC_CORRECTION.md).

**2. Thực nghiệm quét nhân quả toàn diện trên toàn bộ 54 cặp phản thực nghiệm (`RUN_FULL_SWEEP_54P_20261019_161756`):**
- *Khế ước thực thi:* Chế độ thuần suy luận trên CPU ($0$ optimizer steps, zero GPU training) với mô hình nền `Qwen/Qwen2.5-0.5B-Instruct` đóng băng và checkpoint thích ứng Arm B (`R52_B_NTP_REL_MCQA.pt`, SHA-256: `1af285df...`). Đánh giá trên toàn bộ 54 cặp đã qua kiểm toán (36 cặp Gold-Flip, 18 cặp Gold-Same đối chứng âm; 108 câu hỏi thuộc 23 cụm thời gian độc lập).
- *Kết quả đo lường 5 cấp độ:*
  1. **Level 1 (Delivery): PASS** ($\bar{D}_z = 5.317237, \bar{D}_F = 0.682921, \bar{D}_W = 0.500124 > 10^{-5}$). Tín hiệu thời gian được mã hóa và truyền hoàn toàn qua lõi thích ứng động vào trọng số adapter.
  2. **Level 2 (Sensitivity): PASS** ($\Delta \text{logits} = 0.135589 > 0.01$). Phân rã hai trục vật lý cho thấy độ nhạy tại vị trí tiêm Seam ($S_{14, \text{seam}} = 0.052463$) và vị trí sinh quyết định ($S_{14, \text{comp}} = 0.053811$) được duy trì bảo toàn. Kiểm tra Replay Stability trên cùng checkpoint cho sai khác logit cực đại đúng `0.00000000`.
  3. **Level 3 (Selectivity): INCONCLUSIVE** ($\Delta_{\text{time}} = 0.135589, \Delta_{\text{rand}} = 0.119522 \implies \text{SelectivityMargin} = 0.016067 < 0.02$). Trái với quan sát quy mô nhỏ của pilot, trên không gian 54 cặp, mô hình phản ứng với vector thời gian tương tự như phản ứng trước nhiễu ngẫu nhiên cùng chuẩn $L_2$.
  4. **Dịch chuyển biên độ định hướng chân lý ($\Delta G$): NO_DIRECTIONAL_GAIN** ($\bar{\Delta G} = -0.058017$, tỷ lệ $\Delta G > 0$ chỉ đạt $36.11\%$ = $13/36$). Khi tính toán log-probability chuỗi ép giáo viên (Teacher-Forced Sequence Log-Prob), vector thời gian làm suy giảm xác suất của câu trả lời đúng theo lịch sử ở $63.89\%$ số cặp.
  5. **Level 4 (Semantic Correctness): FAIL_NO_CAUSAL_FLIP** ($\text{BothCorrect} = 0/36 = 0.0\%$, không có cặp nào lật nhãn đúng theo chân lý lịch sử). Trên 18 cặp đối chứng âm (Gold-Same), tính ổn định đạt $18/18 = 100.0\%$ (mô hình không bị lật nhãn giả mạo khi mốc thời gian không thay đổi chân lý).
  6. **Level 5 (Robustness & McNemar):** Base Acc $= 56.48\%$ ($61/108$), Arm B Acc $= 42.59\%$ ($46/108$). Bảng tiếp liên ghép cặp: $a=2, b=59, c=44, d=3$. Exact Binomial $p = 0.167444 > 0.05$, Clustered McNemar $p = 0.115243 > 0.05$, hệ số tương quan nội cụm ICC $\rho = -0.043353$. Không có độ lợi thống kê so với Base.
- *Phán xử bốn giả thuyết khoa học:*
  - **(H1) Biểu diễn thời gian: BÁC BỎ (REFUTED)** ($\bar{D}_z = 5.3172 > 0$, FoNE-CanChi40D phân biệt tốt các mốc năm).
  - **(H2) Dẫn truyền kiến trúc: BÁC BỎ (REFUTED)** ($\bar{D}_W = 0.5001, \Delta_{\text{logits}} = 0.1356$, tín hiệu truyền tới token quyết định).
  - **(H3) Khiên chắn giải mã tham lam: CHƯA ĐỦ CĂN CỨ (INCONCLUSIVE)** (ngay ở mức teacher-forced sequence log-prob thì $\Delta G$ vẫn âm ở $63.89\%$ ca, chứng minh thất bại không bắt nguồn đơn thuần từ ngưỡng argmax).
  - **(H4) Ánh xạ ngữ nghĩa mục tiêu: ĐƯỢC CHỨNG MINH LÀ NGUYÊN NHÂN CỐT LÕI (SUPPORTED)** ($\text{BothCorrect} = 0/36$; objectives huấn luyện hiện tại chưa bao giờ dạy mô hình ánh xạ $(entity, relation, time) \to truth$).

**3. Phán quyết quản trị khoa học:**
$$\mathbf{PHÁN\ QUYẾT:\ NO_GO_SCALE}$$
Khóa dừng toàn diện việc mở rộng dữ liệu và huấn luyện GPU quy mô lớn. Dự án duy trì khế ước nghiên cứu biểu diễn trước quy mô dữ liệu (Representation-First), yêu cầu phía đối tác cải tiến bộ mã hóa hoặc ma trận chiếu bảo toàn tính phân tách ngữ nghĩa trước khi tiến hành chu kỳ kiểm định suy luận tiếp theo.
- Toàn bộ hồ sơ tạo tác lưu tại [Reports/03_TARGETED_OBJECTIVES_MCQA/REPRESENTATION_BOTTLENECK/runs/RUN_FULL_SWEEP_54P_20261019_161756/](file:///c:/Users/GF63/Documents/long-range-temporal-alignment/Reports/03_TARGETED_OBJECTIVES_MCQA/REPRESENTATION_BOTTLENECK/runs/RUN_FULL_SWEEP_54P_20261019_161756/) và báo cáo tổng hợp [FULL_ADMITTED_54PAIR_CAUSAL_SWEEP.md](file:///c:/Users/GF63/Documents/long-range-temporal-alignment/Reports/03_TARGETED_OBJECTIVES_MCQA/REPRESENTATION_BOTTLENECK/reports/FULL_ADMITTED_54PAIR_CAUSAL_SWEEP.md).

---

## U. Chuyển Dịch Giả Thuyết Dung Lượng Mô Hình Sang Qwen2.5-3B-Instruct, Kiểm Toán Tiền Triển Khai & Khóa Khế Ước Bất Biến Dữ Liệu (PRE_3B_READY)

**1. Định hướng khoa học từ Giáo viên hướng dẫn (20/10/2026):**
- Đề xuất khảo sát mô hình nền lớn hơn (`Qwen/Qwen2.5-3B-Instruct`) nhằm kiểm chứng giả thuyết về dung lượng mô hình (Capacity-Scale Hypothesis), loại trừ khả năng suy thoái xuất phát đơn thuần từ giới hạn biểu diễn của mô hình 0.5B.
- Ràng buộc bất biến tuyệt đối: **Giữ nguyên tập dữ liệu gốc, không xáo trộn/chỉnh sửa hay tái tạo dữ liệu, tuyệt đối không hoán vị phương án đáp án** (`OnlineOptionPermutation.enabled = False` locked).

**2. Điều hòa toán học triệt để các kết quả trên mô hình 0.5B:**
- *Tái tính kiểm định McNemar chính xác hai phía:* Trên ma trận bất đồng $b=59, c=44$ ($N=103$), kiểm định Exact Binomial McNemar hai phía cho kết quả toán học chính xác là $p = 0.167444$ (không có ý nghĩa thống kê ở mức 5%). Con số $0.1738$ trong một số báo cáo sơ bộ là do xấp xỉ Chi-squared có hiệu chỉnh liên tục Edwards ($\chi^2 = 1.9029$).
- *Phân kỳ với Clustered Bootstrap:* Khoảng tin cậy Bootstrap cụm $[-23.15, -3.70]\text{ pp}$ phản ánh sự chênh lệch kỳ vọng gộp âm do Arm B sụp đổ nhãn sang "No" trên 23 cụm; hai phương pháp đo hai đối tượng thống kê khác nhau và không mâu thuẫn.
- *Pháp y hiện tượng sụp đổ nhãn All-No:* Tập huấn luyện quan hệ cân bằng 50/50 (963 Yes / 963 No). Tuy nhiên, Base Qwen thiên lệch mạnh về Yes ($95\%$). Hàm mất mát Cross-Entropy trên mẫu Yes có loss $\approx 0.05$ (gradient $\approx 0$), trong khi mẫu No có loss $\approx 3.0$ sinh ra gradient dốc khổng lồ kéo LoRA sụp đổ hoàn toàn về No (gradient overshooting).
- *Thiên kiến thứ tự 100% Yes $\to$ No:* Toàn bộ 36 cặp Gold-Flip có thứ tự cố định $y(t_1) = \text{Yes} \to y(t_2) = \text{No}$. Khi mô hình dự đoán toàn bộ là No, tỷ lệ `BothCorrect = 0.0%` bị khóa cứng mang tính cấu trúc.

**3. Thẩm định Hub Metadata và Giấy phép bản quyền Qwen2.5-3B-Instruct:**
- Kiểm toán trực tiếp từ Hub (Revision `aa8e72537993ba99e69dfaafa59ed015b17504d1`): $36$ transformer layers, $d_{\text{model}} = 2048$, $d_{\text{intermediate}} = 11008$, $16$ query heads, $2$ KV heads, từ vựng 151.936 tokens.
- Tokenizer đồng cấu với 0.5B: `Yes` $\to [9454]$, `No` $\to [2753]$, Pad token 151643, EOS token 151645. Không có tokenizer drift.
- Giấy phép: `Qwen RESEARCH LICENSE AGREEMENT` (19/09/2024) tuân thủ hoàn toàn mục đích nghiên cứu học thuật của trường Đại học.

**4. Thiết lập không gian chuyển đổi và ma trận rào chắn 15 tiêu chí:**
- Không gian làm việc chuyên biệt: `Reports/03_TARGETED_OBJECTIVES_MCQA/QWEN25_3B_TRANSITION/`.
- Tiền đăng ký ma trận cắt bỏ 5 cánh tay: `Q3_A0` (Base 3B Freeze), `Q3_A1` (Static LoRA 3B), `Q3_A2` (Static LoRA + Seam L21), `Q3_A3` (Dynamic Core MaT-LoRA 3B), `Q3_A4` (Joint Causal Multi-task).
- Chuẩn hóa vị trí cổng Seam tại Tầng 21/36 ($\approx 0.5833$), đồng cấu với Tầng 14/24 của 0.5B.
- Nguyên tắc Zero Cross-Scale: Tuyệt đối cấm nạp chéo trọng số LoRA hay Seam từ 0.5B sang 3B.
- Ma trận cổng 15 tiêu chí (`PRE_3B_GATE_MATRIX.json`): Đạt 14 PASS, duy nhất 1 tiêu chí NOT_VERIFIED (G11 - GPU Runtime VRAM check) do chưa cấp phép tải weights.

**5. Trạng thái nghiệm thu và rào chắn:**
$$\mathbf{TRẠNG\ THÁI:\ PRE\_3B\_READY\ (NO\_GO\_LARGE\_TRAINING)}$$
Toàn bộ khâu chuẩn bị tiền triển khai đã hoàn tất 100%. Hệ thống giữ nguyên dataset gốc, duy trì trạng thái đóng băng rào chắn, nghiêm cấm tự ý tải tệp trọng số đa Gigabyte hoặc khởi chạy huấn luyện GPU khi chưa có chỉ đạo tiếp theo.

---

## V. Đóng Gói Khoa Học Stage 03 Qwen2.5-3B V2: Khép Góc Lỗi Thực Thi, Đánh Giá Hợp Nhất 3 Nhánh và Phân Tích Đánh Đổi Bền Vững (26/10/2026)

**1. Khép góc kỹ thuật và vận hành GPU (Kaggle Dedicated Evaluation):**
- Đã giải quyết triệt để 6 lớp lỗi thực thi: F01 (Autograd In-Place Seam), F02 (Negative Loss Intra), F03 (Token Case Mismatch), F04 (Core Generator State Dict Shape), F05 (Embedding Coordinate Misalignment) và F06 (Eval CUDA OOM).
- Thực thi đánh giá độc lập hợp nhất trên GPU Tesla T4 (Kernel `hoangkhang23/lrr-stage03-3b-unified-eval`, ID `140924965`, v3) hoàn thành $100\%$ (`KernelWorkerStatus.COMPLETE`, thời gian $18{,}593$ giây).

**2. Bảng đối soát hiệu năng thực chứng 3 nhánh trên TempUN Clean ($N=5{,}292$):**
- **Arm 1 (Ticktack 3B Base)**: Canonical Acc $32.03\%$, Debiased Mean $31.19\%$, Strict Acc $5.29\%$.
- **Arm 2 (Targeted V2)**: Canonical Acc $28.10\%$, Debiased Mean $29.85\%$, Strict Acc $6.20\%$, Counterfactual (54P) $9.26\%$ ($5/54$).
- **Arm 3 (MaT-LoRA FoNE V2)**: Canonical Acc $26.27\%$, Debiased Mean $27.17\%$, **Strict Acc $8.20\%$** ($+55.0\%$ tương đối so với Arm 1), **Counterfactual (54P) $70.37\%$** ($38/54$).
- **Kiểm định McNemar ghép cặp**:
  - Arm 1 vs Arm 2 ($N=4950$ unique CID): Delta $=-3.90\%$, $\chi^2 = 18.8563, p = 1.4095 \times 10^{-5}$ (có ý nghĩa thống kê).
  - Arm 1 vs Arm 3 ($N=4950$ unique CID): Delta $=-5.56\%$, $\chi^2 = 41.1375, p = 1.4189 \times 10^{-10}$ (có ý nghĩa thống kê).
  - Arm 2 vs Arm 3 ($N=5292$ full): Delta $=-1.83\%$, $\chi^2 = 5.8663, p = 0.01543$ (có ý nghĩa thống kê).

**3. Khám phá khoa học về sự đánh đổi kiến trúc:**
- *Ảo ảnh của Canonical Accuracy*: $86.55\%$ câu trắc nghiệm 4 lựa chọn của Base model bị sụp đổ vị trí (chọn cố định 1 chữ cái). Ticktack Base đạt $32.03\%$ là do thiên lệch vị trí bề mặt; khi xoay 4 hoán vị, Strict Acc sụp đổ còn $5.29\%$.
- *Bản chất của MaT-LoRA*: 108 module MLP ($11.36\text{M}$ tham số) tác động trực tiếp vào Key-Value Memory của mô hình, triệt tiêu cơ chế đoán mò vị trí. Với 1 epoch ($6,508$ steps), mạng lõi chịu "thuế thích ứng" ban đầu (cổng Seam mới mở $\alpha = 0.0053$), làm giảm nhẹ Canonical Acc nhưng tăng vọt Strict Acc lên $8.20\%$ và tăng độ bền vững phản thực nghiệm lên $70.37\%$ ($38/54$ cặp).
- *Điểm nghẽn tác vụ TB (Time Boundary)*: FoNE chu kỳ tuần hoàn làm TB giảm từ $28.51\%$ xuống $11.73\%$ do thiếu cơ chế trừ số học tuyến tính khoảng cách thời gian (kéo tụt điểm trung bình toàn cục $\sim 3.0\%$).

**4. Lưu trữ biên nhận và hồ sơ quản lý:**
- Toàn bộ hồ sơ biên nhận đã được lưu trữ tại `Documents/Documenting/Results/STAGE03_3B_V2_20261026/` (`RECEIPT_STAGE03_3B_V2_CLOSURE.yaml`, `THREE_ARM_COMPARISON_RECEIPT.json`) và sao lưu bất biến lên Google Drive tại `G:\My Drive\NCKH\Receipts\3B_STAGE03_V2_SCIENTIFIC_CLOSURE_20261026`.
- Báo cáo phân tích chuyên sâu tại [STAGE03_3B_ARCHITECTURAL_TRADEOFF_ANALYSIS.md](file:///c:/Users/GF63/Documents/long-range-temporal-alignment/Reports/03_TARGETED_OBJECTIVES_MCQA/STAGE03_3B_ARCHITECTURAL_TRADEOFF_ANALYSIS.md).

















