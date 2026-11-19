# Hệ Thống Báo Cáo Phân Tầng Thực Nghiệm (Stage-Wise Research Reports)

Tài liệu này đóng vai trò hướng dẫn đọc, bản đồ bằng chứng khoa học và cẩm nang xác minh độc lập dành cho Giáo viên Hướng dẫn (GVHD) và hội đồng đánh giá đề tài **Nghiên cứu Biểu diễn và Căn chỉnh Thời gian Dài hạn trong Mô hình Ngôn ngữ (Long-Range Temporal Alignment - Learning Track)**.

---

## 1. Mục Đích và Cấu Trúc Báo Cáo

Hệ thống báo cáo được chuẩn hóa thành **3 Stage khoa học độc lập** theo bản chất can thiệp phương pháp, thay vì liệt kê phân mảnh theo mã lượt chạy (run IDs). Cách tiếp cận này giúp phân định rạch ròi đóng góp kỹ thuật, nguồn gốc dữ liệu, giao thức đánh giá và bằng chứng nhân quả của từng giai đoạn.

```
Reports/
├── README.md                                    # Hướng dẫn định vị và bản đồ xác minh độc lập
├── STAGE-INDEX.json                             # Chỉ mục máy đọc được toàn bộ 3 stage và metrics
├── 01_MLP_ONLY_LORA/                            # STAGE 01: MLP-Only LoRA / Static PEFT Baselines
│   ├── STAGE_REPORT.md                          # Báo cáo khoa học chi tiết Stage 01
│   ├── EVIDENCE-INDEX.json                      # Danh mục tạo tác và mã băm SHA256 Stage 01
│   └── CLAIM-EVIDENCE-MATRIX.md                 # Ma trận khẳng định - bằng chứng Stage 01
├── 02_TRADITIONAL_DUAL_CYCLE_MAT_LORA/          # STAGE 02: Traditional Dual-Cycle MaT-LoRA
│   ├── STAGE_REPORT.md                          # Báo cáo khoa học chi tiết Stage 02
│   ├── EVIDENCE-INDEX.json                      # Danh mục tạo tác và mã băm SHA256 Stage 02
│   └── CLAIM-EVIDENCE-MATRIX.md                 # Ma trận khẳng định - bằng chứng Stage 02
├── 03_TARGETED_OBJECTIVES_MCQA/                 # STAGE 03: Targeted Objectives & MCQA Training (Active)
│   ├── STAGE_REPORT.md                          # Báo cáo khoa học chi tiết Stage 03
│   ├── EVIDENCE-INDEX.json                      # Danh mục tạo tác và mã băm SHA256 Stage 03
│   ├── CLAIM-EVIDENCE-MATRIX.md                 # Ma trận khẳng định - bằng chứng Stage 03
│   ├── RUN_TIMELINE.md                          # Dòng thời gian chi tiết chuỗi V2 (Run 01 - Run 05.2)
│   └── OPEN-ISSUES-AND-GATES.md                 # Kiểm toán 7 mâu thuẫn Run 05.2 và bài toán nhãn
└── CROSS-STAGE/                                 # Tổng hợp liên Stage
    ├── LINEAGE_AND_PROTOCOL_COMPATIBILITY.md    # Phân tích tương thích giao thức đánh giá P00 - P12
    ├── RECONCILIATION_RECEIPT.json              # Biên nhận điều hòa số liệu và mã băm toàn dự án
    └── GVHD_PROGRESS_SUMMARY.md                 # Báo cáo tiến độ cô đọng (1 trang) cho GVHD
```

---

## 2. Bản Đồ 3 Stage Khoa Học

| Stage | Tên khoa học | Phạm vi can thiệp | Mô hình & Cấu hình | Kết quả cốt lõi | Phán quyết khoa học |
|---|---|---|---|---|---|
| **STAGE 01** | MLP-Only LoRA / Static PEFT Baselines | Tối ưu hóa hàm mất mát và ranh giới PEFT tĩnh | `Qwen2.5-0.5B-Instruct` + Static LoRA ($r=8, \alpha=16$) | `MASKED_TARGET` triệt tiêu lỗi cú pháp (0% Invalid), đạt 38.89% Acc. MLP-Only giảm 57% tham số nhưng bảo toàn hiệu năng. Tiêm token 0 gây sụp đổ; Gated Seam token 14 giải cứu cú pháp. | `VALIDATED / ANCHOR_ESTABLISHED` |
| **STAGE 02** | Traditional Dual-Cycle MaT-LoRA | Tích hợp vi chu kỳ Can-Chi và đại kỷ nguyên vào lõi $\Delta W(\theta, z)$ | MaT-LoRA ($r=8$) trên 72 module MLP + khế ước FoNE-CanChi 40D | Khắc phục sụp đổ số học NaN; chuẩn hóa giao thức $P_{12}$ Task-Aware phục hồi điểm số 3 seed lên $37.50\% \pm 4.17\%$ (0% Invalid). Đối chứng bài báo TickTack gốc xác lập ranh giới năng lực 0.5B. | `MEASURED_VERIFIED / SUPERSEDED_TO_TARGETED` |
| **STAGE 03** | Targeted Objectives & MCQA Training | Hàm mất mát đa nhiệm (Causal NTP, Relation, Clean MCQA) & Gatekeeper | 2 nhánh huấn luyện A/B độc lập + 7 can thiệp suy luận Arm C | Arm B đạt 50.83% DEV MCQA, 62.03% DEV Relation. Bóc tách 7 mâu thuẫn pháp y: sụp đổ nhãn 'c', bypass 144 câu multi-year, rò rỉ 60 câu nguồn, sụt giảm LoRA vs Base không có ý nghĩa thống kê ($p=0.165$). | `ACTIVE_STAGE / DEFECTS_RESOLVED` |

---

## 3. Hệ Nhãn Phân Loại Tri Thức Chuẩn Mực

Mọi số liệu, khẳng định và tạo tác trong báo cáo đều được gắn nhãn tri thức theo quy định khoa học của dự án:
- `MEASURED_VERIFIED`: Số liệu đã được đo đạc trực tiếp trên máy chủ tính toán, có tệp raw generation tương ứng, có mã băm SHA256 và đã được tái tính toán độc lập.
- `REPORTED_UNVERIFIED`: Số liệu do đối tác báo cáo hoặc trích từ văn bản bài báo gốc nhưng chưa thể tái lập độc lập trong cùng môi trường.
- `HISTORICAL`: Kết quả thực nghiệm lịch sử có giá trị định hướng hoặc phân tích tiến hóa, nhưng không dùng làm mốc so sánh trực tiếp do khác biệt khế ước.
- `SUPERSEDED`: Kết quả đã bị thay thế sau khi phát hiện khiếm khuyết dữ liệu hoặc lỗi giao thức đánh giá.
- `PROVISIONAL`: Số liệu sơ bộ đang trong quá trình đo đạc bổ sung, chưa đủ bằng chứng ghép cặp để đưa ra kết luận chắc chắn.
- `CONFLICT`: Điểm mâu thuẫn giữa văn bản báo cáo cũ và dữ liệu thực tế trên đĩa (đã được ghi nhận vào Ledger và phân giải).
- `NOT_EVALUATED`: Thành phần hoặc nhánh thử nghiệm chưa được cấp phát tài nguyên thực thi.

---

## 4. Quy Trình Xác Minh Độc Lập (Independent Verification)

Người đọc có thể kiểm chứng lại toàn bộ các số liệu trong hệ thống báo cáo bằng cách thực thi các công cụ kiểm toán được cung cấp trong mã nguồn:

```powershell
# 1. Kiểm tra tính toàn vẹn và hợp lệ của cấu trúc tri thức dự án
.venv\Scripts\python.exe scripts/validate_knowledge_vault.py

# 2. Kiểm tra bộ bất biến khế ước của Run 05.2 (20/20 tests PASS)
.venv\Scripts\pytest tests/test_run052_contract_invariants.py

# 3. Chạy lại công cụ kiểm toán pháp y tái tính toán toàn bộ số liệu Run 05.2
.venv\Scripts\python.exe scripts/audit_run052_discrepancies.py

# 4. Kiểm tra các cổng kiểm soát kỹ thuật chung của dự án
.venv\Scripts\python.exe scripts/validate_project.py
```
