# Phân Vùng Chuyển Đổi Quy Mô Dung Lượng: Qwen2.5-3B-Instruct (Stage 03 Transition)

**Dự án**: `long-range-temporal-alignment` — Learning Track — Stage 03  
**Nhánh thực hiện**: `v2-targeted-continual-alignment`  
**Thời điểm ban hành**: `2026-10-20`  
**Trạng thái phân vùng**: **`PRE_3B_READY`** (Sẵn sàng tiền triển khai, khóa dừng toàn diện huấn luyện lớn)

---

## 1. Mục Đích và Định Hướng Khoa Học

Phân vùng `QWEN25_3B_TRANSITION` được thiết lập nhằm triển khai chỉ đạo định hướng nghiên cứu của Giảng viên Hướng dẫn (GVHD) ngày 20/10/2026:
- Thử nghiệm mô hình nền lớn hơn `Qwen/Qwen2.5-3B-Instruct` (~$3.09\text{B}$ tham số) để kiểm chứng giả thuyết về dung lượng mô hình (**Capacity-Scale Hypothesis**).
- Đánh giá khả năng giải tỏa hiện tượng sụp đổ nhãn và nâng cao năng lực tiếp nhận tín hiệu thời gian $z(t)$ trên mô hình có dung lượng lớn.
- **Ranh giới bất biến**: Giữ nguyên $100\%$ tập dữ liệu gốc, không xáo trộn nhãn, không áp dụng hoán vị phương án trắc nghiệm (`OnlineOptionPermutation.enabled = False`), bảo đảm tính toàn vẹn của dữ liệu thực nghiệm.

---

## 2. Cấu Trúc Tài Liệu Phân Vùng

```
Reports/03_TARGETED_OBJECTIVES_MCQA/QWEN25_3B_TRANSITION/
├── README.md                                          <- Tài liệu tổng quan phân vùng
├── config/
│   ├── 3b_research_plan.yaml                         <- Kế hoạch nghiên cứu và định hướng thí nghiệm 3B
│   └── 3b_model_and_resource_contract.yaml           <- Hợp đồng tham số mô hình và ngân sách phần cứng
├── audits/
│   ├── PRE_3B_05B_EVIDENCE_FREEZE.json               <- Bằng chứng đóng băng bất biến mô hình 0.5B
│   ├── 3B_MODEL_CARD_LICENSE_AUDIT.json              <- Kiểm toán chi tiết giấy phép Qwen Research License
│   ├── 3B_MODEL_SHAPE_ADAPTER_INVENTORY.json         <- Danh mục kích thước tensor và 108 module LoRA
│   ├── 3B_RUNTIME_CAPABILITY_AND_BUDGET.json         <- Phân tích VRAM và tính khả thi phần cứng
│   ├── PRE_3B_GATE_MATRIX.json                       <- Ma trận 15 cổng kiểm soát G01 - G15
│   └── RESEARCH_CLAIM_EVIDENCE.jsonl                 <- Phân tầng bằng chứng cho 10 khẳng định nghiên cứu
├── protocols/
│   ├── 3B_CROSS_SCALE_COMPARABILITY.md               <- Giao thức so sánh liên quy mô 0.5B vs 3B
│   └── 3B_ABLATION_AND_EVALUATION_CONTRACT.md        <- Khế ước đánh giá và ma trận cắt bỏ 5 cánh tay
├── reports/
│   ├── 3B_MODEL_ADOPTION_DECISION.md                 <- Quyết định khoa học tiếp nhận Qwen2.5-3B
│   ├── 3B_BACKBONE_COMPATIBILITY_MATRIX.md           <- Ma trận tương thích kiến trúc chi tiết
│   ├── PRE_3B_SCIENTIFIC_RECONCILIATION.md           <- Báo cáo điều hòa bằng chứng và kiểm toán toán học
│   ├── PRE_3B_HANDOFF_AND_BLOCKERS.md                <- Biên bản bàn giao và danh mục rào chắn kỹ thuật
│   └── PARTNER_3B_HANDOFF_CONTRACT.md                <- Hợp đồng giao diện kỹ thuật với đối tác
└── writebacks/
    └── WRITEBACK_LEDGER.jsonl                        <- Sổ cái ghi nhận giao dịch an toàn nhánh
```

---

## 3. Trạng Thái Cổng Kiểm Soát Hiện Thời

- **Số cổng đạt yêu cầu**: **$14 / 15$ cổng PASS** (`G01` – `G10`, `G12` – `G15`).
- **Cổng chưa xác thực thực tế**: `G11_GPU_VRAM_PRECISION_MEASUREMENT` được ghi nhận chuẩn xác là **`NOT_VERIFIED`** do chưa có GPU cấp phát trong phiên chuẩn bị.
- **Lệnh điều hành bắt buộc**: `GO_3B_PREPARATION`, `NO_GO_DATA_MUTATION`, `NO_GO_OPTION_PERMUTATION`, `NO_GO_LARGE_TRAINING`.
