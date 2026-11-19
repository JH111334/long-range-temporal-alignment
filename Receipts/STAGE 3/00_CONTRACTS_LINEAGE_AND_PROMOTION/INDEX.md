# Mục Lục Chỉ Mục Phân Vùng Điểm Nghẽn Biểu Diễn

Bảng chỉ mục liên kết toàn bộ tài liệu, cấu hình, hợp đồng và báo cáo thuộc phân vùng `REPRESENTATION_BOTTLENECK`.

## 1. Cấu Hình và Giao Diện (Configuration & Interfaces)

- [Cấu hình quét thực nghiệm](file:///c:/Users/GF63/Documents/long-range-temporal-alignment/Reports/03_TARGETED_OBJECTIVES_MCQA/REPRESENTATION_BOTTLENECK/config/bottleneck_experiment.yaml): Khung tham số cho ma trận can thiệp và các nhánh bóc tách (ablation arms).
- [Cấu hình OpenTelemetry / Arize Phoenix](file:///c:/Users/GF63/Documents/long-range-temporal-alignment/Reports/03_TARGETED_OBJECTIVES_MCQA/REPRESENTATION_BOTTLENECK/config/otel_phoenix.yaml): Thiết lập endpoint, collector fail-open và phân rã cây span.
- [Mẫu biến môi trường](file:///c:/Users/GF63/Documents/long-range-temporal-alignment/Reports/03_TARGETED_OBJECTIVES_MCQA/REPRESENTATION_BOTTLENECK/config/.env.example): Biến môi trường phục vụ kết nối Phoenix collector.
- [JSON Schema giao diện biểu diễn](file:///c:/Users/GF63/Documents/long-range-temporal-alignment/Reports/03_TARGETED_OBJECTIVES_MCQA/REPRESENTATION_BOTTLENECK/config/representation_interface.schema.json): Đặc tả kiểu dữ liệu, chiều tensor, biên giá trị và chuẩn hóa của vector $z(t)$.
- [Ngưỡng kiểm định khoa học](file:///c:/Users/GF63/Documents/long-range-temporal-alignment/Reports/03_TARGETED_OBJECTIVES_MCQA/REPRESENTATION_BOTTLENECK/config/thresholds.yaml): Các ngưỡng quyết định cho Delivery, Sensitivity, Selectivity và McNemar test.

## 2. Hợp Đồng và Ranh Giới (Contracts & Boundaries)

- [Hợp đồng bàn giao liên nhánh](file:///c:/Users/GF63/Documents/long-range-temporal-alignment/Reports/03_TARGETED_OBJECTIVES_MCQA/REPRESENTATION_BOTTLENECK/contracts/BRANCH_HANDOFF.md): Phân định trách nhiệm giữa nhánh Biểu diễn (Partner) và nhánh Mục tiêu Huấn luyện / Đánh giá Nhân quả.
- [Đặc tả tương thích biểu diễn](file:///c:/Users/GF63/Documents/long-range-temporal-alignment/Reports/03_TARGETED_OBJECTIVES_MCQA/REPRESENTATION_BOTTLENECK/contracts/REPRESENTATION_COMPATIBILITY.md): Điều kiện tương thích giữa $z(t)$ với module Seam Injection và Dynamic Core $F_t$.

## 3. Quy Chuẩn và Phương Pháp (Protocols & Methodology)

- [Định nghĩa toán học các độ đo truyền dẫn](file:///c:/Users/GF63/Documents/long-range-temporal-alignment/Reports/03_TARGETED_OBJECTIVES_MCQA/REPRESENTATION_BOTTLENECK/protocols/TRACING_METRIC_DEFINITIONS.md): Công thức tính toán $D_z, D_F, D_W, S_{\ell, p}, \text{CosineSim}, \Delta \text{logits}, M(t), \Delta M(g_2)$.
- [Hợp đồng đánh giá nhân quả](file:///c:/Users/GF63/Documents/long-range-temporal-alignment/Reports/03_TARGETED_OBJECTIVES_MCQA/REPRESENTATION_BOTTLENECK/protocols/CAUSAL_EVALUATION_CONTRACT.md): Quy trình kiểm định cặp ghép đôi, bảng tiếp liên $2 \times 2$, kiểm định McNemar hiệu chỉnh cụm Obuchowski và bootstrap cụm.
- [Báo cáo kiểm toán tập phản thực nghiệm Gold](file:///c:/Users/GF63/Documents/long-range-temporal-alignment/Reports/03_TARGETED_OBJECTIVES_MCQA/REPRESENTATION_BOTTLENECK/protocols/COUNTERFACTUAL_GOLD_AUDIT.md): Phân loại và nghiệm thu 54 cặp phản thực nghiệm làm ground truth.

## 4. Báo Cáo và Ma Trận Quyết Định (Reports & Decisions)

- [Ma trận quyết định điểm nghẽn](file:///c:/Users/GF63/Documents/long-range-temporal-alignment/Reports/03_TARGETED_OBJECTIVES_MCQA/REPRESENTATION_BOTTLENECK/reports/BOTTLENECK_DECISION_MATRIX.md): Đánh giá phân loại theo 5 cấp độ cho từng nhánh can thiệp.
- [Báo cáo tổng hợp bóc tách biểu diễn](file:///c:/Users/GF63/Documents/long-range-temporal-alignment/Reports/03_TARGETED_OBJECTIVES_MCQA/REPRESENTATION_BOTTLENECK/reports/STAGE03_REPRESENTATION_SWEEP_REPORT.md): Báo cáo học thuật chi tiết phục vụ GVHD và đồng bộ tri thức.

## 5. Quy Trình Đồng Bộ (Writeback Ledger)

- [Quy tắc đồng bộ hồ sơ](file:///c:/Users/GF63/Documents/long-range-temporal-alignment/Reports/03_TARGETED_OBJECTIVES_MCQA/REPRESENTATION_BOTTLENECK/writebacks/WRITEBACK_RULES.md): Nguyên tắc ghi nhận an toàn nhánh và ngăn chặn ô nhiễm tài liệu canonical.
- [Sổ cái ghi nhận đồng bộ](file:///c:/Users/GF63/Documents/long-range-temporal-alignment/Reports/03_TARGETED_OBJECTIVES_MCQA/REPRESENTATION_BOTTLENECK/writebacks/WRITEBACK_LEDGER.jsonl): Bản ghi nối tiếp các giao dịch cập nhật trạng thái kèm mã kiểm tra SHA-256.
