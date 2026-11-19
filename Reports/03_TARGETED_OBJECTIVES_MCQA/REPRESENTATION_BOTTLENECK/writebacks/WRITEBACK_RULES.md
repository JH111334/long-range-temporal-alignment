# Quy Tắc Đồng Bộ Hồ Sơ Tri Thức (Writeback Rules)

Tài liệu này quy định kỷ luật ghi nhận tri thức từ các thực nghiệm bóc tách điểm nghẽn biểu diễn Stage 03 vào hệ thống Living Knowledge Vault và hồ sơ tiến độ của repository.

## 1. Nguyên Tắc Cốt Lõi

1. **Chỉ ghi nhận bằng chứng kiểm chứng được**: Tuyệt đối không ghi nhận nhận định cảm tính, suy đoán hoặc kết quả chưa có mã băm SHA-256 đối chiếu độc lập.
2. **Ngăn chặn ô nhiễm tài liệu Canonical**: Không đưa nhật ký gỡ lỗi (debug narratives) hoặc mã lỗi cục bộ tạm thời vào `Documents/General/RULE.md` hay các tài liệu lý thuyết nền tảng.
3. **Phân loại giao dịch chuẩn tắc**: Mọi giao dịch cập nhật trạng thái phải thuộc một trong các nhóm:
   - `AUDIT_RECEIPT`: Biên bản thẩm định dữ liệu hoặc hạ tầng.
   - `SWEEP_RESULT`: Kết quả quét thực nghiệm bóc tách điểm nghẽn.
   - `GATE_TRANSITION`: Quyết định chuyển trạng thái cổng (GO / NO-GO).
   - `KNOWLEDGE_SYNCHRONIZATION`: Đồng bộ hóa trạng thái sang `Current-State.md` và `EVIDENCE-INDEX.json`.

## 2. Quy Trình Ghi Sổ Cái (Ledger Protocol)

Mỗi lần thực thi một cột mốc hoặc hoàn thành một đợt đánh giá, agent bắt buộc phải ghi một dòng JSON vào `writebacks/WRITEBACK_LEDGER.jsonl` với cấu trúc:
- `entry_id`: Định danh duy nhất (ví dụ: `WB_20261019_001`).
- `timestamp`: Dấu thời gian chuẩn ISO 8601.
- `transaction_type`: Loại giao dịch theo phân loại trên.
- `operator_branch`: Nhánh làm việc thực tế (`v2-targeted-continual-alignment`).
- `source_artifacts`: Danh sách các tệp tạo tác kết quả kèm mã băm SHA-256.
- `target_documents`: Danh sách các tài liệu hồ sơ được cập nhật.
- `evidence_claim`: Tóm tắt mệnh đề khoa học được xác lập từ thực nghiệm.
- `gate_verdict`: Phán quyết cổng kiểm định (`PASS`, `FAIL`, `REJECT`, `GO`, `NO-GO`).
