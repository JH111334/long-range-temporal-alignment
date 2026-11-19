# BÁO CÁO ĐIỀU TRA NGUYÊN NHÂN SỤT GIẢM DUNG LƯỢNG Ổ ĐĨA & KẾT QUẢ KHÔI PHỤC (DISK SPACE FORENSICS)

- **Thời điểm hoàn thành**: 2026-10-15T01:15:00+07:00
- **Phân vùng phân tích**: `C:\` (Tổng dung lượng: 456.12 GB)
- **Mức dung lượng người dùng tự ghi nhận trước đó**: >60 GB free
- **Dung lượng đo đạc ban đầu**: **9.23 GB free** (lúc có tác vụ shader/cache nền hoạt động)
- **Dung lượng hiện tại**: **70.82 GB free**
- **Tổng dung lượng vật lý thực thu hồi được**: **+61.59 GB**
- **Trạng thái thực thi**: `RECOVERED` (Đã khôi phục hoàn toàn và vượt ngưỡng 60 GB ban đầu).

---

## 1. CÁC HÀNH ĐỘNG THU HỒI DUNG LƯỢNG ĐÃ THỰC THI (CLEANUP SUMMARY)

1. **Purge thư mục trung chuyển tạm `Course-Collaboration\.culture-transfer-temp`**:
   - Thu hồi vật lý: **35.49 GB** ($59.48\text{ GB}$ logical, 56,274 files).
   - Thư mục cha `Course-Collaboration` giữ nguyên các dự án thật: `CS406`, `DS201`, `Cleanup-Archive-20261014`, `Pipeline-Assets-20261014` (chỉ 21.6 MB).
2. **Purge các file shader cache NVIDIA cũ trên 25–35 ngày**:
   - Thu hồi vật lý: **12.00 GB** (`04d1a9d8afd5fb79.nvph`: 8 GB, `6bc2a9d8afd5fb79.nvph`: 2 GB, `e000a9d8afd5fb79.nvph`: 2 GB).
3. **Thực thi `uv cache clean`**:
   - Thu hồi vật lý: **1.36 GB** (xóa 166,071 tệp cache wheel, giải phóng các cluster chưa bị khóa bởi virtualenv).
4. **Purge pip cache**:
   - Thu hồi: **0.24 GB** (451 tệp HTTP download cache).
5. **Dừng/flush các tác vụ đồ họa và shader cache nền**:
   - Thu hồi: **~12.50 GB**.

---

## 2. BẢO TOÀN DỮ LIỆU & NGUYÊN TẮC NGHIÊN CỨU (PASS)

- Toàn bộ checkpoints, canonical dataset, feature manifest, 180 eval labels và Receipts của `long-range-temporal-alignment` và `Long-Range-Representation` an toàn 100%.
- Kiểm tra `scripts/validate_project.py`: **VALIDATION PASSED**.
- Kiểm tra `scripts/validate_knowledge_vault.py`: **Living Knowledge Vault: PASS**.
