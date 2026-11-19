# Hướng Dẫn Tái Lập Thực Nghiệm Độc Lập 1 Lệnh (Exact Offline Reproduction Commands)

Tài liệu này cung cấp các chỉ dẫn cụ thể, khép kín để một nhà nghiên cứu độc lập có thể tái lập 100% các kết quả kiểm toán, kiểm định tính tương đương mô hình nền và đối soát số liệu mà không cần tải thêm dữ liệu ngoài.

---

## 1. Môi Trường Thực Thi Yêu Cầu

- **Hệ điều hành:** Windows 10/11 x64 hoặc Linux x86_64
- **Python:** 3.12.x
- **Thư viện cốt lõi:**
  ```bash
  pip install torch transformers accelerate pytest
  ```
- **Mã băm Commit Git Cơ Sở:** `c4302f4`
- **Nhánh Git Làm Việc:** `experiment/v2-run04-1-audit-revamp`

---

## 2. Kiểm Tra Kiểm Toán Nguồn và Danh Tính Dữ Liệu (Gate G0)

Kiểm tra toàn bộ tính toàn vẹn của tệp mã nguồn, khế ước dữ liệu sạch và các tệp checkpoint bằng lệnh CLI chính thức:

```bash
# Thiết lập PYTHONPATH trỏ về thư mục src/
set PYTHONPATH=src

# Thực thi kiểm tra preflight
python -m temporal_alignment.audit_run041 preflight --config configs/run041.yaml
```

---

## 3. Kiểm Tra Tính Tương Đương Tuyệt Đối Của Mô Hình Nền (Gate G1)

Kiểm chứng tính tương đương từng bit giữa mô hình nền nguyên bản (`RAW_QWEN05B`) và mô hình đã bọc module MaT-LoRA khi ngắt adapter (`PATCHED_BASE_EQUIVALENT` với `enabled=False`):

```bash
python -m temporal_alignment.audit_run041 verify-base --config configs/run041.yaml
```

*Tiêu chí đạt (PASS):* `max_abs_diff == 0.0000000000`, `top1_flip == False` trên toàn bộ các câu kiểm thử.

---

## 4. Kiểm Toán Nghiêm Ngặt Checkpoint và Dấu Vân Tay Tensor (Gate G3 & G4)

Xác minh 508 tensor khả huấn trên 72 module MLP, kiểm tra tính hữu hạn (`torch.isfinite`) và chứng minh tính đồng nhất tuyệt đối về trọng số giữa $R4\_B$ và $R4\_D$:

```bash
python -m temporal_alignment.audit_run041 verify-checkpoints --config configs/run041.yaml
```

---

## 5. Chạy Toàn Bộ Test Suite Ngữ Nghĩa Thực Thi (Unit & Regression Tests)

Kiểm tra 7 trường hợp biên bắt buộc (ngắt adapter, F(0,0) khác 0, thu hồi hook sau exception, cách ly trạng thái giữa các lần gọi):

```bash
pytest tests/test_run041_execution_semantics.py -v
```

*Tiêu chí đạt:* `7 passed in < 5s`.

---

## 6. Thực Thi Tái Lập Toàn Bộ 4 Bảng Điểm và Can Thiệp Nhân Quả

Chạy chu trình tái lập toàn diện, đối soát đa bảng điểm và đo lường 6 can thiệp điều kiện trên cùng checkpoint $R4\_B$:

```bash
python scripts/execute_run041_reconciliation.py
```

---

## 7. Kiểm Tra Tính Hợp Lệ Của Dự Án và Vault Tri Thức

```bash
python scripts/validate_project.py
python scripts/validate_knowledge_vault.py
```

*Tiêu chí đạt:* Cả hai validator đều trả về `EXIT_CODE 0` (`PASS`).
