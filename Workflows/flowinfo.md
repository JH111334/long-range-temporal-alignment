# Workflow Nghiên Cứu

1. Ghi nguồn vào `registries/datasets.json` hoặc `registries/models.json`.
2. Chỉ chuyển trạng thái dataset sang `approved` khi đủ evidence gate.
3. Chuẩn hóa BCE/CE, interval, granularity và provenance bằng contracts.
4. Tạo split theo group/time trước mọi preprocessing học từ dữ liệu.
5. Chạy Tier 0 với scalar, Fourier, published Ticktack và hybrid encoder.
6. Kiểm tra va chạm 60 năm, tính đơn điệu, batch-size sensitivity và invalid output.
7. Chạy resource probe trước frozen-model hoặc LoRA; dừng nếu vượt ngân sách.
8. Mỗi run xuất config hash, manifest hash, code revision, resource peak và metric theo lát cắt.
9. Chỉ so sánh model khi cùng dataset revision, split, prompt, parser và evaluator.
10. Cập nhật error analysis, ADR và brief trước buổi họp GVHD.

## Gate Chuyển Tầng

- Tier 0 → Tier 1: toàn bộ test/validator đạt; không có alias collision ngoài baseline dự kiến.
- Tier 1 → Tier 2: resource probe đạt và frozen probe cho tín hiệu hơn baseline đơn giản.
- Tier 2 → Tier 3: giả thuyết có hiệu ứng qua nhiều seed và split chống leakage.
