# Báo Cáo Phân Tích Pháp Y và Bản Vá Tối Thiểu Nạp Trọng Số M0-FC40

## 1. Bản Chất Nguyên Nhân Gốc Rễ (Root Cause Isolated)

Trong đợt đánh giá 180 câu TempUN Small trước đây tại [scripts/evaluate_m0_fc40_180_p12.py](file:///c:/Users/GF63/Documents/long-range-temporal-alignment/scripts/evaluate_m0_fc40_180_p12.py#L232-L239), quy trình nạp trọng số đã gặp lỗi không khớp khóa từ điển (Dictionary Key Mismatch):

```python
# Đoạn mã lỗi trong scripts/evaluate_m0_fc40_180_p12.py (dòng 232-238)
state = torch.load(ckpt_path, map_location=device)
if "adapted_modules" in state:
    for m_name, m_state in state["adapted_modules"].items():
        if m_name in adapted_modules:
            adapted_modules[m_name].load_state_dict(m_state, strict=False)
if "gated_proj" in state:
    gated_proj.load_state_dict(state["gated_proj"], strict=False)
```

Kiểm tra trực tiếp các tệp checkpoint `m0_fc40_seed_{1,2,3}_after_02.pt` tại [02_checkpoint_load_and_state_census.json](file:///c:/Users/GF63/Documents/long-range-temporal-alignment/artifacts/audits/M0_FC40_CAUSAL_INFERENCE_RECOVERY_20261015/02_checkpoint_load_and_state_census.json) xác nhận:
- Khóa thực tế của 72 module MLP LoRA là: `"adapters_state"` (không phải `"adapted_modules"`).
- Khóa thực tế của tầng chiếu seam là: `"gated_proj_state"` (không phải `"gated_proj"`).

Hệ quả pháp y:
1. Cả hai biểu thức điều kiện `if "adapted_modules" in state` và `if "gated_proj" in state` đều nhận giá trị `False` trên toàn bộ 3 seed.
2. Không có bất kỳ trọng số LoRA hay Seam nào được nạp vào mô hình.
3. Vì `DualCycleMaTLoRALinear` khởi tạo ma trận $B$ bằng 0 (`nn.init.zeros_(self.lora_B)`) để đảm bảo $\Delta W = 0$ tại bước khởi tạo, và `GatedResidualProjection` khởi tạo tham số cổng $\alpha = 0$, mô hình thực tế được đem đi suy luận trong script cũ là **MÔ HÌNH NỀN QWEN2.5-0.5B-INSTRUCT NGUYÊN BẢN 100% (BASE MODEL)**.
4. Điều này giải thích hoàn toàn và triệt để lý do tại sao $M_0\text{-FC40}$ trên cả 3 seed trước đó đều đạt đúng $58/180 = 32{,}22\%$ Strict Accuracy, đúng $41/156$ câu MCQA và đúng $17/24$ câu Yes/No trùng khớp tuyệt đối từng ca với Base Model.

---

## 2. Bản Vá Chuẩn Tắc Tối Thiểu (Minimal Canonical Fix)

Thay vì sửa đổi tạm bợ trong script đánh giá, một bộ nạp chuẩn tắc duy nhất đã được thiết lập tại [src/temporal_alignment/models/m0_fc40_loader.py](file:///c:/Users/GF63/Documents/long-range-temporal-alignment/src/temporal_alignment/models/m0_fc40_loader.py) với các rào chắn fail-closed:

1. **Khóa chặt khóa dữ liệu (Mandatory Key Enforcement):** Kiểm tra bắt buộc phải có `"adapters_state"` và `"gated_proj_state"`. Nếu thiếu, ném ngoại lệ `KeyError` ngay lập tức, tuyệt đối không âm thầm bỏ qua hay fallback về Base model.
2. **Kiểm tra quân số module (Module Count Assertion):** Bắt buộc phải có đủ $72/72$ projections MLP (24 layers $\times$ 3: gate, up, down).
3. **Kiểm tra độ hoạt hóa tham số (Parameter Non-Zero Assertion):** Xác nhận $\| \text{lora\_B} \| > 10^{-4}$ trên toàn bộ 72 module nhằm bảo đảm đóng góp forward là thực chất.
4. **Bảo toàn W0 (W0 Invariant):** Đảm bảo $W_0$ của mô hình nền được đóng băng và hoàn toàn không bị biến đổi.

---

## 3. Kiểm Thử Hồi Quy (Regression Tests)

Bộ kiểm thử hồi quy tại [tests/test_m0_fc40_loader_regression.py](file:///c:/Users/GF63/Documents/long-range-temporal-alignment/tests/test_m0_fc40_loader_regression.py) bao gồm 3 bài test độc lập:
- `test_loader_fails_on_missing_file`: Bắn `FileNotFoundError` khi tệp không tồn tại.
- `test_loader_fails_on_legacy_or_missing_keys`: Bắn `KeyError` / `ValueError` khi checkpoint thiếu `"adapters_state"`, thiếu `"gated_proj_state"` hoặc sai số lượng module.
- `test_loader_with_real_seed1_checkpoint`: Nạp thành công checkpoint thật của Seed 1, xác nhận 72 module, xác nhận $\| \text{lora\_A} \| > 0{,}5$, $\| \text{lora\_B} \| > 0{,}5$, và $W_0$ nguyên vẹn.

Kết quả kiểm thử: **3/3 tests PASSED**.
