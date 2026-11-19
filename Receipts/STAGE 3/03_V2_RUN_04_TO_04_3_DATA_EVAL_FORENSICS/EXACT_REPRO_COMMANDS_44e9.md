# Hướng Dẫn Tái Lập Độc Lập Run 04.3 (Exact Reproduction Commands)

Tài liệu này cung cấp toàn bộ các câu lệnh thực thi độc lập nhằm kiểm chứng các kết quả khoa học của Run 04.3.

---

## 1. Kiểm Tra Tính An Toàn Và Trạng Thái Khóa Không Gian Làm Việc

### 1.1. Workspace Đối Tác (Bắt buộc READ-ONLY)
```powershell
cd C:\Users\GF63\Documents\Long-Range-Representation
# Xác nhận commit là 72f4a50c0590ed4656daff5b24fa1048f05eae48 trên nhánh Huy
git rev-parse HEAD
# Xác nhận không có thay đổi tracked nào
git status --porcelain=v1
```

### 1.2. Workspace Dự Án Cá Nhân
```powershell
cd C:\Users\GF63\Documents\long-range-temporal-alignment
# Xác nhận đang ở nhánh experiment/v2-run04-3-authentic-baseline-runtime-grounding
git branch --show-current
git rev-parse HEAD
```

---

## 2. Thực Thi Kiểm Thử Đơn Vị Runtime Year Resolution V2

Chạy bộ kiểm thử đơn vị cho khế ước phân giải thời gian mới:
```powershell
cd C:\Users\GF63\Documents\long-range-temporal-alignment
py -3.12 -m pytest tests/test_year_resolution_runtime.py -v
```
Kết quả kỳ vọng: `8 passed in ~0.12s`.

---

## 3. Thực Thi Toàn Bộ 30 Bài Test Pháp Y Khoa Học (Run 04.1 + Run 04.2 + Run 04.3)

```powershell
py -3.12 -m pytest tests/test_year_index_reference.py `
                  tests/test_none_and_multiyear_policy.py `
                  tests/test_temporal_token_alignment.py `
                  tests/test_lora_target_identity.py `
                  tests/test_cosine_scale_diagnostic.py `
                  tests/test_run041_execution_semantics.py `
                  tests/test_year_resolution_runtime.py -v
```
Kết quả kỳ vọng: `30 passed in ~6.5s`.

---

## 4. Tái Lập Kiểm Toán 180 Mẫu TempUN Small Với Resolver V2

Tái tạo lại tệp `YEAR_RESOLUTION_AUDIT_180.jsonl` và bảng thống kê `YEAR_RESOLUTION_TEST_REPORT.json`:
```powershell
py -3.12 -c @"
import json
from src.temporal_alignment.normalization.year_resolution import resolve_temporal_context, TemporalInputMode

total, fb_1984, single, multi = 0, 0, 0, 0
with open('data/tempun/eval_mcqa.jsonl', 'r', encoding='utf-8') as f:
    for line in f:
        total += 1
        item = json.loads(line)
        leg = resolve_temporal_context(item, mode=TemporalInputMode.LEGACY_SINGLE_YEAR_V1)
        if leg.effective_year and leg.effective_year.astronomical == 1984:
            fb_1984 += 1
        v2 = resolve_temporal_context(item, mode=TemporalInputMode.RESOLVED_SINGLE_YEAR_V2)
        if v2.status.value == 'SINGLE':
            single += 1
        elif v2.status.value == 'MULTI':
            multi += 1

print(f'Total: {total}, Old Fallback 1984: {fb_1984}, V2 Single: {single}, V2 Multi-Bypass: {multi}')
"@
```
Kết quả kỳ vọng: `Total: 180, Old Fallback 1984: 72, V2 Single: 36, V2 Multi-Bypass: 144`.

---

## 5. Kiểm Định Cổng Hệ Thống (Quality Gates)

Chạy kiểm định tính hợp lệ của toàn bộ dự án và living knowledge vault:
```powershell
py -3.12 scripts/validate_project.py
py -3.12 scripts/validate_knowledge_vault.py
```
Kết quả kỳ vọng: Cả hai công cụ đều báo **PASS**.
