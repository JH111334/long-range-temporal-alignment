# Huong Dan Tai Lap Doc Lap Run 04.2 (Exact Reproduction Commands)

Tai lieu nay cung cap toan bo cac cau lenh can thiet de kiem chung doc lap cac ket luan khoa hoc va ket qua kiem toan trong Run 04.2.

---

## 1. Dong Bo Moi Truong Va Git Revisions

### 1.1. Internal Repository (`long-range-temporal-alignment`)
```powershell
cd C:\Users\GF63\Documents\long-range-temporal-alignment
git status
# Xac nhan commit hien tai la c4302f49eef825c5ce9f84a076e2be3707b0a0da hoac HEAD cua experiment/v2-run04-2-partner-paper-fidelity-audit
git rev-parse HEAD
```

### 1.2. Partner Repository (`Long-Range-Representation`)
```powershell
cd C:\Users\GF63\Documents\Long-Range-Representation
git fetch --all
git status
# Xac nhan commit la 72f4a5009a63cf4eeea35b44ec8742ca8f97c84c tren nhanh Huy
git rev-parse HEAD
```

---

## 2. Kiem Tra Bo 5 Unit Tests Phap Y Khoa Hoc

Chay toan bo 15 bai test phap y doc lap duoc phat trien trong Run 04.2:
```powershell
cd C:\Users\GF63\Documents\long-range-temporal-alignment
py -3.12 -m pytest tests/test_year_index_reference.py `
                  tests/test_none_and_multiyear_policy.py `
                  tests/test_temporal_token_alignment.py `
                  tests/test_lora_target_identity.py `
                  tests/test_cosine_scale_diagnostic.py -v
```
Ket qua ky vong: `15 passed in ~5.7s`.

---

## 3. Tai Lap Phap Y Dong Cau Can-Chi (year - 4 vs year - 3)

Xac thuc tinh dong cau so hoc va do lech goc 6 do giua hai he quy uoc:
```powershell
py -3.12 -c @"
for y in [4, 1984, 2024, 2026]:
    idx_0based = (y - 4) % 60
    idx_1based = (y - 3) % 60
    assert (idx_1based - 1) % 60 == idx_0based, f'Mismatch at {y}'
    print(f'Year {y}: 0-based={idx_0based}, 1-based={idx_1based} (Homomorphic)')
print('Chung minh thanh cong: khong bao gio sinh None tren so nguyen.')
"@
```

---

## 4. Tai Lap Chan Doan Suy Bien Goc Cosine Collapse

Kiem chung hien tuong Cosine Collapse khi broadcast khong co ty le thu phong $1/d$:
```powershell
py -3.12 -c @"
import torch
import torch.nn.functional as F

dim = 896
seq_len = 10
E = torch.randn(seq_len, dim)
E = F.normalize(E, p=2, dim=-1) * 0.4505
te_raw = torch.randn(1, dim)
te_raw = F.normalize(te_raw, p=2, dim=-1) * 42.2618

E_collapsed = E + te_raw
cos_collapsed = F.cosine_similarity(E_collapsed[0:1], E_collapsed[1:2]).item()

te_scaled = te_raw * (1.0 / dim)
E_scaled = E + te_scaled
cos_scaled = F.cosine_similarity(E_scaled[0:1], E_scaled[1:2]).item()

print(f'Raw Broadcast Cosine: {cos_collapsed:.4f} (Cosine Collapse > 0.99)')
print(f'Scaled Broadcast Cosine: {cos_scaled:.4f} (Preserved Representation < 0.20)')
"@
```

---

## 5. Tai Lap Phan Tich Khuyet Nam (canonical_year: null)

Kiem tra 144/180 truong hop mang gia tri `null` trong tap `eval_mcqa.jsonl`:
```powershell
py -3.12 -c @"
import json

null_count = 0
total = 0
with open('data/tempun/eval_mcqa.jsonl', 'r', encoding='utf-8') as f:
    for line in f:
        data = json.loads(line)
        total += 1
        if data.get('canonical_year') is None:
            null_count += 1

print(f'Total: {total}, Missing canonical_year: {null_count} ({null_count/total*100:.1f}%)')
"@
```
Ket qua ky vong: `Total: 180, Missing canonical_year: 144 (80.0%)`.

---

## 6. Kiem Tra Toan Ven Project Va Knowledge Vault

Chay cac cong kiem dinh deterministic cua he thong:
```powershell
cd C:\Users\GF63\Documents\long-range-temporal-alignment
py -3.12 scripts/validate_project.py
py -3.12 scripts/validate_knowledge_vault.py
```
