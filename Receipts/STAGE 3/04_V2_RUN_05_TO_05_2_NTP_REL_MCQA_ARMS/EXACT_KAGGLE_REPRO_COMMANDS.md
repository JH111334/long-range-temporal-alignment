# Lệnh Tái Lập Thực Nghiệm Độc Lập — V2 Run 05.1 (Forensic Phoenix Recovery)

**Mã Định Danh Nghiên Cứu:** `V2_RUN05_1_PHOENIX_RECOVERY`  
**Kiến Trúc Đối Tượng:** Frozen `Qwen2.5-0.5B-Instruct` (100% `requires_grad = False` trên base layer) + 72 MLP MaT-LoRA ($r=8, \alpha=16$) + Gated Seam token index 14 + FoNE-CanChi 40D  
**Random Seed:** `42`  
**Checkpoint Kế Thừa Gốc (Parent):** `C1_COMPACT_ADAPTER_ONLY.pt`  

---

## 1. Kiểm Chuẩn Môi Trường Thực Thi (Preflight Environment)

```bash
# Kiểm tra phiên bản Python >= 3.10 (khuyến nghị 3.12)
python3 --version

# Kiểm chuẩn PyTorch và Transformers
python3 -c "import torch, transformers; print('PyTorch:', torch.__version__, 'CUDA:', torch.cuda.is_available(), 'Transformers:', transformers.__version__)"

# Thiết lập đường dẫn import chuẩn xác
export PYTHONPATH="src:$PYTHONPATH"
```

---

## 2. Kiểm Chuẩn Bảng Băm SHA256 Dữ Liệu và Checkpoint Đầu Vào

Tất cả các tệp đầu vào bắt buộc phải khớp chính xác bảng băm mã hóa SHA256 trước khi huấn luyện hoặc đánh giá:

| Tạo Tác / Tệp Tin | Đường Dẫn Tương Đối | Bảng Băm Kỳ Vọng (SHA256) |
| :--- | :--- | :--- |
| **Parent Checkpoint C1** | `artifacts/experiments/V2_CAUSAL_TEMPORAL_ALIGNMENT/run_02/checkpoints/C1_COMPACT_ADAPTER_ONLY.pt` | `30cadb35504e7f15e12922ba4cb6274cb6ee880a16df5f1f7592de48889cbc82` |
| **Tập Kiểm Định TempUN 180** | `data/tempun/eval_mcqa.jsonl` | `4e4a7e407a3711a6afb319db06bf7addca1dc09b5ed1e3dd31989a946957f1e8` |
| **Canary Validation 24 Items** | `artifacts/experiments/V2_CAUSAL_TEMPORAL_ALIGNMENT/run_05_1/CANARY_TRUTH_AND_SPLIT.jsonl` | `874742e7bb0e9b25a32ec699df893c5eb29d6756fd9d6824976cf4cf2330a84d` |
| **Corpus Huấn Luyện NWP** | `data/tempun/train_nwp.jsonl` | `9bba5f40f02758197127e192f15cbfa9b54ec67eb5913cefb99b793132cf07b8` |
| **Clean MCQA Train 400** | `artifacts/experiments/V2_CAUSAL_TEMPORAL_ALIGNMENT/run_04/MCQA4_TRAIN_CLEAN.jsonl` | `e990c6ca2bf2df5d519277d341901c0be16b23b5d3da0fe69c84e1b8c2fa874f` |
| **Clean MCQA Dev 120** | `artifacts/experiments/V2_CAUSAL_TEMPORAL_ALIGNMENT/run_04/MCQA4_DEV_CLEAN.jsonl` | `10e95ffcf0a72fa3dfdfaa6c1f1ec78b0f80bcde89b099dd4be9fc82946be488` |
| **Dev Temporal Relation 60** | `artifacts/experiments/V2_CAUSAL_TEMPORAL_ALIGNMENT/run_02/dev_relation_pairs.jsonl` | `f59990b7eef46cbaee1c2780e071e7d8c7ea49479cb2544ec78ce09f3c193233` |

---

## 3. Lệnh Thực Thi Huấn Luyện Độc Lập Hai Nhánh (Arm A & Arm B)

### Chạy Trên Node CPU / Headless Môi Trường Cục Bộ:

```bash
python3 scripts/run_v2_run05_1_training.py
```

### Chạy Trên Kaggle GPU (T4 / P100 / L4):

```bash
python3 -c "
import sys
from pathlib import Path
sys.path.insert(0, str(Path.cwd() / 'src'))

from temporal_alignment.training_run05_1 import train_run05_1_arms

repo_root = Path.cwd()
res = train_run05_1_arms(repo_root=repo_root, steps=40, device_name='cuda')
print('Kaggle GPU Training finished successfully!')
print('Arm A Checkpoint:', res['ckpt_a'])
print('Arm B Checkpoint:', res['ckpt_b'])
"
```

---

## 4. Lệnh Đánh Giá Toàn Diện & Phân Tích Can Thiệp (Arm A, B, C)

```bash
# Thực thi đánh giá đầy đủ trên TempUN Small 180, DEV MCQA 120, DEV Relation 60 và 6 can thiệp Arm C
python3 scripts/evaluate_v2_run05_1_all_arms.py
```

---

## 5. Kiểm Tra Tự Động Toàn Bộ Test Suite Run 05.1

```bash
# Chạy 100% passed các bài kiểm định đơn vị Run 05.1
pytest tests/test_run051_*.py -v
```
