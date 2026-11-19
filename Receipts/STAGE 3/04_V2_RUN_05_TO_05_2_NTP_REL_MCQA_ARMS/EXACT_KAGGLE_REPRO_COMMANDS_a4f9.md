# Exact Kaggle & Headless Execution Commands — V2 Run 05 Final

**Study ID:** `V2_RUN05_FINAL_AB_NTP_MULTITASK`  
**Target Architecture:** Frozen `Qwen2.5-0.5B-Instruct` + 72 MLP MaT-LoRA ($r=8, \alpha=16$) + Gated Seam token 14 + FoNE-CanChi 40D  
**Random Seed:** `42`  

---

## 1. Environment Preflight

```bash
# Verify Python version >= 3.10
python3 --version

# Verify PyTorch and Transformers
python3 -c "import torch, transformers; print('Torch:', torch.__version__, 'CUDA:', torch.cuda.is_available(), 'Transformers:', transformers.__version__)"

# Ensure PYTHONPATH includes project root and src
export PYTHONPATH="src:$PYTHONPATH"
```

---

## 2. Dataset and Checkpoint Verification

Ensure the following clean input manifests are present with exact SHA256:

| Artifact | Relative Path | Expected SHA256 |
| :--- | :--- | :--- |
| **Parent Checkpoint C1** | `artifacts/experiments/V2_CAUSAL_TEMPORAL_ALIGNMENT/run_02/checkpoints/C1_COMPACT_ADAPTER_ONLY.pt` | `30cadb35504e7f15e12922ba4cb6274cb6ee880a16df5f1f7592de48889cbc82` |
| **TempUN Small 180** | `data/tempun/eval_mcqa.jsonl` | `4e4a7e407a3711a6afb319db06bf7addca1dc09b5ed1e3dd31989a946957f1e8` |
| **Clean MCQA Train 400** | `artifacts/experiments/V2_CAUSAL_TEMPORAL_ALIGNMENT/run_05_ab_ntp/MCQA4_TRAIN_CLEAN.jsonl` | `e990c6ca2bf2df5d519277d341901c0be16b23b5d3da0fe69c84e1b8c2fa874f` |
| **Clean MCQA Dev 120** | `artifacts/experiments/V2_CAUSAL_TEMPORAL_ALIGNMENT/run_05_ab_ntp/MCQA4_DEV_CLEAN.jsonl` | `10e95ffcf0a72fa3dfdfaa6c1f1ec78b0f80bcde89b099dd4be9fc82946be488` |
| **NTP Training Corpus** | `data/tempun/train_nwp.jsonl` | `9bba5f40f02758197127e192f15cbfa9b54ec67eb5913cefb99b793132cf07b8` |

---

## 3. Independent Training Commands

### Execution on Local CPU / Headless Node

```bash
# Execute Arm A and Arm B training (40 optimizer steps each, matched FLOP budget)
python3 -c "
import sys
from pathlib import Path
sys.path.insert(0, str(Path.cwd() / 'src'))

from temporal_alignment.training_run05 import train_run05_arms

repo_root = Path.cwd()
res = train_run05_arms(repo_root=repo_root, steps=40, device_name='cpu')
print('Training finished:', res['divergence_verification'])
"
```

### Execution on Kaggle GPU (T4 / P100)

```bash
python3 -c "
import sys
from pathlib import Path
sys.path.insert(0, str(Path.cwd() / 'src'))

from temporal_alignment.training_run05 import train_run05_arms

repo_root = Path.cwd()
res = train_run05_arms(repo_root=repo_root, steps=40, device_name='cuda')
print('GPU Training finished:', res['divergence_verification'])
"
```

---

## 4. Evaluation and Intervention Commands

```bash
# Execute Full Evaluation across Arms A, B, and the 6 Arm C Interventions
python3 scripts/evaluate_v2_run05_all_arms.py
```

---

## 5. Test Suite Verification

```bash
# Run 100% pass unit tests for Run 05
pytest tests/test_run05_*.py -v
```
