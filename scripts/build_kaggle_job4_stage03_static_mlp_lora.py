#!/usr/bin/env python3
"""Bộ tạo và phát động Kernel Kaggle: Stage 03 Static MLP LoRA + FoNE/Can-Chi 40D trên Qwen2.5-3B (Base Model).

Mục tiêu khoa học:
- Tách biến đơn (Single-Variable Isolation):
  Loại bỏ hoàn toàn bộ sinh lõi động F_core(theta, z) của Arm 3 (MaT-LoRA),
  chỉ áp dụng LoRA tĩnh tiêu chuẩn (Static LoRA, rank=8, alpha=16.0) trên toàn bộ 108 module MLP
  (gate_proj, up_proj, down_proj trên cả 36 tầng biến đổi) kết hợp Safe Gated Seam 40D tại token prefill 14.
- Mục đích đối chứng:
  Xác định chính xác liệu sự sụt giảm Canonical Accuracy ở Arm 3 (26.27% vs 32.03%)
  bắt nguồn từ bộ sinh lõi động F_core (không kịp hội tụ sau 1 epoch)
  hay do việc can thiệp vào tầng MLP (phá vỡ Key-Value Memory của Transformer).
- Cấu hình huấn luyện an toàn VRAM tuyệt đối (Zero OOM Guarantee):
  micro_batch = 8, grad_accum = 2 (cố định effective batch size = 16),
  max_seq_len = 128, bfloat16, gradient checkpointing (use_reentrant=False).
- Đánh giá đa giao thức tách biệt sau huấn luyện:
  Giải phóng optimizer và dọn VRAM trước khi đánh giá,
  batch_size = 2 cases (8 chuỗi/lần forward) trên 5,292 mẫu TempUN Clean + 54 cặp Gold-Flip Counterfactuals.
- Tài khoản phát động: hoangkhang23.
- Slug Kaggle: hoangkhang23/lrr-stage03-3b-static-mlp-lora.
"""

from __future__ import annotations

import base64
import json
import os
import sys
import zipfile
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any, Dict, List

_REPO_ROOT = Path(__file__).resolve().parents[1]
_OUTPUT_DIR = _REPO_ROOT / "artifacts" / "kaggle_bundles" / "stage03_3b_static_mlp_lora"
_CF_FILE = _REPO_ROOT / "data" / "counterfactual" / "COUNTERFACTUAL_GOLD_AUDIT.jsonl"


def get_hoangkhang_token() -> str:
    odt_path = _REPO_ROOT / "HKKAGGLE.odt"
    with zipfile.ZipFile(odt_path, "r") as z:
        xml_content = z.read("content.xml")
    root = ET.fromstring(xml_content)
    full_text = " ".join(elem.text for elem in root.iter() if elem.text).strip()
    return full_text.split("=")[-1].strip()


def build_job4_notebook() -> Dict[str, Any]:
    cells: List[Dict[str, Any]] = []

    cf_b64 = ""
    if _CF_FILE.exists():
        cf_b64 = base64.b64encode(_CF_FILE.read_bytes()).decode("ascii")

    # Cell 1: Header Markdown
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "# [LRR Stage 03 - Job 4] Static MLP LoRA + FoNE/Can-Chi 40D Safe Seam trên Qwen2.5-3B (Base Model)\n",
            "\n",
            "**Thí nghiệm Tách Biến Khoa Học (Single-Variable Ablation Experiment)**:\n",
            "- **Mục tiêu**: Cô lập nguyên nhân sụt giảm Canonical Accuracy giữa Arm 1 (Ticktack Base, 32.03%) và Arm 3 (MaT-LoRA FoNE, 26.27%).\n",
            "- **Kiến trúc Arm 4 (Static MLP LoRA)**:\n",
            "  1. Can thiệp trực tiếp vào **108 module MLP** (`gate_proj`, `up_proj`, `down_proj` trên cả 36 tầng biến đổi) bằng **LoRA tĩnh tiêu chuẩn** (Static LoRA, $r=8, \\alpha=16.0$, $\\Delta W = \\frac{\\alpha}{r} B \\cdot A$) với $11{,}280{,}384$ tham số.\n",
            "  2. **Loại bỏ bộ sinh lõi động** $F_{\\text{core}}(\\theta, z)$, loại bỏ hoàn toàn ma trận xoay động theo token/mẫu.\n",
            "  3. **Giữ nguyên Safe Gated Seam 40D**: Chiếu vector thời gian bất biến FoNE x Can-Chi 40D qua $\\operatorname{Linear}(40, 2048) \\to \\operatorname{LayerNorm}(2048)$ tại token prefill thứ 14 với cổng khởi tạo $\\alpha_0 = 0.0$ (out-of-place safe injection).\n",
            "- **Kiểm soát VRAM và Batch Size Cố Định (Zero OOM Guarantee)**:\n",
            "  - Huấn luyện: `micro_batch = 8`, `grad_accum = 2` (Fixed effective batch size = 16), max_seq_len = 128, bfloat16, gradient checkpointing (use_reentrant=False).\n",
            "  - Đánh giá: Tách rời hoàn toàn sau huấn luyện (`del optimizer`, `torch.cuda.empty_cache()`), `batch_size = 2` cases (8 hoán vị/lần forward).\n",
            "- **Đánh giá đa giao thức nghiêm ngặt**: Canonical (thứ tự gốc), 4-fold Debiased (trung bình 4 hoán vị), Strict Accuracy (đúng toàn bộ 4 hoán vị), phân rã 6 tác vụ TempUN, và 54 cặp Gold-Flip Counterfactuals."
        ]
    })

    # Cell 2: Imports & Environment Setup
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "import os, sys, time, math, gzip, base64, json, random, gc, re\n",
            "from typing import Optional, List, Dict, Any, Tuple, Sequence\n",
            "from collections import Counter, defaultdict\n",
            "from pathlib import Path\n",
            "import numpy as np\n",
            "import torch\n",
            "import torch.nn as nn\n",
            "import torch.nn.functional as F\n",
            "from torch.utils.data import Dataset, DataLoader\n",
            "\n",
            "print('PyTorch Version:', torch.__version__)\n",
            "print('CUDA Available:', torch.cuda.is_available())\n",
            "if torch.cuda.is_available():\n",
            "    device = torch.device('cuda:0')\n",
            "    gpu_name = torch.cuda.get_device_name(0)\n",
            "    total_vram = round(torch.cuda.get_device_properties(0).total_memory / (1024**3), 2)\n",
            "    print(f'GPU: {gpu_name} ({total_vram} GB VRAM)')\n",
            "    torch.backends.cuda.matmul.allow_tf32 = True\n",
            "else:\n",
            "    device = torch.device('cpu')\n",
            "    print('CẢNH BÁO: Đang chạy trên CPU!')\n",
            "\n",
            "def set_seed(seed=42):\n",
            "    random.seed(seed)\n",
            "    np.random.seed(seed)\n",
            "    torch.manual_seed(seed)\n",
            "    if torch.cuda.is_available():\n",
            "        torch.cuda.manual_seed_all(seed)\n",
            "set_seed(42)\n",
            "MAX_TRAIN_STEPS = None\n"
        ]
    })

    # Cell 3: Data Loading
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            f"CF_DATA_B64 = '''{cf_b64}'''\n",
            "\n",
            "def load_dataset_records():\n",
            "    train_path, eval_path = None, None\n",
            "    kaggle_input = Path('/kaggle/input')\n",
            "    if kaggle_input.exists():\n",
            "        c_tr = [p for p in kaggle_input.rglob('train_104k_clean*') if p.is_file()]\n",
            "        c_ev = [p for p in kaggle_input.rglob('eval_5292_clean*') if p.is_file()]\n",
            "        if c_tr:\n",
            "            train_path = sorted(c_tr, key=lambda p: (not p.name.endswith('.gz'), len(str(p))))[0]\n",
            "        if c_ev:\n",
            "            eval_path = sorted(c_ev, key=lambda p: (not p.name.endswith('.gz'), len(str(p))))[0]\n",
            "    if not train_path or not eval_path:\n",
            "        local_roots = [Path('.'), Path('..'), Path('Data'), Path('artifacts/kaggle_datasets/hoangkhang23_stage03_train_bundle')]\n",
            "        for root in local_roots:\n",
            "            if not root.exists(): continue\n",
            "            if not train_path:\n",
            "                c = [p for p in root.rglob('train_104k_clean*') if p.is_file()]\n",
            "                if c: train_path = sorted(c, key=lambda p: (not p.name.endswith('.gz'), len(str(p))))[0]\n",
            "            if not eval_path:\n",
            "                c = [p for p in root.rglob('eval_5292_clean*') if p.is_file()]\n",
            "                if c: eval_path = sorted(c, key=lambda p: (not p.name.endswith('.gz'), len(str(p))))[0]\n",
            "            if train_path and eval_path: break\n",
            "    \n",
            "    def _read_records(p):\n",
            "        if str(p).endswith('.gz'):\n",
            "            with gzip.open(p, 'rt', encoding='utf-8') as f:\n",
            "                return [json.loads(l) for l in f if l.strip()]\n",
            "        with open(p, 'r', encoding='utf-8') as f:\n",
            "            return [json.loads(l) for l in f if l.strip()]\n",
            "    train_r = _read_records(train_path)\n",
            "    eval_r = _read_records(eval_path)\n",
            "    return train_r, eval_r\n",
            "\n",
            "train_cases, eval_cases = load_dataset_records()\n",
            "print(f'Đã nạp {len(train_cases):,} mẫu train NWP và {len(eval_cases):,} mẫu eval MCQA sạch.')\n",
            "\n",
            "cf_pairs = []\n",
            "kaggle_input = Path('/kaggle/input')\n",
            "c_cf = [p for p in kaggle_input.rglob('*counterfactual*.jsonl') if p.is_file()] if kaggle_input.exists() else []\n",
            "if c_cf:\n",
            "    with open(c_cf[0], 'r', encoding='utf-8') as f:\n",
            "        cf_pairs = [json.loads(l) for l in f if l.strip()]\n",
            "elif CF_DATA_B64.strip():\n",
            "    cf_raw = base64.b64decode(CF_DATA_B64.encode('ascii')).decode('utf-8')\n",
            "    cf_pairs = [json.loads(l) for l in cf_raw.splitlines() if l.strip()]\n",
            "print(f'Đã nạp {len(cf_pairs)} cặp phản thực nghiệm lịch sử.')\n"
        ]
    })

    # Cell 4: FoNE x Can-Chi 40D Feature Generator
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# --- Máy Sinh Đặc Trưng Bất Biến FoNE x Can-Chi 40D (FIXED_FONE_CANCHI_V1) ---\n",
            "def compute_fone_32d(year: float, num_frequencies: int = 16, min_period: float = 10.0, max_period: float = 10000.0) -> List[float]:\n",
            "    ln_min = math.log(min_period)\n",
            "    ln_max = math.log(max_period)\n",
            "    step = (ln_max - ln_min) / (num_frequencies - 1) if num_frequencies > 1 else 0.0\n",
            "    features = []\n",
            "    for k in range(num_frequencies):\n",
            "        period = math.exp(ln_min + k * step)\n",
            "        omega = 2.0 * math.pi / period\n",
            "        features.append(math.cos(omega * year))\n",
            "        features.append(math.sin(omega * year))\n",
            "    return features\n",
            "\n",
            "def compute_canchi_8d(year: int, t0: int = 1900) -> List[float]:\n",
            "    offset = year - 4\n",
            "    cycle_idx = offset % 60\n",
            "    stem_idx = offset % 10\n",
            "    branch_idx = offset % 12\n",
            "    theta = 2.0 * math.pi * float(cycle_idx) / 60.0\n",
            "    radius = 1.0 + float(year - t0) / 60.0\n",
            "    x = radius * math.cos(theta)\n",
            "    y = radius * math.sin(theta)\n",
            "    return [\n",
            "        float(x), float(y), float(math.cos(theta)), float(math.sin(theta)),\n",
            "        float(radius), float(stem_idx) / 10.0, float(branch_idx) / 12.0, float(cycle_idx) / 60.0\n",
            "    ]\n",
            "\n",
            "def compute_fixed_fone_canchi_40d(year: int) -> List[float]:\n",
            "    return compute_fone_32d(float(year)) + compute_canchi_8d(year)\n",
            "\n",
            "print('Kiểm tra tính không suy biến FoNE 40D (1947 vs 2007):')\n",
            "v1947 = compute_fixed_fone_canchi_40d(1947)\n",
            "v2007 = compute_fixed_fone_canchi_40d(2007)\n",
            "dist = math.sqrt(sum((a - b)**2 for a, b in zip(v1947, v2007)))\n",
            "print(f'||v1947 - v2007||_2 = {dist:.4f} > 0.0 (Khử triệt để tuần hoàn 60 năm mod 60!)')\n"
        ]
    })

    # Cell 5: Safe Conditioning Seam & Static LoRA Linear Definition
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# --- Safe Conditioning Seam (Gated Residual Injection) ---\n",
            "class GatedResidualProjection(nn.Module):\n",
            "    def __init__(self, in_features: int = 40, out_features: int = 2048):\n",
            "        super().__init__()\n",
            "        self.linear = nn.Linear(in_features, out_features)\n",
            "        self.norm = nn.LayerNorm(out_features, bias=False)\n",
            "        self.alpha = nn.Parameter(torch.zeros(1))\n",
            "        nn.init.normal_(self.linear.weight, mean=0.0, std=0.01)\n",
            "        nn.init.zeros_(self.linear.bias)\n",
            "    \n",
            "    def forward(self, x: torch.Tensor) -> torch.Tensor:\n",
            "        proj = self.linear(x)\n",
            "        normed = self.norm(proj)\n",
            "        return self.alpha * normed\n",
            "\n",
            "def inject_seam_prefill_safe(model, seam_proj, input_ids: torch.Tensor, v40: torch.Tensor, token_idx: int = 14):\n",
            "    embed_tokens = model.get_input_embeddings()\n",
            "    inputs_embeds = embed_tokens(input_ids)\n",
            "    B, S, D = inputs_embeds.shape\n",
            "    if S > token_idx:\n",
            "        inj = seam_proj(v40.to(inputs_embeds.device, dtype=inputs_embeds.dtype))\n",
            "        delta = torch.zeros_like(inputs_embeds)\n",
            "        delta[:, token_idx, :] = inj\n",
            "        inputs_embeds = inputs_embeds + delta\n",
            "    return inputs_embeds\n",
            "\n",
            "# --- Static LoRA Linear Module (Tiêu chuẩn không lõi động F_core) ---\n",
            "class StaticLoRALinear(nn.Module):\n",
            "    def __init__(self, original_linear: nn.Linear, rank: int = 8, alpha: float = 16.0, dropout: float = 0.0):\n",
            "        super().__init__()\n",
            "        self.in_features = original_linear.in_features\n",
            "        self.out_features = original_linear.out_features\n",
            "        self.rank = rank\n",
            "        self.scaling = alpha / float(rank)\n",
            "        self.base_layer = original_linear\n",
            "        self.base_layer.weight.requires_grad = False\n",
            "        if self.base_layer.bias is not None:\n",
            "            self.base_layer.bias.requires_grad = False\n",
            "        \n",
            "        device = original_linear.weight.device\n",
            "        dtype = original_linear.weight.dtype\n",
            "        \n",
            "        self.lora_A = nn.Parameter(torch.empty(rank, self.in_features, device=device, dtype=dtype))\n",
            "        self.lora_B = nn.Parameter(torch.empty(self.out_features, rank, device=device, dtype=dtype))\n",
            "        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))\n",
            "        # Khởi tạo B = 0 bảo đảm Delta W = 0 tại step 0 tuyệt đối\n",
            "        nn.init.zeros_(self.lora_B)\n",
            "        \n",
            "        self.dropout = nn.Dropout(p=dropout) if dropout > 0.0 else nn.Identity()\n",
            "        self.enabled: bool = True\n",
            "    \n",
            "    def forward(self, x: torch.Tensor) -> torch.Tensor:\n",
            "        base_out = self.base_layer(x)\n",
            "        if not self.enabled:\n",
            "            return base_out\n",
            "        \n",
            "        # Low-rank forward: (x @ A^T) @ B^T * scaling\n",
            "        x_a = F.linear(x, self.lora_A)\n",
            "        lora_out = F.linear(self.dropout(x_a), self.lora_B) * self.scaling\n",
            "        return base_out + lora_out\n"
        ]
    })

    # Cell 6: Model Loader & MLP Replacement for 3B Base
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# --- Khởi tạo Qwen2.5-3B Base và Thay Thế Toàn Bộ 108 Module MLP bằng Static LoRA ---\n",
            "from transformers import AutoTokenizer, AutoModelForCausalLM\n",
            "\n",
            "MODEL_ID = 'Qwen/Qwen2.5-3B'\n",
            "print(f'Nạp mô hình nền BASE: {MODEL_ID}...')\n",
            "tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, use_fast=True)\n",
            "if tokenizer.pad_token is None:\n",
            "    tokenizer.pad_token = tokenizer.eos_token\n",
            "\n",
            "def build_stage03_static_mlp_model(rank=8, alpha=16.0):\n",
            "    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=torch.bfloat16, device_map='cuda:0' if torch.cuda.is_available() else 'cpu')\n",
            "    for p in model.parameters():\n",
            "        p.requires_grad = False\n",
            "    \n",
            "    seam_projection = GatedResidualProjection(in_features=40, out_features=2048).to(model.device, dtype=torch.bfloat16)\n",
            "    \n",
            "    target_suffixes = ('gate_proj', 'up_proj', 'down_proj')\n",
            "    mlp_modules = []\n",
            "    \n",
            "    for name, module in list(model.named_modules()):\n",
            "        if not isinstance(module, nn.Linear): continue\n",
            "        if any(name.endswith(suffix) for suffix in target_suffixes):\n",
            "            parts = name.split('.')\n",
            "            parent = model\n",
            "            for p in parts[:-1]: parent = getattr(parent, p)\n",
            "            c_name = parts[-1]\n",
            "            adapted = StaticLoRALinear(module, rank=rank, alpha=alpha, dropout=0.0)\n",
            "            setattr(parent, c_name, adapted)\n",
            "            mlp_modules.append(adapted)\n",
            "    \n",
            "    print(f'Đã gắn Static LoRA trên {len(mlp_modules)} module MLP.')\n",
            "    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})\n",
            "    \n",
            "    seen_ids = set()\n",
            "    trainable_params = []\n",
            "    for p in (list(model.parameters()) + list(seam_projection.parameters())):\n",
            "        if p.requires_grad and id(p) not in seen_ids:\n",
            "            seen_ids.add(id(p))\n",
            "            trainable_params.append(p)\n",
            "            \n",
            "    n_train = sum(p.numel() for p in trainable_params)\n",
            "    print(f'Tổng tham số huấn luyện Static MLP LoRA + Seam: {n_train:,} tham số ({n_train/3.09e9*100:.3f}% tổng mô hình).')\n",
            "    return model, seam_projection, mlp_modules, trainable_params\n"
        ]
    })

    # Cell 7: Dataset & Batch Collate with FoNE 40D
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# --- Dataset & Batch Collate với FoNE 40D ---\n",
            "class Stage03StaticMLPDataset(Dataset):\n",
            "    def __init__(self, samples, tokenizer, max_len=128):\n",
            "        self.examples = []\n",
            "        for s in samples:\n",
            "            query = str(s.get('input', ''))\n",
            "            ans = str(s.get('target', ''))\n",
            "            year = int(s.get('year', 2000))\n",
            "            task_id = str(s.get('task_id', 'Unknown'))\n",
            "            \n",
            "            prompt_text = f'Question: {query.strip()}\\nAnswer:'\n",
            "            full_text = f'{prompt_text} {ans.strip()}'\n",
            "            enc = tokenizer(full_text, truncation=True, max_length=max_len, return_tensors='pt')\n",
            "            input_ids = enc['input_ids'].squeeze(0)\n",
            "            \n",
            "            prefix_enc = tokenizer(prompt_text, truncation=True, max_length=max_len, return_tensors='pt')\n",
            "            prefix_len = prefix_enc['input_ids'].size(1)\n",
            "            \n",
            "            labels = input_ids.clone()\n",
            "            labels[:min(prefix_len, len(labels))] = -100  # Chỉ tính gradient trên token đáp án\n",
            "            \n",
            "            v40 = compute_fixed_fone_canchi_40d(year)\n",
            "            is_rel = int(task_id == 'Rel' or ans.strip().lower() in ['yes', 'no'])\n",
            "            \n",
            "            self.examples.append({\n",
            "                'input_ids': input_ids,\n",
            "                'labels': labels,\n",
            "                'v40': v40,\n",
            "                'year': year,\n",
            "                'is_rel': is_rel\n",
            "            })\n",
            "            \n",
            "    def __len__(self):\n",
            "        return len(self.examples)\n",
            "    def __getitem__(self, idx):\n",
            "        return self.examples[idx]\n",
            "\n",
            "def collate_static_mlp_fn(batch, pad_token_id):\n",
            "    max_len = max(len(x['input_ids']) for x in batch)\n",
            "    b_in, b_lbl, b_v40, b_rel = [], [], [], []\n",
            "    for x in batch:\n",
            "        pad_len = max_len - len(x['input_ids'])\n",
            "        inp = torch.cat([x['input_ids'], torch.full((pad_len,), pad_token_id, dtype=torch.long)])\n",
            "        lbl = torch.cat([x['labels'], torch.full((pad_len,), -100, dtype=torch.long)])\n",
            "        b_in.append(inp)\n",
            "        b_lbl.append(lbl)\n",
            "        b_v40.append(x['v40'])\n",
            "        b_rel.append(x['is_rel'])\n",
            "    return {\n",
            "        'input_ids': torch.stack(b_in),\n",
            "        'labels': torch.stack(b_lbl),\n",
            "        'v40': torch.tensor(b_v40, dtype=torch.float32),\n",
            "        'is_rel': torch.tensor(b_rel, dtype=torch.float32)\n",
            "    }\n"
        ]
    })

    # Cell 8: High-Speed Evaluation Suite
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# --- Bộ Đánh Giá Đa Giao Thức (Canonical, 4-fold Debiased, Strict Acc) ---\n",
            "def generate_option_permutations(prompt_text: str, target_text: str):\n",
            "    tgt = target_text.strip().lower()\n",
            "    if tgt in ['yes', 'no']:\n",
            "        p0 = prompt_text\n",
            "        p1 = re.sub(r'Answer as Yes or No', 'Answer as No or Yes', prompt_text, flags=re.IGNORECASE)\n",
            "        if p1 == p0: p1 = re.sub(r'Yes\\s*(or|/)\\s*No', 'No or Yes', prompt_text, flags=re.IGNORECASE)\n",
            "        return [{'prompt': p0, 'target': tgt, 'rotation': 0}, {'prompt': p1, 'target': tgt, 'rotation': 1}]\n",
            "    suffix = ''\n",
            "    m_suffix = re.search(r'(\\s*\\n?\\s*Answer:\\s*)$', prompt_text, re.IGNORECASE)\n",
            "    if m_suffix:\n",
            "        suffix = m_suffix.group(1)\n",
            "        base_prompt = prompt_text[:len(prompt_text) - len(suffix)]\n",
            "    else: base_prompt = prompt_text\n",
            "    pattern = re.compile(r'^(.*Options:\\s*)\\(a\\)\\s*(.*?)\\s*\\(b\\)\\s*(.*?)\\s*\\(c\\)\\s*(.*?)\\s*\\(d\\)\\s*(.*?)$', re.DOTALL)\n",
            "    m = pattern.search(base_prompt)\n",
            "    if not m: return [{'prompt': prompt_text, 'target': tgt, 'rotation': 0}]\n",
            "    prefix = m.group(1)\n",
            "    opts = [m.group(2).strip(), m.group(3).strip(), m.group(4).strip(), m.group(5).strip()]\n",
            "    letters = ['a', 'b', 'c', 'd']\n",
            "    if tgt not in letters: return [{'prompt': prompt_text, 'target': tgt, 'rotation': 0}]\n",
            "    t_idx = letters.index(tgt)\n",
            "    perms = []\n",
            "    for k in range(4):\n",
            "        r_opts = [opts[(j + k) % 4] for j in range(4)]\n",
            "        r_tgt = letters[(t_idx - k) % 4]\n",
            "        perms.append({'prompt': f'{prefix}(a) {r_opts[0]} (b) {r_opts[1]} (c) {r_opts[2]} (d) {r_opts[3]}{suffix}', 'target': r_tgt, 'rotation': k})\n",
            "    return perms\n",
            "\n",
            "def extract_token_ids(tok, words):\n",
            "    ids = []\n",
            "    for w in words:\n",
            "        for prefix in ['', ' ']:\n",
            "            enc = tok.encode(prefix + w, add_special_tokens=False)\n",
            "            if enc and len(enc) == 1: ids.append(enc[0])\n",
            "    return list(dict.fromkeys(ids))\n",
            "\n",
            "yes_ids = extract_token_ids(tokenizer, ['yes', 'Yes', 'YES'])\n",
            "no_ids = extract_token_ids(tokenizer, ['no', 'No', 'NO'])\n",
            "choice_ids = {k: extract_token_ids(tokenizer, [k, k.upper()]) for k in ['a', 'b', 'c', 'd']}\n",
            "print('Yes IDs:', yes_ids, '| No IDs:', no_ids)\n",
            "print('Choice IDs:', choice_ids)\n",
            "\n",
            "def score_prediction_row(logits_row, target_text, prior_bias_rel=0.0):\n",
            "    t_norm = target_text.strip().lower()\n",
            "    if t_norm in ['yes', 'no']:\n",
            "        s_yes = (max(logits_row[idx].item() for idx in yes_ids) if yes_ids else -1e9) + prior_bias_rel\n",
            "        s_no = max(logits_row[idx].item() for idx in no_ids) if no_ids else -1e9\n",
            "        pred = 'yes' if s_yes >= s_no else 'no'\n",
            "    else:\n",
            "        s = {k: (max(logits_row[idx].item() for idx in c_ids) if c_ids else -1e9) for k, c_ids in choice_ids.items()}\n",
            "        pred = max(s, key=s.get)\n",
            "    return pred, t_norm, int(pred == t_norm)\n",
            "\n",
            "def evaluate_static_mlp_dataset(model, seam_proj, eval_records, batch_size=2):\n",
            "    model.eval()\n",
            "    seam_proj.eval()\n",
            "    tokenizer.padding_side = 'left'  # Cần padding trái khi sinh token cuối cùng\n",
            "    case_results = {}\n",
            "    can_correct, strict_correct, deb_sum = 0, 0, 0.0\n",
            "    task_scores = defaultdict(lambda: {'canonical': 0, 'debiased_sum': 0.0, 'total': 0})\n",
            "    total = len(eval_records)\n",
            "    t0 = time.time()\n",
            "    \n",
            "    for b_start in range(0, total, batch_size):\n",
            "        b_cases = eval_records[b_start:b_start + batch_size]\n",
            "        all_prompts, all_targets, all_years, case_perm_slices = [], [], [], []\n",
            "        for case in b_cases:\n",
            "            yr = case.get('year')\n",
            "            perms = generate_option_permutations(case['input'], case['target'])\n",
            "            s_idx = len(all_prompts)\n",
            "            for p in perms:\n",
            "                all_prompts.append(p['prompt'])\n",
            "                all_targets.append(p['target'])\n",
            "                all_years.append(yr)\n",
            "            e_idx = len(all_prompts)\n",
            "            case_perm_slices.append((case, (s_idx, e_idx)))\n",
            "            \n",
            "        formatted = [f'Question: {p.strip()}\\nAnswer:' if not p.strip().endswith('Answer:') else p.strip() for p in all_prompts]\n",
            "        inputs = tokenizer(formatted, return_tensors='pt', padding=True).to(model.device)\n",
            "        v40_list = [compute_fixed_fone_canchi_40d(int(y) if y is not None else 2000) for y in all_years]\n",
            "        v40_t = torch.tensor(v40_list, dtype=torch.float32, device=model.device)\n",
            "        \n",
            "        with torch.no_grad():\n",
            "            inputs_embeds = inject_seam_prefill_safe(model, seam_proj, inputs['input_ids'], v40_t, token_idx=14)\n",
            "            out = model(inputs_embeds=inputs_embeds, attention_mask=inputs['attention_mask'])\n",
            "            logits = out.logits[:, -1, :]\n",
            "            \n",
            "        b_results = []\n",
            "        for idx, (p_text, gold) in enumerate(zip(all_prompts, all_targets)):\n",
            "            pred, t_norm, is_corr = score_prediction_row(logits[idx], gold)\n",
            "            b_results.append({'pred': pred, 'gold': t_norm, 'is_correct': is_corr})\n",
            "            \n",
            "        for case, (s_idx, e_idx) in case_perm_slices:\n",
            "            cid = case.get('example_id', f'case_{len(case_results)}')\n",
            "            task_id = case.get('task_id', 'Unknown')\n",
            "            outcomes = [r['is_correct'] for r in b_results[s_idx:e_idx]]\n",
            "            is_can = outcomes[0]\n",
            "            d_score = float(np.mean(outcomes))\n",
            "            is_st = all(outcomes)\n",
            "            if is_can: can_correct += 1\n",
            "            if is_st: strict_correct += 1\n",
            "            deb_sum += d_score\n",
            "            task_scores[task_id]['canonical'] += int(is_can)\n",
            "            task_scores[task_id]['debiased_sum'] += d_score\n",
            "            task_scores[task_id]['total'] += 1\n",
            "            safe_key = f\"{len(case_results)}_{cid}\"\n",
            "            case_results[safe_key] = {'cid': cid, 'is_correct': is_can, 'debiased_score': d_score, 'is_strict': is_st, 'task_id': task_id}\n",
            "            \n",
            "        done = min(b_start + batch_size, total)\n",
            "        if done % 500 < batch_size or done == total:\n",
            "            print(f'Eval: {done}/{total} ({done/(time.time()-t0):.2f} câu/s)...')\n",
            "            torch.cuda.empty_cache()\n",
            "            \n",
            "    task_summary = {}\n",
            "    for t, v in sorted(task_scores.items()):\n",
            "        c = v['total']\n",
            "        task_summary[t] = {'canonical_acc': round(v['canonical']/c*100, 2), 'debiased_mean_acc': round(v['debiased_sum']/c*100, 2), 'count': c}\n",
            "    return {\n",
            "        'canonical_acc': round(can_correct / total * 100.0, 2),\n",
            "        'debiased_mean_acc': round(deb_sum / total * 100.0, 2),\n",
            "        'strict_acc': round(strict_correct / total * 100.0, 2),\n",
            "        'total': total,\n",
            "        'task_summary': task_summary,\n",
            "        'case_results': case_results\n",
            "    }\n"
        ]
    })

    # Cell 9: Training Loop with Fixed Batch Size & Zero OOM Controls
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# --- Huấn luyện Stage 03 Static MLP LoRA (Batch Size Cố Định 16) ---\n",
            "def train_static_mlp_model(train_data, epochs=1, lr=2e-4, micro_batch=8, grad_accum=2, max_steps=MAX_TRAIN_STEPS):\n",
            "    print('\\n========== BẮT ĐẦU HUẤN LUYỆN STAGE 03 STATIC MLP LORA (3B) ==========')\n",
            "    set_seed(42)\n",
            "    model, seam_proj, mlp_mods, trainable_params = build_stage03_static_mlp_model(rank=8, alpha=16.0)\n",
            "    model.train()\n",
            "    seam_proj.train()\n",
            "    tokenizer.padding_side = 'right'  # Dùng padding phải cho collate huấn luyện\n",
            "    \n",
            "    train_ds = Stage03StaticMLPDataset(train_data, tokenizer, max_len=128)\n",
            "    train_loader = DataLoader(train_ds, batch_size=micro_batch, shuffle=True, collate_fn=lambda b: collate_static_mlp_fn(b, tokenizer.pad_token_id))\n",
            "    \n",
            "    optimizer = torch.optim.AdamW(trainable_params, lr=lr, weight_decay=0.01)\n",
            "    step = 0\n",
            "    accum_loss = 0.0\n",
            "    t_start = time.time()\n",
            "    \n",
            "    for epoch in range(epochs):\n",
            "        for b_idx, batch in enumerate(train_loader):\n",
            "            input_ids = batch['input_ids'].to(model.device)\n",
            "            labels = batch['labels'].to(model.device)\n",
            "            v40 = batch['v40'].to(model.device)\n",
            "            \n",
            "            # Tiêm Seam tại token prefill 14 (safe out-of-place)\n",
            "            inputs_embeds = inject_seam_prefill_safe(model, seam_proj, input_ids, v40, token_idx=14)\n",
            "            \n",
            "            outputs = model(inputs_embeds=inputs_embeds, labels=labels)\n",
            "            loss = outputs.loss / grad_accum\n",
            "            loss.backward()\n",
            "            accum_loss += loss.item() * grad_accum\n",
            "            \n",
            "            if (b_idx + 1) % grad_accum == 0:\n",
            "                torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)\n",
            "                optimizer.step()\n",
            "                optimizer.zero_grad()\n",
            "                step += 1\n",
            "                if step % 250 == 0:\n",
            "                    print(f'Epoch {epoch+1} | Step {step} | Loss: {accum_loss/250.0:.4f} | Seam alpha: {seam_proj.alpha.item():.5f} | Thời gian: {time.time()-t_start:.1f}s')\n",
            "                    accum_loss = 0.0\n",
            "                if step % 500 == 0:\n",
            "                    torch.cuda.empty_cache()\n",
            "                if max_steps and step >= max_steps: break\n",
            "                \n",
            "    print(f'Huấn luyện Static MLP LoRA hoàn tất sau {time.time()-t_start:.1f}s.')\n",
            "    \n",
            "    ckpt = {\n",
            "        'lora_state': {k: v.cpu() for k, v in model.state_dict().items() if 'lora_' in k},\n",
            "        'seam_projection': {k: v.cpu() for k, v in seam_proj.state_dict().items()},\n",
            "        'metadata': {\n",
            "            'architecture': 'StaticLoRA_MLP108_Seam40D',\n",
            "            'rank': 8,\n",
            "            'alpha': 16.0,\n",
            "            'steps': step,\n",
            "            'final_alpha': seam_proj.alpha.item(),\n",
            "            'training_time_s': time.time() - t_start\n",
            "        }\n",
            "    }\n",
            "    torch.save(ckpt, 'stage03_3b_static_mlp_lora_weights.pt')\n",
            "    print('Đã lưu checkpoint: stage03_3b_static_mlp_lora_weights.pt')\n",
            "    \n",
            "    # Giải phóng hoàn toàn optimizer để dọn sạch bộ nhớ trước khi đánh giá (Chống OOM tuyệt đối)\n",
            "    del optimizer\n",
            "    gc.collect()\n",
            "    torch.cuda.empty_cache()\n",
            "    print('Đã giải phóng optimizer và thu gom rác bộ nhớ.')\n",
            "    return model, seam_proj, ckpt['metadata']\n",
            "\n",
            "trained_model, trained_seam, train_meta = train_static_mlp_model(train_cases, epochs=1)\n"
        ]
    })

    # Cell 10: Evaluation on TempUN Clean & Counterfactuals
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# --- Đánh Giá Độc Lập Sau Huấn Luyện (TempUN Clean 5,292 MCQA + 54 Cặp Counterfactual) ---\n",
            "print('\\n========== BẮT ĐẦU ĐÁNH GIÁ STATIC MLP LORA TRÊN TEMPUN EVAL (5,292 MẪU) ==========')\n",
            "res_arm4 = evaluate_static_mlp_dataset(trained_model, trained_seam, eval_cases, batch_size=2)\n",
            "\n",
            "print(f'\\nKẾT QUẢ ARM 4 (STATIC MLP LORA):')\n",
            "print(f'- Canonical Accuracy: {res_arm4[\"canonical_acc\"]}%')\n",
            "print(f'- Debiased Mean Accuracy: {res_arm4[\"debiased_mean_acc\"]}%')\n",
            "print(f'- Strict Accuracy (4 hoán vị): {res_arm4[\"strict_acc\"]}%')\n",
            "print('Phân rã theo tác vụ (Task Breakdown):', res_arm4['task_summary'])\n",
            "\n",
            "# Chẩn đoán 54 cặp Counterfactual\n",
            "print('\\n========== BẮT ĐẦU ĐÁNH GIÁ 54 CẶP COUNTERFACTUAL (ARM 4) ==========')\n",
            "both_c4 = 0\n",
            "for pair in cf_pairs:\n",
            "    if len(pair) >= 2:\n",
            "        q1, q2 = pair[0], pair[1]\n",
            "        f1 = [f'Question: {q1[\"input\"].strip()}\\nAnswer:']\n",
            "        f2 = [f'Question: {q2[\"input\"].strip()}\\nAnswer:']\n",
            "        i1 = tokenizer(f1, return_tensors='pt').to(trained_model.device)\n",
            "        i2 = tokenizer(f2, return_tensors='pt').to(trained_model.device)\n",
            "        v1 = torch.tensor([compute_fixed_fone_canchi_40d(q1.get('year', 2000))], dtype=torch.float32, device=trained_model.device)\n",
            "        v2 = torch.tensor([compute_fixed_fone_canchi_40d(q2.get('year', 2000))], dtype=torch.float32, device=trained_model.device)\n",
            "        with torch.no_grad():\n",
            "            emb1 = inject_seam_prefill_safe(trained_model, trained_seam, i1['input_ids'], v1, token_idx=14)\n",
            "            emb2 = inject_seam_prefill_safe(trained_model, trained_seam, i2['input_ids'], v2, token_idx=14)\n",
            "            o1 = trained_model(inputs_embeds=emb1, attention_mask=i1['attention_mask']).logits[:, -1, :]\n",
            "            o2 = trained_model(inputs_embeds=emb2, attention_mask=i2['attention_mask']).logits[:, -1, :]\n",
            "            r1 = score_prediction_row(o1[0], q1['target'])[2]\n",
            "            r2 = score_prediction_row(o2[0], q2['target'])[2]\n",
            "        if r1 and r2: both_c4 += 1\n",
            "\n",
            "res_arm4['cf_54p'] = {'both_correct': both_c4, 'rate': round(both_c4/len(cf_pairs)*100, 2) if cf_pairs else 0}\n",
            "print(f'Arm 4 Counterfactual 54P BothCorrect: {both_c4}/{len(cf_pairs)} ({res_arm4[\"cf_54p\"][\"rate\"]}%)')\n",
            "\n",
            "with open('stage03_3b_static_mlp_lora_results.json', 'w', encoding='utf-8') as f:\n",
            "    json.dump(res_arm4, f, indent=2)\n",
            "print('Đã lưu kết quả chi tiết: stage03_3b_static_mlp_lora_results.json')\n",
            "\n",
            "summary = {\n",
            "    'arm': 'Arm 4: Static MLP LoRA (108 modules) + FoNE 40D Safe Seam',\n",
            "    'canonical_acc': res_arm4['canonical_acc'],\n",
            "    'debiased_mean_acc': res_arm4['debiased_mean_acc'],\n",
            "    'strict_acc': res_arm4['strict_acc'],\n",
            "    'cf_54p_both_correct': res_arm4['cf_54p']['both_correct'],\n",
            "    'cf_54p_rate': res_arm4['cf_54p']['rate'],\n",
            "    'task_breakdown': res_arm4['task_summary'],\n",
            "    'train_metadata': train_meta\n",
            "}\n",
            "with open('training_summary.json', 'w', encoding='utf-8') as f:\n",
            "    json.dump(summary, f, indent=2)\n",
            "print('Đã lưu tổng kết: training_summary.json')\n"
        ]
    })

    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "name": "python",
                "version": "3.10.12"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 5
    }


def main():
    _OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    nb_dict = build_job4_notebook()
    nb_path = _OUTPUT_DIR / "stage03_3b_static_mlp_lora.ipynb"
    with open(nb_path, "w", encoding="utf-8") as f:
        json.dump(nb_dict, f, indent=1)
    print(f"Đã tạo Notebook Job 4: {nb_path} ({os.path.getsize(nb_path)/1024:.1f} KB)")

    meta = {
        "id": "hoangkhang23/lrr-stage03-3b-static-mlp-lora",
        "title": "LRR Stage03 3B Static MLP LoRA",
        "code_file": "stage03_3b_static_mlp_lora.ipynb",
        "language": "python",
        "kernel_type": "notebook",
        "is_private": "true",
        "enable_gpu": "true",
        "enable_tpu": "false",
        "enable_internet": "true",
        "dataset_sources": [
            "hoangkhang23/lrr-stage03-train-bundle"
        ],
        "competition_sources": [],
        "kernel_sources": [],
        "model_sources": []
    }
    meta_path = _OUTPUT_DIR / "kernel-metadata.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    print(f"Đã tạo Metadata Job 4 cho hoangkhang23: {meta_path}")

    token = get_hoangkhang_token()
    os.environ["KAGGLE_API_TOKEN"] = token

    from kaggle.api.kaggle_api_extended import KaggleApi
    api = KaggleApi()
    api.authenticate()
    print("Đang phát động Arm 4 (Static MLP LoRA) lên Kaggle GPU bằng tài khoản hoangkhang23...")
    push_res = api.kernels_push(str(_OUTPUT_DIR))
    print(f"Kết quả Push Job 4: {push_res}")


if __name__ == "__main__":
    main()
