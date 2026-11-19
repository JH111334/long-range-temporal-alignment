#!/usr/bin/env python3
"""
Generator script to build the self-contained Kaggle notebook:
  kaggle_ticktack_paper_rebuild/ticktack_paper_rebuild.ipynb
and metadata:
  tnghuy11/ticktack-paper-rebuild-baseline

Implements the 4-Way Control Disentanglement Matrix + Letter Bias Debiasing Benchmark:
- Base (Row 0): Frozen Base Model Zero-shot (Qwen/Qwen2.5-0.5B-Instruct)
- C1 (Raw Broadcast): TE broadcast across all tokens with sigma = 0.5 (Original paper math)
- C2 (Raw Date-token only): TE injected only at date token positions with sigma = 0.5 (Localized injection)
- C3 (Raw with sigma = 0): TE broadcast, but sigma = 0 (Pure NTP, no intra loss)
- C4 (PT + Intra loss): No TE injected, but trained with NTP + sigma * intra loss (sigma = 0.5)

Academic Evaluation Protocol:
- Choice Logit Scoring (Academic Standard, Invalid Rate = 0.0%)
- 4-Fold Option Cyclic Permutation Rotation (ABCD, BCDA, CDAB, DABC) for MCQA
- Bidirectional Ordering ('Yes or No', 'No or Yes') for Yes/No questions
- Measures Canonical Accuracy, Debiased Mean Accuracy, and Consistency (Strict Debiased Accuracy)
"""

import base64
import gzip
import json
import os
import sys
from typing import Optional

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

def _find_clean_path(filename: str) -> str:
    for candidate in [
        os.path.join("Data", "tempun_clean", filename),
        os.path.join("data", "tempun_clean", filename),
        os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "Data", "tempun_clean", filename)),
        os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "data", "tempun_clean", filename)),
    ]:
        if os.path.exists(candidate):
            return candidate
    return os.path.join("Data", "tempun_clean", filename)

CLEAN_TRAIN_PATH = _find_clean_path("train_2461_clean.jsonl")
CLEAN_EVAL_PATH = _find_clean_path("eval_180_clean.jsonl")
def _find_output_dir(default_name: str = "kaggle_ticktack_paper_rebuild") -> str:
    repo_cand = os.path.abspath(os.path.join(_REPO_ROOT, default_name))
    if os.path.exists(repo_cand):
        return repo_cand
    return default_name

OUTPUT_DIR = _find_output_dir("kaggle_ticktack_paper_rebuild")
NOTEBOOK_PATH = os.path.join(OUTPUT_DIR, "ticktack_paper_rebuild.ipynb")
METADATA_PATH = os.path.join(OUTPUT_DIR, "kernel-metadata.json")

def create_kaggle_notebook(
    output_dir: str = OUTPUT_DIR,
    train_path: Optional[str] = None,
    eval_path: Optional[str] = None,
    extra_output_dir: Optional[str] = os.path.join(_REPO_ROOT, "baselines", "ticktack", "notebooks")
) -> str:
    """
    Builds the standalone, fully executable Kaggle Tesla T4 benchmark notebook.
    
    Args:
        output_dir: Target directory to write notebook and kernel-metadata.json.
        train_path: Custom path to train_2461_clean.jsonl.
        eval_path: Custom path to eval_180_clean.jsonl.
        extra_output_dir: Optional second directory (e.g. baselines/ticktack/notebooks) to sync notebook.
        
    Returns:
        Path to generated notebook.
    """
    actual_train_path = train_path or CLEAN_TRAIN_PATH
    actual_eval_path = eval_path or CLEAN_EVAL_PATH
    
    os.makedirs(output_dir, exist_ok=True)
    if extra_output_dir:
        os.makedirs(extra_output_dir, exist_ok=True)
    
    if not os.path.exists(actual_train_path) or not os.path.exists(actual_eval_path):
        raise FileNotFoundError(f"Clean datasets not found at {actual_train_path} or {actual_eval_path}!")
        
    with open(actual_train_path, "r", encoding="utf-8") as f:
        train_content = f.read()
    with open(actual_eval_path, "r", encoding="utf-8") as f:
        eval_content = f.read()

    train_b64 = base64.b64encode(gzip.compress(train_content.encode("utf-8"))).decode("utf-8")
    eval_b64 = base64.b64encode(gzip.compress(eval_content.encode("utf-8"))).decode("utf-8")

    cells = []

    # Cell 1: Markdown introduction
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "# Disentangling Representation Collapse in Ticktack Baseline (Han et al., 2025)\n",
            "\n",
            "This notebook executes a rigorous, independent empirical study to determine the exact root cause of representation collapse and verify letter bias debiasing on TempUN Small (2,461 train, 180 eval):\n",
            "\n",
            "### 4-Way Control Configurations:\n",
            "- **`Base` (Zero-shot)**: Frozen `Qwen/Qwen2.5-0.5B-Instruct` on raw prompt `Question: ...\\nAnswer:`.\n",
            "- **`C1 (Raw Broadcast)`**: TE broadcast across all tokens with $\\sigma = 0.5$ (Literal paper formula Eq. 1-5, Eq. 8).\n",
            "- **`C2 (Raw Date-token only)`**: TE injected only at date token positions with $\\sigma = 0.5$ (Prevents RMSNorm from washing out token identities).\n",
            "- **`C3 (Raw with $\\sigma = 0$)`**: TE broadcast across tokens, but $\\sigma = 0.0$ (Pure NTP, no intra-class contrastive loss).\n",
            "- **`C4 (PT + Intra loss)`**: No TE injected, but trained with NTP + $\\sigma \\times$ intra-loss ($\\sigma = 0.5$, tests intra-loss alone).\n",
            "\n",
            "### Evaluation & Debiasing Protocol:\n",
            "- **Academic Choice Log-Likelihood Scoring** (Invalid Rate = 0.0%).\n",
            "- **4-fold Option Cyclic Permutation Rotation** (Rot 0: ABCD, Rot 1: BCDA, Rot 2: CDAB, Rot 3: DABC) for MCQA.\n",
            "- **Bidirectional Ordering** ('Yes or No', 'No or Yes') for Yes/No questions.\n",
            "- Measures **Canonical Accuracy**, **Debiased Mean Accuracy**, and **Consistency (Strict Debiased Accuracy)**."
        ]
    })

    # Cell 2: Imports & Environment Check
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "import os, sys, time, math, gzip, base64, json, random, gc, re\n",
            "from collections import Counter, defaultdict\n",
            "import numpy as np\n",
            "import torch\n",
            "import torch.nn as nn\n",
            "import torch.nn.functional as F\n",
            "from torch.utils.data import Dataset, DataLoader\n",
            "from scipy import stats\n",
            "\n",
            "print('PyTorch Version:', torch.__version__)\n",
            "print('CUDA Available:', torch.cuda.is_available())\n",
            "if torch.cuda.is_available():\n",
            "    device = torch.device('cuda:0')\n",
            "    print('GPU Device Name:', torch.cuda.get_device_name(0))\n",
            "    print('Total VRAM:', round(torch.cuda.get_device_properties(0).total_memory / (1024**3), 2), 'GB')\n",
            "else:\n",
            "    device = torch.device('cpu')\n",
            "    print('WARNING: GPU not found, running on CPU!')\n"
        ]
    })

    # Cell 3: Data Unpacking & Integrity Verification
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            f"TRAIN_B64 = '{train_b64}'\n",
            f"EVAL_B64 = '{eval_b64}'\n",
            "\n",
            "train_raw = gzip.decompress(base64.b64decode(TRAIN_B64)).decode('utf-8')\n",
            "train_cases = [json.loads(line) for line in train_raw.strip().splitlines() if line.strip()]\n",
            "print(f'Unpacked {len(train_cases)} clean train samples (Expected: 2461).')\n",
            "\n",
            "eval_raw = gzip.decompress(base64.b64decode(EVAL_B64)).decode('utf-8')\n",
            "eval_cases = [json.loads(line) for line in eval_raw.strip().splitlines() if line.strip()]\n",
            "print(f'Unpacked {len(eval_cases)} clean eval samples (Expected: 180).')\n",
            "\n",
            "train_year_dist = Counter(int(x['year']) for x in train_cases)\n",
            "eval_year_dist = Counter(int(x['year']) for x in eval_cases)\n",
            "print('Train year distribution:', dict(sorted(train_year_dist.items())))\n",
            "print('Eval year distribution:', dict(sorted(eval_year_dist.items())))\n",
            "assert len(train_cases) == 2461 and len(eval_cases) == 180\n",
            "assert eval_year_dist == {1947: 48, 1984: 72, 2022: 60}\n"
        ]
    })

    # Cell 4: Ticktack Mathematical Architecture & LoRA
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# --- 1. Ticktack Temporal Encoding Math (Han et al., 2025 Eq. 1-4) ---\n",
            "def compute_canchi_cycle(year: int) -> int:\n",
            "    \"\"\"Gregorian year to 60-year sexagenary cycle index (Eq. 1).\"\"\"\n",
            "    if year >= 4:\n",
            "        return (year - 3) % 60\n",
            "    elif year > 0:\n",
            "        return (60 - abs(year - 3)) % 60\n",
            "    else:\n",
            "        return (60 - abs(year) - 2) % 60\n",
            "\n",
            "def compute_batch_te(years: torch.Tensor, hidden_dim=896, scale_mode='raw', alpha=0.5, beta=0.5, device='cpu', dtype=torch.float32):\n",
            "    \"\"\"Computes vectorized TE in R^{B x 1 x hidden_dim} (Eq. 2, 3, 4).\"\"\"\n",
            "    cycle_indices = ((years.to(torch.int64) - 3) % 60).to(torch.float32)\n",
            "    theta = (cycle_indices * 6.0) * (math.pi / 180.0)\n",
            "    r = alpha + beta * theta\n",
            "    x = (r * torch.cos(theta)).view(-1, 1).to(device)\n",
            "    y = (r * torch.sin(theta)).view(-1, 1).to(device)\n",
            "    \n",
            "    dim_indices = torch.arange(0, hidden_dim, 2, device=device, dtype=torch.float32)\n",
            "    inv_freq = 1.0 / (10000.0 ** (dim_indices / float(hidden_dim))).unsqueeze(0)\n",
            "    \n",
            "    x_inp = x * inv_freq\n",
            "    te_x = torch.stack([torch.sin(x_inp), torch.cos(x_inp)], dim=-1).flatten(start_dim=-2)\n",
            "    \n",
            "    y_inp = y * inv_freq\n",
            "    te_y = torch.stack([torch.sin(y_inp), torch.cos(y_inp)], dim=-1).flatten(start_dim=-2)\n",
            "    \n",
            "    te_raw = (te_x + te_y).unsqueeze(1).to(dtype=dtype)  # [B, 1, D]\n",
            "    if scale_mode == 'raw':\n",
            "        return te_raw\n",
            "    elif scale_mode == 'scaled':\n",
            "        scale = 1.0 / float(hidden_dim)\n",
            "        return (te_raw * scale).to(dtype=dtype)\n",
            "    else:\n",
            "        raise ValueError(f'Unknown scale_mode: {scale_mode}')\n",
            "\n",
            "# --- 2. Standard LoRA Linear Layer (Hu et al., 2021) ---\n",
            "class LoRALinear(nn.Module):\n",
            "    def __init__(self, in_features, out_features, rank=8, alpha=16.0, bias=False):\n",
            "        super().__init__()\n",
            "        self.in_features = in_features\n",
            "        self.out_features = out_features\n",
            "        self.rank = rank\n",
            "        self.alpha = alpha\n",
            "        self.scaling = alpha / float(rank) if rank > 0 else 1.0\n",
            "        self.base_layer = nn.Linear(in_features, out_features, bias=bias)\n",
            "        self.base_layer.weight.requires_grad = False\n",
            "        if self.base_layer.bias is not None:\n",
            "            self.base_layer.bias.requires_grad = False\n",
            "        self.lora_A = nn.Parameter(torch.empty(rank, in_features))\n",
            "        self.lora_B = nn.Parameter(torch.empty(out_features, rank))\n",
            "        self.reset_parameters()\n",
            "\n",
            "    def reset_parameters(self):\n",
            "        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))\n",
            "        nn.init.zeros_(self.lora_B)\n",
            "\n",
            "    @classmethod\n",
            "    def from_linear(cls, linear, rank=8, alpha=16.0):\n",
            "        mod = cls(linear.in_features, linear.out_features, rank=rank, alpha=alpha, bias=(linear.bias is not None))\n",
            "        mod.base_layer.weight.data.copy_(linear.weight.data)\n",
            "        mod.base_layer.weight.requires_grad = False\n",
            "        if linear.bias is not None:\n",
            "            mod.base_layer.bias.data.copy_(linear.bias.data)\n",
            "            mod.base_layer.bias.requires_grad = False\n",
            "        mod.to(device=linear.weight.device, dtype=linear.weight.dtype)\n",
            "        return mod\n",
            "\n",
            "    def forward(self, x):\n",
            "        base_out = self.base_layer(x)\n",
            "        if self.rank == 0:\n",
            "            return base_out\n",
            "        lora_out = F.linear(F.linear(x, self.lora_A.to(dtype=x.dtype, device=x.device)), self.lora_B.to(dtype=x.dtype, device=x.device)) * self.scaling\n",
            "        return base_out + lora_out\n",
            "\n",
            "def apply_lora_to_model(model, target_modules=['q_proj', 'v_proj'], rank=8, alpha=16.0):\n",
            "    adapted = {}\n",
            "    for name, module in model.named_modules():\n",
            "        for t in target_modules:\n",
            "            if name.endswith(t) and isinstance(module, nn.Linear):\n",
            "                parts = name.split('.')\n",
            "                parent = model\n",
            "                for p in parts[:-1]:\n",
            "                    parent = getattr(parent, p)\n",
            "                layer_name = parts[-1]\n",
            "                adapter = LoRALinear.from_linear(module, rank=rank, alpha=alpha)\n",
            "                setattr(parent, layer_name, adapter)\n",
            "                adapted[name] = adapter\n",
            "    return adapted\n",
            "\n",
            "# --- 3. Eq. 8 Intra-class Contrastive Loss with Can Chi Sexagenary Cycle Grouping ---\n",
            "def compute_intra_loss(hidden_states: torch.Tensor, years: torch.Tensor):\n",
            "    \"\"\"Eq. 8 intra-class contrastive loss clustering by Can Chi sexagenary cycle (Han et al., 2025).\"\"\"\n",
            "    B = hidden_states.size(0)\n",
            "    norm_h = F.normalize(hidden_states, p=2, dim=-1)\n",
            "    sim_matrix = torch.matmul(norm_h, norm_h.T)\n",
            "    canchi_cycles = ((years.to(torch.int64) - 3) % 60).to(device=hidden_states.device)\n",
            "    same_cycle_mask = (canchi_cycles.unsqueeze(1) == canchi_cycles.unsqueeze(0)) & (~torch.eye(B, dtype=torch.bool, device=hidden_states.device))\n",
            "    if same_cycle_mask.sum() > 0:\n",
            "        return 1.0 - sim_matrix[same_cycle_mask].mean()\n",
            "    else:\n",
            "        return (hidden_states * 0.0).sum()\n",
            "\n",
            "print('Ticktack architecture math and LoRA modules defined successfully.')\n"
        ]
    })

    # Cell 5: Model Factory, Dataset Collator, Date-Token Masking, Permutation Generator & Unified Runner
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "from transformers import AutoModelForCausalLM, AutoTokenizer\n",
            "\n",
            "MODEL_ID = 'Qwen/Qwen2.5-0.5B-Instruct'\n",
            "REVISION = '7ae557604adf67be50417f59c2c2f167def9a775'\n",
            "\n",
            "print(f'Loading tokenizer for {MODEL_ID}...')\n",
            "tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, revision=REVISION, trust_remote_code=True)\n",
            "if tokenizer.pad_token_id is None:\n",
            "    tokenizer.pad_token_id = tokenizer.eos_token_id\n",
            "\n",
            "def load_base_model():\n",
            "    model = AutoModelForCausalLM.from_pretrained(\n",
            "        MODEL_ID,\n",
            "        revision=REVISION,\n",
            "        torch_dtype=torch.float32,\n",
            "        device_map={'': device},\n",
            "        trust_remote_code=True\n",
            "    )\n",
            "    for p in model.parameters():\n",
            "        p.requires_grad = False\n",
            "    model.eval()\n",
            "    return model\n",
            "\n",
            "def load_lora_model(rank=8, alpha=16.0):\n",
            "    model = load_base_model()\n",
            "    adapted = apply_lora_to_model(model, target_modules=['q_proj', 'v_proj'], rank=rank, alpha=alpha)\n",
            "    return model, adapted\n",
            "\n",
            "# Dataset for Next-Word Prediction\n",
            "class JointNWPDataset(Dataset):\n",
            "    def __init__(self, samples, tokenizer, max_len=128):\n",
            "        self.examples = []\n",
            "        for s in samples:\n",
            "            inp = s['input']\n",
            "            tgt = str(s['target'])\n",
            "            if not tgt.startswith(' ') and not inp.endswith(' '):\n",
            "                tgt = ' ' + tgt\n",
            "            enc_p = tokenizer.encode(inp, add_special_tokens=False)\n",
            "            enc_t = tokenizer.encode(tgt, add_special_tokens=False)\n",
            "            ids = enc_p + enc_t\n",
            "            lbls = [-100] * len(enc_p) + enc_t\n",
            "            if len(ids) > max_len:\n",
            "                ids = ids[:max_len]\n",
            "                lbls = lbls[:max_len]\n",
            "            self.examples.append({\n",
            "                'input_ids': ids,\n",
            "                'labels': lbls,\n",
            "                'year': int(s['year'])\n",
            "            })\n",
            "\n",
            "    def __len__(self):\n",
            "        return len(self.examples)\n",
            "\n",
            "    def __getitem__(self, idx):\n",
            "        return self.examples[idx]\n",
            "\n",
            "def collate_joint_fn(batch, pad_token_id):\n",
            "    max_len = max(len(x['input_ids']) for x in batch)\n",
            "    input_ids, labels, attention_mask, years = [], [], [], []\n",
            "    for x in batch:\n",
            "        pad_len = max_len - len(x['input_ids'])\n",
            "        input_ids.append(x['input_ids'] + [pad_token_id] * pad_len)\n",
            "        labels.append(x['labels'] + [-100] * pad_len)\n",
            "        attention_mask.append([1] * len(x['input_ids']) + [0] * pad_len)\n",
            "        years.append(x['year'])\n",
            "    return {\n",
            "        'input_ids': torch.tensor(input_ids, dtype=torch.long),\n",
            "        'labels': torch.tensor(labels, dtype=torch.long),\n",
            "        'attention_mask': torch.tensor(attention_mask, dtype=torch.long),\n",
            "        'year': torch.tensor(years, dtype=torch.long)\n",
            "    }\n",
            "\n",
            "# --- Date Token Masking & Localized TE Injection ---\n",
            "def find_date_token_mask(input_ids: torch.Tensor, years: torch.Tensor, tokenizer, device=None):\n",
            "    \"\"\"Finds boolean mask [B, S, 1] identifying exact date token sub-sequences.\"\"\"\n",
            "    if device is None:\n",
            "        device = input_ids.device\n",
            "    B, S = input_ids.shape\n",
            "    mask = torch.zeros(B, S, 1, dtype=torch.bool, device=device)\n",
            "    for b in range(B):\n",
            "        yr = int(years[b].item())\n",
            "        sub = tokenizer.encode(str(yr), add_special_tokens=False)\n",
            "        seq = input_ids[b].tolist()\n",
            "        sub_len = len(sub)\n",
            "        found = False\n",
            "        for i in range(len(seq) - sub_len + 1):\n",
            "            if seq[i:i+sub_len] == sub:\n",
            "                mask[b, i:i+sub_len, 0] = True\n",
            "                found = True\n",
            "        if not found:\n",
            "            mask[b, 1:min(5, S), 0] = True\n",
            "    return mask\n",
            "\n",
            "def inject_te_embeddings(model, tokenizer, input_ids: torch.Tensor, years: torch.Tensor, te_mode='broadcast', scale_mode='raw', device=None):\n",
            "    \"\"\"Injects TE according to te_mode ('none', 'broadcast', 'date_token').\"\"\"\n",
            "    if device is None:\n",
            "        device = input_ids.device\n",
            "    token_embeds = model.model.embed_tokens(input_ids)\n",
            "    if te_mode == 'none':\n",
            "        return token_embeds\n",
            "    te_batch = compute_batch_te(years, hidden_dim=model.config.hidden_size, scale_mode=scale_mode, device=device, dtype=model.dtype)\n",
            "    if te_mode == 'broadcast':\n",
            "        return token_embeds + te_batch\n",
            "    elif te_mode == 'date_token':\n",
            "        mask = find_date_token_mask(input_ids, years, tokenizer, device=device)\n",
            "        return torch.where(mask, token_embeds + te_batch, token_embeds)\n",
            "    else:\n",
            "        raise ValueError(f'Unknown te_mode: {te_mode}')\n",
            "\n",
            "# --- Option Permutation Generator (Letter Bias Debiasing) ---\n",
            "def generate_option_permutations(prompt_text: str, target_text: str):\n",
            "    \"\"\"\n",
            "    Generates option permutations to evaluate and debias letter / ordering bias.\n",
            "    - 4-choice MCQA: 4 cyclic rotations (Rot 0: ABCD, Rot 1: BCDA, Rot 2: CDAB, Rot 3: DABC)\n",
            "      and transforms target accordingly.\n",
            "    - Yes/No: 2 orderings ('Answer as Yes or No', 'Answer as No or Yes').\n",
            "    \"\"\"\n",
            "    tgt = target_text.strip().lower()\n",
            "    if tgt in ['yes', 'no']:\n",
            "        p0 = prompt_text\n",
            "        p1 = re.sub(r'Answer as Yes or No', 'Answer as No or Yes', prompt_text)\n",
            "        return [\n",
            "            {'prompt': p0, 'target': tgt, 'rot_idx': 0, 'rot_name': 'Yes/No'},\n",
            "            {'prompt': p1, 'target': tgt, 'rot_idx': 1, 'rot_name': 'No/Yes'}\n",
            "        ]\n",
            "    suffix = ''\n",
            "    m_suffix = re.search(r'(\\s*\\n?\\s*Answer:\\s*)$', prompt_text, re.IGNORECASE)\n",
            "    if m_suffix:\n",
            "        suffix = m_suffix.group(1)\n",
            "        base_prompt = prompt_text[:len(prompt_text) - len(suffix)]\n",
            "    else:\n",
            "        base_prompt = prompt_text\n",
            "    pattern = re.compile(r'^(.*Options:\\s*)\\(a\\)\\s*(.*?)\\s*\\(b\\)\\s*(.*?)\\s*\\(c\\)\\s*(.*?)\\s*\\(d\\)\\s*(.*?)$', re.DOTALL)\n",
            "    m = pattern.search(base_prompt)\n",
            "    if not m:\n",
            "        return [{'prompt': prompt_text, 'target': tgt, 'rot_idx': 0, 'rot_name': 'orig'}]\n",
            "    prefix = m.group(1)\n",
            "    opts = [m.group(2).strip(), m.group(3).strip(), m.group(4).strip(), m.group(5).strip()]\n",
            "    letters = ['a', 'b', 'c', 'd']\n",
            "    if tgt not in letters:\n",
            "        return [{'prompt': prompt_text, 'target': tgt, 'rot_idx': 0, 'rot_name': 'orig'}]\n",
            "    t_idx = letters.index(tgt)\n",
            "    rot_names = ['ABCD', 'BCDA', 'CDAB', 'DABC']\n",
            "    permutations = []\n",
            "    for k in range(4):\n",
            "        rot_opts = [opts[(j + k) % 4] for j in range(4)]\n",
            "        rot_target = letters[(t_idx - k) % 4]\n",
            "        rot_prompt = f'{prefix}(a) {rot_opts[0]} (b) {rot_opts[1]} (c) {rot_opts[2]} (d) {rot_opts[3]}{suffix}'\n",
            "        permutations.append({\n",
            "            'prompt': rot_prompt,\n",
            "            'target': rot_target,\n",
            "            'rot_idx': k,\n",
            "            'rot_name': rot_names[k]\n",
            "        })\n",
            "    return permutations\n",
            "\n",
            "# --- Academic Choice Logit Evaluator with Rotation Support ---\n",
            "def eval_choice_logit_single(model, tokenizer, prompt_text: str, target_text: str, year: int = None, te_mode: str = 'none', te_scale_mode: str = 'raw'):\n",
            "    prompt = f'Question: {prompt_text.strip()}\\nAnswer:' if not prompt_text.strip().endswith('Answer:') else prompt_text.strip()\n",
            "    inputs = tokenizer(prompt, return_tensors='pt').to(device)\n",
            "    with torch.no_grad():\n",
            "        if te_mode != 'none' and year is not None:\n",
            "            y_t = torch.tensor([year], dtype=torch.long, device=device)\n",
            "            inputs_embeds = inject_te_embeddings(model, tokenizer, inputs['input_ids'], y_t, te_mode=te_mode, scale_mode=te_scale_mode, device=device)\n",
            "            out = model(inputs_embeds=inputs_embeds, attention_mask=inputs['attention_mask'])\n",
            "        else:\n",
            "            out = model(**inputs)\n",
            "        logits = out.logits[0, -1, :]\n",
            "    tgt = target_text.strip().lower()\n",
            "    if tgt in ['yes', 'no']:\n",
            "        scores = {\n",
            "            'yes': max(logits[7414].item(), logits[9454].item(), logits[9834].item(), logits[9693].item()),\n",
            "            'no': max(logits[2308].item(), logits[2753].item(), logits[902].item(), logits[2152].item()),\n",
            "        }\n",
            "        pred = 'yes' if scores['yes'] >= scores['no'] else 'no'\n",
            "    else:\n",
            "        scores = {\n",
            "            'a': max(logits[264].item(), logits[362].item(), logits[64].item(), logits[32].item()),\n",
            "            'b': max(logits[293].item(), logits[425].item(), logits[65].item(), logits[33].item()),\n",
            "            'c': max(logits[272].item(), logits[356].item(), logits[66].item(), logits[34].item()),\n",
            "            'd': max(logits[294].item(), logits[422].item(), logits[67].item(), logits[35].item()),\n",
            "        }\n",
            "        pred = max(scores, key=scores.get)\n",
            "    top1_id = logits.argmax().item()\n",
            "    top1_str = tokenizer.decode([top1_id]).strip()\n",
            "    is_correct = int(pred == tgt)\n",
            "    greedy_is_correct = int(top1_str.lower() == tgt)\n",
            "    return {\n",
            "        'prediction': pred,\n",
            "        'is_valid': 1,\n",
            "        'is_correct': is_correct,\n",
            "        'greedy_top1': top1_str,\n",
            "        'greedy_is_correct': greedy_is_correct,\n",
            "        'scores': {k: round(v, 4) for k, v in scores.items()}\n",
            "    }\n",
            "\n",
            "def evaluate_dataset(model, tokenizer, eval_cases, te_mode='none', te_scale_mode='raw', debias=True):\n",
            "    model.eval()\n",
            "    total = len(eval_cases)\n",
            "    case_results = {}\n",
            "    canonical_correct = 0\n",
            "    debiased_mean_correct_sum = 0.0\n",
            "    debiased_strict_correct = 0\n",
            "    year_stats = defaultdict(lambda: {'total': 0, 'canonical_correct': 0, 'debiased_mean_sum': 0.0, 'debiased_strict': 0})\n",
            "    canonical_pred_counts = Counter()\n",
            "    rot_pred_counts = defaultdict(Counter)\n",
            "    rot_correct_counts = defaultdict(lambda: {'total': 0, 'correct': 0})\n",
            "    \n",
            "    for c in eval_cases:\n",
            "        yr = int(c['year'])\n",
            "        perms = generate_option_permutations(c['input'], c['target']) if debias else [{'prompt': c['input'], 'target': c['target'], 'rot_idx': 0, 'rot_name': 'orig'}]\n",
            "        perm_results = []\n",
            "        for p in perms:\n",
            "            res = eval_choice_logit_single(model, tokenizer, p['prompt'], p['target'], year=yr, te_mode=te_mode, te_scale_mode=te_scale_mode)\n",
            "            perm_results.append({\n",
            "                'rot_idx': p['rot_idx'],\n",
            "                'rot_name': p['rot_name'],\n",
            "                'target': p['target'],\n",
            "                'prediction': res['prediction'],\n",
            "                'is_correct': res['is_correct'],\n",
            "                'greedy_top1': res['greedy_top1'],\n",
            "                'scores': res['scores']\n",
            "            })\n",
            "            rot_pred_counts[p['rot_name']][res['prediction']] += 1\n",
            "            rot_correct_counts[p['rot_name']]['total'] += 1\n",
            "            rot_correct_counts[p['rot_name']]['correct'] += res['is_correct']\n",
            "        \n",
            "        canonical_res = perm_results[0]\n",
            "        canonical_is_corr = canonical_res['is_correct']\n",
            "        canonical_pred = canonical_res['prediction']\n",
            "        canonical_correct += canonical_is_corr\n",
            "        canonical_pred_counts[canonical_pred] += 1\n",
            "        \n",
            "        sample_mean_corr = sum(r['is_correct'] for r in perm_results) / float(len(perm_results))\n",
            "        sample_strict_corr = int(all(r['is_correct'] for r in perm_results))\n",
            "        debiased_mean_correct_sum += sample_mean_corr\n",
            "        debiased_strict_correct += sample_strict_corr\n",
            "        \n",
            "        year_stats[yr]['total'] += 1\n",
            "        year_stats[yr]['canonical_correct'] += canonical_is_corr\n",
            "        year_stats[yr]['debiased_mean_sum'] += sample_mean_corr\n",
            "        year_stats[yr]['debiased_strict'] += sample_strict_corr\n",
            "        \n",
            "        case_results[c['example_id']] = {\n",
            "            'year': yr,\n",
            "            'target': c['target'],\n",
            "            'prediction': canonical_pred,\n",
            "            'is_correct': canonical_is_corr,\n",
            "            'greedy_top1': canonical_res['greedy_top1'],\n",
            "            'debiased_mean_acc': sample_mean_corr,\n",
            "            'is_consistent': sample_strict_corr,\n",
            "            'rotations': perm_results\n",
            "        }\n",
            "        \n",
            "    canonical_acc = (canonical_correct / float(total)) * 100.0\n",
            "    debiased_mean_acc = (debiased_mean_correct_sum / float(total)) * 100.0\n",
            "    debiased_strict_acc = (debiased_strict_correct / float(total)) * 100.0\n",
            "    \n",
            "    by_year = {}\n",
            "    for yr, st in sorted(year_stats.items()):\n",
            "        cnt = st['total']\n",
            "        by_year[yr] = {\n",
            "            'canonical_acc': (st['canonical_correct'] / float(cnt)) * 100.0 if cnt > 0 else 0.0,\n",
            "            'debiased_mean_acc': (st['debiased_mean_sum'] / float(cnt)) * 100.0 if cnt > 0 else 0.0,\n",
            "            'debiased_strict_acc': (st['debiased_strict'] / float(cnt)) * 100.0 if cnt > 0 else 0.0,\n",
            "        }\n",
            "        \n",
            "    rot_accuracies = {}\n",
            "    for rname, rdata in rot_correct_counts.items():\n",
            "        rot_accuracies[rname] = (rdata['correct'] / float(rdata['total'])) * 100.0 if rdata['total'] > 0 else 0.0\n",
            "        \n",
            "    return {\n",
            "        'overall_acc': canonical_acc,\n",
            "        'canonical_acc': canonical_acc,\n",
            "        'debiased_mean_acc': debiased_mean_acc,\n",
            "        'debiased_strict_acc': debiased_strict_acc,\n",
            "        'invalid_rate': 0.0,\n",
            "        'by_year': by_year,\n",
            "        'case_results': case_results,\n",
            "        'pred_counts': dict(canonical_pred_counts),\n",
            "        'rot_pred_counts': {k: dict(v) for k, v in rot_pred_counts.items()},\n",
            "        'rot_accuracies': rot_accuracies\n",
            "    }\n",
            "\n",
            "# --- Unified Configuration Training & Evaluation Runner ---\n",
            "EPOCHS = 2\n",
            "BATCH_SIZE = 8\n",
            "GRAD_ACCUM = 2\n",
            "LR = 1e-4\n",
            "WEIGHT_DECAY = 0.01\n",
            "SEEDS = [1, 2, 3]\n",
            "\n",
            "def train_and_eval_configuration(cfg_name, te_mode='broadcast', te_scale_mode='raw', sigma=0.5, seeds=SEEDS):\n",
            "    results = []\n",
            "    pad_id = tokenizer.pad_token_id\n",
            "    print('=' * 60)\n",
            "    print(f'RUNNING: {cfg_name.upper()} (te_mode={te_mode}, scale={te_scale_mode}, sigma={sigma})')\n",
            "    print('=' * 60)\n",
            "    for seed_idx, seed in enumerate(seeds):\n",
            "        print(f'\\n--- Training {cfg_name}: Seed {seed} ({seed_idx+1}/{len(seeds)}) ---')\n",
            "        random.seed(seed)\n",
            "        np.random.seed(seed)\n",
            "        torch.manual_seed(seed)\n",
            "        if torch.cuda.is_available():\n",
            "            torch.cuda.manual_seed_all(seed)\n",
            "        model, adapted = load_lora_model(rank=8, alpha=16.0)\n",
            "        optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=LR, weight_decay=WEIGHT_DECAY)\n",
            "        ds = JointNWPDataset(train_cases, tokenizer)\n",
            "        loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True, collate_fn=lambda b: collate_joint_fn(b, pad_id))\n",
            "        model.train()\n",
            "        t_start = time.time()\n",
            "        for epoch in range(EPOCHS):\n",
            "            epoch_loss, epoch_ntp, epoch_intra, step_cnt = 0.0, 0.0, 0.0, 0\n",
            "            optimizer.zero_grad()\n",
            "            for step, batch in enumerate(loader):\n",
            "                b_in = batch['input_ids'].to(device)\n",
            "                b_lbl = batch['labels'].to(device)\n",
            "                b_msk = batch['attention_mask'].to(device)\n",
            "                b_yr = batch['year'].to(device)\n",
            "                if te_mode != 'none':\n",
            "                    inputs_embeds = inject_te_embeddings(model, tokenizer, b_in, b_yr, te_mode=te_mode, scale_mode=te_scale_mode, device=device)\n",
            "                    out = model(inputs_embeds=inputs_embeds, labels=b_lbl, attention_mask=b_msk, output_hidden_states=True)\n",
            "                else:\n",
            "                    out = model(input_ids=b_in, labels=b_lbl, attention_mask=b_msk, output_hidden_states=True)\n",
            "                loss_ntp = out.loss\n",
            "                if sigma > 0.0:\n",
            "                    seq_lengths = b_msk.sum(dim=1) - 1\n",
            "                    last_hidden = out.hidden_states[-1][torch.arange(b_in.size(0), device=device), seq_lengths, :]\n",
            "                    loss_intra = compute_intra_loss(last_hidden, b_yr)\n",
            "                    loss_total = loss_ntp + sigma * loss_intra\n",
            "                else:\n",
            "                    loss_intra = torch.tensor(0.0, device=device)\n",
            "                    loss_total = loss_ntp\n",
            "                (loss_total / GRAD_ACCUM).backward()\n",
            "                epoch_loss += loss_total.item()\n",
            "                epoch_ntp += loss_ntp.item()\n",
            "                epoch_intra += loss_intra.item()\n",
            "                step_cnt += 1\n",
            "                if (step + 1) % GRAD_ACCUM == 0 or (step + 1) == len(loader):\n",
            "                    optimizer.step()\n",
            "                    optimizer.zero_grad()\n",
            "            avg_loss = epoch_loss / step_cnt\n",
            "            avg_ntp = epoch_ntp / step_cnt\n",
            "            avg_intra = epoch_intra / step_cnt\n",
            "            print(f'  Epoch {epoch+1}/{EPOCHS} - Total: {avg_loss:.4f} (NTP: {avg_ntp:.4f}, Intra: {avg_intra:.4f})')\n",
            "        res = evaluate_dataset(model, tokenizer, eval_cases, te_mode=te_mode, te_scale_mode=te_scale_mode, debias=True)\n",
            "        dur = time.time() - t_start\n",
            "        print(f'  Seed {seed} Result: Canonical ACC = {res[\"canonical_acc\"]:.2f}%, Debiased Mean ACC = {res[\"debiased_mean_acc\"]:.2f}%, Consistency = {res[\"debiased_strict_acc\"]:.2f}% (Time: {dur:.1f}s)')\n",
            "        results.append({\n",
            "            'seed': seed,\n",
            "            'overall_acc': res['overall_acc'],\n",
            "            'canonical_acc': res['canonical_acc'],\n",
            "            'debiased_mean_acc': res['debiased_mean_acc'],\n",
            "            'debiased_strict_acc': res['debiased_strict_acc'],\n",
            "            'by_year': res['by_year'],\n",
            "            'case_results': res['case_results'],\n",
            "            'pred_counts': res['pred_counts'],\n",
            "            'rot_pred_counts': res['rot_pred_counts'],\n",
            "            'rot_accuracies': res['rot_accuracies'],\n",
            "            'duration': dur\n",
            "        })\n",
            "        del model, adapted, optimizer\n",
            "        gc.collect()\n",
            "        if torch.cuda.is_available():\n",
            "            torch.cuda.empty_cache()\n",
            "    return results\n",
            "\n",
            "print('Model loaders, Debiasing Engine, and Unified Runner configured successfully.')\n"
        ]
    })

    # Cell 6: Run 0 - Base Model Zero-shot Evaluation (Anchor Baseline)
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "print('========================================================')\n",
            "print('CONFIGURATION 0: BASE MODEL ZERO-SHOT (ANCHOR BASELINE)')\n",
            "print('========================================================')\n",
            "\n",
            "t0_base = time.time()\n",
            "base_model = load_base_model()\n",
            "base_res = evaluate_dataset(base_model, tokenizer, eval_cases, te_mode='none', debias=True)\n",
            "base_time = time.time() - t0_base\n",
            "\n",
            "print(f'Base Model Evaluation Completed in {base_time:.2f}s:')\n",
            "print(f'  Canonical Accuracy:       {base_res[\"canonical_acc\"]:.2f}% ({sum(r[\"is_correct\"] for r in base_res[\"case_results\"].values())}/{len(eval_cases)})')\n",
            "print(f'  Debiased Mean Accuracy:   {base_res[\"debiased_mean_acc\"]:.2f}%')\n",
            "print(f'  Consistency (Strict Acc): {base_res[\"debiased_strict_acc\"]:.2f}%')\n",
            "print(f'  Canonical Prediction Dist: {base_res[\"pred_counts\"]}')\n",
            "\n",
            "del base_model\n",
            "gc.collect()\n",
            "if torch.cuda.is_available():\n",
            "    torch.cuda.empty_cache()\n"
        ]
    })

    # Cell 7: Run 1 - Control 1 (C1: Raw Broadcast, sigma = 0.5)
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "print('>>> Executing Control 1: Raw Broadcast (Paper Formula, sigma = 0.5) <<<')\n",
            "c1_results = train_and_eval_configuration('C1_raw_broadcast', te_mode='broadcast', te_scale_mode='raw', sigma=0.5, seeds=SEEDS)\n"
        ]
    })

    # Cell 8: Run 2 - Control 2 (C2: Raw Date-token only, sigma = 0.5)
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "print('>>> Executing Control 2: Raw Date-token only (Localized Injection, sigma = 0.5) <<<')\n",
            "c2_results = train_and_eval_configuration('C2_raw_date_token', te_mode='date_token', te_scale_mode='raw', sigma=0.5, seeds=SEEDS)\n"
        ]
    })

    # Cell 9: Run 3 - Control 3 (C3: Raw with sigma = 0, pure NTP)
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "print('>>> Executing Control 3: Raw with sigma = 0 (Pure NTP, no intra loss) <<<')\n",
            "c3_results = train_and_eval_configuration('C3_raw_sigma0', te_mode='broadcast', te_scale_mode='raw', sigma=0.0, seeds=SEEDS)\n"
        ]
    })

    # Cell 10: Run 4 - Control 4 (C4: PT + Intra loss, no TE, sigma = 0.5)
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "print('>>> Executing Control 4: PT + Intra loss (No TE, sigma = 0.5) <<<')\n",
            "c4_results = train_and_eval_configuration('C4_pt_intra_loss', te_mode='none', te_scale_mode='none', sigma=0.5, seeds=SEEDS)\n"
        ]
    })

    # Cell 11: Statistical Audit, Letter Bias Analysis, Paired McNemar Tests & JSON Export
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "print('========================================================')\n",
            "print('FINAL STATISTICAL AUDIT & COMPARATIVE CONTROL SUMMARY')\n",
            "print('========================================================')\n",
            "\n",
            "def compute_mcnemar_table(cases_a, cases_b):\n",
            "    b = sum(1 for cid in cases_a if cases_a[cid]['is_correct'] and not cases_b[cid]['is_correct'])\n",
            "    c = sum(1 for cid in cases_a if not cases_a[cid]['is_correct'] and cases_b[cid]['is_correct'])\n",
            "    a = sum(1 for cid in cases_a if cases_a[cid]['is_correct'] and cases_b[cid]['is_correct'])\n",
            "    d = sum(1 for cid in cases_a if not cases_a[cid]['is_correct'] and not cases_b[cid]['is_correct'])\n",
            "    diff = abs(b - c)\n",
            "    chi2_edwards = ((diff - 1.0) ** 2) / (b + c) if (b + c) > 0 and diff >= 1 else 0.0\n",
            "    p_edwards = stats.chi2.sf(chi2_edwards, df=1) if chi2_edwards > 0 else 1.0\n",
            "    p_exact = stats.binomtest(min(b, c), b + c, 0.5).pvalue if (b + c) > 0 else 1.0\n",
            "    return {\n",
            "        'contingency': {'a_both_correct': a, 'b_only_a': b, 'c_only_b': c, 'd_both_incorrect': d},\n",
            "        'chi2_edwards': float(chi2_edwards),\n",
            "        'p_value_edwards': float(p_edwards),\n",
            "        'p_value_exact': float(p_exact)\n",
            "    }\n",
            "\n",
            "def get_metrics_summary(results_list):\n",
            "    c_accs = [r['canonical_acc'] for r in results_list]\n",
            "    d_accs = [r['debiased_mean_acc'] for r in results_list]\n",
            "    s_accs = [r['debiased_strict_acc'] for r in results_list]\n",
            "    return {\n",
            "        'canonical_mean': float(np.mean(c_accs)),\n",
            "        'canonical_std': float(np.std(c_accs, ddof=1)),\n",
            "        'debiased_mean': float(np.mean(d_accs)),\n",
            "        'debiased_mean_std': float(np.std(d_accs, ddof=1)),\n",
            "        'consistency_mean': float(np.mean(s_accs)),\n",
            "        'consistency_std': float(np.std(s_accs, ddof=1)),\n",
            "        'seeds_canonical': c_accs,\n",
            "        'seeds_debiased': d_accs,\n",
            "        'seeds_consistency': s_accs,\n",
            "    }\n",
            "\n",
            "summary = {\n",
            "    'base': {\n",
            "        'canonical_acc': float(base_res['canonical_acc']),\n",
            "        'debiased_mean_acc': float(base_res['debiased_mean_acc']),\n",
            "        'consistency': float(base_res['debiased_strict_acc']),\n",
            "        'by_year': {str(k): float(v['canonical_acc']) for k, v in base_res['by_year'].items()},\n",
            "    },\n",
            "    'c1_raw_broadcast': get_metrics_summary(c1_results),\n",
            "    'c2_raw_date_token': get_metrics_summary(c2_results),\n",
            "    'c3_raw_sigma0': get_metrics_summary(c3_results),\n",
            "    'c4_pt_intra_loss': get_metrics_summary(c4_results),\n",
            "}\n",
            "\n",
            "print('--- SUMMARY OF 4-WAY CONTROL MATRIX BENCHMARK ---')\n",
            "print(f\"Base (Zero-shot):       Canonical = {summary['base']['canonical_acc']:.2f}%, Debiased = {summary['base']['debiased_mean_acc']:.2f}%, Consistency = {summary['base']['consistency']:.2f}%\")\n",
            "print(f\"C1 (Raw Broadcast):     Canonical = {summary['c1_raw_broadcast']['canonical_mean']:.2f}±{summary['c1_raw_broadcast']['canonical_std']:.2f}%, Debiased = {summary['c1_raw_broadcast']['debiased_mean']:.2f}±{summary['c1_raw_broadcast']['debiased_mean_std']:.2f}%, Consistency = {summary['c1_raw_broadcast']['consistency_mean']:.2f}±{summary['c1_raw_broadcast']['consistency_std']:.2f}%\")\n",
            "print(f\"C2 (Date-token only):   Canonical = {summary['c2_raw_date_token']['canonical_mean']:.2f}±{summary['c2_raw_date_token']['canonical_std']:.2f}%, Debiased = {summary['c2_raw_date_token']['debiased_mean']:.2f}±{summary['c2_raw_date_token']['debiased_mean_std']:.2f}%, Consistency = {summary['c2_raw_date_token']['consistency_mean']:.2f}±{summary['c2_raw_date_token']['consistency_std']:.2f}%\")\n",
            "print(f\"C3 (Raw sigma=0):       Canonical = {summary['c3_raw_sigma0']['canonical_mean']:.2f}±{summary['c3_raw_sigma0']['canonical_std']:.2f}%, Debiased = {summary['c3_raw_sigma0']['debiased_mean']:.2f}±{summary['c3_raw_sigma0']['debiased_mean_std']:.2f}%, Consistency = {summary['c3_raw_sigma0']['consistency_mean']:.2f}±{summary['c3_raw_sigma0']['consistency_std']:.2f}%\")\n",
            "print(f\"C4 (PT + Intra loss):   Canonical = {summary['c4_pt_intra_loss']['canonical_mean']:.2f}±{summary['c4_pt_intra_loss']['canonical_std']:.2f}%, Debiased = {summary['c4_pt_intra_loss']['debiased_mean']:.2f}±{summary['c4_pt_intra_loss']['debiased_mean_std']:.2f}%, Consistency = {summary['c4_pt_intra_loss']['consistency_mean']:.2f}±{summary['c4_pt_intra_loss']['consistency_std']:.2f}%\")\n",
            "\n",
            "# Paired McNemar Tests across Seed 1\n",
            "mcnemar_tests = {\n",
            "    'c1_vs_base': compute_mcnemar_table(c1_results[0]['case_results'], base_res['case_results']),\n",
            "    'c2_vs_base': compute_mcnemar_table(c2_results[0]['case_results'], base_res['case_results']),\n",
            "    'c2_vs_c1':   compute_mcnemar_table(c2_results[0]['case_results'], c1_results[0]['case_results']),\n",
            "    'c3_vs_c1':   compute_mcnemar_table(c3_results[0]['case_results'], c1_results[0]['case_results']),\n",
            "    'c4_vs_base': compute_mcnemar_table(c4_results[0]['case_results'], base_res['case_results']),\n",
            "    'c4_vs_c1':   compute_mcnemar_table(c4_results[0]['case_results'], c1_results[0]['case_results']),\n",
            "}\n",
            "\n",
            "print('\\n--- PAIRED MCNEMAR AUDIT (SEED 1) ---')\n",
            "for k, v in mcnemar_tests.items():\n",
            "    print(f\"{k}: Contingency={v['contingency']}, Chi2={v['chi2_edwards']:.3f}, p_edwards={v['p_value_edwards']:.4e}, p_exact={v['p_value_exact']:.4e}\")\n",
            "\n",
            "def compute_pooled_mcnemar_table(runs_a, runs_b):\n",
            "    b, c, a, d = 0, 0, 0, 0\n",
            "    for r_a, r_b in zip(runs_a, runs_b):\n",
            "        cases_a = r_a['case_results']\n",
            "        cases_b = r_b['case_results']\n",
            "        for cid in cases_a:\n",
            "            ca = bool(cases_a[cid]['is_correct'])\n",
            "            cb = bool(cases_b[cid]['is_correct'])\n",
            "            if ca and not cb: b += 1\n",
            "            elif not ca and cb: c += 1\n",
            "            elif ca and cb: a += 1\n",
            "            else: d += 1\n",
            "    diff = abs(b - c)\n",
            "    chi2_edwards = ((diff - 1.0) ** 2) / (b + c) if (b + c) > 0 and diff >= 1 else 0.0\n",
            "    p_edwards = stats.chi2.sf(chi2_edwards, df=1) if chi2_edwards > 0 else 1.0\n",
            "    p_exact = stats.binomtest(min(b, c), b + c, 0.5).pvalue if (b + c) > 0 else 1.0\n",
            "    return {\n",
            "        'contingency': {'a_both_correct': a, 'b_only_a': b, 'c_only_b': c, 'd_both_incorrect': d},\n",
            "        'chi2_edwards': float(chi2_edwards),\n",
            "        'p_value_edwards': float(p_edwards),\n",
            "        'p_value_exact': float(p_exact)\n",
            "    }\n",
            "\n",
            "base_runs_mock = [{'case_results': base_res['case_results']} for _ in range(len(SEEDS))]\n",
            "pooled_mcnemar_tests = {\n",
            "    'c1_vs_base_pooled': compute_pooled_mcnemar_table(c1_results, base_runs_mock),\n",
            "    'c2_vs_base_pooled': compute_pooled_mcnemar_table(c2_results, base_runs_mock),\n",
            "    'c2_vs_c1_pooled':   compute_pooled_mcnemar_table(c2_results, c1_results),\n",
            "    'c3_vs_c1_pooled':   compute_pooled_mcnemar_table(c3_results, c1_results),\n",
            "    'c4_vs_base_pooled': compute_pooled_mcnemar_table(c4_results, base_runs_mock),\n",
            "    'c4_vs_c1_pooled':   compute_pooled_mcnemar_table(c4_results, c1_results),\n",
            "}\n",
            "\n",
            "print('\\n--- POOLED PAIRED MCNEMAR AUDIT (ALL 3 SEEDS, N=540) ---')\n",
            "for k, v in pooled_mcnemar_tests.items():\n",
            "    print(f\"{k}: Contingency={v['contingency']}, Chi2={v['chi2_edwards']:.3f}, p_edwards={v['p_value_edwards']:.4e}, p_exact={v['p_value_exact']:.4e}\")\n",
            "\n",
            "# Letter Bias Audit: Percentage of 'a' predictions on MCQA\n",
            "letter_bias_audit = {}\n",
            "for name, runs in [('C1_raw_broadcast', c1_results), ('C2_raw_date_token', c2_results), ('C3_raw_sigma0', c3_results), ('C4_pt_intra_loss', c4_results)]:\n",
            "    pct_a = [r['pred_counts'].get('a', 0) / float(len(eval_cases)) * 100.0 for r in runs]\n",
            "    letter_bias_audit[name] = {'pct_a_mean': float(np.mean(pct_a)), 'pct_a_seeds': pct_a}\n",
            "letter_bias_audit['base'] = {'pct_a': base_res['pred_counts'].get('a', 0) / float(len(eval_cases)) * 100.0}\n",
            "print('\\n--- LETTER BIAS AUDIT (% PREDICTED OPTION \"A\") ---')\n",
            "for k, v in letter_bias_audit.items():\n",
            "    print(f\"{k}: {v}\")\n",
            "\n",
            "final_export = {\n",
            "    'metadata': {\n",
            "        'model': MODEL_ID,\n",
            "        'benchmark': 'Ticktack 4-Way Disentanglement Control Matrix + Debiased Permutations',\n",
            "        'evaluation_protocol': 'Choice Logit Scoring with 4-fold Permutation Rotation (Letter Bias Debiasing)',\n",
            "        'seeds': SEEDS,\n",
            "        'epochs': EPOCHS,\n",
            "        'batch_size': BATCH_SIZE,\n",
            "        'learning_rate': LR,\n",
            "        'sigma': 0.5\n",
            "    },\n",
            "    'summary': summary,\n",
            "    'mcnemar_audit': mcnemar_tests,\n",
            "    'pooled_mcnemar_audit': pooled_mcnemar_tests,\n",
            "    'letter_bias_audit': letter_bias_audit,\n",
            "    'raw_runs': {\n",
            "        'base': base_res,\n",
            "        'c1_raw_broadcast': c1_results,\n",
            "        'c2_raw_date_token': c2_results,\n",
            "        'c3_raw_sigma0': c3_results,\n",
            "        'c4_pt_intra_loss': c4_results\n",
            "    }\n",
            "}\n",
            "\n",
            "out_file = 'TICKTACK_CONTROL_MATRIX_RESULTS.json'\n",
            "with open(out_file, 'w', encoding='utf-8') as f:\n",
            "    json.dump(final_export, f, indent=2, ensure_ascii=False)\n",
            "print(f'\\n[DONE] Successfully saved full benchmark results to {out_file}!')\n"
        ]
    })

    notebook = {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "codemirror_mode": {"name": "ipython", "version": 3},
                "file_extension": ".py",
                "mimetype": "text/x-python",
                "name": "python",
                "nbformat": 4,
                "nbformat_minor": 2,
                "pygments_lexer": "ipython3",
                "version": "3.10.12"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 2
    }

    target_nb_path = os.path.join(output_dir, "ticktack_paper_rebuild.ipynb")
    with open(target_nb_path, "w", encoding="utf-8") as f:
        json.dump(notebook, f, indent=2, ensure_ascii=False)
    print(f"Generated notebook at {target_nb_path} ({len(cells)} cells).")

    if extra_output_dir:
        extra_nb_path = os.path.join(extra_output_dir, "ticktack_paper_rebuild.ipynb")
        with open(extra_nb_path, "w", encoding="utf-8") as f:
            json.dump(notebook, f, indent=2, ensure_ascii=False)
        print(f"Synchronized notebook to {extra_nb_path}.")

    metadata = {
        "id": "tnghuy11/ticktack-paper-rebuild-baseline",
        "title": "ticktack-paper-rebuild-baseline",
        "code_file": "ticktack_paper_rebuild.ipynb",
        "language": "python",
        "kernel_type": "notebook",
        "is_private": "true",
        "enable_gpu": "true",
        "enable_tpu": "false",
        "enable_internet": "true",
        "dataset_sources": [],
        "competition_sources": [],
        "kernel_sources": [],
        "model_sources": []
    }
    target_meta_path = os.path.join(output_dir, "kernel-metadata.json")
    with open(target_meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    print(f"Generated kernel metadata at {target_meta_path}.")

    return target_nb_path

if __name__ == "__main__":
    create_kaggle_notebook()
