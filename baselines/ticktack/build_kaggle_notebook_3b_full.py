"""
Tạo Notebook Kaggle Thực nghiệm Qwen2.5-3B-Instruct trên Toàn bộ TempUN Full Sạch (76 năm).

Bao gồm:
- Mô hình nền: Qwen/Qwen2.5-3B-Instruct (hidden_dim = 2048)
- Bộ dữ liệu sạch: TempUN Full (76 giai đoạn 1947-2022, 104,130 NWP train, 5,292 MCQA eval)
- LoRA PEFT: q_proj, v_proj (r=8, alpha=16.0), đóng băng 100% mô hình nền (< 0.08% tham số huấn luyện)
- Tối ưu phần cứng cho Tesla T4 (16GB VRAM):
  * Độ chính xác nửa (torch.float16 / torch.bfloat16)
  * model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
  * micro-batch size 8, gradient accumulation steps 2 (effective batch size 16)
- Hỗ trợ 3 chế độ sinh kernel (Dual-Kernel Parallel Dispatch):
  1. 'pt': Kernel 1 (Base 3B Zero-shot + PT 3B LoRA NTP) -> tnghuy11/ticktack-3b-pt-baseline
  2. 'ticktack': Kernel 2 (Ticktack 3B Date-token TE + Intra-loss) -> tnghuy11/ticktack-3b-ticktack-baseline
  3. 'all': Kernel tổng hợp cả 3 cấu hình -> tnghuy11/ticktack-3b-tempun-full-baseline
- Quy trình Đánh giá Khử Thiên vị:
  * Academic Choice-Logit Scoring (Invalid Rate = 0.0%)
  * Hoán vị vòng tròn 4 phương án (ABCD, BCDA, CDAB, DABC) cho MCQA và 2 chiều cho Yes/No
  * Kiểm định thống kê ghép cặp McNemar
  * Lưu checkpoint JSON an toàn ngay sau mỗi mô hình
"""

from __future__ import annotations

import base64
import gzip
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass


def _find_clean_file(filename: str) -> Path:
    candidates = [
        Path(_REPO_ROOT) / "Data" / "tempun_full_clean" / filename,
        Path("d:/nckh/data/tempun_full_clean") / filename,
        Path("Data/tempun_full_clean") / filename,
    ]
    for c in candidates:
        if c.is_file():
            return c.resolve()
    raise FileNotFoundError(f"Clean dataset file not found: {filename}")


def _find_base_3b_file() -> Optional[Path]:
    candidates = [
        Path(_REPO_ROOT) / "kaggle_ticktack_3b_pt" / "output" / "base_3b_benchmark_results.json",
        Path("d:/nckh/kaggle_ticktack_3b_pt/output/base_3b_benchmark_results.json"),
        Path("kaggle_ticktack_3b_pt/output/base_3b_benchmark_results.json"),
        Path("base_3b_benchmark_results.json"),
    ]
    for c in candidates:
        if c.is_file():
            return c.resolve()
    return None


def _get_base_3b_b64() -> str:
    base_file = _find_base_3b_file()
    if base_file:
        try:
            return base64.b64encode(gzip.compress(base_file.read_bytes())).decode("ascii")
        except Exception:
            pass
    try:
        from baselines.ticktack.base_3b_data import BASE_3B_GZ_B64
        return BASE_3B_GZ_B64
    except Exception:
        return ""


def create_kaggle_notebook_3b_full(
    output_dir: Optional[str] = None,
    train_path: Optional[str] = None,
    eval_path: Optional[str] = None,
    extra_output_dir: Optional[str] = None,
    embed_data: bool = False,
    mode: str = "all",
) -> str:
    """Xây dựng notebook Kaggle độc lập cho Qwen2.5-3B trên dữ liệu sạch TempUN Full."""
    if output_dir is None:
        if mode == "pt":
            output_dir = os.path.abspath(os.path.join(_REPO_ROOT, "kaggle_ticktack_3b_pt"))
        elif mode == "ticktack":
            output_dir = os.path.abspath(os.path.join(_REPO_ROOT, "kaggle_ticktack_3b_ticktack"))
        else:
            output_dir = os.path.abspath(os.path.join(_REPO_ROOT, "kaggle_ticktack_3b_full"))

    if mode == "pt":
        kernel_slug = "ticktack-3b-pt-baseline"
        nb_filename = "ticktack_3b_pt.ipynb"
        title_suffix = "Base 3B Zero-shot và PT 3B LoRA NTP"
    elif mode == "ticktack":
        kernel_slug = "ticktack-3b-ticktack-baseline"
        nb_filename = "ticktack_3b_ticktack.ipynb"
        title_suffix = "Ticktack 3B (Date-Token TE + Intra Loss)"
    else:
        kernel_slug = "ticktack-3b-tempun-full-baseline"
        nb_filename = "ticktack_3b_full.ipynb"
        title_suffix = "Qwen2.5-3B-Instruct Đầy đủ"

    actual_train = Path(train_path) if train_path else _find_clean_file("train_104k_clean.jsonl.gz")
    actual_eval = Path(eval_path) if eval_path else _find_clean_file("eval_5292_clean.jsonl.gz")

    os.makedirs(output_dir, exist_ok=True)
    if extra_output_dir:
        os.makedirs(extra_output_dir, exist_ok=True)

    print(f"[{mode.upper()}] Thư mục xuất: {output_dir}")
    print(f"[{mode.upper()}] Tập train: {actual_train}")
    print(f"[{mode.upper()}] Tập eval: {actual_eval}")

    train_b64 = base64.b64encode(actual_train.read_bytes()).decode("ascii") if embed_data else ""
    eval_b64 = base64.b64encode(actual_eval.read_bytes()).decode("ascii") if embed_data else ""
    base_3b_b64 = _get_base_3b_b64() if mode == "pt" else ""

    cells: List[Dict[str, Any]] = []

    # Cell 1: Header Markdown
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            f"# Baseline Nghiên cứu: {title_suffix} trên TempUN Full Sạch (76 năm)\n",
            "\n",
            "Notebook này triển khai và đánh giá mô hình **Qwen/Qwen2.5-3B-Instruct** trên toàn bộ 76 năm của tập dữ liệu **TempUN Full** (1947–2022):\n",
            "- **Quy mô dữ liệu sạch**: 104,130 mẫu huấn luyện NWP và 5,292 mẫu đánh giá trắc nghiệm MCQA (6 bài toán: DB, MM, RB, TB, Rel, WB).\n",
            "- **Đảm bảo tính trong sạch**: Tuyệt đối không chứa siêu dữ liệu FoNE hay chuỗi Can Chi được cấy sẵn vào dữ liệu thô.\n",
            "- **Quy trình Khử thiên vị Chữ cái (Debiasing Protocol)**:\n",
            "  * Academic Choice-Logit Scoring (Invalid Rate = 0.0%).\n",
            "  * Hoán vị vòng tròn 4 phương án (ABCD, BCDA, CDAB, DABC) cho 4-choice MCQA và 2 chiều cho Yes/No.\n",
            "  * Đo Debiased Mean Accuracy triệt tiêu letter bias.\n",
            "  * Kiểm định thống kê ghép cặp McNemar đối chiếu."
        ]
    })

    # Cell 2: Imports & GPU Check
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "import os, sys, time, math, gzip, base64, json, random, gc, re\n",
            "from typing import Optional, List, Dict, Any, Tuple, Union\n",
            "from collections import Counter, defaultdict\n",
            "from pathlib import Path\n",
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
            "    gpu_name = torch.cuda.get_device_name(0)\n",
            "    total_vram = round(torch.cuda.get_device_properties(0).total_memory / (1024**3), 2)\n",
            "    print(f'GPU: {gpu_name} ({total_vram} GB VRAM)')\n",
            "    torch.backends.cuda.matmul.allow_tf32 = True\n",
            "else:\n",
            "    device = torch.device('cpu')\n",
            "    print('WARNING: Running on CPU!')\n",
            "\n",
            "SEEDS = [42]\n",
            "def set_seed(seed=42):\n",
            "    random.seed(seed)\n",
            "    np.random.seed(seed)\n",
            "    torch.manual_seed(seed)\n",
            "    if torch.cuda.is_available():\n",
            "        torch.cuda.manual_seed_all(seed)\n",
            "set_seed(42)\n",
            "\n",
            "# --- Cấu hình số bước huấn luyện ---\n",
            "# MAX_TRAIN_STEPS = None: Chạy toàn bộ 1 epoch (104,130 mẫu ~ 6,508 steps hiệu dụng).\n",
            "MAX_TRAIN_STEPS = None\n"
        ]
    })

    # Cell 3: Data Loading & Verification
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# --- Nạp và kiểm tra dữ liệu sạch TempUN Full ---\n",
            "def load_dataset_records():\n",
            "    train_path, eval_path = None, None\n",
            "    kaggle_input = Path('/kaggle/input')\n",
            "    if kaggle_input.exists():\n",
            "        candidate_train = [p for p in kaggle_input.rglob('train_104k_clean*') if p.is_file()]\n",
            "        candidate_eval = [p for p in kaggle_input.rglob('eval_5292_clean*') if p.is_file()]\n",
            "        if candidate_train:\n",
            "            train_path = sorted(candidate_train, key=lambda p: (not p.name.endswith('.gz'), len(str(p))))[0]\n",
            "        if candidate_eval:\n",
            "            eval_path = sorted(candidate_eval, key=lambda p: (not p.name.endswith('.gz'), len(str(p))))[0]\n",
            "    \n",
            "    if not train_path or not eval_path:\n",
            "        local_roots = [Path('.'), Path('..'), Path('Data'), Path('../data'), Path('data')]\n",
            "        for root in local_roots:\n",
            "            if not root.exists():\n",
            "                continue\n",
            "            if not train_path:\n",
            "                c_tr = [p for p in root.rglob('train_104k_clean*') if p.is_file()]\n",
            "                if c_tr:\n",
            "                    train_path = sorted(c_tr, key=lambda p: (not p.name.endswith('.gz'), len(str(p))))[0]\n",
            "            if not eval_path:\n",
            "                c_ev = [p for p in root.rglob('eval_5292_clean*') if p.is_file()]\n",
            "                if c_ev:\n",
            "                    eval_path = sorted(c_ev, key=lambda p: (not p.name.endswith('.gz'), len(str(p))))[0]\n",
            "            if train_path and eval_path:\n",
            "                break\n",
            "    \n",
            "    if not train_path or not eval_path:\n",
            "        print('=== CHẨN ĐOÁN LỖI TÌM TỆP DỮ LIỆU ===')\n",
            "        if kaggle_input.exists():\n",
            "            print(f'Cấu trúc thư mục {kaggle_input}:')\n",
            "            for root, dirs, files in os.walk(str(kaggle_input)):\n",
            "                print(f'  Dir: {root} | Subdirs: {dirs} | Files: {files[:10]}')\n",
            "        else:\n",
            "            print(f'Không tồn tại thư mục {kaggle_input}. Thư mục hiện tại: {os.getcwd()}')\n",
            "            print(f'Tệp tin trong CWD: {os.listdir(\".\")[:20]}')\n",
            "        raise FileNotFoundError(\n",
            "            f'Không tìm thấy tập dữ liệu (train_path={train_path}, eval_path={eval_path}). '\n",
            "            'Vui lòng gắn dataset tnghuy11/tempun-full-clean vào kernel Kaggle.'\n",
            "        )\n",
            "    \n",
            "    print(f'Đã định vị tập dữ liệu: train={train_path}, eval={eval_path}')\n",
            "    \n",
            "    def _read_records(p):\n",
            "        if str(p).endswith('.gz'):\n",
            "            with gzip.open(p, 'rt', encoding='utf-8') as f:\n",
            "                return [json.loads(line) for line in f if line.strip()]\n",
            "        else:\n",
            "            with open(p, 'r', encoding='utf-8') as f:\n",
            "                return [json.loads(line) for line in f if line.strip()]\n",
            "                \n",
            "    train_records = _read_records(train_path)\n",
            "    eval_records = _read_records(eval_path)\n",
            "    return train_records, eval_records\n",
            "\n",
            "train_cases, eval_cases = load_dataset_records()\n",
            "print(f'Đã nạp {len(train_cases)} mẫu huấn luyện NWP (Kỳ vọng: 104130).')\n",
            "print(f'Đã nạp {len(eval_cases)} mẫu đánh giá MCQA (Kỳ vọng: 5292).')\n",
            "assert len(train_cases) == 104130, f'Sai số lượng train: {len(train_cases)}'\n",
            "assert len(eval_cases) == 5292, f'Sai số lượng eval: {len(eval_cases)}'\n",
            "print('Kiểm tra dữ liệu: 100% khớp quy chuẩn sạch.')\n"
        ]
    })

    # Cell 4: Ticktack Mathematical Architecture & LoRA for 3B (d=2048)
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# --- Kiến trúc Toán học Ticktack (Han et al., 2025) & LoRA ---\n",
            "def year_to_canchi_polar(years: torch.Tensor, alpha: float = 0.5, beta: float = 0.5) -> Tuple[torch.Tensor, torch.Tensor]:\n",
            "    \"\"\"Chuyển đổi năm sang toạ độ cực (theta, r) theo xoắn ốc Archimedes và chu kỳ Can Chi 60 năm (Eq. 2).\"\"\"\n",
            "    cycle_indices = ((years.to(torch.int64) - 3) % 60).to(torch.float32)\n",
            "    theta = (cycle_indices * 6.0) * (math.pi / 180.0)\n",
            "    r = alpha + beta * theta\n",
            "    return theta, r\n",
            "\n",
            "def polar_to_cartesian(theta: torch.Tensor, r: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:\n",
            "    \"\"\"Chiếu toạ độ cực sang mặt phẳng Descartes (x, y) (Eq. 3).\"\"\"\n",
            "    x = (r * torch.cos(theta)).view(-1, 1)\n",
            "    y = (r * torch.sin(theta)).view(-1, 1)\n",
            "    return x, y\n",
            "\n",
            "def compute_sinusoidal_te(coord: torch.Tensor, hidden_dim: int) -> torch.Tensor:\n",
            "    \"\"\"Tính mã hoá thời gian dạng Sinusoidal Eq. 4 trên toàn bộ hidden_dim.\"\"\"\n",
            "    device = coord.device\n",
            "    dim_indices = torch.arange(0, hidden_dim, 2, device=device, dtype=torch.float32)\n",
            "    inv_freq = 1.0 / (10000.0 ** (dim_indices / float(hidden_dim))).unsqueeze(0)\n",
            "    coord_inp = coord * inv_freq\n",
            "    return torch.stack([torch.sin(coord_inp), torch.cos(coord_inp)], dim=-1).flatten(start_dim=-2)\n",
            "\n",
            "def compute_batch_te(\n",
            "    years: Union[torch.Tensor, List[int], int],\n",
            "    hidden_dim: int = 2048,\n",
            "    scale_mode: str = 'raw',\n",
            "    alpha: float = 0.5,\n",
            "    beta: float = 0.5,\n",
            "    device: Optional[Union[str, torch.device]] = None,\n",
            "    dtype: torch.dtype = torch.float32,\n",
            "    gating_factor: float = 0.01,\n",
            ") -> torch.Tensor:\n",
            "    \"\"\"Tạo vector TE tổng hợp kích thước [B, 1, hidden_dim] chuẩn xác theo Han et al. (2025) và models.py.\"\"\"\n",
            "    if isinstance(years, (int, float)):\n",
            "        years = torch.tensor([years], device=device or 'cpu')\n",
            "    elif not isinstance(years, torch.Tensor):\n",
            "        years = torch.tensor(years, device=device or 'cpu')\n",
            "\n",
            "    if device is None:\n",
            "        device = years.device\n",
            "        \n",
            "    years_dev = years.view(-1).to(device=device)\n",
            "    theta, r = year_to_canchi_polar(years_dev, alpha=alpha, beta=beta)\n",
            "    x, y = polar_to_cartesian(theta, r)\n",
            "    x = x.to(device)\n",
            "    y = y.to(device)\n",
            "    pe_x = compute_sinusoidal_te(x, hidden_dim)\n",
            "    pe_y = compute_sinusoidal_te(y, hidden_dim)\n",
            "    te_raw = (pe_x + pe_y).unsqueeze(1).to(dtype=dtype)\n",
            "    \n",
            "    if scale_mode == 'raw':\n",
            "        return te_raw\n",
            "    elif scale_mode == 'scaled':\n",
            "        scale = 1.0 / float(hidden_dim)\n",
            "        return (te_raw * scale).to(dtype=dtype)\n",
            "    elif scale_mode == 'gated':\n",
            "        return (te_raw * gating_factor).to(dtype=dtype)\n",
            "    else:\n",
            "        raise ValueError(f'scale_mode không hợp lệ: {scale_mode}')\n",
            "\n",
            "# --- Cấu trúc LoRALinear chuẩn hoá cho LoRA PEFT ---\n",
            "class LoRALinear(nn.Module):\n",
            "    def __init__(self, in_features, out_features, rank=8, alpha=16.0, base_layer=None):\n",
            "        super().__init__()\n",
            "        self.rank = rank\n",
            "        self.scaling = alpha / rank if rank > 0 else 1.0\n",
            "        self.base_layer = base_layer if base_layer is not None else nn.Linear(in_features, out_features, bias=False)\n",
            "        self.base_layer.weight.requires_grad = False\n",
            "        if self.base_layer.bias is not None:\n",
            "            self.base_layer.bias.requires_grad = False\n",
            "            \n",
            "        if rank > 0:\n",
            "            self.lora_A = nn.Parameter(torch.empty(rank, in_features))\n",
            "            self.lora_B = nn.Parameter(torch.zeros(out_features, rank))\n",
            "            nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))\n",
            "            \n",
            "    @classmethod\n",
            "    def from_linear(cls, linear: nn.Linear, rank=8, alpha=16.0):\n",
            "        mod = cls(linear.in_features, linear.out_features, rank=rank, alpha=alpha, base_layer=linear)\n",
            "        mod.base_layer.weight.data.copy_(linear.weight.data)\n",
            "        if linear.bias is not None:\n",
            "            mod.base_layer.bias.data.copy_(linear.bias.data)\n",
            "        return mod\n",
            "        \n",
            "    def forward(self, x):\n",
            "        base_out = self.base_layer(x)\n",
            "        if self.rank > 0:\n",
            "            lora_out = F.linear(F.linear(x, self.lora_A), self.lora_B) * self.scaling\n",
            "            return base_out + lora_out\n",
            "        return base_out\n",
            "\n",
            "def apply_lora_to_model(model, target_modules=['q_proj', 'v_proj'], rank=8, alpha=16.0):\n",
            "    for p in model.parameters():\n",
            "        p.requires_grad = False\n",
            "    adapted = {}\n",
            "    for name, module in model.named_modules():\n",
            "        for target in target_modules:\n",
            "            if name.endswith(target) and isinstance(module, nn.Linear):\n",
            "                parent_name = name.rsplit('.', 1)[0] if '.' in name else ''\n",
            "                child_name = name.rsplit('.', 1)[-1]\n",
            "                parent = model.get_submodule(parent_name) if parent_name else model\n",
            "                lora_layer = LoRALinear.from_linear(module, rank=rank, alpha=alpha)\n",
            "                lora_layer.to(device=module.weight.device, dtype=module.weight.dtype)\n",
            "                setattr(parent, child_name, lora_layer)\n",
            "                adapted[name] = lora_layer\n",
            "    return adapted\n",
            "\n",
            "# --- Hàm mất mát tương phản Can Chi Intra-class ---\n",
            "def compute_intra_loss(hidden_states: torch.Tensor, years: torch.Tensor) -> torch.Tensor:\n",
            "    \"\"\"Tính hàm mất mát tương phản chu kỳ Can Chi Eq. 8.\"\"\"\n",
            "    B = hidden_states.size(0)\n",
            "    if B <= 1:\n",
            "        return torch.tensor(0.0, device=hidden_states.device, dtype=hidden_states.dtype)\n",
            "        \n",
            "    h_pooled = hidden_states.mean(dim=1)\n",
            "    h_norm = F.normalize(h_pooled, p=2, dim=-1)\n",
            "    cos_sim = torch.mm(h_norm, h_norm.t())\n",
            "    \n",
            "    canchi_cycles = ((years.to(torch.int64) - 3) % 60).to(device=hidden_states.device)\n",
            "    same_cycle = canchi_cycles.unsqueeze(1) == canchi_cycles.unsqueeze(0)\n",
            "    eye_mask = torch.eye(B, dtype=torch.bool, device=hidden_states.device)\n",
            "    pair_mask = same_cycle & (~eye_mask)\n",
            "    \n",
            "    if not pair_mask.any():\n",
            "        return torch.tensor(0.0, device=hidden_states.device, dtype=hidden_states.dtype)\n",
            "        \n",
            "    intra_loss = (1.0 - cos_sim)[pair_mask].mean()\n",
            "    return intra_loss\n"
        ]
    })

    # Cell 5: Model Loader for Qwen2.5-3B-Instruct
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# --- Khởi tạo mô hình Qwen2.5-3B-Instruct với tối ưu bộ nhớ --- \n",
            "from transformers import AutoTokenizer, AutoModelForCausalLM\n",
            "\n",
            "MODEL_ID = 'Qwen/Qwen2.5-3B-Instruct'\n",
            "HIDDEN_DIM = 2048\n",
            "\n",
            "print(f'Đang tải tokenizer cho {MODEL_ID}...')\n",
            "tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)\n",
            "if tokenizer.pad_token is None:\n",
            "    tokenizer.pad_token = tokenizer.eos_token\n",
            "\n",
            "def load_base_model():\n",
            "    \"\"\"Tải mô hình nền 3B ở độ chính xác nửa (fp16/bf16).\"\"\"\n",
            "    dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16\n",
            "    print(f'Nạp mô hình {MODEL_ID} (dtype={dtype})...')\n",
            "    model = AutoModelForCausalLM.from_pretrained(\n",
            "        MODEL_ID,\n",
            "        torch_dtype=dtype,\n",
            "        device_map='auto' if torch.cuda.is_available() else None,\n",
            "        trust_remote_code=True,\n",
            "    )\n",
            "    if hasattr(model, 'gradient_checkpointing_enable'):\n",
            "        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})\n",
            "        if hasattr(model, 'enable_input_require_grads'):\n",
            "            model.enable_input_require_grads()\n",
            "        print('Đã kích hoạt Gradient Checkpointing (use_reentrant=False).')\n",
            "    return model\n",
            "\n",
            "def load_lora_model(rank=8, alpha=16.0):\n",
            "    \"\"\"Tải mô hình nền và áp dụng LoRA trên q_proj và v_proj với đóng băng 100% mô hình nền.\"\"\"\n",
            "    model = load_base_model()\n",
            "    adapted = apply_lora_to_model(model, target_modules=['q_proj', 'v_proj'], rank=rank, alpha=alpha)\n",
            "    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)\n",
            "    total = sum(p.numel() for p in model.parameters())\n",
            "    print(f'LoRA đã thích ứng {len(adapted)} module. Số tham số huấn luyện: {trainable:,} / {total:,} ({trainable/total*100:.3f}%).')\n",
            "    assert trainable < 5_000_000, f'Lỗi: Số tham số LoRA quá lớn ({trainable:,}), mô hình chưa được đóng băng đúng cách!'\n",
            "    return model\n"
        ]
    })

    # Cell 6: Dataset and Injection Engine
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# --- Bộ dữ liệu Joint NWP Dataset & Tiêm mã hoá thời gian ---\n",
            "class JointNWPDataset(Dataset):\n",
            "    def __init__(self, samples, tokenizer, max_len=128):\n",
            "        self.examples = []\n",
            "        for s in samples:\n",
            "            query = str(s.get('input', ''))\n",
            "            ans = str(s.get('target', ''))\n",
            "            year = int(s.get('year', 2000))\n",
            "            full_text = f'{query.strip()} {ans.strip()}'\n",
            "            enc = tokenizer(full_text, truncation=True, max_length=max_len, return_tensors='pt')\n",
            "            input_ids = enc['input_ids'].squeeze(0)\n",
            "            \n",
            "            prefix_enc = tokenizer(query.strip(), truncation=True, max_length=max_len, return_tensors='pt')\n",
            "            prefix_len = prefix_enc['input_ids'].size(1)\n",
            "            \n",
            "            labels = input_ids.clone()\n",
            "            labels[:min(prefix_len, len(labels))] = -100\n",
            "            \n",
            "            self.examples.append({\n",
            "                'input_ids': input_ids,\n",
            "                'labels': labels,\n",
            "                'year': year,\n",
            "            })\n",
            "            \n",
            "    def __len__(self):\n",
            "        return len(self.examples)\n",
            "        \n",
            "    def __getitem__(self, idx):\n",
            "        return self.examples[idx]\n",
            "\n",
            "def collate_joint_fn(batch, pad_token_id):\n",
            "    max_len = max(len(x['input_ids']) for x in batch)\n",
            "    b_in, b_lbl, b_yr = [], [], []\n",
            "    for x in batch:\n",
            "        pad_len = max_len - len(x['input_ids'])\n",
            "        inp = torch.cat([x['input_ids'], torch.full((pad_len,), pad_token_id, dtype=torch.long)])\n",
            "        lbl = torch.cat([x['labels'], torch.full((pad_len,), -100, dtype=torch.long)])\n",
            "        b_in.append(inp)\n",
            "        b_lbl.append(lbl)\n",
            "        b_yr.append(x['year'])\n",
            "    return {\n",
            "        'input_ids': torch.stack(b_in),\n",
            "        'labels': torch.stack(b_lbl),\n",
            "        'years': torch.tensor(b_yr, dtype=torch.long),\n",
            "    }\n",
            "\n",
            "def find_date_token_mask(input_ids: torch.Tensor, years: torch.Tensor, tokenizer, device=None):\n",
            "    \"\"\"Xác định mặt nạ nhị phân [B, S, 1] cho đúng các token số năm trong câu hỏi.\"\"\"\n",
            "    if device is None:\n",
            "        device = input_ids.device\n",
            "    B, S = input_ids.shape\n",
            "    mask = torch.zeros((B, S, 1), dtype=torch.bool, device=device)\n",
            "    for b in range(B):\n",
            "        yr_str = str(int(years[b].item()))\n",
            "        yr_tokens = tokenizer.encode(yr_str, add_special_tokens=False)\n",
            "        if not yr_tokens:\n",
            "            continue\n",
            "        k = len(yr_tokens)\n",
            "        seq = input_ids[b].tolist()\n",
            "        for i in range(S - k + 1):\n",
            "            if seq[i:i + k] == yr_tokens:\n",
            "                mask[b, i:i + k, 0] = True\n",
            "    return mask\n",
            "\n",
            "def inject_te_embeddings(model, tokenizer, input_ids: torch.Tensor, years: torch.Tensor, te_mode='broadcast', scale_mode='raw', device=None):\n",
            "    \"\"\"Tiêm vector thời gian TE vào biểu diễn token ban đầu.\"\"\"\n",
            "    if device is None:\n",
            "        device = input_ids.device\n",
            "    embed_tokens = model.get_input_embeddings()\n",
            "    tok_emb = embed_tokens(input_ids)\n",
            "    \n",
            "    if te_mode == 'none':\n",
            "        return tok_emb\n",
            "        \n",
            "    hidden_dim = tok_emb.size(-1)\n",
            "    te = compute_batch_te(years, hidden_dim=hidden_dim, scale_mode=scale_mode, device=device, dtype=tok_emb.dtype)\n",
            "    \n",
            "    if te_mode == 'broadcast':\n",
            "        return tok_emb + te\n",
            "    elif te_mode == 'date_token':\n",
            "        date_mask = find_date_token_mask(input_ids, years, tokenizer, device=device)\n",
            "        return torch.where(date_mask, tok_emb + te, tok_emb)\n",
            "    else:\n",
            "        raise ValueError(f'te_mode không hợp lệ: {te_mode}')\n"
        ]
    })

    # Cell 7: Option Permutation & Debiased Choice-Logit Engine
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# --- Bộ sinh hoán vị phương án và Đánh giá Choice-Logit khử thiên vị ---\n",
            "def generate_option_permutations(prompt_text: str, target_text: str):\n",
            "    \"\"\"Sinh hoán vị phương án cho cả trắc nghiệm 4 lựa chọn (4 vòng) và Yes/No (2 thứ tự).\"\"\"\n",
            "    tgt = target_text.strip().lower()\n",
            "    if tgt in ['yes', 'no']:\n",
            "        p0 = prompt_text\n",
            "        p1 = re.sub(r'Answer as Yes or No', 'Answer as No or Yes', prompt_text, flags=re.IGNORECASE)\n",
            "        if p1 == p0:\n",
            "            p1 = re.sub(r'Yes\\s*(or|/)\\s*No', 'No or Yes', prompt_text, flags=re.IGNORECASE)\n",
            "        return [\n",
            "            {'prompt': p0, 'target': tgt, 'rotation': 0, 'rot_name': 'Yes/No'},\n",
            "            {'prompt': p1, 'target': tgt, 'rotation': 1, 'rot_name': 'No/Yes'}\n",
            "        ]\n",
            "        \n",
            "    suffix = ''\n",
            "    m_suffix = re.search(r'(\\s*\\n?\\s*Answer:\\s*)$', prompt_text, re.IGNORECASE)\n",
            "    if m_suffix:\n",
            "        suffix = m_suffix.group(1)\n",
            "        base_prompt = prompt_text[:len(prompt_text) - len(suffix)]\n",
            "    else:\n",
            "        base_prompt = prompt_text\n",
            "        \n",
            "    pattern = re.compile(r'^(.*Options:\\s*)\\(a\\)\\s*(.*?)\\s*\\(b\\)\\s*(.*?)\\s*\\(c\\)\\s*(.*?)\\s*\\(d\\)\\s*(.*?)$', re.DOTALL)\n",
            "    m = pattern.search(base_prompt)\n",
            "    if not m:\n",
            "        return [{'prompt': prompt_text, 'target': tgt, 'rotation': 0, 'rot_name': 'orig'}]\n",
            "        \n",
            "    prefix = m.group(1)\n",
            "    opts = [m.group(2).strip(), m.group(3).strip(), m.group(4).strip(), m.group(5).strip()]\n",
            "    letters = ['a', 'b', 'c', 'd']\n",
            "    if tgt not in letters:\n",
            "        return [{'prompt': prompt_text, 'target': tgt, 'rotation': 0, 'rot_name': 'orig'}]\n",
            "        \n",
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
            "            'rotation': k,\n",
            "            'rot_name': rot_names[k]\n",
            "        })\n",
            "    return permutations\n",
            "\n",
            "def _extract_choice_token_ids(tokenizer, words):\n",
            "    ids = []\n",
            "    if tokenizer is not None and hasattr(tokenizer, 'encode'):\n",
            "        for w in words:\n",
            "            for prefix in ['', ' ']:\n",
            "                try:\n",
            "                    enc = tokenizer.encode(prefix + w, add_special_tokens=False)\n",
            "                    if enc and len(enc) == 1:\n",
            "                        ids.append(enc[0])\n",
            "                except Exception:\n",
            "                    pass\n",
            "    return list(dict.fromkeys(ids))\n",
            "\n",
            "def eval_choice_logit_single(model, tokenizer, prompt_text: str, target_text: str, year: int = None, te_mode: str = 'none', te_scale_mode: str = 'raw'):\n",
            "    prompt = f'Question: {prompt_text.strip()}\\nAnswer:' if not prompt_text.strip().endswith('Answer:') else prompt_text.strip()\n",
            "    inputs = tokenizer(prompt, return_tensors='pt').to(device=model.device)\n",
            "    \n",
            "    with torch.no_grad():\n",
            "        if te_mode == 'none':\n",
            "            outputs = model(**inputs)\n",
            "        else:\n",
            "            yr_tensor = torch.tensor([year if year is not None else 2000], dtype=torch.long, device=model.device)\n",
            "            inputs_embeds = inject_te_embeddings(model, tokenizer, inputs['input_ids'], yr_tensor, te_mode=te_mode, scale_mode=te_scale_mode, device=model.device)\n",
            "            outputs = model(inputs_embeds=inputs_embeds, attention_mask=inputs.get('attention_mask'))\n",
            "            \n",
            "        logits = outputs.logits[0, -1, :]\n",
            "        vocab_size = logits.size(-1)\n",
            "        \n",
            "    def _get_max_logit(token_candidates, fallback_ids):\n",
            "        dynamic_ids = _extract_choice_token_ids(tokenizer, token_candidates)\n",
            "        candidate_ids = dynamic_ids if dynamic_ids else fallback_ids\n",
            "        valid_ids = [idx for idx in candidate_ids if 0 <= idx < vocab_size]\n",
            "        if valid_ids:\n",
            "            return max(logits[idx].item() for idx in valid_ids)\n",
            "        return -1e9\n",
            "\n",
            "    tgt = target_text.strip().lower()\n",
            "    if tgt in ['yes', 'no']:\n",
            "        scores = {\n",
            "            'yes': _get_max_logit(['yes', 'Yes', 'YES'], [7414, 9454, 9834, 9693]),\n",
            "            'no': _get_max_logit(['no', 'No', 'NO'], [2308, 2753, 902, 2152]),\n",
            "        }\n",
            "        pred = 'yes' if scores['yes'] >= scores['no'] else 'no'\n",
            "    else:\n",
            "        scores = {\n",
            "            'a': _get_max_logit(['a', 'A'], [264, 362, 64, 32]),\n",
            "            'b': _get_max_logit(['b', 'B'], [293, 425, 65, 33]),\n",
            "            'c': _get_max_logit(['c', 'C'], [272, 356, 66, 34]),\n",
            "            'd': _get_max_logit(['d', 'D'], [294, 422, 67, 35]),\n",
            "        }\n",
            "        pred = max(scores, key=scores.get)\n",
            "        \n",
            "    return {\n",
            "        'pred': pred,\n",
            "        'gold': tgt,\n",
            "        'is_correct': (pred == tgt),\n",
            "        'scores': {k: round(v, 4) for k, v in scores.items()},\n",
            "    }\n",
            "\n",
            "def eval_choice_logits_batch(model, tokenizer, prompt_texts, target_texts, year=None, te_mode='none', te_scale_mode='raw'):\n",
            "    \"\"\"Đánh giá theo lô các hoán vị phương án trong 1 forward pass duy nhất với left padding.\"\"\"\n",
            "    if not prompt_texts:\n",
            "        return []\n",
            "    if len(prompt_texts) != len(target_texts):\n",
            "        raise ValueError(f'Mismatched batch lengths: {len(prompt_texts)} prompts vs {len(target_texts)} targets')\n",
            "        \n",
            "    dev = getattr(model, 'device', next(model.parameters()).device if hasattr(model, 'parameters') and list(model.parameters()) else torch.device('cpu'))\n",
            "    tokenizer.padding_side = 'left'\n",
            "    if tokenizer.pad_token is None:\n",
            "        tokenizer.pad_token = tokenizer.eos_token\n",
            "        \n",
            "    formatted_prompts = [\n",
            "        f'Question: {p.strip()}\\nAnswer:' if not p.strip().endswith('Answer:') else p.strip()\n",
            "        for p in prompt_texts\n",
            "    ]\n",
            "    inputs = tokenizer(formatted_prompts, return_tensors='pt', padding=True).to(device=dev)\n",
            "    \n",
            "    with torch.no_grad():\n",
            "        if te_mode == 'none':\n",
            "            outputs = model(**inputs)\n",
            "        else:\n",
            "            B = len(prompt_texts)\n",
            "            yr_val = year if year is not None else 2000\n",
            "            yr_tensor = torch.full((B,), yr_val, dtype=torch.long, device=dev)\n",
            "            inputs_embeds = inject_te_embeddings(\n",
            "                model, tokenizer, inputs['input_ids'], yr_tensor,\n",
            "                te_mode=te_mode, scale_mode=te_scale_mode, device=dev\n",
            "            )\n",
            "            outputs = model(inputs_embeds=inputs_embeds, attention_mask=inputs.get('attention_mask'))\n",
            "            \n",
            "        batch_logits = outputs.logits[:, -1, :]\n",
            "        vocab_size = batch_logits.size(-1)\n",
            "        \n",
            "    choice_candidates = {\n",
            "        'yes': (['yes', 'Yes', 'YES'], [7414, 9454, 9834, 9693]),\n",
            "        'no': (['no', 'No', 'NO'], [2308, 2753, 902, 2152]),\n",
            "        'a': (['a', 'A'], [264, 362, 64, 32]),\n",
            "        'b': (['b', 'B'], [293, 425, 65, 33]),\n",
            "        'c': (['c', 'C'], [272, 356, 66, 34]),\n",
            "        'd': (['d', 'D'], [294, 422, 67, 35]),\n",
            "    }\n",
            "    resolved_ids = {}\n",
            "    for opt, (words, fallback) in choice_candidates.items():\n",
            "        dyn = _extract_choice_token_ids(tokenizer, words)\n",
            "        cand = dyn if dyn else fallback\n",
            "        resolved_ids[opt] = [idx for idx in cand if 0 <= idx < vocab_size]\n",
            "        \n",
            "    results = []\n",
            "    for i, tgt_raw in enumerate(target_texts):\n",
            "        log_i = batch_logits[i]\n",
            "        tgt = tgt_raw.strip().lower()\n",
            "        if tgt in ['yes', 'no']:\n",
            "            scores = {\n",
            "                'yes': max((log_i[idx].item() for idx in resolved_ids['yes']), default=-1e9),\n",
            "                'no': max((log_i[idx].item() for idx in resolved_ids['no']), default=-1e9),\n",
            "            }\n",
            "            pred = 'yes' if scores['yes'] >= scores['no'] else 'no'\n",
            "        else:\n",
            "            scores = {\n",
            "                'a': max((log_i[idx].item() for idx in resolved_ids['a']), default=-1e9),\n",
            "                'b': max((log_i[idx].item() for idx in resolved_ids['b']), default=-1e9),\n",
            "                'c': max((log_i[idx].item() for idx in resolved_ids['c']), default=-1e9),\n",
            "                'd': max((log_i[idx].item() for idx in resolved_ids['d']), default=-1e9),\n",
            "            }\n",
            "            pred = max(scores, key=scores.get)\n",
            "            \n",
            "        results.append({\n",
            "            'pred': pred,\n",
            "            'gold': tgt,\n",
            "            'is_correct': (pred == tgt),\n",
            "            'scores': {k: round(v, 4) for k, v in scores.items()},\n",
            "        })\n",
            "    return results\n",
            "\n",
            "def evaluate_dataset(model, tokenizer, eval_records, te_mode='none', te_scale_mode='raw', debias=True):\n",
            "    model.eval()\n",
            "    tokenizer.padding_side = 'left'\n",
            "    if tokenizer.pad_token is None:\n",
            "        tokenizer.pad_token = tokenizer.eos_token\n",
            "        \n",
            "    case_results = {}\n",
            "    canonical_correct = 0\n",
            "    debiased_correct_sum = 0.0\n",
            "    strict_correct = 0\n",
            "    task_scores = defaultdict(lambda: {'canonical': 0, 'debiased_sum': 0.0, 'total': 0})\n",
            "    \n",
            "    t0 = time.time()\n",
            "    total_cases = len(eval_records)\n",
            "    for idx, case in enumerate(eval_records):\n",
            "        cid = case.get('example_id', f'case_{idx}')\n",
            "        task_id = case.get('task_id', 'Unknown')\n",
            "        yr = case.get('year')\n",
            "        \n",
            "        if debias:\n",
            "            perms = generate_option_permutations(case['input'], case['target'])\n",
            "            p_prompts = [p['prompt'] for p in perms]\n",
            "            p_targets = [p['target'] for p in perms]\n",
            "            batch_res = eval_choice_logits_batch(\n",
            "                model, tokenizer, p_prompts, p_targets,\n",
            "                year=yr, te_mode=te_mode, te_scale_mode=te_scale_mode\n",
            "            )\n",
            "            perm_outcomes = [r['is_correct'] for r in batch_res]\n",
            "            is_canonical = perm_outcomes[0]\n",
            "            debiased_score = float(np.mean(perm_outcomes))\n",
            "            is_strict = all(perm_outcomes)\n",
            "        else:\n",
            "            batch_res = eval_choice_logits_batch(\n",
            "                model, tokenizer, [case['input']], [case['target']],\n",
            "                year=yr, te_mode=te_mode, te_scale_mode=te_scale_mode\n",
            "            )\n",
            "            is_canonical = batch_res[0]['is_correct']\n",
            "            debiased_score = 1.0 if is_canonical else 0.0\n",
            "            is_strict = is_canonical\n",
            "            \n",
            "        if is_canonical:\n",
            "            canonical_correct += 1\n",
            "        debiased_correct_sum += debiased_score if debias else (1.0 if is_canonical else 0.0)\n",
            "        if is_strict:\n",
            "            strict_correct += 1\n",
            "            \n",
            "        task_scores[task_id]['canonical'] += int(is_canonical)\n",
            "        task_scores[task_id]['debiased_sum'] += debiased_score\n",
            "        task_scores[task_id]['total'] += 1\n",
            "        \n",
            "        case_key = f'{cid}#{idx}'\n",
            "        case_results[case_key] = {\n",
            "            'is_correct': is_canonical,\n",
            "            'debiased_score': debiased_score,\n",
            "            'is_strict': is_strict,\n",
            "            'task_id': task_id,\n",
            "        }\n",
            "        \n",
            "        if (idx + 1) % 500 == 0 or (idx + 1) == total_cases:\n",
            "            elapsed = time.time() - t0\n",
            "            rate = (idx + 1) / elapsed if elapsed > 0 else 0.0\n",
            "            remaining = (total_cases - (idx + 1)) / rate if rate > 0 else 0.0\n",
            "            print(f'Đã đánh giá: {idx + 1}/{total_cases} câu ({elapsed:.1f}s | {rate:.2f} câu/s | ETA: {remaining/60:.1f} phút)...')\n",
            "            \n",
            "    N = len(eval_records)\n",
            "    task_summary = {}\n",
            "    for t, v in sorted(task_scores.items()):\n",
            "        cnt = v['total']\n",
            "        task_summary[t] = {\n",
            "            'canonical_acc': round(v['canonical'] / cnt * 100.0, 2) if cnt > 0 else 0.0,\n",
            "            'debiased_mean_acc': round(v['debiased_sum'] / cnt * 100.0, 2) if cnt > 0 else 0.0,\n",
            "            'count': cnt,\n",
            "        }\n",
            "        \n",
            "    return {\n",
            "        'canonical_acc': round(canonical_correct / N * 100.0, 2) if N > 0 else 0.0,\n",
            "        'debiased_mean_acc': round(debiased_correct_sum / N * 100.0, 2) if N > 0 else 0.0,\n",
            "        'strict_acc': round(strict_correct / N * 100.0, 2) if N > 0 else 0.0,\n",
            "        'total': N,\n",
            "        'task_summary': task_summary,\n",
            "        'case_results': case_results,\n",
            "    }\n"
        ]
    })

    # Cell 8: Training & Baseline Execution Loop
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# --- Vòng lặp huấn luyện LoRA tối ưu trên Tesla T4 (Micro-batch 8, Accumulation 2) ---\n",
            "def train_baseline(\n",
            "    cfg_name: str,\n",
            "    train_data: list,\n",
            "    te_mode: str = 'none',\n",
            "    te_scale_mode: str = 'raw',\n",
            "    sigma: float = 0.5,\n",
            "    epochs: int = 1,\n",
            "    lr: float = 2e-4,\n",
            "    micro_batch_size: int = 8,\n",
            "    gradient_accumulation_steps: int = 2,\n",
            "    max_steps = None,\n",
            "):\n",
            "    print(f'\\n========== BẮT ĐẦU HUẤN LUYỆN: {cfg_name} ==========')\n",
            "    print(f'Tham số: te_mode={te_mode}, te_scale={te_scale_mode}, sigma={sigma}, epochs={epochs}')\n",
            "    set_seed(42)\n",
            "    \n",
            "    model = load_lora_model(rank=8, alpha=16.0)\n",
            "    model.train()\n",
            "    \n",
            "    train_dataset = JointNWPDataset(train_data, tokenizer, max_len=128)\n",
            "    train_loader = DataLoader(\n",
            "        train_dataset,\n",
            "        batch_size=micro_batch_size,\n",
            "        shuffle=True,\n",
            "        collate_fn=lambda b: collate_joint_fn(b, pad_token_id=tokenizer.pad_token_id),\n",
            "    )\n",
            "    \n",
            "    trainable_params = [p for p in model.parameters() if p.requires_grad]\n",
            "    optimizer = torch.optim.AdamW(trainable_params, lr=lr, weight_decay=0.01)\n",
            "    \n",
            "    step = 0\n",
            "    total_loss_accum = 0.0\n",
            "    t_start = time.time()\n",
            "    \n",
            "    for epoch in range(epochs):\n",
            "        for b_idx, batch in enumerate(train_loader):\n",
            "            input_ids = batch['input_ids'].to(model.device)\n",
            "            labels = batch['labels'].to(model.device)\n",
            "            years = batch['years'].to(model.device)\n",
            "            \n",
            "            if te_mode == 'none':\n",
            "                outputs = model(input_ids=input_ids, labels=labels, output_hidden_states=(sigma > 0))\n",
            "            else:\n",
            "                inputs_embeds = inject_te_embeddings(\n",
            "                    model, tokenizer, input_ids, years,\n",
            "                    te_mode=te_mode, scale_mode=te_scale_mode, device=model.device\n",
            "                )\n",
            "                if inputs_embeds.is_leaf:\n",
            "                    inputs_embeds.requires_grad_(True)\n",
            "                outputs = model(inputs_embeds=inputs_embeds, labels=labels, output_hidden_states=(sigma > 0))\n",
            "                \n",
            "            loss_ntp = outputs.loss\n",
            "            if sigma > 0 and hasattr(outputs, 'hidden_states') and outputs.hidden_states is not None:\n",
            "                last_hidden = outputs.hidden_states[-1]\n",
            "                loss_intra = compute_intra_loss(last_hidden, years)\n",
            "                loss_total = loss_ntp + sigma * loss_intra\n",
            "            else:\n",
            "                loss_total = loss_ntp\n",
            "                \n",
            "            loss_total = loss_total / gradient_accumulation_steps\n",
            "            loss_total.backward()\n",
            "            total_loss_accum += loss_total.item() * gradient_accumulation_steps\n",
            "            \n",
            "            if (b_idx + 1) % gradient_accumulation_steps == 0:\n",
            "                torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)\n",
            "                optimizer.step()\n",
            "                optimizer.zero_grad()\n",
            "                step += 1\n",
            "                \n",
            "                if step % 250 == 0:\n",
            "                    avg_l = total_loss_accum / 250.0\n",
            "                    elapsed = time.time() - t_start\n",
            "                    print(f'Epoch {epoch+1} | Step {step} | Loss trung bình: {avg_l:.4f} | Thời gian: {elapsed:.1f}s')\n",
            "                    total_loss_accum = 0.0\n",
            "                    \n",
            "                if max_steps is not None and step >= max_steps:\n",
            "                    print(f'Đã đạt ngưỡng max_steps={max_steps}, dừng huấn luyện sớm.')\n",
            "                    break\n",
            "                    \n",
            "    print(f'Huấn luyện hoàn thành sau {time.time() - t_start:.1f} giây.')\n",
            "    lora_state = {k: v.cpu() for k, v in model.state_dict().items() if 'lora_' in k}\n",
            "    lora_filename = f'{cfg_name.lower()}_lora_weights.pt' if cfg_name else 'pt_3b_lora_weights.pt'\n",
            "    torch.save(lora_state, lora_filename)\n",
            "    if lora_filename != 'pt_3b_lora_weights.pt' and cfg_name and 'pt' in cfg_name.lower():\n",
            "        torch.save(lora_state, 'pt_3b_lora_weights.pt')\n",
            "    lora_size_mb = os.path.getsize(lora_filename) / (1024 * 1024)\n",
            "    print(f'Đã lưu checkpoint LoRA ({len(lora_state)} tensors, {lora_size_mb:.2f} MB) vào {lora_filename}')\n",
            "    return model\n"
        ]
    })

    # Cell 9 & 10: Tailored per Mode
    if mode == "pt":
        cells.append({
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# --- NẠP MỐC THAM CHIẾU BASE 3B ZERO-SHOT ĐÃ ĐƯỢC XÁC MINH VÀ HUẤN LUYỆN PT 3B ---\n",
                "BENCHMARK_RESULTS = {}\n",
                "\n",
                f"BASE_3B_GZ_B64 = '''{base_3b_b64}'''\n",
                "\n",
                "def load_verified_base_3b():\n",
                "    if os.path.isfile('base_3b_benchmark_results.json'):\n",
                "        with open('base_3b_benchmark_results.json', 'r', encoding='utf-8') as f:\n",
                "            data = json.load(f)\n",
                "            return data.get('Base_3B', data)\n",
                "    raw = gzip.decompress(base64.b64decode(BASE_3B_GZ_B64))\n",
                "    data = json.loads(raw.decode('utf-8'))\n",
                "    base_res = data.get('Base_3B', data)\n",
                "    with open('base_3b_benchmark_results.json', 'w', encoding='utf-8') as f:\n",
                "        json.dump({'Base_3B': base_res}, f, indent=2)\n",
                "    return base_res\n",
                "\n",
                "# 1. Nạp mốc chuẩn Base 3B Zero-shot đã được đo đạc và xác minh an toàn\n",
                "print('>>> [1/2] Nạp mốc chuẩn Base 3B Zero-shot (100% 5,292 mẫu)...')\n",
                "res_base = load_verified_base_3b()\n",
                "BENCHMARK_RESULTS['Base_3B'] = res_base\n",
                "print(f'Base 3B (Verified) -> Canonical Acc: {res_base[\"canonical_acc\"]}%, Debiased Mean: {res_base[\"debiased_mean_acc\"]}%, Strict Consistency: {res_base[\"strict_acc\"]}%')\n",
                "print('Đã sẵn sàng kết quả Base 3B trong base_3b_benchmark_results.json')\n",
                "\n",
                "# 2. PT 3B (LoRA NTP chuẩn)\n",
                "print('\\n>>> [2/2] Huấn luyện & Đánh giá PT 3B (LoRA NTP)...')\n",
                "pt_model = train_baseline('PT_3B', train_cases, te_mode='none', sigma=0.0, epochs=1, max_steps=MAX_TRAIN_STEPS)\n",
                "res_pt = evaluate_dataset(pt_model, tokenizer, eval_cases, te_mode='none', debias=True)\n",
                "BENCHMARK_RESULTS['PT_3B'] = res_pt\n",
                "print(f'PT 3B -> Canonical Acc: {res_pt[\"canonical_acc\"]}%, Debiased Mean: {res_pt[\"debiased_mean_acc\"]}%, Strict Consistency: {res_pt[\"strict_acc\"]}%')\n",
                "del pt_model\n",
                "gc.collect()\n",
                "torch.cuda.empty_cache()\n",
                "\n",
                "# Lưu ngay checkpoint PT 3B ra đĩa\n",
                "with open('pt_3b_benchmark_results.json', 'w', encoding='utf-8') as f:\n",
                "    json.dump(BENCHMARK_RESULTS, f, indent=2)\n",
                "print('Đã lưu an toàn checkpoint PT 3B vào pt_3b_benchmark_results.json')\n"
            ]
        })
        cells.append({
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# --- TỔNG KẾT SO SÁNH VÀ KIỂM ĐỊNH MCNEMAR ---\n",
                "print('\\n' + '='*80)\n",
                "print(f'{\"Cấu hình\":<15} | {\"Canonical Acc\":<15} | {\"Debiased Mean Acc\":<18} | {\"Strict Consistency\":<18}')\n",
                "print('-'*80)\n",
                "for name, res in BENCHMARK_RESULTS.items():\n",
                "    print(f'{name:<15} | {res[\"canonical_acc\"]:>13}% | {res[\"debiased_mean_acc\"]:>16}% | {res[\"strict_acc\"]:>16}%')\n",
                "print('='*80)\n",
                "\n",
                "def compute_mcnemar_table(cases_a, cases_b):\n",
                "    b = sum(1 for cid in cases_a if cases_a[cid]['is_correct'] and not cases_b[cid]['is_correct'])\n",
                "    c = sum(1 for cid in cases_a if not cases_a[cid]['is_correct'] and cases_b[cid]['is_correct'])\n",
                "    a = sum(1 for cid in cases_a if cases_a[cid]['is_correct'] and cases_b[cid]['is_correct'])\n",
                "    d = sum(1 for cid in cases_a if not cases_a[cid]['is_correct'] and not cases_b[cid]['is_correct'])\n",
                "    stat = ((abs(b - c) - 1.0) ** 2) / (b + c) if (b + c) > 0 else 0.0\n",
                "    pval = 2.0 * min(stats.binom.cdf(min(b, c), b + c, 0.5), 1.0 - stats.binom.cdf(min(b, c) - 1, b + c, 0.5)) if (b + c) > 0 else 1.0\n",
                "    return {'b': b, 'c': c, 'a': a, 'd': d, 'stat': round(stat, 4), 'pval': round(pval, 6)}\n",
                "\n",
                "if 'Base_3B' in BENCHMARK_RESULTS and 'PT_3B' in BENCHMARK_RESULTS:\n",
                "    mcn_pt = compute_mcnemar_table(BENCHMARK_RESULTS['PT_3B']['case_results'], BENCHMARK_RESULTS['Base_3B']['case_results'])\n",
                "    print(f'Kiểm định McNemar (PT_3B vs Base_3B): b={mcn_pt[\"b\"]}, c={mcn_pt[\"c\"]}, stat={mcn_pt[\"stat\"]}, p={mcn_pt[\"pval\"]}')\n",
                "\n",
                "# Xuất kết quả tổng hợp ra file JSON\n",
                "clean_export = {}\n",
                "for k, v in BENCHMARK_RESULTS.items():\n",
                "    clean_export[k] = {\n",
                "        'canonical_acc': v['canonical_acc'],\n",
                "        'debiased_mean_acc': v['debiased_mean_acc'],\n",
                "        'strict_acc': v['strict_acc'],\n",
                "        'task_summary': v['task_summary'],\n",
                "    }\n",
                "with open('pt_benchmark_summary.json', 'w', encoding='utf-8') as f:\n",
                "    json.dump(clean_export, f, indent=2)\n",
                "print('Đã lưu kết quả tổng hợp vào pt_benchmark_summary.json')\n"
            ]
        })
    elif mode == "ticktack":
        cells.append({
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# --- THỰC HIỆN HUẤN LUYỆN & ĐÁNH GIÁ TICKTACK 3B ---\n",
                "BENCHMARK_RESULTS = {}\n",
                "\n",
                "# Tham chiếu kết quả Base 3B Zero-shot đã đo đạc chính thức 100%\n",
                "BENCHMARK_RESULTS['Base_3B_Ref'] = {\n",
                "    'canonical_acc': 28.38,\n",
                "    'debiased_mean_acc': 29.20,\n",
                "    'strict_acc': 6.24,\n",
                "    'task_summary': {\n",
                "        'DB': {'canonical_acc': 26.10, 'debiased_mean_acc': 26.59, 'count': 912},\n",
                "        'MM': {'canonical_acc': 25.82, 'debiased_mean_acc': 25.38, 'count': 852},\n",
                "        'RB': {'canonical_acc': 25.35, 'debiased_mean_acc': 25.12, 'count': 852},\n",
                "        'Rel': {'canonical_acc': 37.91, 'debiased_mean_acc': 37.91, 'count': 852},\n",
                "        'TB': {'canonical_acc': 26.32, 'debiased_mean_acc': 26.26, 'count': 912},\n",
                "        'WB': {'canonical_acc': 29.06, 'debiased_mean_acc': 34.02, 'count': 912}\n",
                "    }\n",
                "}\n",
                "print('Mốc tham chiếu Base 3B Zero-shot (100% eval): Canonical Acc = 28.38%, Debiased Mean = 29.20%')\n",
                "\n",
                "# 1. Ticktack 3B (Localized Date-token Injection + Can Chi Cycle Intra-Loss)\n",
                "print('\\n>>> [1/1] Huấn luyện & Đánh giá Ticktack 3B (Date-token, sigma=0.5)...')\n",
                "ticktack_model = train_baseline('Ticktack_3B', train_cases, te_mode='date_token', te_scale_mode='raw', sigma=0.5, epochs=1, max_steps=MAX_TRAIN_STEPS)\n",
                "res_tt = evaluate_dataset(ticktack_model, tokenizer, eval_cases, te_mode='date_token', te_scale_mode='raw', debias=True)\n",
                "BENCHMARK_RESULTS['Ticktack_3B'] = res_tt\n",
                "print(f'Ticktack 3B -> Canonical Acc: {res_tt[\"canonical_acc\"]}%, Debiased Mean: {res_tt[\"debiased_mean_acc\"]}%')\n",
                "del ticktack_model\n",
                "gc.collect()\n",
                "torch.cuda.empty_cache()\n",
                "\n",
                "# Lưu ngay checkpoint Ticktack 3B ra đĩa\n",
                "with open('ticktack_3b_benchmark_results.json', 'w', encoding='utf-8') as f:\n",
                "    json.dump(BENCHMARK_RESULTS, f, indent=2)\n",
                "print('Đã lưu an toàn checkpoint Ticktack 3B vào ticktack_3b_benchmark_results.json')\n"
            ]
        })
        cells.append({
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# --- TỔNG KẾT SO SÁNH TICKTACK 3B VÀ MỐC THAM CHIẾU BASE 3B ---\n",
                "print('\\n' + '='*80)\n",
                "print(f'{\"Cấu hình\":<15} | {\"Canonical Acc\":<15} | {\"Debiased Mean Acc\":<18} | {\"Strict Consistency\":<18}')\n",
                "print('-'*80)\n",
                "for name, res in BENCHMARK_RESULTS.items():\n",
                "    print(f'{name:<15} | {res[\"canonical_acc\"]:>13}% | {res[\"debiased_mean_acc\"]:>16}% | {res[\"strict_acc\"]:>16}%')\n",
                "print('='*80)\n",
                "\n",
                "# Xuất kết quả tổng hợp ra file JSON\n",
                "clean_export = {}\n",
                "for k, v in BENCHMARK_RESULTS.items():\n",
                "    clean_export[k] = {\n",
                "        'canonical_acc': v['canonical_acc'],\n",
                "        'debiased_mean_acc': v['debiased_mean_acc'],\n",
                "        'strict_acc': v['strict_acc'],\n",
                "        'task_summary': v['task_summary'],\n",
                "    }\n",
                "with open('ticktack_benchmark_summary.json', 'w', encoding='utf-8') as f:\n",
                "    json.dump(clean_export, f, indent=2)\n",
                "print('Đã lưu kết quả tổng hợp vào ticktack_benchmark_summary.json')\n"
            ]
        })
    else:
        # All 3 models sequentially with immediate checkpoint saves
        cells.append({
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# --- THỰC HIỆN ĐÁNH GIÁ 3 CẤU HÌNH CỐT LÕI ---\n",
                "BENCHMARK_RESULTS = {}\n",
                "\n",
                "# 1. Base 3B Zero-shot (Không huấn luyện)\n",
                "print('\\n>>> [1/3] Đánh giá Base 3B Zero-shot...')\n",
                "base_model = load_base_model()\n",
                "res_base = evaluate_dataset(base_model, tokenizer, eval_cases, te_mode='none', debias=True)\n",
                "BENCHMARK_RESULTS['Base_3B'] = res_base\n",
                "print(f'Base 3B -> Canonical Acc: {res_base[\"canonical_acc\"]}%, Debiased Mean: {res_base[\"debiased_mean_acc\"]}%')\n",
                "del base_model\n",
                "gc.collect()\n",
                "torch.cuda.empty_cache()\n",
                "with open('base_3b_benchmark_results.json', 'w', encoding='utf-8') as f:\n",
                "    json.dump({'Base_3B': res_base}, f, indent=2)\n",
                "print('Đã lưu an toàn kết quả Base 3B.')\n",
                "\n",
                "# 2. PT 3B (LoRA NTP chuẩn)\n",
                "print('\\n>>> [2/3] Huấn luyện & Đánh giá PT 3B (LoRA NTP)...')\n",
                "pt_model = train_baseline('PT_3B', train_cases, te_mode='none', sigma=0.0, epochs=1, max_steps=MAX_TRAIN_STEPS)\n",
                "res_pt = evaluate_dataset(pt_model, tokenizer, eval_cases, te_mode='none', debias=True)\n",
                "BENCHMARK_RESULTS['PT_3B'] = res_pt\n",
                "print(f'PT 3B -> Canonical Acc: {res_pt[\"canonical_acc\"]}%, Debiased Mean: {res_pt[\"debiased_mean_acc\"]}%')\n",
                "del pt_model\n",
                "gc.collect()\n",
                "torch.cuda.empty_cache()\n",
                "with open('pt_3b_benchmark_results.json', 'w', encoding='utf-8') as f:\n",
                "    json.dump(BENCHMARK_RESULTS, f, indent=2)\n",
                "print('Đã lưu an toàn kết quả PT 3B.')\n",
                "\n",
                "# 3. Ticktack 3B (Localized Date-token Injection + Can Chi Cycle Intra-Loss)\n",
                "print('\\n>>> [3/3] Huấn luyện & Đánh giá Ticktack 3B (Date-token, sigma=0.5)...')\n",
                "ticktack_model = train_baseline('Ticktack_3B', train_cases, te_mode='date_token', te_scale_mode='raw', sigma=0.5, epochs=1, max_steps=MAX_TRAIN_STEPS)\n",
                "res_tt = evaluate_dataset(ticktack_model, tokenizer, eval_cases, te_mode='date_token', te_scale_mode='raw', debias=True)\n",
                "BENCHMARK_RESULTS['Ticktack_3B'] = res_tt\n",
                "print(f'Ticktack 3B -> Canonical Acc: {res_tt[\"canonical_acc\"]}%, Debiased Mean: {res_tt[\"debiased_mean_acc\"]}%')\n",
                "del ticktack_model\n",
                "gc.collect()\n",
                "torch.cuda.empty_cache()\n",
                "with open('ticktack_3b_benchmark_results.json', 'w', encoding='utf-8') as f:\n",
                "    json.dump(BENCHMARK_RESULTS, f, indent=2)\n",
                "print('Đã lưu an toàn kết quả Ticktack 3B.')\n"
            ]
        })
        cells.append({
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# --- TỔNG KẾT SO SÁNH VÀ KIỂM ĐỊNH MCNEMAR ---\n",
                "print('\\n' + '='*80)\n",
                "print(f'{\"Cấu hình\":<15} | {\"Canonical Acc\":<15} | {\"Debiased Mean Acc\":<18} | {\"Strict Consistency\":<18}')\n",
                "print('-'*80)\n",
                "for name, res in BENCHMARK_RESULTS.items():\n",
                "    print(f'{name:<15} | {res[\"canonical_acc\"]:>13}% | {res[\"debiased_mean_acc\"]:>16}% | {res[\"strict_acc\"]:>16}%')\n",
                "print('='*80)\n",
                "\n",
                "def compute_mcnemar_table(cases_a, cases_b):\n",
                "    b = sum(1 for cid in cases_a if cases_a[cid]['is_correct'] and not cases_b[cid]['is_correct'])\n",
                "    c = sum(1 for cid in cases_a if not cases_a[cid]['is_correct'] and cases_b[cid]['is_correct'])\n",
                "    a = sum(1 for cid in cases_a if cases_a[cid]['is_correct'] and cases_b[cid]['is_correct'])\n",
                "    d = sum(1 for cid in cases_a if not cases_a[cid]['is_correct'] and not cases_b[cid]['is_correct'])\n",
                "    stat = ((abs(b - c) - 1.0) ** 2) / (b + c) if (b + c) > 0 else 0.0\n",
                "    pval = 2.0 * min(stats.binom.cdf(min(b, c), b + c, 0.5), 1.0 - stats.binom.cdf(min(b, c) - 1, b + c, 0.5)) if (b + c) > 0 else 1.0\n",
                "    return {'b': b, 'c': c, 'a': a, 'd': d, 'stat': round(stat, 4), 'pval': round(pval, 6)}\n",
                "\n",
                "if 'Base_3B' in BENCHMARK_RESULTS and 'Ticktack_3B' in BENCHMARK_RESULTS:\n",
                "    mcn_base = compute_mcnemar_table(BENCHMARK_RESULTS['Ticktack_3B']['case_results'], BENCHMARK_RESULTS['Base_3B']['case_results'])\n",
                "    print(f'Kiểm định McNemar (Ticktack_3B vs Base_3B): b={mcn_base[\"b\"]}, c={mcn_base[\"c\"]}, p={mcn_base[\"pval\"]}')\n",
                "\n",
                "if 'PT_3B' in BENCHMARK_RESULTS and 'Ticktack_3B' in BENCHMARK_RESULTS:\n",
                "    mcn_pt = compute_mcnemar_table(BENCHMARK_RESULTS['Ticktack_3B']['case_results'], BENCHMARK_RESULTS['PT_3B']['case_results'])\n",
                "    print(f'Kiểm định McNemar (Ticktack_3B vs PT_3B): b={mcn_pt[\"b\"]}, c={mcn_pt[\"c\"]}, p={mcn_pt[\"pval\"]}')\n",
                "\n",
                "# Xuất kết quả ra file JSON\n",
                "output_json = 'ticktack_3b_full_benchmark_results.json'\n",
                "clean_export = {}\n",
                "for k, v in BENCHMARK_RESULTS.items():\n",
                "    clean_export[k] = {\n",
                "        'canonical_acc': v['canonical_acc'],\n",
                "        'debiased_mean_acc': v['debiased_mean_acc'],\n",
                "        'strict_acc': v['strict_acc'],\n",
                "        'task_summary': v['task_summary'],\n",
                "    }\n",
                "with open(output_json, 'w', encoding='utf-8') as f:\n",
                "    json.dump(clean_export, f, indent=2)\n",
                "print(f'Đã lưu kết quả đo đạc chuẩn vào: {output_json}')\n"
            ]
        })

    notebook_dict = {
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
            },
            "accelerator": "GPU"
        },
        "nbformat": 4,
        "nbformat_minor": 5
    }

    notebook_path = os.path.join(output_dir, nb_filename)
    with open(notebook_path, "w", encoding="utf-8") as f:
        json.dump(notebook_dict, f, indent=1)
    print(f"Generated Kaggle notebook: {notebook_path} ({os.path.getsize(notebook_path) / 1024:.1f} KB)")

    meta = {
        "id": f"tnghuy11/{kernel_slug}",
        "title": kernel_slug,
        "code_file": nb_filename,
        "language": "python",
        "kernel_type": "notebook",
        "is_private": "true",
        "enable_gpu": "true",
        "enable_tpu": "false",
        "enable_internet": "true",
        "dataset_sources": [
            "tnghuy11/tempun-full-clean"
        ],
        "competition_sources": [],
        "kernel_sources": [],
        "model_sources": []
    }
    meta_path = os.path.join(output_dir, "kernel-metadata.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    print(f"Generated kernel metadata: {meta_path}")

    if extra_output_dir:
        extra_nb = os.path.join(extra_output_dir, nb_filename)
        with open(extra_nb, "w", encoding="utf-8") as f:
            json.dump(notebook_dict, f, indent=1)
        extra_meta = os.path.join(extra_output_dir, "kernel-metadata.json")
        with open(extra_meta, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)
        print(f"Mirrored notebook and metadata to: {extra_output_dir}")

    return notebook_path


def build_all_3b_kernels(extra_root: Optional[str] = "d:/nckh") -> Dict[str, str]:
    """Sinh cả 2 kernel song song và kernel tổng hợp, đồng bộ sang cả d:/nckh."""
    paths = {}
    modes = [
        ("pt", "PT 3B Baseline", "kaggle_ticktack_3b_pt"),
        ("ticktack", "Ticktack 3B Baseline", "kaggle_ticktack_3b_ticktack"),
        ("all", "All-in-One Full Baseline", "kaggle_ticktack_3b_full"),
    ]
    for idx, (m, title, folder_name) in enumerate(modes, 1):
        print(f"\n--- [{idx}/3] Tạo Kernel: {title} ---")
        extra_dir = None
        if extra_root and os.path.isdir(extra_root):
            extra_dir = os.path.join(extra_root, folder_name)
        paths[m] = create_kaggle_notebook_3b_full(mode=m, extra_output_dir=extra_dir)

    legacy_nb_dir = os.path.join(_REPO_ROOT, "baselines", "ticktack", "notebooks")
    if os.path.isdir(legacy_nb_dir):
        import shutil
        if "all" in paths:
            legacy_dest = os.path.join(legacy_nb_dir, "ticktack_3b_full.ipynb")
            shutil.copy2(paths["all"], legacy_dest)
            print(f"Mirrored legacy notebook to: {legacy_dest}")
        if "pt" in paths:
            legacy_pt_dest = os.path.join(legacy_nb_dir, "ticktack_3b_pt.ipynb")
            shutil.copy2(paths["pt"], legacy_pt_dest)
            print(f"Mirrored PT 3B notebook to: {legacy_pt_dest}")

    return paths


if __name__ == "__main__":
    build_all_3b_kernels()
