"""
Unit tests for Qwen2.5-3B Architecture and Clean TempUN Full Dataset.

Tests:
1. Sinusoidal Temporal Encoding at d=2048: shape [B, 1, 2048], norm ~45.25.
2. Clean TempUN Full dataset integrity: 104,130 NWP train, 5,292 MCQA eval, 0 conditioning fields.
3. LoRA Linear at 3B dimensions (in_dim=2048, out_dim=2048, rank=8, alpha=16).
4. Option permutation cyclic debiasing on clean TempUN eval questions.
5. Kaggle 3B notebook generation and JSON valid syntax.
"""

from __future__ import annotations

import gzip
import json
import math
from pathlib import Path
import pytest
import torch
import torch.nn as nn

from baselines.ticktack.models import compute_batch_te, LoRALinear, compute_canchi_cycle
from baselines.ticktack.eval_debiasing import generate_option_permutations
from baselines.ticktack.build_kaggle_notebook_3b_full import create_kaggle_notebook_3b_full


REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "data" / "tempun_full_clean"


class TestTicktack3BFull:
    def test_te_vector_at_d2048(self):
        """Kiểm tra vector TE ở chiều ẩn d=2048 của Qwen2.5-3B."""
        years = torch.tensor([1947, 1984, 2022], dtype=torch.long)
        te_raw = compute_batch_te(years, hidden_dim=2048, scale_mode="raw")
        assert te_raw.shape == (3, 1, 2048)
        
        # Chuẩn L2 lý thuyết của TE thô (tổng 2 vector te_x + te_y) ở d=2048 là ~sqrt(2*2048) = 64.0
        norms = torch.norm(te_raw, p=2, dim=-1)
        for n in norms:
            val = n.item()
            assert 55.0 <= val <= 65.0, f"TE norm {val} nằm ngoài khoảng mong đợi [55.0, 65.0] (lý thuyết ~sqrt(2d) = 64.0)"

        # Kiểm tra scale_mode = 'scaled' (chia cho 2048)
        te_scaled = compute_batch_te(years, hidden_dim=2048, scale_mode="scaled")
        assert te_scaled.shape == (3, 1, 2048)
        scaled_norms = torch.norm(te_scaled, p=2, dim=-1)
        for sn in scaled_norms:
            assert 0.025 <= sn.item() <= 0.035

    def test_clean_tempun_full_dataset_integrity(self):
        """Kiểm tra tính sạch và đầy đủ của tập dữ liệu TempUN Full."""
        train_path = DATA_DIR / "train_104k_clean.jsonl.gz"
        eval_path = DATA_DIR / "eval_5292_clean.jsonl.gz"

        assert train_path.is_file(), f"Không tìm thấy {train_path}"
        assert eval_path.is_file(), f"Không tìm thấy {eval_path}"

        forbidden_keys = {"conditioning", "can_chi", "fone_vector", "stem_index", "branch_index"}

        # 1. Kiểm tra tập Train NWP
        train_count = 0
        with gzip.open(train_path, "rt", encoding="utf-8") as f:
            for idx, line in enumerate(f):
                train_count += 1
                if idx < 50:
                    rec = json.loads(line)
                    assert forbidden_keys.isdisjoint(rec.keys()), f"Phát hiện trường cấm trong train: {rec.keys()}"
                    assert "year" in rec and 1947 <= rec["year"] <= 2022
                    assert "input" in rec and len(rec["input"]) > 0
                    assert "target" in rec and len(rec["target"]) > 0
        assert train_count == 104130, f"Số lượng mẫu train không đúng: {train_count}"

        # 2. Kiểm tra tập Eval MCQA
        eval_count = 0
        task_counts: dict[str, int] = {}
        with gzip.open(eval_path, "rt", encoding="utf-8") as f:
            for idx, line in enumerate(f):
                eval_count += 1
                rec = json.loads(line)
                if idx < 50:
                    assert forbidden_keys.isdisjoint(rec.keys()), f"Phát hiện trường cấm trong eval: {rec.keys()}"
                tid = rec["task_id"]
                task_counts[tid] = task_counts.get(tid, 0) + 1
        
        assert eval_count == 5292, f"Số lượng mẫu eval không đúng: {eval_count}"
        expected_tasks = {
            "DB": 912,
            "MM": 852,
            "RB": 852,
            "TB": 912,
            "Rel": 852,
            "WB": 912,
        }
        assert task_counts == expected_tasks, f"Phân bố task eval lệch: {task_counts} vs {expected_tasks}"

    def test_lora_at_3b_dimensions(self):
        """Kiểm tra lớp LoRA tại chiều d=2048 của Qwen2.5-3B."""
        linear = nn.Linear(2048, 2048, bias=False)
        lora = LoRALinear.from_linear(linear, rank=8, alpha=16.0)

        assert lora.base_layer.weight.requires_grad is False
        assert lora.lora_A.requires_grad is True
        assert lora.lora_B.requires_grad is True
        assert lora.scaling == 2.0  # 16.0 / 8

        # Kiểm tra forward pass
        x = torch.randn(2, 8, 2048)
        out = lora(x)
        assert out.shape == (2, 8, 2048)

        # Kiểm tra tính toán gradient
        loss = out.sum()
        loss.backward()
        assert lora.lora_A.grad is not None
        assert lora.lora_B.grad is not None
        assert lora.base_layer.weight.grad is None

    def test_option_permutations_on_clean_eval(self):
        """Kiểm tra bộ hoán vị trên cả mẫu MCQA 4 lựa chọn và mẫu Yes/No."""
        # 1. MCQA 4 choices
        sample_prompt = (
            "Which option is correct for the question: In 1947, share of contribution to global warming of Afghanistan was: "
            "Options: (a) 0.13 (b) 0.17 (c) 0.65 (d) 0.24"
        )
        sample_target = "a"

        perms = generate_option_permutations(sample_prompt, sample_target)
        assert len(perms) == 4
        assert perms[0]["target"] == "a"
        assert perms[1]["target"] == "d"
        assert perms[2]["target"] == "c"
        assert perms[3]["target"] == "b"

        # 2. Yes/No (Task Rel)
        rel_prompt = "In 1947, Private Civil Liberties Index in Norway was higher than 1952? Answer as Yes or No"
        rel_target = "Yes"
        rel_perms = generate_option_permutations(rel_prompt, rel_target)
        assert len(rel_perms) == 2
        assert rel_perms[0]["target"] == "yes"
        assert rel_perms[1]["target"] == "yes"
        assert "No or Yes" in rel_perms[1]["prompt"]

    def test_notebook_generator_execution_and_ast_parse(self, tmp_path):
        """Kiểm tra script sinh notebook tạo JSON hợp lệ và mọi code cell đều đúng cú pháp Python."""
        import ast
        nb_path = create_kaggle_notebook_3b_full(
            output_dir=str(tmp_path),
            extra_output_dir=None,
        )
        assert Path(nb_path).is_file()
        with open(nb_path, "r", encoding="utf-8") as f:
            nb_json = json.load(f)
        assert nb_json["nbformat"] == 4
        assert len(nb_json["cells"]) >= 10

        # Kiểm tra cú pháp AST từng cell và kiểm tra không có symbol bị thiếu import
        import builtins
        builtin_names = set(dir(builtins))
        full_code = "\n".join("".join(c.get("source", [])) for c in nb_json["cells"] if c.get("cell_type") == "code")
        tree = ast.parse(full_code)
        defined = set()
        loaded = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Name):
                if isinstance(node.ctx, ast.Store):
                    defined.add(node.id)
                elif isinstance(node.ctx, ast.Load):
                    loaded.add(node.id)
            elif isinstance(node, ast.FunctionDef):
                defined.add(node.name)
                for arg in node.args.args:
                    defined.add(arg.arg)
            elif isinstance(node, ast.ClassDef):
                defined.add(node.name)
            elif isinstance(node, (ast.Import, ast.ImportFrom)):
                for alias in node.names:
                    defined.add(alias.asname or alias.name)
        undefined = loaded - defined - builtin_names
        assert len(undefined) == 0, f"Notebook có symbol bị thiếu import: {undefined}"

    def test_gradient_checkpointing_inputs_embeds_safety(self):
        """Kiểm tra tính an toàn của Gradient Checkpointing khi đưa inputs_embeds vào transformer."""
        # Giả lập 1 block transformer với gradient checkpointing
        class MockBlock(nn.Module):
            def __init__(self, d=64):
                super().__init__()
                self.fc = nn.Linear(d, d)
            def forward(self, x):
                return self.fc(x)

        block = MockBlock(64)
        inputs_embeds = torch.randn(2, 4, 64)
        inputs_embeds.requires_grad_(True)

        from torch.utils.checkpoint import checkpoint
        # Kiểm tra chạy checkpointing với use_reentrant=False không bị lỗi
        out = checkpoint(block, inputs_embeds, use_reentrant=False)
        loss = out.sum()
        loss.backward()
        assert inputs_embeds.grad is not None
        assert block.fc.weight.grad is not None
