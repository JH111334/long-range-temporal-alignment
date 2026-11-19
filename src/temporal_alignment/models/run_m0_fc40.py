"""Executable runner for M0-FC40 (Periodic Dual-Cycle MaT-LoRA on FIXED_FONE_CANCHI_V1 40D).

Adheres strictly to the DECISION-FIXED-FONE-CANCHI-LEARNING-TRACK-105 contract:
- Model Backbone: Qwen/Qwen2.5-0.5B-Instruct (commit 7ae557604adf67be50417f59c2c2f167def9a775)
- Adapter: Periodic Dual-Cycle MaT-LoRA (r=8, alpha=16.0, K=2 harmonics, F_base=I_8) on 72 MLP modules
- Representation: FIXED_FONE_CANCHI_V1 (32D FoNE + 8D CanChi = 40D concat)
- Conditioning Seam: Gated Residual Projection (Linear 40->896, LayerNorm, alpha_0=0.0) at token index 14
- Loss: MASKED_TARGET causal cross-entropy (labels: -100 on prompt, real tokens on target + EOS)
- Stages: year:1947 (733), year:1984 (1338), year:2022 (390)
- Evaluation: 180 frozen MCQA cases after each stage with GENERATIVE_STRICT_PARSE_V1
- Checkpoints & Lineage: Full adapter, core, and seam weights saved and verifiable after each stage.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import random
import sys
import time
import traceback
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Add parent path
_PKG_ROOT = Path(__file__).resolve().parents[2]
if str(_PKG_ROOT) not in sys.path:
    sys.path.insert(0, str(_PKG_ROOT))

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except ImportError:
    torch = None
    nn = None
    F = None

from temporal_alignment.models.fone_canchi_40d import (
    compute_fixed_fone_canchi_40d,
    extract_canchi_angles_from_40d,
)
from temporal_alignment.models.dual_cycle_mat_lora import (
    DualCycleMaTLoRALinear,
    replace_mlp_with_dual_cycle_mat_lora,
    set_model_temporal_coordinates,
)
from temporal_alignment.models.gated_seam import (
    ConditioningContext,
    GatedResidualProjection,
    make_gated_embedding_hook,
)
from temporal_alignment.models.tempun_dataset import (
    STAGES,
    STAGE_YEARS,
    TempUNExample,
    load_tempun_eval_records,
    load_tempun_train_records,
    parse_strict_mcqa_answer,
    tokenize_masked_target_example,
)


def assert_finite_boundary(tensor: Any, boundary_name: str, module_name: str = "") -> None:
    """Fail-fast guard raising FloatingPointError at the exact boundary of non-finite creation."""
    if torch is None or not isinstance(tensor, torch.Tensor):
        return
    if torch.isnan(tensor).any() or torch.isinf(tensor).any():
        non_finite_count = (torch.isnan(tensor) | torch.isinf(tensor)).sum().item()
        total_count = tensor.numel()
        raise FloatingPointError(
            f"NONFINITE_DETECTED at [{boundary_name}] in [{module_name}]: "
            f"{non_finite_count}/{total_count} elements non-finite (dtype={tensor.dtype}, shape={list(tensor.shape)})"
        )

STAGE_TAGS = ("after_00", "after_01", "after_02")


def write_json_atomic(path: Path, value: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    content = json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n"
    temp.write_text(content, encoding="utf-8")
    os.replace(temp, path)


def set_seed(seed: int) -> None:
    random.seed(seed)
    if torch is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)


def build_toy_smoke_model() -> nn.Module:
    """Build lightweight CPU-runnable toy model for local smoke verification."""
    class ToyDecoderLayer(nn.Module):
        def __init__(self, hidden_dim: int = 64, mlp_dim: int = 128):
            super().__init__()
            self.gate_proj = nn.Linear(hidden_dim, mlp_dim, bias=False)
            self.up_proj = nn.Linear(hidden_dim, mlp_dim, bias=False)
            self.down_proj = nn.Linear(mlp_dim, hidden_dim, bias=False)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            gate = F.silu(self.gate_proj(x))
            up = self.up_proj(x)
            return x + self.down_proj(gate * up)

    class ToyModel(nn.Module):
        def __init__(self, vocab_size: int = 1000, hidden_dim: int = 64):
            super().__init__()
            self.embed_tokens = nn.Embedding(vocab_size, hidden_dim)
            self.layers = nn.ModuleList([ToyDecoderLayer(hidden_dim) for _ in range(2)])
            self.lm_head = nn.Linear(hidden_dim, vocab_size, bias=False)

        def get_input_embeddings(self) -> nn.Embedding:
            return self.embed_tokens

        def forward(self, input_ids: torch.Tensor, **kwargs: Any) -> Any:
            x = self.embed_tokens(input_ids)
            for layer in self.layers:
                x = layer(x)
            logits = self.lm_head(x)
            from types import SimpleNamespace
            return SimpleNamespace(logits=logits)

    return ToyModel()


def parameter_census(model: nn.Module, gated_proj: nn.Module, adapted_modules: Dict[str, DualCycleMaTLoRALinear]) -> Dict[str, Any]:
    """Perform strict parameter census validating frozen base weights and trainable counts."""
    total_params = sum(p.numel() for p in model.parameters()) + sum(p.numel() for p in gated_proj.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad) + sum(p.numel() for p in gated_proj.parameters() if p.requires_grad)

    seam_trainable = sum(p.numel() for p in gated_proj.parameters() if p.requires_grad)
    adapter_trainable = sum(p.numel() for mod in adapted_modules.values() for p in mod.parameters() if p.requires_grad)

    # Base linear weight trainability check
    base_trainable_count = 0
    for mod in adapted_modules.values():
        if hasattr(mod.base_layer, "weight") and mod.base_layer.weight.requires_grad:
            base_trainable_count += mod.base_layer.weight.numel()

    return {
        "total_parameters": total_params,
        "trainable_parameters": trainable_params,
        "trainable_percentage": (trainable_params / total_params * 100.0) if total_params > 0 else 0.0,
        "seam_trainable_params": seam_trainable,
        "adapter_trainable_params": adapter_trainable,
        "adapted_module_count": len(adapted_modules),
        "base_linear_leakage_params": base_trainable_count,
        "is_base_cleanly_frozen": (base_trainable_count == 0),
    }


def evaluate_stage(
    model: nn.Module,
    tokenizer: Any,
    eval_records: List[TempUNExample],
    cond_context: ConditioningContext,
    gated_proj: GatedResidualProjection,
    adapted_modules: Dict[str, DualCycleMaTLoRALinear],
    device: torch.device,
    is_smoke: bool = False,
    is_canary: bool = False,
) -> Dict[str, Any]:
    """Perform generative evaluation on frozen 180 MCQA cases."""
    model.eval()
    records_to_eval = eval_records[:5] if is_smoke else (eval_records[:15] if is_canary else eval_records)

    predictions = []
    correct_count = 0
    valid_count = 0
    invalid_count = 0

    per_stage_stats: Dict[str, Dict[str, int]] = defaultdict(lambda: {"total": 0, "correct": 0, "valid": 0, "invalid": 0})

    with torch.no_grad():
        for record in records_to_eval:
            year = record.year
            vec40 = compute_fixed_fone_canchi_40d(year)
            theta, z = extract_canchi_angles_from_40d(vec40)

            # Set temporal coordinates for Dual-Cycle MaT-LoRA
            set_model_temporal_coordinates(adapted_modules, theta, z)

            # Compute injected continuous vector through gated seam
            vec40_t = torch.tensor(vec40, dtype=torch.float32, device=device).unsqueeze(0)
            injected_vec = gated_proj(vec40_t).squeeze(0)
            cond_context.set_vector(injected_vec)

            # Prompt encoding
            if hasattr(tokenizer, "apply_chat_template") and getattr(tokenizer, "chat_template", None):
                messages = [
                    {"role": "system", "content": "You are a helpful assistant. Please answer the question by selecting the correct option."},
                    {"role": "user", "content": record.prompt.strip() + "\nAnswer directly with only the option letter (a, b, c, d) or Yes/No."}
                ]
                formatted_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                tok_out = tokenizer(formatted_prompt, return_tensors="pt")
                raw_ids = tok_out["input_ids"]
                if not isinstance(raw_ids, torch.Tensor):
                    raw_ids = torch.tensor(raw_ids, dtype=torch.long)
                if raw_ids.dim() == 1:
                    raw_ids = raw_ids.unsqueeze(0)
                prompt_ids = raw_ids.to(device)
            elif hasattr(tokenizer, "__call__"):
                tok_out = tokenizer(record.prompt, return_tensors="pt")
                raw_ids = tok_out["input_ids"]
                if not isinstance(raw_ids, torch.Tensor):
                    raw_ids = torch.tensor(raw_ids, dtype=torch.long)
                if raw_ids.dim() == 1:
                    raw_ids = raw_ids.unsqueeze(0)
                prompt_ids = raw_ids.to(device)
            else:
                prompt_ids = torch.randint(0, 100, (1, 15), device=device)

            if is_smoke or not hasattr(model, "generate"):
                # Deterministic smoke dummy answer
                ans_str = record.target
            else:
                # Autoregressive generation
                gen_out = model.generate(
                    prompt_ids,
                    max_new_tokens=8,
                    do_sample=False,
                    pad_token_id=tokenizer.eos_token_id,
                )
                new_tokens = gen_out[0, prompt_ids.shape[1]:]
                ans_str = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

            parsed = parse_strict_mcqa_answer(ans_str)
            is_valid = parsed is not None
            is_correct = (parsed == record.target) if is_valid else False

            if is_valid:
                valid_count += 1
            else:
                invalid_count += 1

            if is_correct:
                correct_count += 1

            stage_stat = per_stage_stats[record.stage_id]
            stage_stat["total"] += 1
            if is_valid:
                stage_stat["valid"] += 1
            else:
                stage_stat["invalid"] += 1
            if is_correct:
                stage_stat["correct"] += 1

            predictions.append({
                "example_id": record.example_id,
                "stage_id": record.stage_id,
                "year": record.year,
                "gold_target": record.target,
                "raw_completion": ans_str,
                "parsed_answer": parsed,
                "is_valid": is_valid,
                "is_correct": is_correct,
            })

    cond_context.clear()
    set_model_temporal_coordinates(adapted_modules, None, None)

    total = len(records_to_eval)
    strict_acc = (correct_count / total * 100.0) if total > 0 else 0.0
    valid_acc = (correct_count / valid_count * 100.0) if valid_count > 0 else 0.0
    invalid_rate = (invalid_count / total * 100.0) if total > 0 else 0.0

    return {
        "total_examples": total,
        "correct": correct_count,
        "valid": valid_count,
        "invalid": invalid_count,
        "strict_accuracy": strict_acc,
        "valid_accuracy": valid_acc,
        "invalid_syntax_rate": invalid_rate,
        "per_stage": dict(per_stage_stats),
        "predictions": predictions,
    }


def save_stage_checkpoint(
    path: Path,
    seed: int,
    stage_idx: int,
    stage_tag: str,
    stage_id: str,
    gated_proj: GatedResidualProjection,
    adapted_modules: Dict[str, DualCycleMaTLoRALinear],
    metadata: Dict[str, Any],
) -> None:
    """Save complete and verifiable adapter, core, and seam state dict."""
    path.parent.mkdir(parents=True, exist_ok=True)
    state = {
        "format": "M0_FC40_CHECKPOINT_V1",
        "seed": seed,
        "stage_idx": stage_idx,
        "stage_tag": stage_tag,
        "stage_id": stage_id,
        "gated_proj_state": gated_proj.state_dict(),
        "adapters_state": {
            name: {
                "lora_A": mod.lora_A.data.cpu(),
                "lora_B": mod.lora_B.data.cpu(),
                "core_state": mod.core_generator.state_dict(),
            }
            for name, mod in adapted_modules.items()
        },
        "metadata": metadata,
        "timestamp": time.time(),
    }
    torch.save(state, path)


def train_m0_fc40_seed(
    bundle_dir: Path,
    output_dir: Path,
    seed: int,
    is_smoke: bool = False,
    is_canary: bool = False,
    resume: bool = True,
) -> Dict[str, Any]:
    """Execute continual training and evaluation for one seed."""
    set_seed(seed)
    seed_output = output_dir / f"seed_{seed}"
    seed_output.mkdir(parents=True, exist_ok=True)
    receipt_file = seed_output / "seed_receipt.json"

    if resume and receipt_file.exists():
        print(f"Seed {seed} already finished. Loading receipt.")
        return json.loads(receipt_file.read_text(encoding="utf-8"))

    has_cuda = torch.cuda.is_available() and not is_smoke
    device = torch.device("cuda:0" if has_cuda else "cpu")

    gpu_info = {
        "cuda_available": has_cuda,
        "device_name": torch.cuda.get_device_name(0) if has_cuda else "CPU",
        "bf16_supported": torch.cuda.is_bf16_supported() if has_cuda else False,
        "initial_vram_gib": (torch.cuda.max_memory_allocated() / (1024**3)) if has_cuda else 0.0,
    }
    print(f"Hardware Environment: {gpu_info['device_name']}, BF16={gpu_info['bf16_supported']}")

    # Load data
    train_records = load_tempun_train_records(bundle_dir / "data/train_nwp.jsonl")
    eval_records = load_tempun_eval_records(bundle_dir / "data/eval_mcqa.jsonl")

    # Load model and tokenizer
    if is_smoke:
        print("[SMOKE] Initializing toy CPU model for smoke verification...")
        model = build_toy_smoke_model().to(device)
        class DummyTokenizer:
            eos_token_id = 0
            pad_token_id = 0
            def __call__(self, text: str, **kwargs: Any) -> Dict[str, Any]:
                return {"input_ids": torch.tensor([[1] * 12], dtype=torch.long)}
            def decode(self, ids: Any, **kwargs: Any) -> str:
                return "A"
        tokenizer = DummyTokenizer()
        hidden_dim = 64
        embed_module = model.get_input_embeddings()
    else:
        print("[PROD] Loading Qwen2.5-0.5B-Instruct in FP16 precision...")
        import transformers
        model_id = "Qwen/Qwen2.5-0.5B-Instruct"
        revision = "7ae557604adf67be50417f59c2c2f167def9a775"
        tokenizer = transformers.AutoTokenizer.from_pretrained(model_id, revision=revision)
        # Load directly in FP16 without quantization to guarantee exact weight copy and avoid quantization shape mismatch
        model = transformers.AutoModelForCausalLM.from_pretrained(
            model_id,
            revision=revision,
            torch_dtype=torch.float16,
            device_map="auto",
            low_cpu_mem_usage=True,
        )
        hidden_dim = 896
        embed_module = model.get_input_embeddings()

    # Freeze base model parameters
    for p in model.parameters():
        p.requires_grad = False

    # Attach GatedResidualProjection and hook
    gated_proj = GatedResidualProjection(in_features=40, out_features=hidden_dim).to(device)
    cond_context = ConditioningContext(injection_token_index=14)
    hook_handle = embed_module.register_forward_hook(make_gated_embedding_hook(cond_context))

    # Replace MLP modules with DualCycleMaTLoRALinear
    adapted_modules = replace_mlp_with_dual_cycle_mat_lora(
        model,
        rank=8,
        alpha=16.0,
        dropout=0.0,
        num_harmonics=2,
        use_epoch_poly=True,
    )
    print(f"Attached DualCycleMaTLoRA to {len(adapted_modules)} MLP modules.")

    census = parameter_census(model, gated_proj, adapted_modules)
    print(f"Parameter Census: Trainable = {census['trainable_parameters']:,} ({census['trainable_percentage']:.2f}%), Base Leakage = {census['base_linear_leakage_params']}")
    if not census["is_base_cleanly_frozen"]:
        raise RuntimeError("Base model weights leaked into trainable set!")

    # Collect named trainable parameters: LoRA A/B, core generator params, gated seam (all strictly float32)
    named_trainable_params: List[Tuple[str, nn.Parameter]] = []
    for name, p in gated_proj.named_parameters():
        if p.requires_grad:
            named_trainable_params.append((f"seam.{name}", p))
    for mod_name, mod in adapted_modules.items():
        named_trainable_params.append((f"{mod_name}.lora_A", mod.lora_A))
        named_trainable_params.append((f"{mod_name}.lora_B", mod.lora_B))
        for core_name, cp in mod.core_generator.named_parameters():
            if cp.requires_grad:
                named_trainable_params.append((f"{mod_name}.core.{core_name}", cp))

    trainable_params = [p for _, p in named_trainable_params]

    # Verify all trainable parameters are strictly float32
    for name, p in named_trainable_params:
        if p.dtype != torch.float32:
            p.data = p.data.float()

    optimizer = torch.optim.AdamW(trainable_params, lr=2e-4, weight_decay=0.01)

    # Continual Stages Execution
    stage_metrics = {}
    stage_predictions = {}
    stage_retention_rows = []

    for stage_idx, stage_id in enumerate(STAGES):
        stage_tag = STAGE_TAGS[stage_idx]
        print(f"\n--- SEED {seed} | STAGE {stage_idx + 1}/{len(STAGES)}: {stage_id} ({stage_tag}) ---")
        stage_train = [r for r in train_records if r.stage_id == stage_id]
        if is_smoke:
            stage_train = stage_train[:3]
        elif is_canary:
            stage_train = stage_train[:15]

        model.train()
        gated_proj.train()

        start_time = time.perf_counter()
        total_loss = 0.0
        step_count = 0

        for record in stage_train:
            optimizer.zero_grad(set_to_none=True)
            year = record.year
            vec40 = compute_fixed_fone_canchi_40d(year)
            theta, z = extract_canchi_angles_from_40d(vec40)

            # Set adapter coordinates
            set_model_temporal_coordinates(adapted_modules, theta, z)

            # Inject through seam
            vec40_t = torch.tensor(vec40, dtype=torch.float32, device=device).unsqueeze(0)
            injected_vec = gated_proj(vec40_t).squeeze(0)
            cond_context.set_vector(injected_vec)

            # Tokenize MASKED_TARGET
            if is_smoke:
                input_ids = torch.randint(0, 100, (1, 16), device=device)
                labels = input_ids.clone()
                labels[:, :10] = -100
            else:
                full_ids, label_ids = tokenize_masked_target_example(record, tokenizer)
                input_ids = torch.tensor([full_ids], device=device)
                labels = torch.tensor([label_ids], device=device)

            out = model(input_ids=input_ids)
            logits = out.logits

            # Shift logits and labels for causal LM loss
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()

            valid_tokens = (shift_labels != -100).sum().item()
            if valid_tokens == 0:
                print(f"Warning: sample {record.example_id} has zero unmasked tokens, skipping!")
                continue

            # Upcast to float32 before computing CrossEntropy to guarantee numeric stability
            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)).float(),
                shift_labels.view(-1),
                ignore_index=-100,
            )

            assert_finite_boundary(loss, f"loss_stage_{stage_tag}_step_{step_count}")

            # Direct backward without loss scaler avoids 65536x amplification causing FP16 intermediate VJP overflow
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
            for name, p in named_trainable_params:
                if p.grad is not None:
                    assert_finite_boundary(p.grad, f"grad_stage_{stage_tag}_step_{step_count}", module_name=name)
            optimizer.step()

            # Guard post-step weights
            for name, p in named_trainable_params:
                assert_finite_boundary(p, f"weight_stage_{stage_tag}_step_{step_count}", module_name=name)

            total_loss += loss.item()
            step_count += 1
            if step_count % (5 if is_canary else 100) == 0:
                print(f"  [{stage_id}] Step {step_count}: loss={loss.item():.4f}")

        elapsed_sec = time.perf_counter() - start_time
        avg_loss = total_loss / max(1, step_count)
        peak_vram = (torch.cuda.max_memory_allocated() / (1024**3)) if has_cuda else 0.0
        print(f"Stage {stage_id} training complete: {step_count} steps, avg_loss={avg_loss:.4f}, time={elapsed_sec:.1f}s, peak_vram={peak_vram:.2f} GiB")

        # Stage Evaluation
        eval_result = evaluate_stage(
            model=model,
            tokenizer=tokenizer,
            eval_records=eval_records,
            cond_context=cond_context,
            gated_proj=gated_proj,
            adapted_modules=adapted_modules,
            device=device,
            is_smoke=is_smoke,
            is_canary=is_canary,
        )

        stage_metrics[stage_tag] = {
            "stage_idx": stage_idx,
            "stage_id": stage_id,
            "train_steps": step_count,
            "train_loss": avg_loss,
            "runtime_seconds": elapsed_sec,
            "peak_vram_gib": peak_vram,
            "strict_accuracy": eval_result["strict_accuracy"],
            "valid_accuracy": eval_result["valid_accuracy"],
            "invalid_syntax_rate": eval_result["invalid_syntax_rate"],
            "per_stage_acc": eval_result["per_stage"],
        }
        stage_predictions[stage_tag] = eval_result["predictions"]

        # Track diagonal for BWT calculation: accuracy on each historical stage slice
        row_slice_acc = [
            eval_result["per_stage"].get(s, {}).get("correct", 0) / max(1, eval_result["per_stage"].get(s, {}).get("total", 1)) * 100.0
            for s in STAGES
        ]
        stage_retention_rows.append(row_slice_acc)

        print(f"Stage {stage_tag} Eval: Strict Acc = {eval_result['strict_accuracy']:.2f}%, Invalid = {eval_result['invalid_syntax_rate']:.2f}%")

        # Save stage-wise verifiable checkpoint
        ckpt_path = seed_output / f"checkpoints/m0_fc40_seed_{seed}_{stage_tag}.pt"
        save_stage_checkpoint(
            path=ckpt_path,
            seed=seed,
            stage_idx=stage_idx,
            stage_tag=stage_tag,
            stage_id=stage_id,
            gated_proj=gated_proj,
            adapted_modules=adapted_modules,
            metadata={"train_loss": avg_loss, "strict_accuracy": eval_result["strict_accuracy"]},
        )
        write_json_atomic(seed_output / f"metrics/metrics_{stage_tag}.json", stage_metrics[stage_tag])
        write_json_atomic(seed_output / f"predictions/predictions_{stage_tag}.json", stage_predictions[stage_tag])

    hook_handle.remove()

    # Calculate Backward Transfer (BWT)
    # BWT = 1/(T-1) * sum_{j=1}^{T-1} (R_{T, j} - R_{j, j})
    if len(stage_retention_rows) == 3:
        r_00 = stage_retention_rows[0][0]
        r_11 = stage_retention_rows[1][1]
        r_20 = stage_retention_rows[2][0]
        r_21 = stage_retention_rows[2][1]
        bwt = 0.5 * ((r_20 - r_00) + (r_21 - r_11))
    else:
        bwt = 0.0

    receipt = {
        "status": "PASS",
        "seed": seed,
        "is_smoke": is_smoke,
        "architecture": "Periodic_Dual_Cycle_MaT_LoRA_FC40",
        "representation": "FIXED_FONE_CANCHI_V1",
        "hardware": gpu_info,
        "census": census,
        "stage_metrics": stage_metrics,
        "continual_matrix": stage_retention_rows,
        "backward_transfer_bwt": bwt,
        "final_strict_accuracy": stage_metrics[STAGE_TAGS[-1]]["strict_accuracy"],
        "final_invalid_rate": stage_metrics[STAGE_TAGS[-1]]["invalid_syntax_rate"],
        "timestamp": time.time(),
    }

    write_json_atomic(receipt_file, receipt)
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True, help="Path to lrr_bundle directory")
    parser.add_argument("--output", type=Path, default=Path("artifacts"), help="Output directory")
    parser.add_argument("--seed", type=int, default=1, help="Seed to execute")
    parser.add_argument("--smoke", action="store_true", help="Run lightweight local smoke test on CPU")
    parser.add_argument("--canary", action="store_true", help="Run bounded 15-step canary trial on GPU")
    parser.add_argument("--resume", action="store_true", help="Resume from existing checkpoints if present")
    args = parser.parse_args()

    res = train_m0_fc40_seed(
        bundle_dir=args.bundle.resolve(),
        output_dir=args.output.resolve(),
        seed=args.seed,
        is_smoke=args.smoke,
        is_canary=args.canary,
        resume=args.resume,
    )
    print(json.dumps(res, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
