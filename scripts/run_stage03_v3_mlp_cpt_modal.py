"""STAGE 03 V3 MLP-ONLY CPT TRAINING + EVALUATION LAUNCHER (MODAL GPU A10G)

Locked Scope & Contract:
- Goal: G1 (YEAR-only closed-answer temporal QA: TempUN MCQA + Rel)
- Branch: v2-targeted-continual-alignment
- Transfer Paradigm: Zero-Shot Cross-Task Transfer (Train pure NWP -> Eval MCQA/Rel)
- Adapter Constraint: MLP Linear ONLY (gate_proj, up_proj, down_proj; NO Attention, NO DoRA)
- Locked Improvements Set:
  1. Non-linear asinh compression for z + multi-period FoNE 40D (powers10 + asinh radius)
  2. Token-level multi-year injection (MULTI_YEAR_SPAN) synchronized across train & eval
  3. AdamW + LR Scheduling (Linear Warmup + Cosine Decay / WSD) for MLP LoRA

Safety Guard:
- Defaults to --mode preflight (CPU-only dry run; zero GPU spend).
- Remote GPU dispatch requires BOTH --mode train_eval AND --confirm-gpu.
- Mounts local src/temporal_alignment directly into container (/root/temporal_alignment)
  so Modal worker runs the exact CPU-verified modules with zero train-eval mismatch.
"""

from __future__ import annotations

import gzip
import io
import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import modal

# Ensure local src/ is importable during local entrypoint preflight
_REPO_ROOT = Path(__file__).resolve().parent.parent
_SRC_DIR = _REPO_ROOT / "src"
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from temporal_alignment.stage03_eval.preflight_v3 import (  # noqa: E402
    V3ModalLaunchSpec,
    verify_v3_modal_inputs,
)

app = modal.App("stage03-v3-mlp-cpt-loggingfuture")

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "torch>=2.4.0",
        "transformers>=4.45.0",
        "accelerate>=0.34.0",
        "numpy>=1.26.0",
        "scipy",
    )
    .env({"PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"})
    .add_local_dir(
        str(_SRC_DIR / "temporal_alignment"),
        remote_path="/root/temporal_alignment",
        ignore=["__pycache__", "*.pyc"],
    )
)

volume = modal.Volume.from_name("stage03-v3-artifacts", create_if_missing=True)


@app.function(
    image=image,
    gpu="A10G",
    timeout=21600,
    retries=0,
    max_containers=1,
    volumes={"/vol": volume},
)
def run_stage03_v3_train_and_eval(
    spec_dict: Dict[str, Any],
    train_bytes: bytes,
    eval_bytes: bytes,
    manifest_bytes: bytes,
) -> Tuple[Dict[str, Any], bytes]:
    """Execute MLP-only V3 CPT training followed by MULTI_YEAR_SPAN freegen evaluation."""
    import random
    import sys as _sys
    import numpy as np
    import torch
    import torch.nn.functional as F
    from torch.utils.data import DataLoader, Dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if "/root" not in _sys.path:
        _sys.path.insert(0, "/root")

    from temporal_alignment.stage03_eval.contracts import (
        EvaluationConfig,
        ParserContract,
        TemporalPolicy,
    )
    from temporal_alignment.stage03_eval.model_adapter import (
        attach_adapters_to_model,
        matlora_temporal_context,
    )
    from temporal_alignment.stage03_eval.preflight_v3 import V3ModalLaunchSpec
    from temporal_alignment.stage03_eval.run_eval import execute_evaluation_loop
    from temporal_alignment.stage03_eval.temporal_injector import (
        GatedResidualProjection,
        compute_theta_z,
        inject_multi_year_temporal_features,
    )
    from temporal_alignment.stage03_eval.temporal_locator import resolve_all_year_mentions
    from temporal_alignment.training.cpt_optimization import (
        build_cpt_optimizer_param_groups,
        create_cpt_lr_scheduler,
    )

    spec = V3ModalLaunchSpec(**spec_dict)
    spec.validate()

    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)

    with gzip.GzipFile(fileobj=io.BytesIO(train_bytes), mode="rb") as f:
        train_cases = [json.loads(line) for line in f if line.strip()]
    with gzip.GzipFile(fileobj=io.BytesIO(eval_bytes), mode="rb") as f:
        eval_all = [json.loads(line) for line in f if line.strip()]
    manifest = json.loads(manifest_bytes.decode("utf-8"))
    frozen_indices = [s["original_eval_index"] for s in manifest["samples"][: spec.n_eval_samples]]
    frozen_cases = [eval_all[idx] for idx in frozen_indices]

    model_id = "Qwen/Qwen2.5-3B"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16).to("cuda:0")
    for p in model.parameters():
        p.requires_grad = False

    hidden_size = model.config.hidden_size
    seam_proj = GatedResidualProjection(
        in_features=40, out_features=hidden_size, gate_mode="linear"
    ).to("cuda:0", dtype=torch.bfloat16)

    # Strictly MLP-only (gate_proj, up_proj, down_proj; fails closed on Attention)
    _, core_gen = attach_adapters_to_model(
        model,
        arm=spec.arm,
        rank=spec.rank,
        alpha=spec.alpha,
    )
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

    class V3MultiYearNWPDataset(Dataset):
        def __init__(self, samples: List[Dict[str, Any]], tok: Any, max_l: int = 128):
            self.items: List[Dict[str, Any]] = []
            for s in samples:
                query = str(s.get("input", ""))
                ans = str(s.get("target", ""))
                year = int(s.get("year", 2000))
                prompt_text = f"Question: {query.strip()}\nAnswer:"
                full_text = f"{prompt_text} {ans.strip()}"

                enc = tok(full_text, truncation=True, max_length=max_l, add_special_tokens=False, return_tensors="pt")
                input_ids = enc["input_ids"].squeeze(0)
                prefix_enc = tok(prompt_text, truncation=True, max_length=max_l, add_special_tokens=False, return_tensors="pt")
                prefix_len = prefix_enc["input_ids"].shape[1]

                labels = input_ids.clone()
                labels[:prefix_len] = -100

                # Resolve ALL year mentions ONLY inside prompt_text (never target tokens)
                mentions = resolve_all_year_mentions(tok, prompt_text, anchor_year=year)
                theta, z = compute_theta_z(year, z_mode=spec.z_mode)

                self.items.append({
                    "input_ids": input_ids,
                    "labels": labels,
                    "mentions": mentions,
                    "theta": theta,
                    "z": z,
                    "year": year,
                })

        def __len__(self) -> int:
            return len(self.items)

        def __getitem__(self, idx: int) -> Dict[str, Any]:
            return self.items[idx]

    def collate_v3(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        max_l = max(len(x["input_ids"]) for x in batch)
        b_in, b_lbl, b_att, b_th, b_z, b_yr, b_men = [], [], [], [], [], [], []
        pad_id = tokenizer.pad_token_id
        for x in batch:
            seq_len = len(x["input_ids"])
            pad_len = max_l - seq_len
            b_in.append(torch.cat([x["input_ids"], torch.full((pad_len,), pad_id, dtype=torch.long)]))
            b_lbl.append(torch.cat([x["labels"], torch.full((pad_len,), -100, dtype=torch.long)]))
            b_att.append(torch.cat([torch.ones(seq_len, dtype=torch.long), torch.zeros(pad_len, dtype=torch.long)]))
            b_th.append(x["theta"])
            b_z.append(x["z"])
            b_yr.append(x["year"])
            b_men.append(x["mentions"])
        return {
            "input_ids": torch.stack(b_in),
            "labels": torch.stack(b_lbl),
            "attention_mask": torch.stack(b_att),
            "theta": torch.tensor(b_th, dtype=torch.float32),
            "z": torch.tensor(b_z, dtype=torch.float32),
            "years": torch.tensor(b_yr, dtype=torch.long),
            "mentions": b_men,
        }

    t_ds0 = time.time()
    train_ds = V3MultiYearNWPDataset(train_cases, tokenizer, max_l=128)
    train_loader = DataLoader(train_ds, batch_size=spec.micro_batch, shuffle=True, collate_fn=collate_v3)
    print(f"Prepared {len(train_ds):,} NWP multi-year samples in {time.time() - t_ds0:.1f}s ({len(train_loader):,} micro-batches/epoch).")

    param_groups = build_cpt_optimizer_param_groups(
        model,
        seam_proj=seam_proj,
        core_gen=core_gen,
        base_lr=spec.base_lr,
        loraplus_ratio=1.0,
        adapter_head_lr=spec.adapter_head_lr,
        weight_decay=spec.weight_decay,
    )
    optimizer = torch.optim.AdamW(param_groups)
    opt_steps_per_epoch = max(1, (len(train_loader) + spec.grad_accum - 1) // spec.grad_accum)
    total_opt_steps = opt_steps_per_epoch * spec.epochs
    scheduler = create_cpt_lr_scheduler(
        optimizer,
        total_steps=total_opt_steps,
        warmup_ratio=spec.warmup_ratio,
        schedule_type=spec.schedule_type,
        min_lr_ratio=spec.min_lr_ratio,
    )

    all_trainable = [p for g in param_groups for p in g["params"]]
    model.train()
    seam_proj.train()
    if core_gen is not None:
        core_gen.train()

    def compute_contrastive_terms(hidden_states: torch.Tensor, years: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        dev = hidden_states.device
        B_cur = hidden_states.size(0)
        if B_cur <= 1:
            z0 = torch.tensor(0.0, device=dev, dtype=torch.float32)
            return z0, z0
        h_pool = F.normalize(hidden_states.mean(dim=1).float(), p=2, dim=-1)
        sim_matrix = torch.matmul(h_pool, h_pool.T)
        cycles = (years.to(dev) - 3) % 60
        labels_eq = torch.eq(cycles.unsqueeze(0), cycles.unsqueeze(1))
        mask_self = torch.eye(B_cur, dtype=torch.bool, device=dev)
        pos_mask = labels_eq & (~mask_self)
        neg_mask = (~labels_eq) & (~mask_self)
        if pos_mask.any():
            sim_scaled = sim_matrix / 0.07
            sim_max = torch.max(sim_scaled, dim=-1, keepdim=True).values.detach()
            exp_sim = torch.exp(sim_scaled - sim_max)
            denom = (exp_sim * (~mask_self).float()).sum(dim=-1, keepdim=True) + 1e-8
            log_prob = (sim_scaled - sim_max) - torch.log(denom)
            pos_counts = pos_mask.sum(dim=-1)
            valid_rows = pos_counts > 0
            loss_intra = (-(log_prob * pos_mask.float()).sum(dim=-1) / (pos_counts.float() + 1e-8))[valid_rows].mean()
        else:
            loss_intra = torch.tensor(0.0, device=dev, dtype=torch.float32)
        if neg_mask.any():
            loss_inter = torch.clamp(sim_matrix - spec.margin_inter, min=0.0)[neg_mask].mean()
        else:
            loss_inter = torch.tensor(0.0, device=dev, dtype=torch.float32)
        return loss_intra, loss_inter

    t_start = time.time()
    global_step = 0
    need_contrastive = (spec.lambda_intra > 0.0) or (spec.lambda_inter > 0.0)
    print(f"Starting V3 5-epoch training: arm={spec.arm}, total_opt_steps={total_opt_steps}, schedule={spec.schedule_type}, base_lr={spec.base_lr}")
    for epoch in range(spec.epochs):
        optimizer.zero_grad()
        micro_count = 0
        running_loss = 0.0
        running_steps = 0
        t_ep0 = time.time()
        for b_idx, batch in enumerate(train_loader):
            input_ids = batch["input_ids"].to("cuda:0")
            labels = batch["labels"].to("cuda:0")
            attn_mask = batch["attention_mask"].to("cuda:0")
            theta = batch["theta"].to("cuda:0")
            z = batch["z"].to("cuda:0")
            years = batch["years"].to("cuda:0")

            embeds = model.get_input_embeddings()(input_ids)
            injected_embeds = inject_multi_year_temporal_features(
                embeds,
                seam_proj,
                batch["mentions"],
                padding_side="right",
                attention_mask=attn_mask,
                fone_mode=spec.fone_mode,
                radius_mode="asinh",
            )

            ctx = matlora_temporal_context(model, theta, z) if spec.arm == "arm5" else matlora_temporal_context(model, None, None)
            with ctx:
                outputs = model(
                    inputs_embeds=injected_embeds,
                    attention_mask=attn_mask,
                    labels=labels,
                    output_hidden_states=need_contrastive,
                )
                loss = outputs.loss
                if need_contrastive:
                    l_intra, l_inter = compute_contrastive_terms(outputs.hidden_states[-1], years)
                    loss = loss + (spec.lambda_intra * l_intra) + (spec.lambda_inter * l_inter)

                running_loss += float(loss.item())
                (loss / spec.grad_accum).backward()
            micro_count += 1
            if micro_count % spec.grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(all_trainable, 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1
                running_steps += 1
                if global_step % 200 == 0 or global_step == total_opt_steps:
                    avg_l = running_loss / max(1, running_steps * spec.grad_accum)
                    cur_lr = optimizer.param_groups[0]["lr"]
                    print(
                        f"[HEARTBEAT] Epoch {epoch + 1}/{spec.epochs} | Step {global_step}/{total_opt_steps} | "
                        f"Loss: {avg_l:.4f} | LR: {cur_lr:.6f} | Alpha: {seam_proj.alpha.item():.5f} | "
                        f"Elapsed: {time.time() - t_start:.1f}s"
                    )
                    running_loss = 0.0
                    running_steps = 0

        if micro_count % spec.grad_accum != 0:
            torch.nn.utils.clip_grad_norm_(all_trainable, 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            global_step += 1

        print(f"[EPOCH {epoch + 1}/{spec.epochs} COMPLETE] EpochTime: {(time.time() - t_ep0)/60.0:.2f}m | Alpha: {seam_proj.alpha.item():.5f}")

        # Save per-epoch checkpoint on Modal Volume
        ep_path = f"/vol/v3_{spec.arm}_ep{epoch + 1}_weights.pt"
        torch.save(
            {
                "lora_state": {k: v.cpu() for k, v in model.state_dict().items() if "lora_" in k},
                "seam_projection": {k: v.cpu() for k, v in seam_proj.state_dict().items()},
                "core_generator": {k: v.cpu() for k, v in core_gen.state_dict().items()} if core_gen is not None else None,
                "epoch": epoch + 1,
                "spec": spec_dict,
            },
            ep_path,
        )
        volume.commit()

    # Evaluate using canonical shared execute_evaluation_loop on frozen 128 cases
    model.eval()
    seam_proj.eval()
    if core_gen is not None:
        core_gen.eval()

    eval_cfg = EvaluationConfig(
        contract_version="EVAL_CONTRACT_G1_FREEGEN_V3",
        arm=spec.arm,
        temporal_policy=TemporalPolicy(spec.temporal_policy),
        parser_contract=ParserContract.LABEL_ONLY,
        device="cuda:0",
        z_mode=spec.z_mode,
        fone_mode=spec.fone_mode,
        adapter_scope="MLP_ONLY",
    )
    eval_summary = execute_evaluation_loop(
        config=eval_cfg,
        model=model,
        tokenizer=tokenizer,
        eval_cases=frozen_cases,
        seam_proj=seam_proj,
        core_gen=core_gen,
    )

    final_path = f"/vol/v3_{spec.arm}_final_weights.pt"
    torch.save(
        {
            "lora_state": {k: v.cpu() for k, v in model.state_dict().items() if "lora_" in k},
            "seam_projection": {k: v.cpu() for k, v in seam_proj.state_dict().items()},
            "core_generator": {k: v.cpu() for k, v in core_gen.state_dict().items()} if core_gen is not None else None,
            "spec": spec_dict,
        },
        final_path,
    )
    volume.commit()

    runtime_sec = round(time.time() - t_start, 2)
    receipt = {
        "timestamp": datetime.now().isoformat(),
        "spec": spec_dict,
        "global_steps": global_step,
        "runtime_seconds": runtime_sec,
        "eval_summary": eval_summary,
    }
    return receipt, json.dumps(receipt, indent=2, ensure_ascii=False).encode("utf-8")


@app.function(
    image=image,
    gpu="A10G",
    timeout=3600,
    retries=0,
    max_containers=1,
    volumes={"/vol": volume},
)
def run_stage03_v3_eval_only(
    spec_dict: Dict[str, Any],
    checkpoint_path: str,
    eval_bytes: bytes,
    manifest_bytes: bytes,
) -> Tuple[Dict[str, Any], bytes]:
    """Evaluate a saved V3 checkpoint on Modal Volume using frozen 128 cases."""
    import sys as _sys
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if "/root" not in _sys.path:
        _sys.path.insert(0, "/root")

    from temporal_alignment.stage03_eval.contracts import (
        EvaluationConfig,
        ParserContract,
        TemporalPolicy,
    )
    from temporal_alignment.stage03_eval.model_adapter import attach_adapters_to_model
    from temporal_alignment.stage03_eval.preflight_v3 import V3ModalLaunchSpec
    from temporal_alignment.stage03_eval.run_eval import execute_evaluation_loop
    from temporal_alignment.stage03_eval.temporal_injector import GatedResidualProjection

    t_start = time.time()
    volume.reload()
    spec = V3ModalLaunchSpec(**spec_dict)
    spec.validate()

    with gzip.GzipFile(fileobj=io.BytesIO(eval_bytes), mode="rb") as f:
        eval_all = [json.loads(line) for line in f if line.strip()]
    manifest = json.loads(manifest_bytes.decode("utf-8"))
    frozen_indices = [s["original_eval_index"] for s in manifest["samples"][: spec.n_eval_samples]]
    frozen_cases = [eval_all[idx] for idx in frozen_indices]

    model_id = "Qwen/Qwen2.5-3B"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16).to("cuda:0")
    for p in model.parameters():
        p.requires_grad = False

    hidden_size = model.config.hidden_size
    seam_proj = GatedResidualProjection(
        in_features=40, out_features=hidden_size, gate_mode="linear"
    ).to("cuda:0", dtype=torch.bfloat16)

    _, core_gen = attach_adapters_to_model(
        model,
        arm=spec.arm,
        rank=spec.rank,
        alpha=spec.alpha,
    )

    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if ckpt.get("lora_state"):
        model.load_state_dict(ckpt["lora_state"], strict=False)
    if ckpt.get("seam_projection"):
        seam_proj.load_state_dict(ckpt["seam_projection"])
    if core_gen is not None and ckpt.get("core_generator"):
        core_gen.load_state_dict(ckpt["core_generator"])

    model.eval()
    seam_proj.eval()
    if core_gen is not None:
        core_gen.eval()

    eval_cfg = EvaluationConfig(
        contract_version="EVAL_CONTRACT_G1_FREEGEN_V3",
        arm=spec.arm,
        temporal_policy=TemporalPolicy(spec.temporal_policy),
        parser_contract=ParserContract.LABEL_ONLY,
        device="cuda:0",
        z_mode=spec.z_mode,
        fone_mode=spec.fone_mode,
        adapter_scope="MLP_ONLY",
    )
    eval_summary = execute_evaluation_loop(
        config=eval_cfg,
        model=model,
        tokenizer=tokenizer,
        eval_cases=frozen_cases,
        seam_proj=seam_proj,
        core_gen=core_gen,
    )
    runtime_sec = round(time.time() - t_start, 2)
    receipt = {
        "timestamp": datetime.now().isoformat(),
        "mode": "eval_only",
        "checkpoint_path": checkpoint_path,
        "checkpoint_epoch": ckpt.get("epoch"),
        "spec": spec_dict,
        "runtime_seconds": runtime_sec,
        "eval_summary": eval_summary,
    }
    return receipt, json.dumps(receipt, indent=2, ensure_ascii=False).encode("utf-8")


@app.local_entrypoint()
def main(
    arm: str = "arm5",
    epochs: int = 5,
    schedule_type: str = "cosine",
    base_lr: float = 1e-4,
    warmup_ratio: float = 0.04,
    lambda_intra: float = 0.5,
    lambda_inter: float = 0.5,
    n_eval_samples: int = 128,
    max_budget_usd: float = 5.0,
    profile: str = "loggingfuture",
    mode: str = "preflight",
    checkpoint_path: str = "",
    confirm_gpu: bool = False,
    out_dir: str = "Reports/6-11Report/G1_V3_MLP_CPT",
):
    spec = V3ModalLaunchSpec(
        arm=arm,
        epochs=epochs,
        schedule_type=schedule_type,
        base_lr=base_lr,
        warmup_ratio=warmup_ratio,
        lambda_intra=lambda_intra,
        lambda_inter=lambda_inter,
        n_eval_samples=n_eval_samples,
        max_budget_usd=max_budget_usd,
        profile=profile,
    )
    preflight = verify_v3_modal_inputs(spec)
    print(json.dumps(preflight, indent=2, ensure_ascii=False))

    if mode not in ("train_eval", "eval_only"):
        print("\n[PREFLIGHT ONLY] Zero GPU jobs launched.")
        return
    if not confirm_gpu:
        raise RuntimeError(
            "Refusing to launch GPU job without --confirm-gpu. Re-run with --mode train_eval --confirm-gpu."
        )

    eval_bytes = Path("data/tempun_full_clean/eval_5292_clean.jsonl.gz").read_bytes()
    manifest_bytes = Path("Reports/6-11Report/_POST_FORENSIC_OPEN_GATES_20261105/REPLAY_SAMPLE_MANIFEST.json").read_bytes()

    from dataclasses import asdict
    if mode == "eval_only":
        if not checkpoint_path:
            raise ValueError("--checkpoint-path is required when --mode eval_only")
        receipt, receipt_bytes = run_stage03_v3_eval_only.remote(
            asdict(spec),
            checkpoint_path,
            eval_bytes,
            manifest_bytes,
        )
        ep_tag = f"{epochs}EP"
        out_p = Path(out_dir)
        out_p.mkdir(parents=True, exist_ok=True)
        out_file = out_p / f"V3_{arm.upper()}_{ep_tag}_EVAL_RECEIPT.json"
        out_file.write_bytes(receipt_bytes)
        print(f"Saved eval receipt to {out_file} (runtime: {receipt['runtime_seconds']}s)")
        return

    train_bytes = Path("data/tempun_full_clean/train_104k_clean.jsonl.gz").read_bytes()
    receipt, receipt_bytes = run_stage03_v3_train_and_eval.remote(
        asdict(spec),
        train_bytes,
        eval_bytes,
        manifest_bytes,
    )
    out_p = Path(out_dir)
    out_p.mkdir(parents=True, exist_ok=True)
    out_file = out_p / f"V3_{arm.upper()}_{epochs}EP_RECEIPT.json"
    out_file.write_bytes(receipt_bytes)
    print(f"Saved receipt to {out_file} (runtime: {receipt['runtime_seconds']}s)")
