"""STAGE 03 G1 MLP-ONLY CPT TRAINING, SMOKE, AND EVALUATION LAUNCHER (MODAL GPU A10G)

Locked Scope & Contract:
- Goal: G1 (YEAR-only closed-answer temporal QA: TempUN MCQA + Rel)
- Branch: v2-targeted-continual-alignment
- Transfer Paradigm: Zero-Shot Cross-Task Transfer (NWP + non-target general replay -> Eval MCQA/Rel)
- Adapter Constraint: MLP Linear ONLY (gate_proj, up_proj, down_proj; NO Attention, NO DoRA)
- Repaired Recipes (V2 scientific baseline + P0 fixes A/B/C):
  1. G1_SINGLE_YEAR_REPLAY_FIXED (R1): dyadic/linear 40D + linear z, YEAR_SPAN single question-body anchor,
     15% non-temporal SQuAD v1.1 general replay, masked prompt contrastive loss, uniform AdamW LR=2e-4.
  2. G1_SINGLE_YEAR_REPLAY_COSINE (R2): identical to R1 except 4% linear warmup + cosine decay to 0.10*peak (2e-5).
  3. LEGACY_V3_HISTORICAL: retained for archived V3 checkpoint eval_only inspection only.

Safety Guard:
- Defaults to --mode preflight (CPU-only dry run; zero GPU spend).
- Remote GPU dispatch requires BOTH --mode in {smoke, train_eval, eval_only} AND --confirm-gpu.
- Mounts local src/temporal_alignment directly into container (/root/temporal_alignment) and verifies code digest.
"""

from __future__ import annotations

import gzip
import hashlib
import io
import json
import sys
import time
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import modal

_REPO_ROOT = Path(__file__).resolve().parent.parent
_SRC_DIR = _REPO_ROOT / "src"
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from temporal_alignment.stage03_eval.preflight_v3 import (  # noqa: E402
    LOCKED_G1_REPAIR_RECIPES,
    PINNED_NWP_COUNT,
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


def compute_source_tree_sha256(src_root: Path) -> str:
    """Compute a deterministic SHA-256 across all Python modules under src_root."""
    h = hashlib.sha256()
    for p in sorted(src_root.rglob("*.py"), key=lambda path: path.relative_to(src_root).as_posix()):
        rel = p.relative_to(src_root).as_posix()
        h.update(rel.encode("utf-8"))
        h.update(hashlib.sha256(p.read_bytes()).digest())
    return h.hexdigest()


def _state_digest(
    model: Any,
    seam_proj: Any,
    core_gen: Any = None,
) -> str:
    import torch

    h = hashlib.sha256()
    items: List[Tuple[str, torch.Tensor]] = [
        (f"lora.{k}", v.detach().cpu().to(torch.float32))
        for k, v in sorted(model.state_dict().items())
        if "lora_" in k
    ]
    if seam_proj is not None:
        items.extend(
            (f"seam.{k}", v.detach().cpu().to(torch.float32))
            for k, v in sorted(seam_proj.state_dict().items())
        )
    if core_gen is not None:
        items.extend(
            (f"core.{k}", v.detach().cpu().to(torch.float32))
            for k, v in sorted(core_gen.state_dict().items())
        )
    for k, tensor in items:
        h.update(k.encode("utf-8"))
        h.update(tensor.contiguous().numpy().tobytes())
    return h.hexdigest()


def _build_remote_runtime_helpers():
    import torch
    import torch.nn.functional as F
    from torch.utils.data import DataLoader, Dataset

    from temporal_alignment.stage03_eval.contracts import (
        EvaluationConfig,
        ParserContract,
        TemporalPolicy,
    )
    from temporal_alignment.stage03_eval.model_adapter import (
        attach_adapters_to_model,
        load_checkpoint_weights,
        matlora_temporal_context,
    )
    from temporal_alignment.stage03_eval.run_eval import execute_evaluation_loop
    from temporal_alignment.stage03_eval.temporal_injector import (
        GatedResidualProjection,
        compute_temporal_40d,
        compute_theta_z,
        inject_multi_year_temporal_features,
    )
    from temporal_alignment.stage03_eval.temporal_locator import (
        build_temporal_mask,
        resolve_all_year_mentions,
        resolve_question_body_anchor,
    )
    from temporal_alignment.training.cpt_optimization import (
        build_cpt_optimizer_param_groups,
        compute_masked_temporal_contrastive_loss,
        create_cpt_lr_scheduler,
        interleave_general_replay,
        tokenize_supervised_pair,
    )

    class PreTokenizedMixedDataset(Dataset):
        def __init__(self, items: List[Dict[str, Any]]):
            self.items = items

        def __len__(self) -> int:
            return len(self.items)

        def __getitem__(self, idx: int) -> Dict[str, Any]:
            return self.items[idx]

    def collate_mixed(batch: List[Dict[str, Any]], pad_id: int) -> Dict[str, Any]:
        max_l = max(len(x["input_ids"]) for x in batch)
        b_in, b_lbl, b_att, b_pm, b_act = [], [], [], [], []
        b_th, b_z, b_yr, b_men, b_src, b_idx = [], [], [], [], [], []
        for x in batch:
            seq_len = len(x["input_ids"])
            pad_len = max_l - seq_len
            b_in.append(torch.cat([x["input_ids"], torch.full((pad_len,), pad_id, dtype=torch.long)]))
            b_lbl.append(torch.cat([x["labels"], torch.full((pad_len,), -100, dtype=torch.long)]))
            b_pm.append(torch.cat([x["prompt_mask"], torch.zeros((pad_len,), dtype=torch.bool)]))
            b_att.append(torch.cat([torch.ones(seq_len, dtype=torch.long), torch.zeros(pad_len, dtype=torch.long)]))
            b_act.append(bool(x["temporal_active"]))
            b_th.append(float(x["theta"]))
            b_z.append(float(x["z"]))
            b_yr.append(int(x["year"]))
            b_men.append(x["anchor_mention"])
            b_src.append(x["source_stream"])
            b_idx.append(int(x["original_idx"]))
        return {
            "input_ids": torch.stack(b_in),
            "labels": torch.stack(b_lbl),
            "attention_mask": torch.stack(b_att),
            "prompt_mask": torch.stack(b_pm),
            "temporal_active": torch.tensor(b_act, dtype=torch.bool),
            "theta": torch.tensor(b_th, dtype=torch.float32),
            "z": torch.tensor(b_z, dtype=torch.float32),
            "years": torch.tensor(b_yr, dtype=torch.long),
            "anchor_mentions": b_men,
            "source_stream": b_src,
            "original_idx": b_idx,
        }

    def pretokenize_pools(
        nwp_cases: Sequence[Dict[str, Any]],
        replay_cases: Sequence[Dict[str, Any]],
        tokenizer: Any,
        spec: V3ModalLaunchSpec,
        *,
        limit_nwp: Optional[int] = None,
        limit_replay: Optional[int] = None,
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, Any]]:
        nwp_slice = list(nwp_cases[:limit_nwp] if limit_nwp is not None else nwp_cases)
        rep_slice = list(replay_cases[:limit_replay] if limit_replay is not None else replay_cases)

        nwp_items: List[Dict[str, Any]] = []
        nwp_meta: List[Dict[str, Any]] = []
        nwp_sup_tokens = 0
        role_counts: Dict[str, int] = {}

        for idx, s in enumerate(nwp_slice):
            query = str(s.get("input", ""))
            ans = str(s.get("target", ""))
            year = int(s["year"])
            prompt_text = f"Question: {query.strip()}\nAnswer:"
            tok_res = tokenize_supervised_pair(tokenizer, prompt_text, ans, max_length=128)
            mention = resolve_question_body_anchor(
                tokenizer,
                tok_res["full_text"],
                task_id="NWP",
                metadata_year=year,
                prompt_char_limit=tok_res["prompt_char_len"],
                offsets=tok_res["offsets"],
            )
            role_counts[mention.role] = role_counts.get(mention.role, 0) + 1
            theta, z = compute_theta_z(int(mention.year), z_mode=spec.z_mode)
            nwp_sup_tokens += int((tok_res["labels"] != -100).sum().item())
            nwp_items.append(
                {
                    "input_ids": tok_res["input_ids"],
                    "labels": tok_res["labels"],
                    "prompt_mask": tok_res["prompt_mask"],
                    "temporal_active": True,
                    "theta": theta,
                    "z": z,
                    "year": int(mention.year),
                    "anchor_mention": mention,
                    "source_stream": "nwp",
                    "original_idx": idx,
                }
            )
            nwp_meta.append(
                {
                    "_pool": "nwp",
                    "_pool_idx": idx,
                    "id": str(s.get("example_id", f"nwp_{idx}")),
                    "input": query,
                }
            )

        replay_items: List[Dict[str, Any]] = []
        replay_meta: List[Dict[str, Any]] = []
        replay_sup_tokens = 0
        for idx, r in enumerate(rep_slice):
            prompt_in = str(r.get("input", "")).strip()
            ans = str(r.get("target", "")).strip()
            prompt_text = f"{prompt_in}\nAnswer:"
            tok_res = tokenize_supervised_pair(tokenizer, prompt_text, ans, max_length=128)
            replay_sup_tokens += int((tok_res["labels"] != -100).sum().item())
            replay_items.append(
                {
                    "input_ids": tok_res["input_ids"],
                    "labels": tok_res["labels"],
                    "prompt_mask": tok_res["prompt_mask"],
                    "temporal_active": False,
                    "theta": 0.0,
                    "z": 0.0,
                    "year": -1,
                    "anchor_mention": None,
                    "source_stream": "general_replay",
                    "original_idx": idx,
                }
            )
            replay_meta.append(
                {
                    "_pool": "replay",
                    "_pool_idx": idx,
                    "source_id": str(r.get("source_id", f"rep_{idx}")),
                    "input": prompt_in,
                }
            )

        census = {
            "nwp_count": len(nwp_items),
            "nwp_supervised_tokens": nwp_sup_tokens,
            "nwp_anchor_roles": role_counts,
            "replay_count": len(replay_items),
            "replay_supervised_tokens": replay_sup_tokens,
        }
        return nwp_items, nwp_meta, replay_items, replay_meta, census

    def scatter_inject_year_span(
        embeds: torch.Tensor,
        seam_proj: Any,
        batch: Dict[str, Any],
        attn_mask: torch.Tensor,
        prompt_mask: torch.Tensor,
        spec: V3ModalLaunchSpec,
    ) -> torch.Tensor:
        active_t = batch["temporal_active"]
        active_list = [bool(v) for v in active_t.tolist()]
        temp_mask = build_temporal_mask(
            batch["input_ids"].to(embeds.device),
            attn_mask,
            TemporalPolicy(spec.temporal_policy),
            anchor_mentions=batch["anchor_mentions"],
            padding_side="right",
            temporal_active=active_list,
            prompt_mask=prompt_mask,
        )
        # Invariant assertions: active sum == 1, inactive/padding/target sum == 0
        row_sums = temp_mask.sum(dim=1)
        if active_t.any():
            if not torch.allclose(row_sums[active_t], torch.ones_like(row_sums[active_t]), atol=1e-4):
                raise RuntimeError(f"Temporal mask active row sum != 1.0: {row_sums[active_t][:5]}")
        if (~active_t).any():
            if not torch.allclose(row_sums[~active_t], torch.zeros_like(row_sums[~active_t]), atol=1e-6):
                raise RuntimeError("Temporal mask non-zero on general replay row!")
        non_prompt = (~prompt_mask.to(embeds.device)) | (attn_mask == 0)
        if float((temp_mask * non_prompt.float()).abs().sum().item()) > 1e-6:
            raise RuntimeError("Temporal injection mask leaked onto padding or target tokens!")

        active_indices = torch.nonzero(active_t, as_tuple=False).squeeze(-1)
        if active_indices.numel() == 0:
            return embeds

        v40_rows = [
            compute_temporal_40d(
                int(batch["anchor_mentions"][i].year),
                fone_mode=spec.fone_mode,
                radius_mode=spec.radius_mode,
            )
            for i in active_indices.tolist()
        ]
        v40_active = torch.tensor(v40_rows, dtype=torch.float32, device=embeds.device)
        delta_active = seam_proj(v40_active).to(dtype=embeds.dtype)  # (N_act, D)
        delta_full = torch.zeros((embeds.size(0), embeds.size(2)), dtype=embeds.dtype, device=embeds.device)
        delta_full[active_indices] = delta_active
        mask_exp = temp_mask.to(dtype=embeds.dtype).unsqueeze(-1)
        return embeds + (mask_exp * delta_full.unsqueeze(1))

    @torch.no_grad()
    def eval_text_probe(
        model: Any,
        tokenizer: Any,
        probe_cases: Sequence[Dict[str, Any]],
        *,
        is_temporal: bool,
        seam_proj: Any,
        spec: V3ModalLaunchSpec,
        max_gen_tokens: int = 16,
    ) -> Dict[str, Any]:
        import math

        model.eval()
        if seam_proj is not None:
            seam_proj.eval()
        dev = next(model.parameters()).device
        total_loss = 0.0
        total_toks = 0
        exact_matches = 0
        for case in probe_cases:
            q = str(case.get("input", "")).strip()
            tgt = str(case.get("target", "")).strip()
            prompt_text = f"Question: {q}\nAnswer:" if is_temporal else f"{q}\nAnswer:"
            tok_res = tokenize_supervised_pair(tokenizer, prompt_text, tgt, max_length=128)
            in_ids = tok_res["input_ids"].unsqueeze(0).to(dev)
            lbls = tok_res["labels"].unsqueeze(0).to(dev)
            att = torch.ones_like(in_ids)
            pm = tok_res["prompt_mask"].unsqueeze(0).to(dev)

            embeds = model.get_input_embeddings()(in_ids)
            if is_temporal and seam_proj is not None:
                yr = int(case["year"])
                mention = resolve_question_body_anchor(
                    tokenizer, tok_res["full_text"], task_id="NWP", metadata_year=yr, prompt_char_limit=tok_res["prompt_char_len"], offsets=tok_res["offsets"]
                )
                th_v, z_v = compute_theta_z(int(mention.year), z_mode=spec.z_mode)
                batch_fake = {
                    "input_ids": in_ids,
                    "anchor_mentions": [mention],
                    "temporal_active": torch.tensor([True]),
                }
                inj_embeds = scatter_inject_year_span(embeds, seam_proj, batch_fake, att, pm, spec)
                th_t = torch.tensor([th_v], dtype=torch.float32, device=dev)
                z_t = torch.tensor([z_v], dtype=torch.float32, device=dev)
                act_t = torch.tensor([True], dtype=torch.bool, device=dev)
            else:
                inj_embeds = embeds
                th_t = torch.tensor([0.0], dtype=torch.float32, device=dev)
                z_t = torch.tensor([0.0], dtype=torch.float32, device=dev)
                act_t = torch.tensor([False], dtype=torch.bool, device=dev)

            with matlora_temporal_context(model, th_t, z_t, temporal_active=act_t):
                out = model(inputs_embeds=inj_embeds, attention_mask=att, return_dict=True)
                logits_shift = out.logits[:, :-1, :].float()
                lbls_shift = lbls[:, 1:]
                valid = lbls_shift != -100
                n_tok = int(valid.sum().item())
                if n_tok > 0:
                    ce = F.cross_entropy(logits_shift[valid], lbls_shift[valid], reduction="sum")
                    total_loss += float(ce.item())
                    total_toks += n_tok

                # Prompt-only prefix greedy generation for EM
                p_enc = tokenizer(prompt_text, return_tensors="pt", add_special_tokens=False).to(dev)
                p_ids = p_enc["input_ids"]
                p_att = p_enc["attention_mask"]
                p_emb = model.get_input_embeddings()(p_ids)
                if is_temporal and seam_proj is not None:
                    p_men = resolve_question_body_anchor(tokenizer, prompt_text, task_id="NWP", metadata_year=int(case["year"]))
                    p_pm = torch.ones_like(p_att, dtype=torch.bool)
                    p_inj = scatter_inject_year_span(
                        p_emb, seam_proj, {"input_ids": p_ids, "anchor_mentions": [p_men], "temporal_active": torch.tensor([True])}, p_att, p_pm, spec
                    )
                else:
                    p_inj = p_emb
                gen_ids: List[int] = []
                g_out = model(inputs_embeds=p_inj, attention_mask=p_att, use_cache=True, return_dict=True)
                pkv = g_out.past_key_values
                nxt = int(torch.argmax(g_out.logits[:, -1, :], dim=-1).item())
                gen_ids.append(nxt)
                cur_att = p_att
                for _ in range(1, max_gen_tokens):
                    if nxt == tokenizer.eos_token_id:
                        break
                    cur_txt = tokenizer.decode(gen_ids, skip_special_tokens=True)
                    if "\n" in cur_txt:
                        break
                    cur_att = torch.cat([cur_att, torch.ones((1, 1), dtype=cur_att.dtype, device=dev)], dim=1)
                    s_out = model(input_ids=torch.tensor([[nxt]], device=dev), attention_mask=cur_att, past_key_values=pkv, use_cache=True, return_dict=True)
                    pkv = s_out.past_key_values
                    nxt = int(torch.argmax(s_out.logits[:, -1, :], dim=-1).item())
                    gen_ids.append(nxt)
                pred_str = tokenizer.decode(gen_ids, skip_special_tokens=True).split("\n", 1)[0].strip().lower()
                if pred_str == tgt.strip().lower():
                    exact_matches += 1

        mean_ce = total_loss / float(max(1, total_toks))
        return {
            "num_cases": len(probe_cases),
            "supervised_tokens": total_toks,
            "mean_ce": round(mean_ce, 4),
            "perplexity": round(math.exp(min(20.0, mean_ce)), 4),
            "exact_match_rate": round(float(exact_matches) / float(max(1, len(probe_cases))), 4),
            "exact_match_count": exact_matches,
        }

    return {
        "EvaluationConfig": EvaluationConfig,
        "ParserContract": ParserContract,
        "TemporalPolicy": TemporalPolicy,
        "GatedResidualProjection": GatedResidualProjection,
        "attach_adapters_to_model": attach_adapters_to_model,
        "load_checkpoint_weights": load_checkpoint_weights,
        "matlora_temporal_context": matlora_temporal_context,
        "execute_evaluation_loop": execute_evaluation_loop,
        "compute_theta_z": compute_theta_z,
        "resolve_all_year_mentions": resolve_all_year_mentions,
        "inject_multi_year_temporal_features": inject_multi_year_temporal_features,
        "build_cpt_optimizer_param_groups": build_cpt_optimizer_param_groups,
        "create_cpt_lr_scheduler": create_cpt_lr_scheduler,
        "interleave_general_replay": interleave_general_replay,
        "compute_masked_temporal_contrastive_loss": compute_masked_temporal_contrastive_loss,
        "PreTokenizedMixedDataset": PreTokenizedMixedDataset,
        "collate_mixed": collate_mixed,
        "pretokenize_pools": pretokenize_pools,
        "scatter_inject_year_span": scatter_inject_year_span,
        "eval_text_probe": eval_text_probe,
        "DataLoader": DataLoader,
    }


def _train_loop_core(
    *,
    rt: Dict[str, Any],
    spec: V3ModalLaunchSpec,
    model: Any,
    seam_proj: Any,
    core_gen: Any,
    tokenizer: Any,
    nwp_items: List[Dict[str, Any]],
    nwp_meta: List[Dict[str, Any]],
    replay_items: List[Dict[str, Any]],
    replay_meta: List[Dict[str, Any]],
    eval_cases_full: Sequence[Dict[str, Any]],
    frozen_cases: Sequence[Dict[str, Any]],
    holdout_cases: Sequence[Dict[str, Any]],
    nwp_probe_cases: Sequence[Dict[str, Any]],
    save_prefix_vol: str,
    max_steps: Optional[int] = None,
) -> Dict[str, Any]:
    import math
    import torch
    import torch.nn.functional as F

    forbidden_eval_ids = {str(r.get("example_id", "")) for r in eval_cases_full if r.get("example_id")}
    forbidden_eval_ids.update(f"eval_row_{i}" for i in range(len(eval_cases_full)))
    forbidden_hashes = {
        hashlib.sha256(str(r.get("input", "")).strip().encode("utf-8")).hexdigest()
        for r in eval_cases_full
    }

    param_groups = rt["build_cpt_optimizer_param_groups"](
        model,
        seam_proj=seam_proj,
        core_gen=core_gen,
        base_lr=spec.base_lr,
        loraplus_ratio=1.0,
        adapter_head_lr=spec.adapter_head_lr,
        weight_decay=spec.weight_decay,
    )
    optimizer = torch.optim.AdamW(param_groups, lr=spec.base_lr, betas=(0.9, 0.999), eps=1e-8)

    planned = spec.planned_records_per_epoch(len(nwp_items))
    total_opt_steps = max_steps if max_steps is not None else planned["total_opt_steps"]
    scheduler = rt["create_cpt_lr_scheduler"](
        optimizer,
        total_steps=total_opt_steps,
        warmup_ratio=spec.warmup_ratio,
        schedule_type=spec.schedule_type,
        min_lr_ratio=spec.min_lr_ratio,
    )

    all_trainable = [p for g in param_groups for p in g["params"]]
    dev = next(model.parameters()).device

    epoch_history: List[Dict[str, Any]] = []
    global_step = 0
    step_wall_times: List[float] = []
    total_nwp_toks = 0
    total_replay_toks = 0
    n_epochs_to_run = 1 if max_steps is not None else spec.epochs

    Path(save_prefix_vol).mkdir(parents=True, exist_ok=True)
    need_contrastive = (spec.lambda_intra > 0.0) or (spec.lambda_inter > 0.0)

    for epoch in range(n_epochs_to_run):
        mixed_meta = rt["interleave_general_replay"](
            nwp_meta,
            replay_meta,
            replay_ratio=spec.replay_ratio,
            seed=spec.seed,
            epoch=epoch,
            allow_oversample=False,
            forbidden_eval_ids=forbidden_eval_ids,
            forbidden_prompt_hashes=forbidden_hashes,
        )
        epoch_items = [
            nwp_items[m["_pool_idx"]] if m["_pool"] == "nwp" else replay_items[m["_pool_idx"]]
            for m in mixed_meta
        ]
        ds = rt["PreTokenizedMixedDataset"](epoch_items)
        loader = rt["DataLoader"](
            ds,
            batch_size=spec.micro_batch,
            shuffle=False,
            collate_fn=lambda b: rt["collate_mixed"](b, tokenizer.pad_token_id),
        )
        n_batches = len(loader)
        optimizer.zero_grad()
        model.train()
        seam_proj.train()
        if core_gen is not None:
            core_gen.train()

        t_ep0 = time.time()
        accum_nwp_ce, accum_nwp_tok = 0.0, 0
        accum_rep_ce, accum_rep_tok = 0.0, 0
        accum_intra, accum_inter, accum_pairs = 0.0, 0.0, 0
        t_step0 = time.time()

        for b_idx, batch in enumerate(loader):
            # Exact micro-batch group size so trailing partial batch is properly scaled
            group_start = (b_idx // spec.grad_accum) * spec.grad_accum
            cur_group_size = min(spec.grad_accum, n_batches - group_start)
            pos_in_group = (b_idx - group_start) + 1

            input_ids = batch["input_ids"].to(dev)
            labels = batch["labels"].to(dev)
            attn_mask = batch["attention_mask"].to(dev)
            prompt_mask = batch["prompt_mask"].to(dev)
            temporal_active = batch["temporal_active"].to(dev)
            theta = batch["theta"].to(dev)
            z = batch["z"].to(dev)
            years = batch["years"].to(dev)

            embeds = model.get_input_embeddings()(input_ids)
            injected_embeds = rt["scatter_inject_year_span"](embeds, seam_proj, batch, attn_mask, prompt_mask, spec)

            with rt["matlora_temporal_context"](model, theta, z, temporal_active=temporal_active):
                outputs = model(
                    inputs_embeds=injected_embeds,
                    attention_mask=attn_mask,
                    labels=labels,
                    output_hidden_states=need_contrastive,
                )
                loss = outputs.loss

                with torch.no_grad():
                    logits_shift = outputs.logits[:, :-1, :].float()
                    lbl_shift = labels[:, 1:]
                    for mask_src, is_nwp in ((temporal_active, True), (~temporal_active, False)):
                        if mask_src.any():
                            v_sub = lbl_shift[mask_src] != -100
                            n_t = int(v_sub.sum().item())
                            if n_t > 0:
                                ce_sum = float(F.cross_entropy(logits_shift[mask_src][v_sub], lbl_shift[mask_src][v_sub], reduction="sum").item())
                                if is_nwp:
                                    accum_nwp_ce += ce_sum
                                    accum_nwp_tok += n_t
                                    total_nwp_toks += n_t
                                else:
                                    accum_rep_ce += ce_sum
                                    accum_rep_tok += n_t
                                    total_replay_toks += n_t

                if need_contrastive:
                    l_intra, l_inter, c_stats = rt["compute_masked_temporal_contrastive_loss"](
                        outputs.hidden_states[-1],
                        years,
                        temporal_active,
                        prompt_mask,
                        attn_mask,
                        temperature=0.07,
                        margin=spec.margin_inter,
                    )
                    loss = loss + (spec.lambda_intra * l_intra) + (spec.lambda_inter * l_inter)
                    accum_intra += float(l_intra.item())
                    accum_inter += float(l_inter.item())
                    accum_pairs += int(c_stats["positive_pairs"])

                if not torch.isfinite(loss):
                    raise FloatingPointError(f"Non-finite training loss detected at step {global_step}: {loss.item()}")
                (loss / float(cur_group_size)).backward()

            if pos_in_group == cur_group_size:
                torch.nn.utils.clip_grad_norm_(all_trainable, 1.0)
                used_lr = float(optimizer.param_groups[0]["lr"])
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1
                step_wall_times.append(time.time() - t_step0)
                t_step0 = time.time()

                if global_step % 200 == 0 or global_step == total_opt_steps or (max_steps is not None):
                    nwp_ce_avg = accum_nwp_ce / float(max(1, accum_nwp_tok))
                    rep_ce_avg = accum_rep_ce / float(max(1, accum_rep_tok))
                    print(
                        f"[HEARTBEAT] Ep {epoch + 1}/{n_epochs_to_run} | Step {global_step}/{total_opt_steps} | "
                        f"NWP_CE: {nwp_ce_avg:.4f} | Rep_CE: {rep_ce_avg:.4f} | "
                        f"UsedLR: {used_lr:.6f} | Alpha: {seam_proj.alpha.item():.5f}"
                    )
                    accum_nwp_ce, accum_nwp_tok = 0.0, 0
                    accum_rep_ce, accum_rep_tok = 0.0, 0
                    accum_intra, accum_inter, accum_pairs = 0.0, 0.0, 0

                if max_steps is not None and global_step >= max_steps:
                    break

        if max_steps is not None:
            break

        # Save per-epoch immutable checkpoint + cheap per-epoch DEV & retention probes
        ep_ckpt_path = f"{save_prefix_vol}/ep{epoch + 1}_weights.pt"
        torch.save(
            {
                "lora_state": {k: v.cpu() for k, v in model.state_dict().items() if "lora_" in k},
                "seam_projection": {k: v.cpu() for k, v in seam_proj.state_dict().items()},
                "core_generator": {k: v.cpu() for k, v in core_gen.state_dict().items()} if core_gen is not None else None,
                "epoch": epoch + 1,
                "global_step": global_step,
                "spec": asdict(spec),
            },
            ep_ckpt_path,
        )
        ep_dev_eval = rt["execute_evaluation_loop"](
            config=spec.eval_config(checkpoint_path=ep_ckpt_path),
            model=model,
            tokenizer=tokenizer,
            eval_cases=list(frozen_cases[:32]),
            seam_proj=seam_proj,
            core_gen=core_gen,
        )
        ep_rep_probe = rt["eval_text_probe"](
            model, tokenizer, list(holdout_cases[:32]), is_temporal=False, seam_proj=None, spec=spec
        )
        ep_nwp_probe = rt["eval_text_probe"](
            model, tokenizer, list(nwp_probe_cases[:32]), is_temporal=True, seam_proj=seam_proj, spec=spec
        )
        volume.commit()
        epoch_history.append(
            {
                "epoch": epoch + 1,
                "global_step": global_step,
                "epoch_minutes": round((time.time() - t_ep0) / 60.0, 2),
                "seam_alpha": float(seam_proj.alpha.item()),
                "dev_32_metrics": ep_dev_eval["metrics"],
                "replay_holdout_32": ep_rep_probe,
                "nwp_heldin_32": ep_nwp_probe,
                "checkpoint_path": ep_ckpt_path,
            }
        )

    return {
        "global_steps": global_step,
        "mean_opt_step_seconds": round(sum(step_wall_times) / float(max(1, len(step_wall_times))), 4),
        "supervised_token_share": {
            "nwp_tokens": total_nwp_toks,
            "general_replay_tokens": total_replay_toks,
            "replay_token_fraction": round(
                float(total_replay_toks) / float(max(1, total_nwp_toks + total_replay_toks)), 4
            ),
        },
        "epoch_history": epoch_history,
        "final_used_lr": float(optimizer.param_groups[0]["lr"]),
    }


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
    local_src_sha256: str,
    train_bytes: bytes,
    replay_bytes: bytes,
    replay_holdout_bytes: bytes,
    eval_bytes: bytes,
    manifest_bytes: bytes,
) -> Tuple[Dict[str, Any], bytes, bytes]:
    """Execute full 5-epoch MLP-only training + frozen 128 eval + retention probes."""
    import random
    import numpy as np
    import torch
    import transformers
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if "/root" not in sys.path:
        sys.path.insert(0, "/root")

    remote_src_sha = compute_source_tree_sha256(Path("/root/temporal_alignment"))
    if remote_src_sha != local_src_sha256:
        raise RuntimeError(f"Code tree SHA-256 mismatch: remote={remote_src_sha} != local={local_src_sha256}")

    spec = V3ModalLaunchSpec(**spec_dict)
    spec.validate()
    run_id = spec.run_id or spec.recipe_id.lower()

    random.seed(spec.seed)
    np.random.seed(spec.seed)
    torch.manual_seed(spec.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(spec.seed)

    rt = _build_remote_runtime_helpers()

    with gzip.GzipFile(fileobj=io.BytesIO(train_bytes), mode="rb") as f:
        train_cases = [json.loads(line) for line in f if line.strip()]
    with gzip.GzipFile(fileobj=io.BytesIO(replay_bytes), mode="rb") as f:
        replay_cases = [json.loads(line) for line in f if line.strip()]
    with gzip.GzipFile(fileobj=io.BytesIO(replay_holdout_bytes), mode="rb") as f:
        holdout_cases = [json.loads(line) for line in f if line.strip()]
    with gzip.GzipFile(fileobj=io.BytesIO(eval_bytes), mode="rb") as f:
        eval_all = [json.loads(line) for line in f if line.strip()]
    manifest = json.loads(manifest_bytes.decode("utf-8"))
    frozen_cases = [eval_all[s["original_eval_index"]] for s in manifest["samples"][: spec.n_eval_samples]]

    model_id = "Qwen/Qwen2.5-3B"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16).to("cuda:0")
    for p in model.parameters():
        p.requires_grad = False

    hidden_size = model.config.hidden_size
    seam_proj = rt["GatedResidualProjection"](
        in_features=40, out_features=hidden_size, gate_mode="linear"
    ).to("cuda:0", dtype=torch.bfloat16)
    _, core_gen = rt["attach_adapters_to_model"](model, arm=spec.arm, rank=spec.rank, alpha=spec.alpha)
    init_sha = _state_digest(model, seam_proj, core_gen)

    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

    t_start = time.time()
    nwp_items, nwp_meta, replay_items, replay_meta, census = rt["pretokenize_pools"](
        train_cases, replay_cases, tokenizer, spec
    )
    save_prefix_vol = f"/vol/{spec.checkpoint_namespace}/{run_id}"
    train_stats = _train_loop_core(
        rt=rt,
        spec=spec,
        model=model,
        seam_proj=seam_proj,
        core_gen=core_gen,
        tokenizer=tokenizer,
        nwp_items=nwp_items,
        nwp_meta=nwp_meta,
        replay_items=replay_items,
        replay_meta=replay_meta,
        eval_cases_full=eval_all,
        frozen_cases=frozen_cases,
        holdout_cases=holdout_cases,
        nwp_probe_cases=train_cases[:32],
        save_prefix_vol=save_prefix_vol,
    )

    final_path = f"{save_prefix_vol}/final_weights.pt"
    torch.save(
        {
            "lora_state": {k: v.cpu() for k, v in model.state_dict().items() if "lora_" in k},
            "seam_projection": {k: v.cpu() for k, v in seam_proj.state_dict().items()},
            "core_generator": {k: v.cpu() for k, v in core_gen.state_dict().items()} if core_gen is not None else None,
            "epoch": spec.epochs,
            "global_step": train_stats["global_steps"],
            "init_state_sha256": init_sha,
            "spec": spec_dict,
        },
        final_path,
    )
    volume.commit()
    final_sha = _state_digest(model, seam_proj, core_gen)

    # Full frozen 128 evaluation using strict persisted path + raw JSONL capture
    eval_out_dir = f"{save_prefix_vol}/eval_outputs"
    model.eval()
    seam_proj.eval()
    if core_gen is not None:
        core_gen.eval()
    eval_summary = rt["execute_evaluation_loop"](
        config=spec.eval_config(checkpoint_path=final_path),
        model=model,
        tokenizer=tokenizer,
        eval_cases=frozen_cases,
        output_dir=eval_out_dir,
        seam_proj=seam_proj,
        core_gen=core_gen,
    )
    retention_holdout_256 = rt["eval_text_probe"](
        model, tokenizer, holdout_cases, is_temporal=False, seam_proj=None, spec=spec
    )
    nwp_heldin_64 = rt["eval_text_probe"](
        model, tokenizer, train_cases[:64], is_temporal=True, seam_proj=seam_proj, spec=spec
    )
    volume.commit()

    raw_jsonl_bytes = Path(f"{eval_out_dir}/PREDICTION_RECORDS.jsonl").read_bytes()
    runtime_sec = round(time.time() - t_start, 2)
    receipt = {
        "timestamp": datetime.now().isoformat(),
        "status": "TRAIN_COMPLETE",
        "recipe_id": spec.recipe_id,
        "run_id": run_id,
        "spec": spec_dict,
        "provenance": {
            "source_sha256": remote_src_sha,
            "model_id": model_id,
            "model_commit_hash": getattr(model.config, "_commit_hash", None),
            "torch_version": torch.__version__,
            "transformers_version": transformers.__version__,
            "init_state_sha256": init_sha,
            "final_state_sha256": final_sha,
            "final_checkpoint_path": final_path,
        },
        "data_census": census,
        "train_stats": train_stats,
        "runtime_seconds": runtime_sec,
        "eval_summary": eval_summary,
        "retention_holdout_256": retention_holdout_256,
        "nwp_heldin_64": nwp_heldin_64,
    }
    return receipt, json.dumps(receipt, indent=2, ensure_ascii=False).encode("utf-8"), raw_jsonl_bytes


@app.function(
    image=image,
    gpu="A10G",
    timeout=1800,
    retries=0,
    max_containers=1,
    volumes={"/vol": volume},
)
def run_stage03_g1_smoke(
    spec_dict: Dict[str, Any],
    local_src_sha256: str,
    train_bytes: bytes,
    replay_bytes: bytes,
    replay_holdout_bytes: bytes,
    eval_bytes: bytes,
    manifest_bytes: bytes,
) -> Tuple[Dict[str, Any], bytes]:
    """Bounded A10G smoke: full data/anchor census, Base probe, 4 train steps, strict roundtrip load, 16-case eval & cost forecast."""
    import random
    import numpy as np
    import torch
    import transformers
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if "/root" not in sys.path:
        sys.path.insert(0, "/root")

    remote_src_sha = compute_source_tree_sha256(Path("/root/temporal_alignment"))
    if remote_src_sha != local_src_sha256:
        raise RuntimeError(f"Code tree SHA-256 mismatch: remote={remote_src_sha} != local={local_src_sha256}")

    spec = V3ModalLaunchSpec(**spec_dict)
    spec.validate()
    t0 = time.time()

    random.seed(spec.seed)
    np.random.seed(spec.seed)
    torch.manual_seed(spec.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(spec.seed)

    rt = _build_remote_runtime_helpers()

    with gzip.GzipFile(fileobj=io.BytesIO(train_bytes), mode="rb") as f:
        train_cases = [json.loads(line) for line in f if line.strip()]
    with gzip.GzipFile(fileobj=io.BytesIO(replay_bytes), mode="rb") as f:
        replay_cases = [json.loads(line) for line in f if line.strip()]
    with gzip.GzipFile(fileobj=io.BytesIO(replay_holdout_bytes), mode="rb") as f:
        holdout_cases = [json.loads(line) for line in f if line.strip()]
    with gzip.GzipFile(fileobj=io.BytesIO(eval_bytes), mode="rb") as f:
        eval_all = [json.loads(line) for line in f if line.strip()]
    manifest = json.loads(manifest_bytes.decode("utf-8"))

    # Stratified 16-case smoke slice across 6 tasks
    tasks = ["DB", "TB", "WB", "Rel", "MM", "RB"]
    quotas = {"DB": 3, "TB": 3, "WB": 3, "Rel": 3, "MM": 2, "RB": 2}
    buckets: Dict[str, List[Dict[str, Any]]] = {t: [] for t in tasks}
    for s in manifest["samples"]:
        t = s.get("task_id")
        if t in buckets and len(buckets[t]) < quotas[t]:
            buckets[t].append(eval_all[s["original_eval_index"]])
    smoke_eval_cases = [c for t in tasks for c in buckets[t]]

    model_id = "Qwen/Qwen2.5-3B"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    t_load0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16).to("cuda:0")
    for p in model.parameters():
        p.requires_grad = False
    t_load_sec = round(time.time() - t_load0, 2)

    # Full corpus anchor + tokenization audit (104k train + 20k replay + 5292 eval)
    from temporal_alignment.stage03_eval.temporal_locator import resolve_question_body_anchor
    eval_anchors_ok = 0
    for c in eval_all:
        resolve_question_body_anchor(tokenizer, f"Question: {c['input'].strip()}\nAnswer:", task_id=c["task_id"], metadata_year=int(c["year"]))
        eval_anchors_ok += 1

    # Base baseline probes before attaching adapters
    base_holdout_32 = rt["eval_text_probe"](model, tokenizer, holdout_cases[:32], is_temporal=False, seam_proj=None, spec=spec)
    base_nwp_32 = rt["eval_text_probe"](model, tokenizer, train_cases[:32], is_temporal=True, seam_proj=None, spec=spec)

    hidden_size = model.config.hidden_size
    seam_proj = rt["GatedResidualProjection"](in_features=40, out_features=hidden_size, gate_mode="linear").to("cuda:0", dtype=torch.bfloat16)
    _, core_gen = rt["attach_adapters_to_model"](model, arm=spec.arm, rank=spec.rank, alpha=spec.alpha)
    init_sha = _state_digest(model, seam_proj, core_gen)

    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

    # Pre-tokenize smoke slice matching exact mixed stream ratio (1024 NWP + 256 replay)
    nwp_items, nwp_meta, replay_items, replay_meta, census = rt["pretokenize_pools"](
        train_cases, replay_cases, tokenizer, spec, limit_nwp=1024, limit_replay=256
    )

    smoke_vol_dir = f"/vol/{spec.checkpoint_namespace}/smoke_{spec.recipe_id.lower()}"
    train_stats = _train_loop_core(
        rt=rt,
        spec=spec,
        model=model,
        seam_proj=seam_proj,
        core_gen=core_gen,
        tokenizer=tokenizer,
        nwp_items=nwp_items,
        nwp_meta=nwp_meta,
        replay_items=replay_items,
        replay_meta=replay_meta,
        eval_cases_full=eval_all,
        frozen_cases=smoke_eval_cases,
        holdout_cases=holdout_cases[:32],
        nwp_probe_cases=train_cases[:32],
        save_prefix_vol=smoke_vol_dir,
        max_steps=4,
    )

    smoke_ckpt = f"{smoke_vol_dir}/smoke_weights.pt"
    Path(smoke_vol_dir).mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "lora_state": {k: v.cpu() for k, v in model.state_dict().items() if "lora_" in k},
            "seam_projection": {k: v.cpu() for k, v in seam_proj.state_dict().items()},
            "core_generator": {k: v.cpu() for k, v in core_gen.state_dict().items()} if core_gen is not None else None,
            "epoch": 0,
            "global_step": train_stats["global_steps"],
            "init_state_sha256": init_sha,
            "spec": spec_dict,
        },
        smoke_ckpt,
    )
    volume.commit()
    pre_reload_sha = _state_digest(model, seam_proj, core_gen)

    # Fresh model strict roundtrip verification
    del model, seam_proj, core_gen
    torch.cuda.empty_cache()
    model_rt = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16).to("cuda:0")
    for p in model_rt.parameters():
        p.requires_grad = False
    _, core_rt = rt["attach_adapters_to_model"](model_rt, arm=spec.arm, rank=spec.rank, alpha=spec.alpha)
    seam_rt = rt["GatedResidualProjection"](in_features=40, out_features=hidden_size, gate_mode="linear").to("cuda:0", dtype=torch.bfloat16)
    load_receipt = rt["load_checkpoint_weights"](model_rt, seam_rt, core_rt, smoke_ckpt, arm=spec.arm, strict=True)
    post_reload_sha = _state_digest(model_rt, seam_rt, core_rt)
    if pre_reload_sha != post_reload_sha:
        raise RuntimeError(f"Checkpoint roundtrip digest mismatch: {pre_reload_sha} != {post_reload_sha}")

    model_rt.eval()
    seam_rt.eval()
    core_rt.eval()
    t_eval0 = time.time()
    smoke_eval = rt["execute_evaluation_loop"](
        config=spec.eval_config(checkpoint_path=smoke_ckpt),
        model=model_rt,
        tokenizer=tokenizer,
        eval_cases=smoke_eval_cases,
        seam_proj=seam_rt,
        core_gen=core_rt,
    )
    smoke_holdout_32 = rt["eval_text_probe"](model_rt, tokenizer, holdout_cases[:32], is_temporal=False, seam_proj=None, spec=spec)
    smoke_nwp_32 = rt["eval_text_probe"](model_rt, tokenizer, train_cases[:32], is_temporal=True, seam_proj=seam_rt, spec=spec)
    t_eval_sec = round(time.time() - t_eval0, 2)

    # Forecast full 5-epoch cost for R1 & R2 from measured step time
    t_step = float(train_stats["mean_opt_step_seconds"])
    planned_full = spec.planned_records_per_epoch(PINNED_NWP_COUNT)
    est_train_sec = planned_full["total_opt_steps"] * t_step
    est_total_sec_one_run = t_load_sec + est_train_sec + (t_eval_sec * 8.0) + 180.0
    est_hours_one_run = round(est_total_sec_one_run / 3600.0, 3)
    est_usd_one_run = round(est_hours_one_run * 1.10, 3)

    payload = {
        "timestamp": datetime.now().isoformat(),
        "status": "GPU_SMOKE_PASS",
        "provenance": {
            "source_sha256": remote_src_sha,
            "model_id": model_id,
            "model_commit_hash": getattr(model_rt.config, "_commit_hash", None),
            "torch_version": torch.__version__,
            "transformers_version": transformers.__version__,
            "init_state_sha256": init_sha,
            "roundtrip_state_sha256": post_reload_sha,
            "eval_5292_anchors_resolved": eval_anchors_ok,
        },
        "smoke_census": census,
        "load_receipt": load_receipt,
        "train_stats": train_stats,
        "base_probes": {"holdout_32": base_holdout_32, "nwp_32": base_nwp_32},
        "smoke_probes": {"holdout_32": smoke_holdout_32, "nwp_32": smoke_nwp_32},
        "smoke_eval_16": smoke_eval["metrics"],
        "cost_forecast_5epochs": {
            "measured_t_load_seconds": t_load_sec,
            "measured_t_step_seconds": t_step,
            "planned_total_opt_steps_per_run": planned_full["total_opt_steps"],
            "estimated_hours_per_run": est_hours_one_run,
            "estimated_usd_per_run": est_usd_one_run,
            "estimated_usd_two_runs_r1_plus_r2": round(est_usd_one_run * 2.0, 3),
        },
        "total_smoke_seconds": round(time.time() - t0, 2),
    }
    return payload, json.dumps(payload, indent=2, ensure_ascii=False).encode("utf-8")


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
    wrong_year_offset: int = 0,
    adapter_mode: str = "ACTIVE",
) -> Tuple[Dict[str, Any], bytes, bytes]:
    """Evaluate a saved checkpoint strictly on Modal Volume using frozen eval cases."""
    import torch
    import transformers
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if "/root" not in sys.path:
        sys.path.insert(0, "/root")

    rt = _build_remote_runtime_helpers()
    t_start = time.time()
    volume.reload()
    spec = V3ModalLaunchSpec(**spec_dict)
    spec.validate()

    with gzip.GzipFile(fileobj=io.BytesIO(eval_bytes), mode="rb") as f:
        eval_all = [json.loads(line) for line in f if line.strip()]
    manifest = json.loads(manifest_bytes.decode("utf-8"))
    frozen_cases = [eval_all[s["original_eval_index"]] for s in manifest["samples"][: spec.n_eval_samples]]

    model_id = "Qwen/Qwen2.5-3B"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16).to("cuda:0")
    for p in model.parameters():
        p.requires_grad = False

    hidden_size = model.config.hidden_size
    seam_proj = rt["GatedResidualProjection"](
        in_features=40, out_features=hidden_size, gate_mode="linear"
    ).to("cuda:0", dtype=torch.bfloat16)
    _, core_gen = rt["attach_adapters_to_model"](model, arm=spec.arm, rank=spec.rank, alpha=spec.alpha)
    load_receipt = rt["load_checkpoint_weights"](model, seam_proj, core_gen, checkpoint_path, arm=spec.arm, strict=True)

    model.eval()
    seam_proj.eval()
    if core_gen is not None:
        core_gen.eval()

    base_cfg = spec.eval_config(checkpoint_path=checkpoint_path)
    eval_cfg = rt["EvaluationConfig"](
        contract_version=base_cfg.contract_version,
        arm=base_cfg.arm,
        temporal_policy=base_cfg.temporal_policy,
        parser_contract=base_cfg.parser_contract,
        seed=base_cfg.seed,
        device="cuda:0",
        checkpoint_path=checkpoint_path,
        z_mode=base_cfg.z_mode,
        fone_mode=base_cfg.fone_mode,
        radius_mode=base_cfg.radius_mode,
        wrong_year_offset=wrong_year_offset,
        adapter_mode=adapter_mode,
        adapter_scope="MLP_ONLY",
    )
    out_eval_dir = "/tmp/eval_only_out"
    eval_summary = rt["execute_evaluation_loop"](
        config=eval_cfg,
        model=model,
        tokenizer=tokenizer,
        eval_cases=frozen_cases,
        output_dir=out_eval_dir,
        seam_proj=seam_proj,
        core_gen=core_gen,
    )
    raw_jsonl_bytes = Path(f"{out_eval_dir}/PREDICTION_RECORDS.jsonl").read_bytes()
    runtime_sec = round(time.time() - t_start, 2)
    receipt = {
        "timestamp": datetime.now().isoformat(),
        "mode": "eval_only",
        "checkpoint_path": checkpoint_path,
        "load_receipt": load_receipt,
        "torch_version": torch.__version__,
        "transformers_version": transformers.__version__,
        "spec": spec_dict,
        "runtime_seconds": runtime_sec,
        "eval_summary": eval_summary,
    }
    return receipt, json.dumps(receipt, indent=2, ensure_ascii=False).encode("utf-8"), raw_jsonl_bytes


@app.local_entrypoint()
def main(
    recipe_id: str = "G1_SINGLE_YEAR_REPLAY_FIXED",
    run_id: str = "",
    arm: str = "",
    epochs: int = 0,
    schedule_type: str = "",
    base_lr: float = 0.0,
    warmup_ratio: float = -1.0,
    replay_ratio: float = -1.0,
    temporal_policy: str = "",
    wrong_year_offset: int = 0,
    adapter_mode: str = "ACTIVE",
    n_eval_samples: int = 128,
    max_budget_usd: float = 10.0,
    profile: str = "loggingfuture",
    mode: str = "preflight",
    checkpoint_path: str = "",
    confirm_gpu: bool = False,
    out_dir: str = "Reports/6-11Report/G1_REPAIR_2RUNS",
):
    overrides: Dict[str, Any] = {
        "profile": profile,
        "n_eval_samples": n_eval_samples,
        "max_budget_usd": max_budget_usd,
    }
    if run_id:
        overrides["run_id"] = run_id
    if arm:
        overrides["arm"] = arm
    if epochs > 0:
        overrides["epochs"] = epochs
    if schedule_type:
        overrides["schedule_type"] = schedule_type
    if base_lr > 0.0:
        overrides["base_lr"] = base_lr
        overrides["adapter_head_lr"] = base_lr
    if warmup_ratio >= 0.0:
        overrides["warmup_ratio"] = warmup_ratio
    if replay_ratio >= 0.0:
        overrides["replay_ratio"] = replay_ratio
    if temporal_policy:
        overrides["temporal_policy"] = temporal_policy

    spec = V3ModalLaunchSpec.from_recipe(recipe_id, **overrides)
    preflight = verify_v3_modal_inputs(spec)
    local_src_sha = compute_source_tree_sha256(_SRC_DIR / "temporal_alignment")
    preflight["source_sha256"] = local_src_sha
    print(json.dumps(preflight, indent=2, ensure_ascii=False))

    if mode not in ("smoke", "train_eval", "eval_only"):
        print("\n[PREFLIGHT ONLY] Zero GPU jobs launched.")
        return
    if not confirm_gpu:
        raise RuntimeError(
            "Refusing to launch GPU job without --confirm-gpu. Re-run with --mode smoke|train_eval|eval_only --confirm-gpu."
        )

    eval_bytes = Path("data/tempun_full_clean/eval_5292_clean.jsonl.gz").read_bytes()
    manifest_bytes = Path("Reports/6-11Report/_POST_FORENSIC_OPEN_GATES_20261105/REPLAY_SAMPLE_MANIFEST.json").read_bytes()
    train_bytes = Path("data/tempun_full_clean/train_104k_clean.jsonl.gz").read_bytes()
    replay_bytes = Path(spec.replay_path).read_bytes()
    replay_holdout_bytes = Path(spec.replay_holdout_path).read_bytes()

    out_p = Path(out_dir)
    out_p.mkdir(parents=True, exist_ok=True)
    effective_run_id = spec.run_id or spec.recipe_id.lower()

    if mode == "smoke":
        smoke_payload, smoke_bytes = run_stage03_g1_smoke.remote(
            asdict(spec),
            local_src_sha,
            train_bytes,
            replay_bytes,
            replay_holdout_bytes,
            eval_bytes,
            manifest_bytes,
        )
        out_file = out_p / f"{effective_run_id}_SMOKE_RECEIPT.json"
        out_file.write_bytes(smoke_bytes)
        print(f"Saved smoke receipt to {out_file} (elapsed: {smoke_payload['total_smoke_seconds']}s)")
        return

    if mode == "eval_only":
        if not checkpoint_path:
            raise ValueError("--checkpoint-path is required when --mode eval_only")
        receipt, receipt_bytes, raw_jsonl = run_stage03_v3_eval_only.remote(
            asdict(spec),
            checkpoint_path,
            eval_bytes,
            manifest_bytes,
            wrong_year_offset=wrong_year_offset,
            adapter_mode=adapter_mode,
        )
        tag = f"{effective_run_id}_{spec.temporal_policy}"
        if wrong_year_offset != 0:
            tag += f"_wy{wrong_year_offset}"
        if adapter_mode != "ACTIVE":
            tag += f"_{adapter_mode.lower()}"
        (out_p / f"{tag}_EVAL_RECEIPT.json").write_bytes(receipt_bytes)
        (out_p / f"{tag}_PREDICTION_RECORDS.jsonl").write_bytes(raw_jsonl)
        print(f"Saved eval receipt + raw JSONL to {out_p} (runtime: {receipt['runtime_seconds']}s)")
        return

    receipt, receipt_bytes, raw_jsonl = run_stage03_v3_train_and_eval.remote(
        asdict(spec),
        local_src_sha,
        train_bytes,
        replay_bytes,
        replay_holdout_bytes,
        eval_bytes,
        manifest_bytes,
    )
    (out_p / f"{effective_run_id}_RECEIPT.json").write_bytes(receipt_bytes)
    (out_p / f"{effective_run_id}_PREDICTION_RECORDS.jsonl").write_bytes(raw_jsonl)
    print(f"Saved receipt + raw JSONL to {out_p} (runtime: {receipt['runtime_seconds']}s)")
