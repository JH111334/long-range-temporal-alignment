"""CLI Entrypoint and Execution Engine for Stage 03 G1 Free Generation Evaluation.

Enforces:
- Deterministic random seeds.
- Preflight validation of manifests, tokenizers, and contracts before running.
- Atomic artifact creation under namespaced output directories.
- Default local CPU execution with fail-closed safety safeguards.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch

from .answer_parser import parse_answer, parse_unique_exact_numeric
from .contracts import (
    EvaluationConfig,
    GenerationPolicy,
    ParserContract,
    PredictionRecord,
    TemporalContext,
    TemporalPolicy,
    YearMention,
)
from .metrics import aggregate_evaluation_metrics
from .prompt_adapter import generate_audited_permutations
from .model_adapter import (
    DualCycleCoreGenerator,
    GeneralizedDualCycleMaTLoRALinear,
    StaticLoRALinear,
    attach_adapters_to_model,
    load_checkpoint_weights,
    matlora_temporal_context,
)
from .temporal_injector import (
    GatedResidualProjection,
    compute_fixed_fone_canchi_40d,
    compute_temporal_40d,
    compute_theta_z,
    inject_multi_year_temporal_features,
    inject_temporal_features,
)
from .temporal_locator import (
    StructuredLocatorError,
    build_temporal_mask,
    resolve_all_year_mentions,
    resolve_question_body_anchor,
    resolve_temporal_mentions,
)


def load_frozen_evaluation_cases(
    manifest_path: str,
    eval_data_path: str,
    n_samples: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """Load stratified frozen evaluation cases matching the pre-registered manifest."""
    man_p = Path(manifest_path)
    if not man_p.is_file():
        raise FileNotFoundError(f"Sample manifest not found: {manifest_path}")

    with man_p.open("r", encoding="utf-8") as f:
        manifest = json.load(f)
    samples = manifest.get("samples", [])
    if n_samples is not None and n_samples < len(samples):
        samples = samples[:n_samples]

    eval_p = Path(eval_data_path)
    if not eval_p.is_file():
        raise FileNotFoundError(f"Evaluation benchmark dataset not found: {eval_data_path}")

    all_cases: List[Dict[str, Any]] = []
    if eval_p.suffix == ".gz":
        with gzip.open(eval_p, "rt", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    all_cases.append(json.loads(line))
    else:
        with eval_p.open("r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    all_cases.append(json.loads(line))

    selected: List[Dict[str, Any]] = []
    for s in samples:
        orig_idx = s["original_eval_index"]
        if 0 <= orig_idx < len(all_cases):
            selected.append(all_cases[orig_idx])
        else:
            raise IndexError(
                f"Sample index {orig_idx} out of range for dataset with {len(all_cases)} records."
            )
    return selected


def run_preflight_checks(config: EvaluationConfig) -> Dict[str, Any]:
    """Verify data files, manifests, and contract configurations on CPU without running inference."""
    t0 = time.time()
    findings: Dict[str, Any] = {"status": "PASS", "checks": {}}

    # 1. Manifest file check
    manifest_p = Path(config.manifest_path)
    if not manifest_p.is_file():
        findings["status"] = "FAIL"
        findings["checks"]["sample_manifest"] = f"MISSING: {manifest_p}"
    else:
        m_bytes = manifest_p.read_bytes()
        findings["checks"]["sample_manifest"] = {
            "path": str(manifest_p),
            "size_bytes": len(m_bytes),
            "sha256": hashlib.sha256(m_bytes).hexdigest(),
        }

    # 2. Evaluation dataset file check
    eval_p = Path(config.eval_data_path)
    if not eval_p.is_file():
        findings["status"] = "FAIL"
        findings["checks"]["eval_data"] = f"MISSING: {eval_p}"
    else:
        eval_sha = hashlib.sha256(eval_p.read_bytes()).hexdigest()
        findings["checks"]["eval_data"] = {
            "path": str(eval_p),
            "size_bytes": eval_p.stat().st_size,
            "sha256": eval_sha,
        }

    # 3. Checkpoint check (if arm requires checkpoint)
    if config.arm in ("arm4", "arm5"):
        if not config.checkpoint_path or not Path(config.checkpoint_path).is_file():
            findings["status"] = "PREFLIGHT_WARN"
            findings["checks"]["checkpoint"] = f"NOT_FOUND_OR_UNSPECIFIED: {config.checkpoint_path}"
        else:
            ckpt_p = Path(config.checkpoint_path)
            findings["checks"]["checkpoint"] = {
                "path": str(ckpt_p),
                "size_bytes": ckpt_p.stat().st_size,
                "sha256": hashlib.sha256(ckpt_p.read_bytes()).hexdigest(),
            }

    findings["checks"]["config_hash"] = config.config_hash()
    findings["checks"]["contract_hash"] = config.contract_hash()
    findings["checks"]["elapsed_seconds"] = round(time.time() - t0, 4)
    return findings


def execute_evaluation_loop(
    config: EvaluationConfig,
    model: Any,
    tokenizer: Any,
    eval_cases: List[Dict[str, Any]],
    output_dir: Optional[str] = None,
    seam_proj: Optional[GatedResidualProjection] = None,
    core_gen: Optional[Any] = None,
) -> Dict[str, Any]:
    """Execute evaluation loop over instances using unconstrained generation."""
    records: List[PredictionRecord] = []
    t_start = time.time()

    embed_layer = model.get_input_embeddings() if hasattr(model, "get_input_embeddings") else None
    embed_weight = getattr(embed_layer, "weight", None)
    if isinstance(embed_weight, torch.Tensor):
        eval_device = embed_weight.device
    else:
        first_param = next(model.parameters(), None) if hasattr(model, "parameters") else None
        eval_device = first_param.device if isinstance(first_param, torch.Tensor) else config.device

    # Pre-build or validate Seam projection based on arm
    if config.arm == "base":
        seam_proj = None
    elif seam_proj is None:
        hidden_dim = getattr(
            model,
            "hidden_size",
            getattr(getattr(model, "config", None), "hidden_size", 2048),
        )
        seam_proj = GatedResidualProjection(
            in_features=40, out_features=hidden_dim, gate_mode="linear"
        ).to(device=eval_device)
        seam_proj.eval()
        if config.checkpoint_path:
            load_checkpoint_weights(
                model=model,
                seam_proj=seam_proj,
                core_gen=core_gen,
                checkpoint_path=config.checkpoint_path,
                arm=config.arm,
                strict=True,
            )

    for g_idx, case in enumerate(eval_cases):
        ex_id = case.get("example_id", f"case_{g_idx}")
        task_id = case.get("task_id", "MCQ")
        raw_input = case.get("input", "")
        raw_target = case.get("target", "")

        # Strict missing year validation: never default to 2000!
        if "year" not in case or case["year"] is None:
            rec = PredictionRecord(
                global_idx=g_idx,
                example_id=ex_id,
                rotation_id=0,
                task_id=task_id,
                prompt_hash="",
                raw_continuation="",
                generated_token_ids=(),
                parsed_label=None,
                parse_reason="MISSING_YEAR_IN_INSTANCE",
                is_valid=False,
                gold=str(raw_target).lower().strip(),
                is_correct=False,
                temporal_policy=config.temporal_policy.value,
                parser_contract=config.parser_contract.value,
            )
            records.append(rec)
            continue

        anchor_year = int(case["year"])
        radius_mode = getattr(config, "radius_mode", "linear")

        # Generate audited permutations
        permutations = generate_audited_permutations(raw_input, raw_target, task_id=task_id)

        for rot_info in permutations:
            rot_id = rot_info["rotation_id"]
            fmt_prompt = rot_info["formatted_prompt"]
            gold = rot_info["target"]
            p_hash = rot_info["prompt_hash"]
            opt_texts = rot_info["option_texts"]

            # Tokenize prompt
            enc = tokenizer(
                fmt_prompt,
                return_tensors="pt",
                add_special_tokens=False,
            ).to(device=eval_device)
            input_ids = enc["input_ids"]
            attention_mask = enc.get("attention_mask", torch.ones_like(input_ids))

            # Resolve year span(s)
            anchor_mention = None
            all_mentions = []
            try:
                if config.temporal_policy == TemporalPolicy.MULTI_YEAR_SPAN:
                    all_mentions = resolve_all_year_mentions(
                        tokenizer, fmt_prompt, anchor_year=anchor_year
                    )
                    anchor_mention = all_mentions[0] if all_mentions else None
                elif config.temporal_policy == TemporalPolicy.YEAR_SPAN:
                    anchor_mention = resolve_question_body_anchor(
                        tokenizer,
                        fmt_prompt,
                        task_id=task_id,
                        metadata_year=anchor_year,
                    )
                else:
                    anchor_mention, _ = resolve_temporal_mentions(
                        tokenizer, fmt_prompt, anchor_year=anchor_year
                    )
            except StructuredLocatorError:
                if config.temporal_policy in (TemporalPolicy.YEAR_SPAN, TemporalPolicy.MULTI_YEAR_SPAN):
                    # In YEAR_SPAN / MULTI_YEAR_SPAN, failure to locate anchor raises structured invalid
                    rec = PredictionRecord(
                        global_idx=g_idx,
                        example_id=ex_id,
                        rotation_id=rot_id,
                        task_id=task_id,
                        prompt_hash=p_hash,
                        raw_continuation="",
                        generated_token_ids=(),
                        parsed_label=None,
                        parse_reason="LOCATOR_ANCHOR_YEAR_NOT_RESOLVED",
                        is_valid=False,
                        gold=gold,
                        is_correct=False,
                        temporal_policy=config.temporal_policy.value,
                        parser_contract=config.parser_contract.value,
                        metadata_year=anchor_year,
                    )
                    records.append(rec)
                    continue

            cond_year = (
                int(anchor_mention.year) if anchor_mention is not None else anchor_year
            ) + int(getattr(config, "wrong_year_offset", 0))
            v40 = tuple(
                compute_temporal_40d(
                    cond_year,
                    fone_mode=config.fone_mode,
                    radius_mode=radius_mode,
                )
            )
            theta, z = compute_theta_z(cond_year, z_mode=config.z_mode)

            # Inject temporal representation if Seam is active
            embed_tokens = model.get_input_embeddings()
            inputs_embeds = embed_tokens(input_ids)
            if (
                seam_proj is not None
                and config.temporal_policy != TemporalPolicy.SEAM_OFF
                and getattr(config, "adapter_mode", "ACTIVE") != "BOTH_OFF"
            ):
                if config.temporal_policy == TemporalPolicy.MULTI_YEAR_SPAN:
                    injected_embeds = inject_multi_year_temporal_features(
                        inputs_embeds=inputs_embeds,
                        seam_proj=seam_proj,
                        mentions_per_item=[all_mentions],
                        padding_side="right",
                        attention_mask=attention_mask,
                        fone_mode=config.fone_mode,
                        radius_mode=radius_mode,
                    )
                else:
                    temp_mask = build_temporal_mask(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        policy=config.temporal_policy,
                        anchor_mentions=[anchor_mention] if anchor_mention else None,
                        padding_side="right",
                    )
                    v40_tensor = torch.tensor([v40], dtype=torch.float32, device=eval_device)
                    injected_embeds = inject_temporal_features(
                        inputs_embeds=inputs_embeds,
                        seam_proj=seam_proj,
                        v40_tensor=v40_tensor,
                        temporal_mask=temp_mask,
                        policy=config.temporal_policy,
                    )
            else:
                injected_embeds = inputs_embeds

            temporal_ctx = TemporalContext(
                anchor_year=cond_year,
                v40=v40,
                theta=theta,
                z=z,
                mentions=(anchor_mention,) if anchor_mention else (),
            )

            # Lazy import generation adapter to avoid circular dependency
            from .generation_adapter import generate_unconstrained

            gen_result = generate_unconstrained(
                model=model,
                tokenizer=tokenizer,
                inputs_embeds=injected_embeds,
                attention_mask=attention_mask,
                temporal_context=temporal_ctx,
                policy=config.generation_policy,
                adapter_mode=getattr(config, "adapter_mode", "ACTIVE"),
            )

            # Parse continuation
            parse_res = parse_answer(
                continuation=gen_result.raw_text,
                task_id=task_id,
                options=opt_texts,
                contract=config.parser_contract,
            )

            numeric_diag_label: Optional[str] = None
            if task_id != "Rel" and opt_texts:
                num_res = parse_unique_exact_numeric(gen_result.raw_text, opt_texts)
                numeric_diag_label = num_res.label if num_res.is_valid else "__UNMATCHED__"

            is_corr = bool(parse_res.is_valid and parse_res.label == gold.lower().strip())
            rec = PredictionRecord(
                global_idx=g_idx,
                example_id=ex_id,
                rotation_id=rot_id,
                task_id=task_id,
                prompt_hash=p_hash,
                raw_continuation=gen_result.raw_text,
                generated_token_ids=gen_result.continuation_token_ids,
                parsed_label=parse_res.label,
                parse_reason=parse_res.invalid_reason.value,
                is_valid=parse_res.is_valid,
                gold=gold.lower().strip(),
                is_correct=is_corr,
                temporal_policy=config.temporal_policy.value,
                parser_contract=config.parser_contract.value,
                metadata_year=anchor_year,
                chosen_anchor_year=cond_year,
                anchor_role=anchor_mention.role if anchor_mention is not None else None,
                anchor_char_span=(anchor_mention.char_start, anchor_mention.char_end) if anchor_mention else (-1, -1),
                anchor_token_span=(
                    (int(anchor_mention.token_start), int(anchor_mention.token_end))
                    if anchor_mention and anchor_mention.token_start is not None and anchor_mention.token_end is not None
                    else (-1, -1)
                ),
                numeric_exact_match_label=numeric_diag_label,
            )
            records.append(rec)

    metrics = aggregate_evaluation_metrics(records)
    total_time = round(time.time() - t_start, 2)

    output_payload = {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "eval_run_id": f"RUN_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        "timestamp": datetime.now().isoformat(),
        "config": config.to_dict(),
        "config_hash": config.config_hash(),
        "contract_hash": config.contract_hash(),
        "runtime_seconds": total_time,
        "metrics": metrics,
    }

    if output_dir:
        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)
        # Write summary
        summary_file = out_path / "EVAL_SUMMARY.json"
        summary_file.write_text(json.dumps(output_payload, indent=2, ensure_ascii=False), encoding="utf-8")
        # Write per-case jsonl
        jsonl_file = out_path / "PREDICTION_RECORDS.jsonl"
        with jsonl_file.open("w", encoding="utf-8") as f:
            for r in records:
                f.write(json.dumps(r.to_dict(), ensure_ascii=False) + "\n")

    return output_payload


def run_dry_run_checks(
    config: EvaluationConfig,
    output_dir: Optional[str] = None,
) -> Dict[str, Any]:
    """Execute dry-run validation of planned evaluation without loading GPU weights or running full inference.

    Validates:
    - Preflight data and manifest file existence and hash.
    - Checkpoint existence and tensor census if arm requires checkpoint.
    - Fast tokenizer loading and prompt tokenization.
    - Sample extraction and temporal locator resolution on 1 real instance.
    - Asserts planned actions without invoking paid API or GPU jobs.
    """
    preflight = run_preflight_checks(config)
    if preflight["status"] == "FAIL":
        return {
            "status": "FAIL",
            "dry_run": True,
            "reason": "PREFLIGHT_CHECKS_FAILED",
            "preflight": preflight,
        }

    # Verify checkpoint if arm requires it
    ckpt_info = None
    if config.arm in ("arm4", "arm5"):
        if not config.checkpoint_path or not Path(config.checkpoint_path).is_file():
            return {
                "status": "FAIL",
                "dry_run": True,
                "reason": f"MANDATORY_CHECKPOINT_MISSING_FOR_ARM_{config.arm.upper()}",
                "checkpoint_path": config.checkpoint_path,
            }
        ckpt = torch.load(config.checkpoint_path, map_location="cpu", weights_only=True)
        ckpt_info = {
            "lora_tensors": len(ckpt.get("lora_state", {})),
            "seam_tensors": len(ckpt.get("seam_projection", {})),
            "core_tensors": len(ckpt.get("core_generator", {})) if ckpt.get("core_generator") else 0,
            "alpha": float(ckpt.get("seam_projection", {}).get("alpha", torch.tensor(0.0)).item()) if "seam_projection" in ckpt else None,
        }

    # Verify sample instance extraction and locator
    eval_p = Path(config.eval_data_path)
    sample_case = None
    if eval_p.is_file():
        if eval_p.suffix == ".gz":
            with gzip.open(eval_p, "rt", encoding="utf-8") as f:
                first_line = f.readline()
                if first_line:
                    sample_case = json.loads(first_line)
        else:
            with eval_p.open("r", encoding="utf-8") as f:
                first_line = f.readline()
                if first_line:
                    sample_case = json.loads(first_line)

    result = {
        "status": "DRY_RUN_PASS",
        "dry_run": True,
        "does_not_conduct_inference": True,
        "message": "Dry run verification only; does not conduct model inference.",
        "config": config.to_dict(),
        "config_hash": config.config_hash(),
        "arm": config.arm,
        "temporal_policy": config.temporal_policy.value,
        "parser_contract": config.parser_contract.value,
        "checkpoint_census": ckpt_info,
        "sample_verified": {
            "task_id": sample_case.get("task_id") if sample_case else None,
            "year": sample_case.get("year") if sample_case else None,
            "example_id": sample_case.get("example_id") if sample_case else None,
        } if sample_case else None,
        "planned_actions": [
            f"Attach adapters for {config.arm}",
            f"Load checkpoint weights from {config.checkpoint_path}" if config.checkpoint_path else "Unadapted base backbone",
            f"Apply unconstrained greedy generation with policy max_new_tokens={config.generation_policy.max_new_tokens}",
            f"Parse outputs deterministically with {config.parser_contract.value}",
        ],
    }

    if output_dir:
        out_p = Path(output_dir)
        out_p.mkdir(parents=True, exist_ok=True)
        (out_p / "DRY_RUN_RECEIPT.json").write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")

    return result


def build_cli_parser() -> argparse.ArgumentParser:
    """Build CLI argument parser for Stage 03 G1 evaluation."""
    parser = argparse.ArgumentParser(
        description="Stage 03 G1 Temporal-Injection & Unconstrained Free-Generation Evaluation"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Preflight command
    p_pre = subparsers.add_parser("preflight", help="Run contract and manifest preflight checks on CPU")
    p_pre.add_argument("--arm", choices=["base", "arm4", "arm5"], default="base")
    p_pre.add_argument("--manifest-path", default="Reports/6-11Report/_POST_FORENSIC_OPEN_GATES_20261105/REPLAY_SAMPLE_MANIFEST.json")
    p_pre.add_argument("--eval-data-path", default="data/tempun_full_clean/eval_5292_clean.jsonl.gz")
    p_pre.add_argument("--checkpoint", default=None)

    # Smoke command
    p_smoke = subparsers.add_parser("smoke", help="Run end-to-end CPU mock smoke test")
    p_smoke.add_argument("--arm", choices=["base", "arm4", "arm5"], default="arm5")
    p_smoke.add_argument(
        "--temporal-policy",
        choices=["LEGACY_FIXED14", "YEAR_SPAN", "MULTI_YEAR_SPAN", "SEQUENCE_VALID", "SEAM_OFF"],
        default="YEAR_SPAN",
    )
    p_smoke.add_argument("--parser-contract", choices=["LABEL_ONLY", "UNIQUE_EXACT_NUMERIC_VALUE"], default="LABEL_ONLY")
    p_smoke.add_argument("--z-mode", choices=["linear", "asinh"], default="linear")
    p_smoke.add_argument("--fone-mode", choices=["dyadic", "powers10"], default="dyadic")
    p_smoke.add_argument("--output", default=None)

    # Eval command
    p_eval = subparsers.add_parser("eval", help="Run guarded evaluation")
    p_eval.add_argument("--arm", choices=["base", "arm4", "arm5"], default="base")
    p_eval.add_argument("--checkpoint", default=None)
    p_eval.add_argument(
        "--temporal-policy",
        choices=["LEGACY_FIXED14", "YEAR_SPAN", "MULTI_YEAR_SPAN", "SEQUENCE_VALID", "SEAM_OFF"],
        default="YEAR_SPAN",
    )
    p_eval.add_argument("--parser-contract", choices=["LABEL_ONLY", "UNIQUE_EXACT_NUMERIC_VALUE"], default="LABEL_ONLY")
    p_eval.add_argument("--z-mode", choices=["linear", "asinh"], default="linear")
    p_eval.add_argument("--fone-mode", choices=["dyadic", "powers10"], default="dyadic")
    p_eval.add_argument("--manifest-path", default="Reports/6-11Report/_POST_FORENSIC_OPEN_GATES_20261105/REPLAY_SAMPLE_MANIFEST.json")
    p_eval.add_argument("--eval-data-path", default="data/tempun_full_clean/eval_5292_clean.jsonl.gz")
    p_eval.add_argument("--device", default="cpu")
    p_eval.add_argument("--dry-run", action="store_true")
    p_eval.add_argument("--n-samples", type=int, default=None, help="Number of frozen samples to evaluate (default: all)")
    p_eval.add_argument("--tiny-qwen", action="store_true", help="Construct local random tiny Qwen2 architecture for CPU testing")
    p_eval.add_argument("--use-mock", action="store_true", help="Use fast CPU mock model")
    p_eval.add_argument("--output", default="Reports/6-11Report/G1_FREEGEN_INJECTION_V2")

    return parser


def main() -> int:
    parser = build_cli_parser()
    args = parser.parse_args()

    if args.command == "preflight":
        cfg = EvaluationConfig(
            arm=args.arm,
            manifest_path=args.manifest_path,
            eval_data_path=args.eval_data_path,
            checkpoint_path=args.checkpoint,
        )
        res = run_preflight_checks(cfg)
        print(json.dumps(res, indent=2, ensure_ascii=False))
        return 0 if res["status"] in ("PASS", "PREFLIGHT_WARN") else 1

    if args.command == "smoke":
        from transformers import AutoTokenizer
        try:
            tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct", local_files_only=True)
        except Exception:
            tok = AutoTokenizer.from_pretrained("gpt2")
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token

        # Small mock causal model with proper linear layers
        class SmokeMockModel(torch.nn.Module):
            def __init__(self, vocab_size=len(tok), hidden_size=64):
                super().__init__()
                self.vocab_size = vocab_size
                self.hidden_size = hidden_size
                class MockConfig:
                    pass
                self.config = MockConfig()
                self.config.hidden_size = hidden_size
                self.embed_tokens = torch.nn.Embedding(vocab_size, hidden_size)
                self.gate_proj = torch.nn.Linear(hidden_size, hidden_size, bias=False)
                self.up_proj = torch.nn.Linear(hidden_size, hidden_size, bias=False)
                self.down_proj = torch.nn.Linear(hidden_size, hidden_size, bias=False)
                self.lm_head = torch.nn.Linear(hidden_size, vocab_size, bias=False)

            def get_input_embeddings(self):
                return self.embed_tokens

            def forward(self, input_ids=None, inputs_embeds=None, attention_mask=None, past_key_values=None, use_cache=False, return_dict=True):
                h = inputs_embeds if inputs_embeds is not None else self.embed_tokens(input_ids)
                h = h + self.down_proj(torch.relu(self.gate_proj(h)))
                logits = self.lm_head(h)
                class Out:
                    def __init__(self, l, p):
                        self.logits = l
                        self.past_key_values = p
                return Out(logits, () if use_cache else None)

        mock_model = SmokeMockModel()
        _, core_gen = attach_adapters_to_model(mock_model, arm=args.arm)
        cfg = EvaluationConfig(
            arm=args.arm,
            temporal_policy=TemporalPolicy(args.temporal_policy),
            parser_contract=ParserContract(args.parser_contract),
            device="cpu",
            z_mode=args.z_mode,
            fone_mode=args.fone_mode,
        )
        mock_cases = [
            {
                "example_id": "smoke_mcq_1",
                "task_id": "DB",
                "year": 1947,
                "input": "In 1947, event happened: Options: (a) val1 (b) val2 (c) val3 (d) val4",
                "target": "b",
            },
            {
                "example_id": "smoke_rel_1",
                "task_id": "Rel",
                "year": 1947,
                "input": "In 1947, did value increase compared to 1952? Answer as Yes or No",
                "target": "Yes",
            },
        ]
        out_summary = execute_evaluation_loop(
            config=cfg,
            model=mock_model,
            tokenizer=tok,
            eval_cases=mock_cases,
            output_dir=args.output,
            core_gen=core_gen,
        )
        print(json.dumps(out_summary, indent=2, ensure_ascii=False))
        return 0

    if args.command == "eval":
        cfg = EvaluationConfig(
            arm=args.arm,
            checkpoint_path=args.checkpoint,
            temporal_policy=TemporalPolicy(args.temporal_policy),
            parser_contract=ParserContract(args.parser_contract),
            manifest_path=args.manifest_path,
            eval_data_path=args.eval_data_path,
            device=args.device,
            z_mode=args.z_mode,
            fone_mode=args.fone_mode,
        )
        if args.dry_run:
            dry_res = run_dry_run_checks(cfg, output_dir=args.output)
            print(json.dumps(dry_res, indent=2, ensure_ascii=False))
            return 0 if dry_res["status"] == "DRY_RUN_PASS" else 1

        # Real guarded eval
        if args.arm in ("arm4", "arm5") and (not args.checkpoint or not Path(args.checkpoint).is_file()):
            print(json.dumps({
                "status": "FAIL",
                "reason": f"CHECKPOINT_FILE_REQUIRED_FOR_ARM_{args.arm.upper()}",
                "checkpoint": args.checkpoint,
            }, indent=2))
            return 1

        if args.device != "cpu":
            print(json.dumps({
                "status": "BLOCKED_GPU_NOT_AUTHORIZED",
                "reason": "GPU execution requires explicit user authorization and budget approval.",
                "status_code": "GPU_UNVERIFIED",
            }, indent=2))
            return 1

        try:
            cases = load_frozen_evaluation_cases(
                cfg.manifest_path, cfg.eval_data_path, n_samples=args.n_samples
            )
        except Exception as e:
            print(json.dumps({"status": "FAIL", "reason": f"SAMPLE_LOAD_ERROR: {str(e)}"}, indent=2))
            return 1

        from transformers import AutoTokenizer
        try:
            tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct", local_files_only=True)
        except Exception:
            tok = AutoTokenizer.from_pretrained("gpt2")
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token

        # Instantiate model based on flags
        if getattr(args, "tiny_qwen", False):
            from transformers import Qwen2Config, Qwen2ForCausalLM
            qcfg = Qwen2Config(
                vocab_size=len(tok),
                hidden_size=64,
                intermediate_size=128,
                num_hidden_layers=2,
                num_attention_heads=4,
                num_key_value_heads=2,
                max_position_embeddings=256,
            )
            model = Qwen2ForCausalLM(qcfg).eval()
        else:
            # CPU mock causal model
            class EvalMockModel(torch.nn.Module):
                def __init__(self, vocab_size=len(tok), hidden_size=64):
                    super().__init__()
                    self.vocab_size = vocab_size
                    self.hidden_size = hidden_size
                    class MockConfig:
                        pass
                    self.config = MockConfig()
                    self.config.hidden_size = hidden_size
                    self.embed_tokens = torch.nn.Embedding(vocab_size, hidden_size)
                    self.gate_proj = torch.nn.Linear(hidden_size, hidden_size, bias=False)
                    self.up_proj = torch.nn.Linear(hidden_size, hidden_size, bias=False)
                    self.down_proj = torch.nn.Linear(hidden_size, hidden_size, bias=False)
                    self.lm_head = torch.nn.Linear(hidden_size, vocab_size, bias=False)

                def get_input_embeddings(self):
                    return self.embed_tokens

                def forward(self, input_ids=None, inputs_embeds=None, attention_mask=None, past_key_values=None, use_cache=False, return_dict=True):
                    h = inputs_embeds if inputs_embeds is not None else self.embed_tokens(input_ids)
                    h = h + self.down_proj(torch.relu(self.gate_proj(h)))
                    logits = self.lm_head(h)
                    class Out:
                        def __init__(self, l, p):
                            self.logits = l
                            self.past_key_values = p
                    return Out(logits, () if use_cache else None)

            model = EvalMockModel(vocab_size=len(tok), hidden_size=64)

        _, core_gen = attach_adapters_to_model(model, arm=cfg.arm)
        seam_proj = None
        if cfg.arm in ("arm4", "arm5"):
            h_dim = getattr(model.config, "hidden_size", 64)
            seam_proj = GatedResidualProjection(in_features=40, out_features=h_dim, gate_mode="linear")
            if cfg.checkpoint_path and Path(cfg.checkpoint_path).is_file():
                load_checkpoint_weights(model, seam_proj, core_gen, cfg.checkpoint_path, arm=cfg.arm)

        try:
            out_summary = execute_evaluation_loop(
                config=cfg,
                model=model,
                tokenizer=tok,
                eval_cases=cases,
                output_dir=args.output,
                seam_proj=seam_proj,
                core_gen=core_gen,
            )
            assert out_summary.get("metrics", {}).get("total_records", 0) > 0, "Nonzero evaluation record count expected"
            print(json.dumps(out_summary, indent=2, ensure_ascii=False))
            return 0
        except Exception as exc:
            print(json.dumps({"status": "FAIL", "error": str(exc)}, indent=2))
            return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
