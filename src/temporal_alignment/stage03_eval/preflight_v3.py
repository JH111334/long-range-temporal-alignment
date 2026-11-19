"""CPU-verifiable preflight contract for Stage 03 G1 MLP-only CPT + Eval on Modal.

Enforces G0/G1/G2/G3 readiness checks before any remote Modal GPU dispatch:
- Dataset, general replay, and manifest SHA-256 verification (reporting raw & CRLF hashes).
- Locked scope validation (G1 YEAR-only, Zero-Shot NWP->MCQA/Rel, MLP-only NO Attention, NO DoRA).
- Explicit recipe resolution (`G1_SINGLE_YEAR_REPLAY_FIXED`, `G1_SINGLE_YEAR_REPLAY_COSINE`, and historical archive).
- Planned mixed record/optimizer step accounting and explicit GPU budget caps.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Tuple

from .contracts import EvaluationConfig, ParserContract, TemporalPolicy

EXPECTED_SHA256: Dict[str, str] = {
    "train": "111d2d0908533a17ac10c98533802053631b708a23846524daa224bd1465697d",
    "eval": "a624d8d258f9676f00e504d16c35dc401cc1cce96613ca56af970daa32a12fc3",
    "manifest": "0fadc3d24278575cf0d64047c7d0c293b423e08d948699272b5b9497b9736ff3",
    "replay_train": "04a212557ea4160f83ab60d635704df82cd6aea098aa6b5521a8df24f4f2a5f7",
    "replay_holdout": "5338e1c507d25eed448982327aa52fd3d736455a066d92c39c6c369e0f8a7709",
}

A10G_USD_PER_HOUR: float = 1.10
PINNED_NWP_COUNT: int = 104130

LOCKED_G1_REPAIR_RECIPES: Dict[str, Dict[str, Any]] = {
    "G1_SINGLE_YEAR_REPLAY_FIXED": {
        "arm": "arm5",
        "epochs": 5,
        "micro_batch": 32,
        "grad_accum": 4,
        "base_lr": 2.0e-4,
        "adapter_head_lr": 2.0e-4,
        "warmup_ratio": 0.0,
        "schedule_type": "constant",
        "min_lr_ratio": 1.0,
        "weight_decay": 0.01,
        "rank": 8,
        "alpha": 16.0,
        "lambda_intra": 0.5,
        "lambda_inter": 0.5,
        "margin_inter": 0.1,
        "z_mode": "linear",
        "fone_mode": "dyadic",
        "radius_mode": "linear",
        "temporal_policy": TemporalPolicy.YEAR_SPAN.value,
        "replay_ratio": 0.15,
        "seed": 42,
        "checkpoint_namespace": "g1_repair",
    },
    "G1_SINGLE_YEAR_REPLAY_COSINE": {
        "arm": "arm5",
        "epochs": 5,
        "micro_batch": 32,
        "grad_accum": 4,
        "base_lr": 2.0e-4,
        "adapter_head_lr": 2.0e-4,
        "warmup_ratio": 0.04,
        "schedule_type": "cosine",
        "min_lr_ratio": 0.10,
        "weight_decay": 0.01,
        "rank": 8,
        "alpha": 16.0,
        "lambda_intra": 0.5,
        "lambda_inter": 0.5,
        "margin_inter": 0.1,
        "z_mode": "linear",
        "fone_mode": "dyadic",
        "radius_mode": "linear",
        "temporal_policy": TemporalPolicy.YEAR_SPAN.value,
        "replay_ratio": 0.15,
        "seed": 42,
        "checkpoint_namespace": "g1_repair",
    },
    "LEGACY_V3_HISTORICAL": {
        "arm": "arm5",
        "epochs": 4,
        "micro_batch": 32,
        "grad_accum": 4,
        "base_lr": 1.0e-4,
        "adapter_head_lr": 2.0e-4,
        "warmup_ratio": 0.04,
        "schedule_type": "cosine",
        "min_lr_ratio": 0.10,
        "weight_decay": 0.01,
        "rank": 8,
        "alpha": 16.0,
        "lambda_intra": 0.5,
        "lambda_inter": 0.5,
        "margin_inter": 0.1,
        "z_mode": "asinh",
        "fone_mode": "powers10",
        "radius_mode": "asinh",
        "temporal_policy": TemporalPolicy.MULTI_YEAR_SPAN.value,
        "replay_ratio": 0.0,
        "seed": 42,
        "checkpoint_namespace": "legacy_v3",
    },
}


@dataclass(frozen=True, slots=True)
class V3ModalLaunchSpec:
    """Validated launch specification for Stage 03 G1 MLP-only Modal execution."""

    recipe_id: str = "G1_SINGLE_YEAR_REPLAY_FIXED"
    run_id: str = ""
    arm: str = "arm5"  # "arm4" | "arm5" (MLP-only ONLY)
    epochs: int = 5
    micro_batch: int = 32
    grad_accum: int = 4
    base_lr: float = 2.0e-4
    adapter_head_lr: float = 2.0e-4
    warmup_ratio: float = 0.0
    schedule_type: str = "constant"
    min_lr_ratio: float = 1.0
    weight_decay: float = 0.01
    rank: int = 8
    alpha: float = 16.0
    lambda_intra: float = 0.5
    lambda_inter: float = 0.5
    margin_inter: float = 0.1
    z_mode: str = "linear"
    fone_mode: str = "dyadic"
    radius_mode: str = "linear"
    temporal_policy: str = TemporalPolicy.YEAR_SPAN.value
    replay_ratio: float = 0.15
    replay_path: str = "data/general_replay/g1_open_qa_train.jsonl.gz"
    replay_holdout_path: str = "data/general_replay/g1_open_qa_holdout.jsonl.gz"
    replay_manifest_path: str = "data/receipts/g1_general_replay_manifest.json"
    checkpoint_namespace: str = "g1_repair"
    seed: int = 42
    n_eval_samples: int = 128
    max_budget_usd: float = 5.0
    profile: str = "loggingfuture"

    @classmethod
    def from_recipe(cls, recipe_id: str, **overrides: Any) -> "V3ModalLaunchSpec":
        if recipe_id not in LOCKED_G1_REPAIR_RECIPES:
            raise ValueError(
                f"Unknown recipe_id '{recipe_id}'. Expected one of {sorted(LOCKED_G1_REPAIR_RECIPES.keys())}."
            )
        cfg = dict(LOCKED_G1_REPAIR_RECIPES[recipe_id])
        cfg["recipe_id"] = recipe_id
        if not overrides.get("run_id"):
            cfg["run_id"] = recipe_id.lower()
        for k, v in overrides.items():
            if v is not None:
                cfg[k] = v
        spec = cls(**cfg)
        spec.validate()
        return spec

    def validate(self) -> None:
        if self.recipe_id not in LOCKED_G1_REPAIR_RECIPES:
            raise ValueError(f"Unsupported recipe_id '{self.recipe_id}'.")
        if self.arm not in ("arm4", "arm5"):
            raise ValueError(f"Unsupported V3 arm '{self.arm}'. Only MLP-only 'arm4' or 'arm5' permitted.")
        if self.epochs < 1 or self.epochs > 10:
            raise ValueError(f"epochs must be in [1, 10], got {self.epochs}")
        if self.z_mode not in ("linear", "asinh"):
            raise ValueError(f"Unsupported z_mode '{self.z_mode}'")
        if self.fone_mode not in ("dyadic", "powers10"):
            raise ValueError(f"Unsupported fone_mode '{self.fone_mode}'")
        if self.radius_mode not in ("linear", "asinh"):
            raise ValueError(f"Unsupported radius_mode '{self.radius_mode}'")
        if self.temporal_policy not in (TemporalPolicy.MULTI_YEAR_SPAN.value, TemporalPolicy.YEAR_SPAN.value):
            raise ValueError(
                f"temporal_policy must be 'YEAR_SPAN' or 'MULTI_YEAR_SPAN', got '{self.temporal_policy}'"
            )
        if self.recipe_id in ("G1_SINGLE_YEAR_REPLAY_FIXED", "G1_SINGLE_YEAR_REPLAY_COSINE"):
            if self.z_mode != "linear" or self.fone_mode != "dyadic" or self.radius_mode != "linear":
                raise ValueError(
                    f"Recipe {self.recipe_id} requires z_mode='linear', fone_mode='dyadic', radius_mode='linear'."
                )
            if self.temporal_policy != TemporalPolicy.YEAR_SPAN.value:
                raise ValueError(f"Recipe {self.recipe_id} requires temporal_policy='YEAR_SPAN'.")
        if self.schedule_type not in ("constant", "cosine", "wsd"):
            raise ValueError(f"schedule_type must be 'constant', 'cosine', or 'wsd', got '{self.schedule_type}'")
        if not 0.0 <= self.replay_ratio <= 0.5:
            raise ValueError(f"replay_ratio must be in [0.0, 0.5], got {self.replay_ratio}")
        if self.max_budget_usd <= 0.0:
            raise ValueError("max_budget_usd must be positive")

    def planned_records_per_epoch(self, n_nwp: int = PINNED_NWP_COUNT) -> Dict[str, int]:
        n_replay = (
            int(round(n_nwp * self.replay_ratio / max(1e-8, 1.0 - self.replay_ratio)))
            if self.replay_ratio > 0.0
            else 0
        )
        total = n_nwp + n_replay
        micro_batches = int(math.ceil(float(total) / float(self.micro_batch)))
        opt_steps_ep = int(math.ceil(float(micro_batches) / float(self.grad_accum)))
        return {
            "nwp_records": n_nwp,
            "replay_records": n_replay,
            "total_records_per_epoch": total,
            "micro_batches_per_epoch": micro_batches,
            "opt_steps_per_epoch": opt_steps_ep,
            "total_opt_steps": opt_steps_ep * int(self.epochs),
        }

    def estimated_gpu_hours(self) -> float:
        # Scale proportionally by epochs and mixed-corpus exposure relative to 104k baseline (~3.4-3.5h for 5ep NWP-only)
        rec_scale = float(self.planned_records_per_epoch()["total_records_per_epoch"]) / float(PINNED_NWP_COUNT)
        return round(3.5 * (float(self.epochs) / 5.0) * rec_scale, 3)

    def estimated_cost_usd(self) -> float:
        return round(self.estimated_gpu_hours() * A10G_USD_PER_HOUR, 3)

    def eval_config(self, *, checkpoint_path: str | None = None) -> EvaluationConfig:
        contract_ver = (
            "EVAL_CONTRACT_G1_FREEGEN_V2_REPAIR"
            if self.recipe_id in ("G1_SINGLE_YEAR_REPLAY_FIXED", "G1_SINGLE_YEAR_REPLAY_COSINE")
            else "EVAL_CONTRACT_G1_FREEGEN_V3"
        )
        return EvaluationConfig(
            contract_version=contract_ver,
            arm=self.arm,
            temporal_policy=TemporalPolicy(self.temporal_policy),
            parser_contract=ParserContract.LABEL_ONLY,
            seed=self.seed,
            checkpoint_path=checkpoint_path,
            z_mode=self.z_mode,
            fone_mode=self.fone_mode,
            radius_mode=self.radius_mode,
            adapter_scope="MLP_ONLY",
        )


def verify_v3_modal_inputs(
    spec: V3ModalLaunchSpec,
    *,
    train_path: str = "data/tempun_full_clean/train_104k_clean.jsonl.gz",
    eval_path: str = "data/tempun_full_clean/eval_5292_clean.jsonl.gz",
    manifest_path: str = "Reports/6-11Report/_POST_FORENSIC_OPEN_GATES_20261105/REPLAY_SAMPLE_MANIFEST.json",
    src_root: str = "src/temporal_alignment",
) -> Dict[str, Any]:
    """Run deterministic CPU preflight checks on datasets, replay corpus, manifests, hashes, and budget cap."""
    spec.validate()
    paths = {
        "train": Path(train_path),
        "eval": Path(eval_path),
        "manifest": Path(manifest_path),
    }
    hashes: Dict[str, str] = {}
    manifest_hash_detail: Dict[str, str] = {}
    for key, p in paths.items():
        if not p.is_file():
            raise FileNotFoundError(f"Required input file missing ({key}): {p}")
        raw_bytes = p.read_bytes()
        raw_digest = hashlib.sha256(raw_bytes).hexdigest()
        expected = EXPECTED_SHA256[key]
        if key == "manifest":
            crlf_digest = hashlib.sha256(raw_bytes.replace(b"\r\n", b"\n").replace(b"\n", b"\r\n")).hexdigest()
            manifest_hash_detail = {
                "raw_sha256": raw_digest,
                "crlf_normalized_sha256": crlf_digest,
                "expected_historical_sha256": expected,
            }
            if raw_digest != expected and crlf_digest != expected:
                raise ValueError(f"SHA-256 mismatch for manifest ({p}): raw={raw_digest}, crlf={crlf_digest}")
            hashes[key] = expected
        else:
            if raw_digest != expected:
                raise ValueError(f"SHA-256 mismatch for {key} ({p}): got {raw_digest}, expected {expected}")
            hashes[key] = raw_digest

    replay_info: Dict[str, Any] = {"replay_ratio": spec.replay_ratio, "enabled": spec.replay_ratio > 0.0}
    if spec.replay_ratio > 0.0:
        for r_key, r_path_str, exp_key in (
            ("replay_train", spec.replay_path, "replay_train"),
            ("replay_holdout", spec.replay_holdout_path, "replay_holdout"),
        ):
            rp = Path(r_path_str)
            if not rp.is_file():
                raise FileNotFoundError(f"Required replay file missing ({r_key}): {rp}")
            r_digest = hashlib.sha256(rp.read_bytes()).hexdigest()
            if r_digest != EXPECTED_SHA256[exp_key]:
                raise ValueError(f"SHA-256 mismatch for {r_key} ({rp}): got {r_digest}, expected {EXPECTED_SHA256[exp_key]}")
            hashes[r_key] = r_digest
        rm_p = Path(spec.replay_manifest_path)
        if not rm_p.is_file():
            raise FileNotFoundError(f"Required replay manifest missing: {rm_p}")
        hashes["replay_manifest"] = hashlib.sha256(rm_p.read_bytes()).hexdigest()
        replay_info["manifest"] = json.loads(rm_p.read_text(encoding="utf-8")).get("split_disjointness", {})

    src_dir = Path(src_root)
    if not src_dir.is_dir():
        raise FileNotFoundError(f"Required source tree missing for Modal mount: {src_dir}")

    est_cost = spec.estimated_cost_usd()
    if est_cost > spec.max_budget_usd:
        raise ValueError(
            f"Estimated cost ${est_cost:.3f} USD exceeds max_budget_usd=${spec.max_budget_usd:.3f} USD."
        )

    eval_cfg = spec.eval_config()
    return {
        "status": "CPU_PREFLIGHT_PASS",
        "scope": {
            "goal": "G1_YEAR_ONLY",
            "branch": "v2-targeted-continual-alignment",
            "transfer_mode": (
                "ZERO_SHOT_NWP_PLUS_GENERAL_REPLAY_TO_MCQA"
                if spec.replay_ratio > 0.0
                else "ZERO_SHOT_NWP_TO_MCQA"
            ),
            "adapter_scope": "MLP_ONLY",
            "forbidden_modules": ["q_proj", "k_proj", "v_proj", "o_proj"],
            "dora_enabled": False,
        },
        "spec": asdict(spec),
        "training_budget": spec.planned_records_per_epoch(),
        "replay_verification": replay_info,
        "eval_config_hash": eval_cfg.config_hash(),
        "input_sha256": hashes,
        "manifest_hash_detail": manifest_hash_detail,
        "estimated_a10g_hours": spec.estimated_gpu_hours(),
        "estimated_cost_usd": est_cost,
        "budget_cap_usd": spec.max_budget_usd,
        "modal_profile_target": spec.profile,
    }
