"""CPU-verifiable preflight contract for Stage 03 V3 MLP-only CPT + Eval on Modal.

Enforces G0/G1/G2/G3 readiness checks before any remote Modal GPU dispatch:
- Dataset & manifest SHA-256 verification.
- Locked scope validation (G1 YEAR-only, Zero-Shot NWP->MCQA, MLP-only NO Attention, NO DoRA).
- Explicit GPU confirmation & budget bound checks.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict

from .contracts import EvaluationConfig, ParserContract, TemporalPolicy

EXPECTED_SHA256: Dict[str, str] = {
    "train": "111d2d0908533a17ac10c98533802053631b708a23846524daa224bd1465697d",
    "eval": "a624d8d258f9676f00e504d16c35dc401cc1cce96613ca56af970daa32a12fc3",
    "manifest": "0fadc3d24278575cf0d64047c7d0c293b423e08d948699272b5b9497b9736ff3",
}

A10G_USD_PER_HOUR: float = 1.10


@dataclass(frozen=True, slots=True)
class V3ModalLaunchSpec:
    """Validated launch specification for Stage 03 V3 Modal execution."""

    arm: str = "arm5"  # "arm4" | "arm5" (MLP-only ONLY)
    epochs: int = 5
    micro_batch: int = 32
    grad_accum: int = 4
    base_lr: float = 1.0e-4
    adapter_head_lr: float = 2.0e-4
    warmup_ratio: float = 0.04
    schedule_type: str = "cosine"
    min_lr_ratio: float = 0.1
    weight_decay: float = 0.01
    rank: int = 8
    alpha: float = 16.0
    lambda_intra: float = 0.5
    lambda_inter: float = 0.5
    margin_inter: float = 0.1
    z_mode: str = "asinh"
    fone_mode: str = "powers10"
    temporal_policy: str = TemporalPolicy.MULTI_YEAR_SPAN.value
    n_eval_samples: int = 128
    max_budget_usd: float = 5.0
    profile: str = "loggingfuture"

    def validate(self) -> None:
        if self.arm not in ("arm4", "arm5"):
            raise ValueError(f"Unsupported V3 arm '{self.arm}'. Only MLP-only 'arm4' or 'arm5' permitted.")
        if self.epochs < 1 or self.epochs > 10:
            raise ValueError(f"epochs must be in [1, 10], got {self.epochs}")
        if self.z_mode != "asinh":
            raise ValueError(f"V3 requires z_mode='asinh', got '{self.z_mode}'")
        if self.fone_mode != "powers10":
            raise ValueError(f"V3 requires fone_mode='powers10', got '{self.fone_mode}'")
        if self.temporal_policy != TemporalPolicy.MULTI_YEAR_SPAN.value:
            raise ValueError(f"V3 requires temporal_policy='MULTI_YEAR_SPAN', got '{self.temporal_policy}'")
        if self.schedule_type not in ("cosine", "wsd"):
            raise ValueError(f"schedule_type must be 'cosine' or 'wsd', got '{self.schedule_type}'")
        if self.max_budget_usd <= 0.0:
            raise ValueError("max_budget_usd must be positive")

    def estimated_gpu_hours(self) -> float:
        # Reference: prior 5-epoch A10G run took ~3.4h (~$3.74); scale proportionally with epochs
        return round(3.5 * (float(self.epochs) / 5.0), 3)

    def estimated_cost_usd(self) -> float:
        return round(self.estimated_gpu_hours() * A10G_USD_PER_HOUR, 3)

    def eval_config(self) -> EvaluationConfig:
        return EvaluationConfig(
            contract_version="EVAL_CONTRACT_G1_FREEGEN_V3",
            arm=self.arm,
            temporal_policy=TemporalPolicy(self.temporal_policy),
            parser_contract=ParserContract.LABEL_ONLY,
            z_mode=self.z_mode,
            fone_mode=self.fone_mode,
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
    """Run deterministic CPU preflight checks on datasets, manifests, hashes, and budget cap."""
    spec.validate()
    paths = {
        "train": Path(train_path),
        "eval": Path(eval_path),
        "manifest": Path(manifest_path),
    }
    hashes: Dict[str, str] = {}
    for key, p in paths.items():
        if not p.is_file():
            raise FileNotFoundError(f"Required input file missing ({key}): {p}")
        digest = hashlib.sha256(p.read_bytes()).hexdigest()
        expected = EXPECTED_SHA256[key]
        if digest != expected:
            raise ValueError(f"SHA-256 mismatch for {key} ({p}): got {digest}, expected {expected}")
        hashes[key] = digest

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
            "transfer_mode": "ZERO_SHOT_NWP_TO_MCQA",
            "adapter_scope": "MLP_ONLY",
            "forbidden_modules": ["q_proj", "k_proj", "v_proj", "o_proj"],
            "dora_enabled": False,
        },
        "spec": asdict(spec),
        "eval_config_hash": eval_cfg.config_hash(),
        "input_sha256": hashes,
        "estimated_a10g_hours": spec.estimated_gpu_hours(),
        "estimated_cost_usd": est_cost,
        "budget_cap_usd": spec.max_budget_usd,
        "modal_profile_target": spec.profile,
    }
