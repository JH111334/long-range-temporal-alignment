"""Versioned Evaluation Contracts and Data Structures for Stage 03 G1 Free Generation.

Canonical Problem & Architectural Contract:
- Goal: G1 = YEAR-only closed-answer temporal QA (MCQA 4-choice A/B/C/D + binary Yes/No).
- Branch: v2-targeted-continual-alignment.
- Paradigm: Zero-Shot Cross-Task Transfer (Train pure numeric NWP -> Eval MCQA/Rel, no MCQA SFT).
- Adapter Scope Lock: MLP Linear layers ONLY (gate_proj, up_proj, down_proj across 108 modules).
  NO Attention intervention (q_proj, k_proj, v_proj, o_proj are strictly forbidden).
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple


class TemporalPolicy(str, Enum):
    """Temporal representation injection policy."""

    LEGACY_FIXED14 = "LEGACY_FIXED14"
    YEAR_SPAN = "YEAR_SPAN"
    MULTI_YEAR_SPAN = "MULTI_YEAR_SPAN"
    SEQUENCE_VALID = "SEQUENCE_VALID"
    SEAM_OFF = "SEAM_OFF"


class AnswerMode(str, Enum):
    """Answer task modality."""

    MCQA = "MCQA"
    BINARY_REL = "BINARY_REL"


class ParserContract(str, Enum):
    """Answer extraction parser contract."""

    LABEL_ONLY = "LABEL_ONLY"
    UNIQUE_EXACT_NUMERIC_VALUE = "UNIQUE_EXACT_NUMERIC_VALUE"


class InvalidReason(str, Enum):
    """Taxonomy of invalid free-generation outputs."""

    VALID = "VALID"
    EMPTY_OUTPUT = "EMPTY_OUTPUT"
    LENGTH_EXHAUSTION = "LENGTH_EXHAUSTION"
    NO_RECOGNIZABLE_LABEL = "NO_RECOGNIZABLE_LABEL"
    MULTIPLE_CONTRADICTORY_LABELS = "MULTIPLE_CONTRADICTORY_LABELS"
    HALLUCINATED_OPTION = "HALLUCINATED_OPTION"
    REFUSAL_OR_UNSUPPORTED = "REFUSAL_OR_UNSUPPORTED"
    UNPARSABLE_FORMAT = "UNPARSABLE_FORMAT"


@dataclass(frozen=True, slots=True)
class YearMention:
    """Exact character and token span of a year mention in formatted prompt."""

    year: int
    role: str = "anchor"
    char_start: int = -1
    char_end: int = -1
    token_start: int | None = None
    token_end: int | None = None

    def is_resolved(self) -> bool:
        return (
            self.char_start >= 0
            and self.char_end > self.char_start
            and self.token_start is not None
            and self.token_end is not None
            and self.token_end > self.token_start
        )


@dataclass(frozen=True, slots=True)
class TemporalContext:
    """Temporal conditioning metadata for a single instance."""

    anchor_year: int
    v40: Tuple[float, ...] = field(default_factory=tuple)
    theta: float = 0.0
    z: float = 0.0
    mentions: Tuple[YearMention, ...] = field(default_factory=tuple)
    unsupported_reason: Optional[str] = None

    @property
    def is_valid(self) -> bool:
        return self.unsupported_reason is None and len(self.v40) == 40


@dataclass(frozen=True, slots=True)
class GenerationPolicy:
    """Unconstrained deterministic greedy decoding policy."""

    max_new_tokens: int = 16
    do_sample: bool = False
    num_beams: int = 1
    temperature: float = 0.0
    stop_strings: Tuple[str, ...] = ("\n", "\n\n", "Question:", "Answer:")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "max_new_tokens": self.max_new_tokens,
            "do_sample": self.do_sample,
            "num_beams": self.num_beams,
            "temperature": self.temperature,
            "stop_strings": list(self.stop_strings),
        }


@dataclass(frozen=True, slots=True)
class PredictionRecord:
    """Full per-case evaluation record with exact provenance."""

    global_idx: int
    example_id: str
    rotation_id: int
    task_id: str
    prompt_hash: str
    raw_continuation: str
    generated_token_ids: Tuple[int, ...]
    parsed_label: Optional[str]
    parse_reason: str
    is_valid: bool
    gold: str
    is_correct: bool
    temporal_policy: str
    parser_contract: str
    metadata_year: Optional[int] = None
    chosen_anchor_year: Optional[int] = None
    anchor_role: Optional[str] = None
    anchor_char_span: Tuple[int, int] = (-1, -1)
    anchor_token_span: Tuple[int, int] = (-1, -1)
    numeric_exact_match_label: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "global_idx": self.global_idx,
            "example_id": self.example_id,
            "rotation_id": self.rotation_id,
            "task_id": self.task_id,
            "prompt_hash": self.prompt_hash,
            "raw_continuation": self.raw_continuation,
            "generated_token_ids": list(self.generated_token_ids),
            "parsed_label": self.parsed_label,
            "parse_reason": self.parse_reason,
            "is_valid": self.is_valid,
            "gold": self.gold,
            "is_correct": self.is_correct,
            "temporal_policy": self.temporal_policy,
            "parser_contract": self.parser_contract,
        }
        if (
            self.metadata_year is not None
            or self.chosen_anchor_year is not None
            or self.anchor_role is not None
            or self.numeric_exact_match_label is not None
        ):
            payload.update(
                {
                    "metadata_year": self.metadata_year,
                    "chosen_anchor_year": self.chosen_anchor_year,
                    "anchor_role": self.anchor_role,
                    "anchor_char_span": list(self.anchor_char_span),
                    "anchor_token_span": list(self.anchor_token_span),
                    "numeric_exact_match_label": self.numeric_exact_match_label,
                }
            )
        return payload


@dataclass(frozen=True, slots=True)
class EvaluationConfig:
    """Versioned evaluation contract configuration."""

    contract_version: str = "EVAL_CONTRACT_G1_FREEGEN_V2"
    arm: str = "base"  # "base" | "arm4" | "arm5"
    temporal_policy: TemporalPolicy = TemporalPolicy.YEAR_SPAN
    parser_contract: ParserContract = ParserContract.LABEL_ONLY
    generation_policy: GenerationPolicy = field(default_factory=GenerationPolicy)
    seed: int = 42
    batch_size: int = 1
    device: str = "cpu"
    checkpoint_path: Optional[str] = None
    manifest_path: str = "Reports/6-11Report/_POST_FORENSIC_OPEN_GATES_20261105/REPLAY_SAMPLE_MANIFEST.json"
    eval_data_path: str = "data/tempun_full_clean/eval_5292_clean.jsonl.gz"
    z_mode: str = "linear"  # "linear" | "asinh"
    fone_mode: str = "dyadic"  # "dyadic" | "powers10"
    radius_mode: str = "linear"  # "linear" | "asinh"
    wrong_year_offset: int = 0
    adapter_mode: str = "ACTIVE"  # "ACTIVE" | "NEUTRAL_CORE" | "BOTH_OFF"
    adapter_scope: str = "MLP_ONLY"  # Locked invariant: MLP Linear only, NO Attention

    def to_dict(self) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "contract_version": self.contract_version,
            "arm": self.arm,
            "temporal_policy": self.temporal_policy.value,
            "parser_contract": self.parser_contract.value,
            "generation_policy": self.generation_policy.to_dict(),
            "seed": self.seed,
            "batch_size": self.batch_size,
            "device": self.device,
            "checkpoint_path": self.checkpoint_path,
            "manifest_path": self.manifest_path,
            "eval_data_path": self.eval_data_path,
            "z_mode": self.z_mode,
            "fone_mode": self.fone_mode,
            "adapter_scope": self.adapter_scope,
        }
        if self.radius_mode != "linear":
            payload["radius_mode"] = self.radius_mode
        if self.wrong_year_offset != 0:
            payload["wrong_year_offset"] = int(self.wrong_year_offset)
        if self.adapter_mode != "ACTIVE":
            payload["adapter_mode"] = self.adapter_mode
        return payload

    def config_hash(self) -> str:
        """Deterministic SHA-256 fingerprint of the configuration."""
        canonical_str = json.dumps(self.to_dict(), sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(canonical_str.encode("utf-8")).hexdigest()
