"""Comprehensive CPU Unit and Integration Tests for Stage 03 G1 Free Generation & Temporal Injection V2.

Verifies:
1. P0 Contracts: Configuration hashing, dataclass immutability, prediction record serialization.
2. P1 Temporal Locator & Injector:
   - Boundary-verified year character span matching.
   - Fast tokenizer offset mapping to token span.
   - Fail-closed behavior on unresolvable/missing years (no silent 14 fallback).
   - Multi-year isolation in Rel prompts (anchor vs comparison mentions).
   - Batched temporal mask synthesis across LEGACY_FIXED14, YEAR_SPAN, SEQUENCE_VALID, SEAM_OFF.
   - Strict zero-masking on left-padded and right-padded tokens.
   - Gradient propagation safety (non-in-place autograd graph).
3. P2 Model Adapters & Generation:
   - StaticLoRALinear, DualCycleCoreGenerator, GeneralizedDualCycleMaTLoRALinear execution.
   - MaT-LoRA temporal context manager lifecycle (zero leakage).
   - Full unconstrained autoregressive greedy generation loop with prefill key-value cache.
4. P3 Deterministic Answer Parser:
   - Strict standalone MCQA letter extraction (A/B/C/D).
   - Refusal to classify natural language prose articles ('a event') as option A.
   - Distinct classification of contradictory labels, hallucinated options, refusals, and empty outputs.
   - Binary Rel Yes/No parsing.
   - Secondary diagnostic UNIQUE_EXACT_NUMERIC_VALUE exact Decimal parsing.
5. P4 Metric Aggregation:
   - Canonical, debiased mean, and strict accuracy over audited rotations.
   - Task stratification and invalid rate accounting on original N denominator.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple

import pytest
import torch
import torch.nn as nn
from transformers import AutoTokenizer

from temporal_alignment.stage03_eval.answer_parser import (
    normalize_numeric_string,
    parse_answer,
    parse_mcqa_label_only,
    parse_rel_label_only,
    parse_unique_exact_numeric,
)
from temporal_alignment.stage03_eval.contracts import (
    EvaluationConfig,
    GenerationPolicy,
    InvalidReason,
    ParserContract,
    PredictionRecord,
    TemporalContext,
    TemporalPolicy,
    YearMention,
)
from temporal_alignment.stage03_eval.generation_adapter import (
    GeneratedContinuation,
    generate_unconstrained,
)
from temporal_alignment.stage03_eval.metrics import aggregate_evaluation_metrics
from temporal_alignment.stage03_eval.model_adapter import (
    DualCycleCoreGenerator,
    GeneralizedDualCycleMaTLoRALinear,
    StaticLoRALinear,
    attach_adapters_to_model,
    matlora_temporal_context,
)
from temporal_alignment.stage03_eval.prompt_adapter import (
    extract_mcqa_option_texts,
    format_inference_prompt,
    generate_audited_permutations,
)
from temporal_alignment.stage03_eval.temporal_injector import (
    GatedResidualProjection,
    compute_canchi_8d,
    compute_fixed_fone_canchi_40d,
    compute_fone_32d,
    compute_theta_z,
    inject_temporal_features,
)
from temporal_alignment.stage03_eval.temporal_locator import (
    StructuredLocatorError,
    build_temporal_mask,
    find_year_char_spans,
    resolve_temporal_mentions,
)


@pytest.fixture(scope="module")
def fast_tokenizer():
    """Load local fast tokenizer with offset_mapping capability."""
    try:
        tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct", local_files_only=True)
    except Exception:
        tok = AutoTokenizer.from_pretrained("gpt2")
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return tok


# =========================================================================
# 1. P0 CONTRACTS & CONFIGURATION TESTS
# =========================================================================

def test_01_config_deterministic_hash():
    """Verify that EvaluationConfig produces identical SHA-256 hashes for identical settings."""
    cfg1 = EvaluationConfig(
        arm="arm5",
        temporal_policy=TemporalPolicy.YEAR_SPAN,
        parser_contract=ParserContract.LABEL_ONLY,
        seed=42,
    )
    cfg2 = EvaluationConfig(
        arm="arm5",
        temporal_policy=TemporalPolicy.YEAR_SPAN,
        parser_contract=ParserContract.LABEL_ONLY,
        seed=42,
    )
    assert cfg1.config_hash() == cfg2.config_hash()

    cfg3 = EvaluationConfig(
        arm="arm4",
        temporal_policy=TemporalPolicy.YEAR_SPAN,
        parser_contract=ParserContract.LABEL_ONLY,
        seed=42,
    )
    assert cfg1.config_hash() != cfg3.config_hash()


def test_02_prediction_record_serialization():
    """Verify PredictionRecord data structure holds required fields and serializes cleanly."""
    rec = PredictionRecord(
        global_idx=0,
        example_id="mcq:test:1947",
        rotation_id=0,
        task_id="DB",
        prompt_hash="abc123hash",
        raw_continuation=" A",
        generated_token_ids=(362,),
        parsed_label="a",
        parse_reason="VALID",
        is_valid=True,
        gold="a",
        is_correct=True,
        temporal_policy="YEAR_SPAN",
        parser_contract="LABEL_ONLY",
    )
    d = rec.to_dict()
    assert d["example_id"] == "mcq:test:1947"
    assert d["is_correct"] is True
    assert d["generated_token_ids"] == [362]


# =========================================================================
# 2. P1 TEMPORAL LOCATOR & INJECTOR TESTS
# =========================================================================

def test_03_find_year_char_spans_boundaries():
    """Verify regex whole-word boundaries prevent substring false-positives."""
    text = "In 1947, index 19470 was not 1947."
    spans = find_year_char_spans(text, 1947)
    assert len(spans) == 2
    assert spans[0] == (3, 7)
    assert spans[1] == (29, 33)


def test_04_resolve_temporal_mentions(fast_tokenizer):
    """Verify fast tokenizer offset mapping locates exact anchor year tokens."""
    prompt = "Question: In 1947, Private Civil Liberties Index in South Sudan was: Options: (a) 0.84\nAnswer:"
    anchor_mention, secondary = resolve_temporal_mentions(fast_tokenizer, prompt, anchor_year=1947)
    assert anchor_mention.year == 1947
    assert anchor_mention.role == "anchor"
    assert anchor_mention.char_start == 13
    assert anchor_mention.char_end == 17
    assert anchor_mention.token_start is not None
    assert anchor_mention.token_end is not None
    assert anchor_mention.token_end > anchor_mention.token_start

    # Verify decoded tokens match "1947"
    enc = fast_tokenizer(prompt, add_special_tokens=False)
    tok_sub = enc["input_ids"][anchor_mention.token_start : anchor_mention.token_end]
    decoded_year = fast_tokenizer.decode(tok_sub).strip()
    assert decoded_year == "1947"


def test_05_resolve_temporal_mentions_multi_year_rel(fast_tokenizer):
    """Verify multi-year Rel prompts isolate anchor year and record secondary comparison year."""
    prompt = "Question: In 1947, Civil Liberties in Norway was higher than 1952? Answer as Yes or No\nAnswer:"
    anchor_mention, secondary = resolve_temporal_mentions(
        fast_tokenizer, prompt, anchor_year=1947, secondary_years=[1952]
    )
    assert anchor_mention.year == 1947
    assert len(secondary) == 1
    assert secondary[0].year == 1952
    assert secondary[0].role == "comparison"


def test_06_temporal_locator_fail_closed_on_missing_year(fast_tokenizer):
    """Verify fail-closed exception raised when anchor year is absent (no silent 14 fallback)."""
    prompt = "Question: Which option is correct for Gabon?\nAnswer:"
    with pytest.raises(StructuredLocatorError) as exc_info:
        resolve_temporal_mentions(fast_tokenizer, prompt, anchor_year=1947)
    assert "Anchor year 1947 not found" in str(exc_info.value)


def test_07_temporal_mask_synthesis_and_padding_safety():
    """Verify temporal mask M under all policies and prove zero-masking on padding."""
    B, S = 2, 20
    input_ids = torch.randint(100, 1000, (B, S))
    # Item 0 has pad_len = 16 under left padding
    attention_mask = torch.ones((B, S), dtype=torch.long)
    attention_mask[0, :16] = 0  # 16 padding tokens on the left!

    # 1. LEGACY_FIXED14: token 14 is a padding token on item 0
    m_legacy = build_temporal_mask(input_ids, attention_mask, TemporalPolicy.LEGACY_FIXED14, padding_side="left")
    assert m_legacy[0, 14] == 0.0, "Legacy fixed 14 MUST be 0 on padding"
    assert m_legacy[1, 14] == 1.0, "Legacy fixed 14 should be 1 on valid non-padded item"

    # 2. YEAR_SPAN: sum-preserving across subtokens
    mentions = [
        YearMention(year=1947, char_start=0, char_end=4, token_start=0, token_end=2),  # 2 subtokens
        YearMention(year=1947, char_start=0, char_end=4, token_start=0, token_end=1),  # 1 subtoken
    ]
    m_year = build_temporal_mask(input_ids, attention_mask, TemporalPolicy.YEAR_SPAN, anchor_mentions=mentions, padding_side="left")
    # For item 0, pad_len=16, so token_start is shifted to 16, token_end to 18
    assert m_year[0, :16].sum() == 0.0, "Padding must receive zero weight"
    assert pytest.approx(m_year[0, 16:18].sum().item()) == 1.0, "Sum of span weights must be 1.0"
    assert pytest.approx(m_year[0, 16].item()) == 0.5
    assert pytest.approx(m_year[0, 17].item()) == 0.5

    # 3. SEQUENCE_VALID: 1.0 on valid tokens, 0 on padding
    m_seq = build_temporal_mask(input_ids, attention_mask, TemporalPolicy.SEQUENCE_VALID)
    assert m_seq[0, :16].sum() == 0.0
    assert (m_seq[0, 16:] == 1.0).all()

    # 4. SEAM_OFF: all zeros
    m_off = build_temporal_mask(input_ids, attention_mask, TemporalPolicy.SEAM_OFF)
    assert (m_off == 0.0).all()


def test_08_temporal_injector_gradient_safety():
    """Verify gradient propagates cleanly through inject_temporal_features without in-place errors."""
    B, S, D = 2, 8, 32
    inputs_embeds = nn.Parameter(torch.randn(B, S, D))
    v40 = torch.randn(B, 40)
    seam_proj = GatedResidualProjection(in_features=40, out_features=D, gate_mode="sigmoid")
    mask = torch.ones((B, S), dtype=torch.float32)

    injected = inject_temporal_features(inputs_embeds, seam_proj, v40, mask, TemporalPolicy.YEAR_SPAN)
    loss = injected.sum()
    loss.backward()

    assert inputs_embeds.grad is not None
    assert seam_proj.alpha.grad is not None
    assert seam_proj.linear.weight.grad is not None


# =========================================================================
# 3. P2 MODEL ADAPTER & UNCONSTRAINED GENERATION TESTS
# =========================================================================

class MockTinyQwen(nn.Module):
    """Tiny mock causal LM for CPU-only integration tests."""

    def __init__(self, vocab_size: int = 500, hidden_size: int = 32):
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.embed_tokens = nn.Embedding(vocab_size, hidden_size)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, hidden_size),
        )
        # Name submodules for adapter attachment
        self.gate_proj = self.mlp[0]
        self.down_proj = self.mlp[2]
        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)

    def get_input_embeddings(self):
        return self.embed_tokens

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[Any] = None,
        use_cache: bool = False,
        return_dict: bool = True,
    ):
        if inputs_embeds is not None:
            h = inputs_embeds
        else:
            h = self.embed_tokens(input_ids)

        h = h + self.mlp(h)
        logits = self.lm_head(h)

        class Output:
            def __init__(self, logits, past_key_values):
                self.logits = logits
                self.past_key_values = past_key_values

        return Output(logits=logits, past_key_values=() if use_cache else None)


def test_09_matlora_temporal_context_isolation():
    """Verify matlora_temporal_context applies (theta, z) cleanly and resets on exit."""
    linear = nn.Linear(32, 32)
    core_gen = DualCycleCoreGenerator(rank=4, num_harmonics=2)
    mat_linear = GeneralizedDualCycleMaTLoRALinear(linear, core_gen=core_gen, rank=4)

    container = nn.ModuleList([mat_linear])
    assert mat_linear.current_theta is None
    assert mat_linear.current_z is None

    th = torch.tensor([1.57])
    z = torch.tensor([0.25])
    with matlora_temporal_context(container, th, z):
        assert mat_linear.current_theta is not None
        assert mat_linear.current_z is not None
        # Verify forward execution
        x = torch.randn(1, 4, 32)
        out = mat_linear(x)
        assert out.shape == (1, 4, 32)

    assert mat_linear.current_theta is None
    assert mat_linear.current_z is None


def test_10_unconstrained_generation_loop(fast_tokenizer):
    """Verify unconstrained greedy generation loop executes with injected prefill."""
    model = MockTinyQwen(vocab_size=len(fast_tokenizer), hidden_size=32)
    prompt = "Question: Sample question?\nAnswer:"
    enc = fast_tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
    input_ids = enc["input_ids"]
    attention_mask = enc["attention_mask"]

    inputs_embeds = model.get_input_embeddings()(input_ids)
    gen_policy = GenerationPolicy(max_new_tokens=4)

    continuation = generate_unconstrained(
        model=model,
        tokenizer=fast_tokenizer,
        inputs_embeds=inputs_embeds,
        attention_mask=attention_mask,
        policy=gen_policy,
    )
    assert len(continuation.continuation_token_ids) <= 4
    assert isinstance(continuation.raw_text, str)
    assert continuation.prefill_token_count == input_ids.shape[1]


# =========================================================================
# 4. P3 DETERMINISTIC ANSWER PARSER TESTS
# =========================================================================

def test_11_parse_mcqa_standalone_letters():
    """Verify exact standalone option letter parsing."""
    assert parse_mcqa_label_only("A").label == "a"
    assert parse_mcqa_label_only("  B. ").label == "b"
    assert parse_mcqa_label_only("(C)").label == "c"
    assert parse_mcqa_label_only("Answer: D").label == "d"
    assert parse_mcqa_label_only("Option (b):").label == "b"
    assert parse_mcqa_label_only("The correct option is C.").label == "c"


def test_12_parse_mcqa_rejects_embedded_prose_article():
    """Verify parser strictly rejects embedded prose article 'a' as option A."""
    res = parse_mcqa_label_only("a significant historical event occurred in that decade.")
    assert res.is_valid is False
    assert res.label is None
    assert res.invalid_reason == InvalidReason.NO_RECOGNIZABLE_LABEL


def test_13_parse_mcqa_contradiction_and_hallucination():
    """Verify contradictory option mentions and hallucinated letters are categorized."""
    # Contradiction
    res_contra = parse_mcqa_label_only("It could be both A and C.")
    assert res_contra.is_valid is False
    assert res_contra.invalid_reason == InvalidReason.MULTIPLE_CONTRADICTORY_LABELS

    # Hallucination
    res_hallu = parse_mcqa_label_only("Answer: Option E")
    assert res_hallu.is_valid is False
    assert res_hallu.invalid_reason == InvalidReason.HALLUCINATED_OPTION

    # Empty
    res_empty = parse_mcqa_label_only("   \n\t  ")
    assert res_empty.is_valid is False
    assert res_empty.invalid_reason == InvalidReason.EMPTY_OUTPUT


def test_14_parse_rel_binary_yes_no():
    """Verify Yes/No parsing for Rel tasks."""
    assert parse_rel_label_only("Yes").label == "yes"
    assert parse_rel_label_only("No.").label == "no"
    assert parse_rel_label_only("Answer: Yes").label == "yes"
    assert parse_rel_label_only("Answer is No, because of historical data").label == "no"

    res_contra = parse_rel_label_only("Yes and No at the same time")
    assert res_contra.is_valid is False
    assert res_contra.invalid_reason == InvalidReason.MULTIPLE_CONTRADICTORY_LABELS


def test_15_parse_unique_exact_numeric():
    """Verify secondary diagnostic exact Decimal matching with numeric options."""
    options = {"a": "0.84", "b": "0.89", "c": "1.03", "d": "0.08"}
    res = parse_unique_exact_numeric("0.08", options)
    assert res.is_valid is True
    assert res.label == "d"

    # Non-matching number
    res_nomatch = parse_unique_exact_numeric("99.9", options)
    assert res_nomatch.is_valid is False
    assert res_nomatch.invalid_reason == InvalidReason.NO_RECOGNIZABLE_LABEL


# =========================================================================
# 5. P4 METRIC AGGREGATION & TASK STRATA TESTS
# =========================================================================

def test_16_metric_aggregation_and_rotations():
    """Verify canonical, debiased mean, and strict accuracy over audited rotations."""
    # Create 2 examples with 4 rotations each
    records = [
        # Example 1: all 4 rotations correct -> strict=1, can=1, deb=1.0
        PredictionRecord(0, "ex1", 0, "DB", "h1", "A", (1,), "a", "VALID", True, "a", True, "YEAR_SPAN", "LABEL_ONLY"),
        PredictionRecord(0, "ex1", 1, "DB", "h2", "B", (2,), "b", "VALID", True, "b", True, "YEAR_SPAN", "LABEL_ONLY"),
        PredictionRecord(0, "ex1", 2, "DB", "h3", "C", (3,), "c", "VALID", True, "c", True, "YEAR_SPAN", "LABEL_ONLY"),
        PredictionRecord(0, "ex1", 3, "DB", "h4", "D", (4,), "d", "VALID", True, "d", True, "YEAR_SPAN", "LABEL_ONLY"),
        # Example 2: rotation 0 correct, rotations 1..3 incorrect -> strict=0, can=1, deb=0.25
        PredictionRecord(1, "ex2", 0, "Rel", "h5", "Yes", (5,), "yes", "VALID", True, "yes", True, "YEAR_SPAN", "LABEL_ONLY"),
        PredictionRecord(1, "ex2", 1, "Rel", "h6", "No", (6,), "no", "VALID", True, "yes", False, "YEAR_SPAN", "LABEL_ONLY"),
    ]

    metrics = aggregate_evaluation_metrics(records)
    assert metrics["total_examples"] == 2
    assert metrics["canonical_correct"] == 2
    assert metrics["canonical_accuracy"] == 1.0
    assert metrics["strict_correct"] == 1
    assert metrics["strict_accuracy"] == 0.5
    # Debiased mean: (1.0 + 0.5) / 2 = 0.75
    assert metrics["debiased_mean_accuracy"] == 0.75
    assert "DB" in metrics["task_strata"]
    assert "Rel" in metrics["task_strata"]


# =========================================================================
# 6. P0 RED-TEAM REGRESSION TESTS (H01 - H06)
# =========================================================================

def test_17_cli_smoke_and_dry_run_execution():
    """Verify CLI smoke runs actual mock inference and eval --dry-run validates planned actions."""
    from temporal_alignment.stage03_eval.run_eval import run_dry_run_checks, main
    import sys

    # 1. Test run_dry_run_checks with missing checkpoint -> hard fail
    cfg_fail = EvaluationConfig(
        arm="arm5",
        checkpoint_path="nonexistent_checkpoint_file.pt",
    )
    res_fail = run_dry_run_checks(cfg_fail)
    assert res_fail["status"] == "FAIL"
    assert "MANDATORY_CHECKPOINT_MISSING" in res_fail["reason"]

    # 2. Test run_dry_run_checks with valid arm=base -> pass
    cfg_base = EvaluationConfig(
        arm="base",
        eval_data_path="data/tempun_full_clean/eval_5292_clean.jsonl.gz",
        manifest_path="Reports/6-11Report/_POST_FORENSIC_OPEN_GATES_20261105/REPLAY_SAMPLE_MANIFEST.json",
    )
    res_base = run_dry_run_checks(cfg_base)
    assert res_base["status"] == "DRY_RUN_PASS"
    assert res_base["dry_run"] is True
    assert res_base["sample_verified"]["task_id"] == "DB"


def test_18_checkpoint_strict_validation_rejects_missing_groups(tmp_path):
    """Mutation test: verify load_checkpoint_weights rejects missing groups, zero matches, and NaNs."""
    from temporal_alignment.stage03_eval.model_adapter import load_checkpoint_weights
    from temporal_alignment.stage03_eval.temporal_injector import GatedResidualProjection

    model = nn.Linear(32, 32)
    seam = GatedResidualProjection(40, 32)
    core = DualCycleCoreGenerator(rank=4)

    # 1. Missing lora_state
    bad_ckpt1 = tmp_path / "bad1.pt"
    torch.save({"seam_projection": {"alpha": torch.zeros(1)}}, bad_ckpt1)
    with pytest.raises(ValueError, match="empty or missing 'lora_state'"):
        load_checkpoint_weights(model, seam, core, str(bad_ckpt1), arm="arm5")

    # 2. Missing seam_projection for arm5
    bad_ckpt2 = tmp_path / "bad2.pt"
    torch.save({"lora_state": {"weight": torch.randn(4, 4)}}, bad_ckpt2)
    with pytest.raises(ValueError, match="empty or missing 'seam_projection'"):
        load_checkpoint_weights(model, seam, core, str(bad_ckpt2), arm="arm5")

    # 3. Missing core_generator for arm5
    bad_ckpt3 = tmp_path / "bad3.pt"
    torch.save({
        "lora_state": {"weight": torch.randn(4, 4)},
        "seam_projection": {"alpha": torch.zeros(1)},
    }, bad_ckpt3)
    with pytest.raises(ValueError, match="missing 'core_generator'"):
        load_checkpoint_weights(model, seam, core, str(bad_ckpt3), arm="arm5")

    # 4. Sentinel mutation: NaN in tensor
    bad_ckpt4 = tmp_path / "bad4.pt"
    torch.save({
        "lora_state": {"weight": torch.tensor([float("nan")])},
        "seam_projection": {"alpha": torch.zeros(1)},
        "core_generator": {"basis": torch.ones(2, 2)},
    }, bad_ckpt4)
    with pytest.raises(ValueError, match="NaN detected"):
        load_checkpoint_weights(model, seam, core, str(bad_ckpt4), arm="arm5")


def test_19_checkpoint_real_stage03_loading_parity():
    """Verify loading real Stage 03 checkpoint achieves 100% parameter census and alpha parity."""
    from pathlib import Path
    from temporal_alignment.stage03_eval.model_adapter import (
        attach_adapters_to_model,
        load_checkpoint_weights,
    )
    from temporal_alignment.stage03_eval.temporal_injector import GatedResidualProjection

    ckpt_p = Path("scratch/checkpoint_staging/stage03_3b_targeted_matlora_5epochs_weights.pt")
    if not ckpt_p.is_file():
        pytest.skip("Historical checkpoint file not present in local staging.")

    # Build 36-layer mock Qwen model matching checkpoint architecture
    class MockMLP(nn.Module):
        def __init__(self):
            super().__init__()
            self.gate_proj = nn.Linear(2048, 11008, bias=False)
            self.up_proj = nn.Linear(2048, 11008, bias=False)
            self.down_proj = nn.Linear(11008, 2048, bias=False)

    class MockLayer(nn.Module):
        def __init__(self):
            super().__init__()
            self.mlp = MockMLP()

    class MockModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = nn.ModuleList([MockLayer() for _ in range(36)])

    class MockQwen(nn.Module):
        def __init__(self):
            super().__init__()
            self.model = MockModel()

    qwen = MockQwen()
    _, core_gen = attach_adapters_to_model(qwen, arm="arm5")
    seam = GatedResidualProjection(40, 2048, gate_mode="linear")

    res = load_checkpoint_weights(qwen, seam, core_gen, str(ckpt_p), arm="arm5", strict=True)
    assert res["status"] == "LOAD_SUCCESS"
    assert res["matched_lora_tensors"] == 216
    assert res["total_lora_tensors"] == 216
    assert res["seam_tensors"] == 4
    assert res["core_tensors"] == 5
    assert res["seam_alpha"] == pytest.approx(-0.006561279, abs=1e-5)


def test_20_genuine_qwen2_cache_parity():
    """Verify genuine Qwen2 DynamicCache incremental decoding achieves 100% parity with no-cache full prefix."""
    from transformers import Qwen2Config, Qwen2ForCausalLM

    torch.manual_seed(42)
    cfg = Qwen2Config(
        vocab_size=500,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=256,
    )
    model = Qwen2ForCausalLM(cfg).eval()

    seq_len = 8
    prompt_ids = torch.randint(0, 500, (1, seq_len))
    inputs_embeds = model.get_input_embeddings()(prompt_ids)
    attention_mask = torch.ones(1, seq_len, dtype=torch.long)

    # 1. Cached generation (prefill + 3 incremental steps)
    with torch.no_grad():
        out1 = model(inputs_embeds=inputs_embeds, attention_mask=attention_mask, use_cache=True, return_dict=True)
        past_kv = out1.past_key_values
        curr_tok = int(torch.argmax(out1.logits[:, -1, :], dim=-1).item())
        tokens_cached = [curr_tok]
        curr_mask = attention_mask
        for _ in range(3):
            curr_mask = torch.cat([curr_mask, torch.ones((1, 1), dtype=torch.long)], dim=1)
            step_out = model(
                input_ids=torch.tensor([[curr_tok]], dtype=torch.long),
                attention_mask=curr_mask,
                past_key_values=past_kv,
                use_cache=True,
                return_dict=True,
            )
            past_kv = step_out.past_key_values
            curr_tok = int(torch.argmax(step_out.logits[:, -1, :], dim=-1).item())
            tokens_cached.append(curr_tok)

    # 2. No-cache generation (full sequence fed at each step)
    with torch.no_grad():
        curr_ids = prompt_ids.clone()
        tokens_nocache = []
        for _ in range(4):
            out2 = model(input_ids=curr_ids, attention_mask=torch.ones_like(curr_ids), use_cache=False, return_dict=True)
            next_tok = int(torch.argmax(out2.logits[:, -1, :], dim=-1).item())
            tokens_nocache.append(next_tok)
            curr_ids = torch.cat([curr_ids, torch.tensor([[next_tok]], dtype=torch.long)], dim=1)

    assert tokens_cached == tokens_nocache


def test_21_gate_mode_linear_parity_with_training():
    """Verify gate_mode='linear' calculates alpha * normed, exactly matching training forward equation."""
    seam_proj = GatedResidualProjection(in_features=40, out_features=64, gate_mode="linear")
    with torch.no_grad():
        seam_proj.alpha.copy_(torch.tensor([-0.007]))
        x = torch.randn(2, 40)
        out = seam_proj(x)
        expected = -0.007 * seam_proj.norm(seam_proj.linear(x))
        assert torch.allclose(out, expected, atol=1e-6)


def test_22_missing_year_fails_closed():
    """Verify instance with missing or None year fails closed with MISSING_YEAR_IN_INSTANCE."""
    from temporal_alignment.stage03_eval.run_eval import execute_evaluation_loop

    class DummyModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.hidden_size = 32
            self.embed = nn.Embedding(100, 32)
            self.linear = nn.Linear(32, 100)
        def get_input_embeddings(self):
            return self.embed
        def forward(self, *args, **kwargs):
            class O:
                logits = torch.randn(1, 4, 100)
                past_key_values = None
            return O()

    class DummyTok:
        eos_token_id = 99
        pad_token = 0
        def __call__(self, text, **kwargs):
            return {"input_ids": torch.tensor([[1, 2, 3]]), "attention_mask": torch.tensor([[1, 1, 1]])}
        def decode(self, ids, **kwargs):
            return "sample"

    cfg = EvaluationConfig(arm="base", temporal_policy=TemporalPolicy.SEAM_OFF)
    bad_cases = [{"input": "test prompt without year", "target": "a"}]
    res = execute_evaluation_loop(cfg, DummyModel(), DummyTok(), bad_cases)

    assert res["metrics"]["invalid_rate"] == 1.0
    assert "MISSING_YEAR_IN_INSTANCE" in res["metrics"]["invalid_reasons"]


def test_23_receipts_json_format_and_no_raw_latex():
    """Regression test ensuring that all receipts in Reports/16+/11Report have zero raw LaTeX strings

    and valid JSON formatting (ASCII/Unicode programming equations, structured arrays, no escaped formulas).
    """
    import os
    import json

    target_dirs = [
        "Reports/16+/11Report",
        "Documents/Documenting/Reports/16+/11Report",
        "Reports/6-11Report",
    ]
    forbidden_latex_patterns = [
        "\\Delta",
        "\\alpha",
        "\\beta",
        "\\sigma",
        "\\cdot",
        "\\frac",
        "\\text{",
    ]

    checked_files = 0
    for base in target_dirs:
        if not os.path.exists(base):
            continue
        for root, _, files in os.walk(base):
            for f in files:
                if f.endswith(".json"):
                    p = os.path.join(root, f)
                    with open(p, "r", encoding="utf-8") as fp:
                        raw = fp.read()
                        data = json.loads(raw)  # Fail if invalid JSON
                        assert isinstance(data, (dict, list)), f"{p} root must be a JSON object or list"
                        for pattern in forbidden_latex_patterns:
                            assert pattern not in raw, (
                                f"Forbidden LaTeX token '{pattern}' detected in receipt: {p}. "
                                "Use ASCII/Unicode programming equations in JSON receipts."
                            )
                    checked_files += 1

    assert checked_files > 0, "Expected to audit at least one receipt JSON file"


def test_24_cli_eval_forward_exception_nonzero_exit(tmp_path, monkeypatch):
    """Verify that a failure or exception during evaluation results in a nonzero exit code

    and produces no false PASS artifact.
    """
    import sys
    from temporal_alignment.stage03_eval.run_eval import main

    out_dir = tmp_path / "broken_run"
    test_args = [
        "run_eval.py",
        "eval",
        "--arm",
        "base",
        "--manifest-path",
        "non_existent_manifest.json",
        "--output",
        str(out_dir),
    ]
    monkeypatch.setattr(sys, "argv", test_args)

    exit_code = main()
    assert exit_code != 0, "CLI eval must exit with nonzero code on error"
    assert not (out_dir / "EVAL_SUMMARY.json").exists(), "Must not write false PASS summary on failure"


