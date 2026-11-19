"""Integration, illumination, and mismatch checks for Stage 03 V3 locked upgrades.

Locked Problem & Architectural Scope:
- Goal: G1 YEAR-only closed QA (Zero-Shot NWP -> MCQA/Rel transfer)
- Branch: v2-targeted-continual-alignment
- Adapter Scope: MLP Linear ONLY (gate_proj, up_proj, down_proj; NO Attention, NO DoRA)
- Improvements Set:
  1. Non-linear asinh compression for z + multi-period FoNE 40D
  2. Token-level multi-year injection (MULTI_YEAR_SPAN) across train & eval
  3. AdamW + LR Scheduling (Warmup + Cosine / WSD) for MLP LoRA (+ optional General Replay)
"""

from __future__ import annotations

import hashlib
import math

import pytest
import torch
import torch.nn as nn
from transformers import AutoTokenizer, Qwen2Config, Qwen2ForCausalLM

from temporal_alignment.stage03_eval.contracts import (
    EvaluationConfig,
    ParserContract,
    TemporalPolicy,
)
from temporal_alignment.stage03_eval.model_adapter import (
    GeneralizedDualCycleMaTLoRALinear,
    StaticLoRALinear,
    attach_adapters_to_model,
    matlora_temporal_context,
)
from temporal_alignment.stage03_eval.run_eval import execute_evaluation_loop
from temporal_alignment.stage03_eval.temporal_injector import (
    GatedResidualProjection,
    compute_fone_powers10_32d,
    compute_temporal_40d,
    compute_theta_z,
    inject_multi_year_temporal_features,
)
from temporal_alignment.stage03_eval.temporal_locator import (
    StructuredLocatorError,
    resolve_all_year_mentions,
)
from temporal_alignment.training import (
    build_cpt_optimizer_param_groups,
    create_cpt_lr_scheduler,
    interleave_general_replay,
)


@pytest.fixture(scope="module")
def fast_tokenizer():
    try:
        tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct", local_files_only=True)
    except Exception:
        tok = AutoTokenizer.from_pretrained("gpt2")
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return tok


def test_01_asinh_z_compression_and_powers10_fone():
    years = [-500, 100, 1800, 1947, 2026, 2500]
    zs_asinh = [compute_theta_z(y, z_mode="asinh")[1] for y in years]
    zs_linear = [compute_theta_z(y, z_mode="linear")[1] for y in years]

    assert all(zs_asinh[i] < zs_asinh[i + 1] for i in range(len(years) - 1))
    assert abs(zs_asinh[0]) < 2.5
    assert abs(zs_linear[0]) > 30.0

    v40 = compute_temporal_40d(1947, fone_mode="powers10", radius_mode="asinh")
    assert len(v40) == 40
    assert all(math.isfinite(x) for x in v40)
    assert len(compute_fone_powers10_32d(1947.0)) == 32

    with pytest.raises(ValueError):
        compute_theta_z(1947, z_mode="asinh", scale=0.0)
    with pytest.raises(ValueError):
        compute_temporal_40d(1947, fone_mode="unknown")


def test_02_multi_year_token_level_injection_and_pad_safety(fast_tokenizer):
    prompt = "Question: Did the treaty of 1914 occur before the summit of 1945?\nAnswer:"
    mentions = resolve_all_year_mentions(fast_tokenizer, prompt, anchor_year=1914)
    assert [m.year for m in mentions] == [1914, 1945]
    assert mentions[0].role == "anchor"
    assert mentions[1].role == "comparison"

    with pytest.raises(StructuredLocatorError):
        resolve_all_year_mentions(fast_tokenizer, prompt, anchor_year=1999)

    enc = fast_tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
    ids = enc["input_ids"]
    S = ids.shape[1]
    pad_len = 4
    attn_mask = torch.cat([torch.zeros((1, pad_len), dtype=torch.long), torch.ones((1, S), dtype=torch.long)], dim=1)

    embeds = torch.zeros((1, S + pad_len, 64), dtype=torch.float32)
    seam = GatedResidualProjection(in_features=40, out_features=64, gate_mode="linear")
    with torch.no_grad():
        seam.alpha.fill_(0.5)

    out = inject_multi_year_temporal_features(
        embeds,
        seam,
        mentions_per_item=[mentions],
        padding_side="left",
        attention_mask=attn_mask,
        fone_mode="powers10",
        radius_mode="asinh",
    )

    assert torch.allclose(out[:, :pad_len, :], torch.zeros_like(out[:, :pad_len, :]))

    m1_s, m1_e = mentions[0].token_start + pad_len, mentions[0].token_end + pad_len
    m2_s, m2_e = mentions[1].token_start + pad_len, mentions[1].token_end + pad_len
    v1 = out[0, m1_s:m1_e, :].sum(dim=0)
    v2 = out[0, m2_s:m2_e, :].sum(dim=0)
    assert v1.abs().sum().item() > 0.0
    assert v2.abs().sum().item() > 0.0
    assert not torch.allclose(v1, v2)

    non_year_indices = [
        i for i in range(pad_len, S + pad_len)
        if not (m1_s <= i < m1_e) and not (m2_s <= i < m2_e)
    ]
    assert torch.allclose(out[0, non_year_indices, :], torch.zeros((len(non_year_indices), 64)))


def test_03_mlp_only_invariant_attention_forbidden_and_adamw_groups(fast_tokenizer):
    qcfg = Qwen2Config(
        vocab_size=len(fast_tokenizer),
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=128,
    )

    # Attention intervention MUST fail-closed
    m_bad = Qwen2ForCausalLM(qcfg)
    for attn_target in ("q_proj", "k_proj", "v_proj", "o_proj"):
        with pytest.raises(ValueError, match="Attention intervention is strictly forbidden"):
            attach_adapters_to_model(m_bad, arm="arm5", target_submodules=[attn_target])

    # Arm 4: MLP-only Static LoRA (2 layers * 3 MLP = 6 modules)
    m_arm4 = Qwen2ForCausalLM(qcfg)
    arm4_mods, core_none = attach_adapters_to_model(m_arm4, arm="arm4", rank=4, alpha=8.0)
    assert core_none is None
    assert len(arm4_mods) == 6
    assert all(isinstance(mod, StaticLoRALinear) for mod in arm4_mods)

    # Arm 5: MLP-only MaT-LoRA (2 layers * 3 MLP = 6 modules)
    m_arm5 = Qwen2ForCausalLM(qcfg)
    arm5_mods, core_gen = attach_adapters_to_model(m_arm5, arm="arm5", rank=4, alpha=8.0)
    assert core_gen is not None
    assert len(arm5_mods) == 6
    assert all(isinstance(mod, GeneralizedDualCycleMaTLoRALinear) for mod in arm5_mods)

    # Verify Attention projections remain untouched nn.Linear
    for layer in m_arm5.model.layers:
        assert isinstance(layer.self_attn.q_proj, nn.Linear)
        assert isinstance(layer.self_attn.v_proj, nn.Linear)

    seam = GatedResidualProjection(in_features=40, out_features=64, gate_mode="linear")
    groups = build_cpt_optimizer_param_groups(
        m_arm5,
        seam_proj=seam,
        core_gen=core_gen,
        base_lr=1.0e-4,
        loraplus_ratio=1.0,
        adapter_head_lr=2.0e-4,
        weight_decay=0.01,
    )
    lr_by_name = {g["name"]: g["lr"] for g in groups}
    wd_by_name = {g["name"]: g["weight_decay"] for g in groups}
    assert math.isclose(lr_by_name["lora_A"], 1.0e-4)
    assert math.isclose(lr_by_name["lora_B"], 1.0e-4)
    assert math.isclose(lr_by_name["temporal_head_nodecay"], 2.0e-4)
    assert wd_by_name["temporal_head_nodecay"] == 0.0

    # Context manager guarantees temporal state cleanup AND supports gradient checkpointing backward
    m_arm5.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    m_arm5.train()
    th = torch.tensor([0.5])
    zv = torch.tensor([0.1])
    inp_ids = torch.randint(0, len(fast_tokenizer), (1, 8))
    emb = m_arm5.get_input_embeddings()(inp_ids) + 0.01 * seam(torch.zeros((1, 40))).unsqueeze(1)
    with matlora_temporal_context(m_arm5, th, zv):
        ckpt_out = m_arm5(inputs_embeds=emb, labels=inp_ids)
        ckpt_out.loss.backward()
    assert core_gen.cos_basis.grad is not None
    assert arm5_mods[0].current_theta is None


def test_04_adamw_cpt_lr_scheduler_and_replay_leakage_guard():
    p = nn.Parameter(torch.ones(4, 4))
    opt = torch.optim.AdamW([{"params": [p], "lr": 1e-4}], weight_decay=0.01)
    sched = create_cpt_lr_scheduler(opt, total_steps=100, warmup_ratio=0.05, schedule_type="cosine", min_lr_ratio=0.1)
    lrs = []
    for _ in range(100):
        opt.step()
        lrs.append(opt.param_groups[0]["lr"])
        sched.step()
    assert lrs[0] < lrs[4]
    assert math.isclose(lrs[4], 1e-4, rel_tol=1e-4)
    assert lrs[-1] < lrs[50]
    assert lrs[-1] >= 1e-5 - 1e-8

    opt_wsd = torch.optim.AdamW([{"params": [p], "lr": 1e-4}], weight_decay=0.01)
    sched_wsd = create_cpt_lr_scheduler(opt_wsd, total_steps=100, warmup_ratio=0.05, schedule_type="wsd", stable_ratio=0.7)
    lrs_wsd = []
    for _ in range(100):
        opt_wsd.step()
        lrs_wsd.append(opt_wsd.param_groups[0]["lr"])
        sched_wsd.step()
    assert math.isclose(lrs_wsd[20], 1e-4, rel_tol=1e-4)
    assert math.isclose(lrs_wsd[60], 1e-4, rel_tol=1e-4)
    assert lrs_wsd[-1] < lrs_wsd[60]

    nwp = [{"id": f"nwp_{i}", "input": f"In {1900 + i}, fact {i}."} for i in range(85)]
    replay = [{"id": f"rep_{i}", "input": f"General text {i}."} for i in range(20)]
    mixed = interleave_general_replay(nwp, replay, replay_ratio=0.15, seed=42)
    assert len(mixed) == 100
    assert sum(1 for r in mixed if r["_source_stream"] == "general_replay") == 15

    with pytest.raises(ValueError, match="Leakage detected"):
        interleave_general_replay(nwp, replay, replay_ratio=0.15, forbidden_eval_ids={"rep_3"})

    rep0_hash = hashlib.sha256("General text 0.".encode("utf-8")).hexdigest()
    with pytest.raises(ValueError, match="Leakage detected"):
        interleave_general_replay(nwp, replay, replay_ratio=0.15, forbidden_prompt_hashes={rep0_hash})


def test_05_tiny_qwen_end_to_end_v3_illumination(fast_tokenizer):
    qcfg = Qwen2Config(
        vocab_size=len(fast_tokenizer),
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=256,
    )
    model = Qwen2ForCausalLM(qcfg).eval()
    _, core_gen = attach_adapters_to_model(model, arm="arm5", rank=4, alpha=8.0)
    cfg_v3 = EvaluationConfig(
        arm="arm5",
        temporal_policy=TemporalPolicy.MULTI_YEAR_SPAN,
        parser_contract=ParserContract.LABEL_ONLY,
        device="cpu",
        z_mode="asinh",
        fone_mode="powers10",
    )
    cfg_v2 = EvaluationConfig(arm="arm5", temporal_policy=TemporalPolicy.YEAR_SPAN, device="cpu")
    assert cfg_v3.config_hash() != cfg_v2.config_hash()
    assert cfg_v3.adapter_scope == "MLP_ONLY"

    cases = [
        {
            "example_id": "v3_rel_multi_year",
            "task_id": "Rel",
            "year": 1914,
            "input": "Did the treaty of 1914 occur before the summit of 1945? Answer as Yes or No",
            "target": "Yes",
        },
        {
            "example_id": "v3_db_single_year",
            "task_id": "DB",
            "year": 1947,
            "input": "In 1947, event happened: Options: (a) 10 (b) 20 (c) 30 (d) 40",
            "target": "b",
        },
    ]
    res = execute_evaluation_loop(
        config=cfg_v3,
        model=model,
        tokenizer=fast_tokenizer,
        eval_cases=cases,
        core_gen=core_gen,
    )
    assert res["metrics"]["total_records"] == 6
    assert res["config_hash"] == cfg_v3.config_hash()
    assert res["config"]["z_mode"] == "asinh"
    assert res["config"]["fone_mode"] == "powers10"
    assert res["config"]["adapter_scope"] == "MLP_ONLY"


def test_06_v3_modal_preflight_and_launcher_wiring():
    import ast
    from pathlib import Path
    from temporal_alignment.stage03_eval.preflight_v3 import (
        V3ModalLaunchSpec,
        verify_v3_modal_inputs,
    )

    spec = V3ModalLaunchSpec(arm="arm5", epochs=5, profile="loggingfuture", max_budget_usd=5.0)
    preflight = verify_v3_modal_inputs(spec)
    assert preflight["status"] == "CPU_PREFLIGHT_PASS"
    assert preflight["scope"]["adapter_scope"] == "MLP_ONLY"
    assert preflight["scope"]["dora_enabled"] is False
    assert preflight["modal_profile_target"] == "loggingfuture"
    assert preflight["estimated_cost_usd"] <= 5.0

    # Negative controls: DoRA/invalid arm and insufficient budget cap fail closed
    with pytest.raises(ValueError, match="Unsupported V3 arm"):
        V3ModalLaunchSpec(arm="arm5_dora").validate()
    with pytest.raises(ValueError, match="exceeds max_budget_usd"):
        verify_v3_modal_inputs(V3ModalLaunchSpec(arm="arm5", epochs=5, max_budget_usd=0.5))

    # AST & wiring audit of the Modal launcher script
    launcher_path = Path("scripts/run_stage03_v3_mlp_cpt_modal.py")
    assert launcher_path.is_file()
    source = launcher_path.read_text(encoding="utf-8")
    ast.parse(source)

    for required_symbol in (
        "add_local_dir",
        "resolve_all_year_mentions",
        "inject_multi_year_temporal_features",
        "compute_theta_z",
        "attach_adapters_to_model",
        "build_cpt_optimizer_param_groups",
        "create_cpt_lr_scheduler",
        "execute_evaluation_loop",
        "verify_v3_modal_inputs",
        "confirm_gpu",
    ):
        assert required_symbol in source, f"Missing required symbol in Modal launcher: {required_symbol}"

    for forbidden_symbol in ("token_idx=14", "DoRALinear", "MaTDoRA", "q_proj", "v_proj", "k_proj", "o_proj"):
        assert forbidden_symbol not in source, f"Forbidden legacy/out-of-scope symbol in Modal launcher: {forbidden_symbol}"


def test_07_mcqa_answer_parser_parenthesized_option_delimiter_regression():
    from temporal_alignment.stage03_eval.answer_parser import parse_mcqa_label_only
    from temporal_alignment.stage03_eval.contracts import InvalidReason

    for text, expected_label in (
        ("(a) 0.84", "a"),
        ("(B) Some text", "b"),
        ("a) 0.84", "a"),
        ("[c] 1995", "c"),
    ):
        res = parse_mcqa_label_only(text)
        assert res.is_valid is True, f"Expected valid parse for {text!r}, got {res}"
        assert res.label == expected_label
        assert res.invalid_reason == InvalidReason.VALID


def test_08_question_body_semantic_anchor_and_full_corpus_roles(fast_tokenizer):
    import gzip
    import json
    from pathlib import Path
    from temporal_alignment.stage03_eval.temporal_locator import (
        build_temporal_mask,
        resolve_question_body_anchor,
    )

    # 1. WB case where metadata_year=1948 is NOT present before Options:, plus citation year 2023 and year in option (a)
    wb_prompt = (
        "Question: What is the correct option for the following query: From 1947 to 1955, "
        "share of contribution to global warming (Jones et al. 2023) was: "
        "Options: (a) In 1948, 0.5 (b) In 1952, 0.9 (c) In 1953, 1.1 (d) In 1955, 1.4\nAnswer:"
    )
    m_wb = resolve_question_body_anchor(fast_tokenizer, wb_prompt, task_id="WB", metadata_year=1948)
    assert m_wb.year == 1947
    assert m_wb.role == "range_start"
    assert m_wb.char_end <= wb_prompt.index("Options:")

    # 2. MM / RB choose range_end, TB chooses range_start, DB/Rel choose subject_year
    tb_prompt = "Question: In 1966-1971, highest rate (Jones et al. 2023)? Options: (a) 1971\nAnswer:"
    assert resolve_question_body_anchor(fast_tokenizer, tb_prompt, task_id="TB", metadata_year=1971).year == 1966

    mm_prompt = "Question: In range 1982-1987, value (Jones et al. 2023)? Options: (a) 1982\nAnswer:"
    m_mm = resolve_question_body_anchor(fast_tokenizer, mm_prompt, task_id="MM", metadata_year=1982)
    assert m_mm.year == 1987 and m_mm.role == "range_end"

    rb_prompt = "Question: In range 1955-1960, mean value? Options: (a) 0.1\nAnswer:"
    assert resolve_question_body_anchor(fast_tokenizer, rb_prompt, task_id="RB", metadata_year=1960).year == 1960

    rel_prompt = "Question: In 2012, share (Jones et al. 2023) was higher than 2017? Answer as Yes or No\nAnswer:"
    m_rel = resolve_question_body_anchor(fast_tokenizer, rel_prompt, task_id="Rel", metadata_year=2012)
    assert m_rel.year == 2012 and m_rel.role == "subject_year"

    # Missing visible body year even when year appears inside Options: MUST fail closed
    bad_prompt = "Question: Which period had peak growth? Options: (a) 1947 (b) 1950 (c) 1960 (d) 1970\nAnswer:"
    with pytest.raises(StructuredLocatorError):
        resolve_question_body_anchor(fast_tokenizer, bad_prompt, task_id="DB", metadata_year=1947)

    # Full eval_5292 corpus anchor resolution check
    eval_path = Path("data/tempun_full_clean/eval_5292_clean.jsonl.gz")
    if eval_path.is_file():
        with gzip.open(eval_path, "rt", encoding="utf-8") as f:
            for idx, line in enumerate(f):
                row = json.loads(line)
                m = resolve_question_body_anchor(
                    fast_tokenizer,
                    f"Question: {row['input'].strip()}\nAnswer:",
                    task_id=row["task_id"],
                    metadata_year=int(row["year"]),
                )
                assert m.is_resolved()
                if idx >= 600:
                    # Verify full 5292 rows with light char-span check beyond sample tokenized 600
                    q_body = f"Question: {row['input'].strip()}\nAnswer:".split("Options:", 1)[0]
                    assert any(c.isdigit() for c in q_body)

    # General replay row gets exact M=0 mask without needing an anchor
    enc = fast_tokenizer(rel_prompt, return_tensors="pt", add_special_tokens=False)
    pm = torch.ones_like(enc["attention_mask"], dtype=torch.bool)
    mask_mix = build_temporal_mask(
        torch.cat([enc["input_ids"], enc["input_ids"]], dim=0),
        torch.cat([enc["attention_mask"], enc["attention_mask"]], dim=0),
        TemporalPolicy.YEAR_SPAN,
        anchor_mentions=[m_rel, None],
        temporal_active=[True, False],
        prompt_mask=torch.cat([pm, pm], dim=0),
    )
    assert math.isclose(mask_mix[0].sum().item(), 1.0, rel_tol=1e-5)
    assert mask_mix[1].sum().item() == 0.0


def test_09_tokenize_supervised_pair_and_hardened_replay(fast_tokenizer):
    from temporal_alignment.training.cpt_optimization import tokenize_supervised_pair

    res = tokenize_supervised_pair(fast_tokenizer, "Question: In 1947, value was\nAnswer:", "68.52", max_length=128)
    assert res["prompt_mask"].sum().item() > 0
    assert (res["labels"] != -100).sum().item() > 0
    assert not bool((res["prompt_mask"] & (res["labels"] != -100)).any().item())
    # Target truncation must fail closed
    with pytest.raises(ValueError, match="Target truncation detected"):
        tokenize_supervised_pair(fast_tokenizer, "Question: " + ("word " * 80) + "\nAnswer:", "68.52", max_length=32)

    nwp = [{"id": f"n_{i}", "input": f"In {1947 + (i % 60)}, metric was"} for i in range(40)]
    rep = [{"source_id": f"s_{i}", "input": f"Context: General fact {i}\nQuestion: What is {i}?"} for i in range(20)]

    # Fail closed on empty replay pool when ratio > 0
    with pytest.raises(ValueError, match="non-empty replay corpus"):
        interleave_general_replay(nwp, [], replay_ratio=0.15)
    # Fail closed on insufficient unique pool unless allow_oversample=True
    with pytest.raises(ValueError, match="Insufficient clean general replay"):
        interleave_general_replay(nwp, rep[:3], replay_ratio=0.15, allow_oversample=False)
    # Fail closed on fake year metadata or MCQA options on replay
    with pytest.raises(ValueError, match="year metadata"):
        interleave_general_replay(nwp, [{"source_id": "bad", "input": "General Q?", "year": 1950}], replay_ratio=0.15, allow_oversample=True)
    with pytest.raises(ValueError, match="MCQA option formatting"):
        interleave_general_replay(nwp, [{"source_id": "bad", "input": "Pick: Options: (a) x (b) y"}], replay_ratio=0.15, allow_oversample=True)

    m_ep0_a = interleave_general_replay(nwp, rep, replay_ratio=0.15, seed=42, epoch=0)
    m_ep0_b = interleave_general_replay(nwp, rep, replay_ratio=0.15, seed=42, epoch=0)
    m_ep1 = interleave_general_replay(nwp, rep, replay_ratio=0.15, seed=42, epoch=1)
    assert [x.get("id", x.get("source_id")) for x in m_ep0_a] == [x.get("id", x.get("source_id")) for x in m_ep0_b]
    assert [x.get("id", x.get("source_id")) for x in m_ep0_a] != [x.get("id", x.get("source_id")) for x in m_ep1]


def test_10_tiny_qwen_replay_gradients_and_masked_contrastive(fast_tokenizer):
    from temporal_alignment.stage03_eval.temporal_injector import inject_temporal_features
    from temporal_alignment.stage03_eval.temporal_locator import build_temporal_mask, resolve_question_body_anchor
    from temporal_alignment.training.cpt_optimization import compute_masked_temporal_contrastive_loss, tokenize_supervised_pair

    qcfg = Qwen2Config(
        vocab_size=len(fast_tokenizer),
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=128,
    )
    model = Qwen2ForCausalLM(qcfg)
    for p in model.parameters():
        p.requires_grad = False
    mods, core_gen = attach_adapters_to_model(model, arm="arm5", rank=4, alpha=8.0)
    seam = GatedResidualProjection(in_features=40, out_features=64, gate_mode="linear")
    # Make lora_B and seam alpha nonzero so first backward exercises both A and B gradients cleanly
    with torch.no_grad():
        for m in mods:
            m.lora_B.normal_(mean=0.0, std=0.05)
        seam.alpha.fill_(0.05)

    r_res = tokenize_supervised_pair(fast_tokenizer, "Context: Water freezes at zero C.\nQuestion: At what C does water freeze?\nAnswer:", "zero degrees", max_length=128)
    t_res1 = tokenize_supervised_pair(fast_tokenizer, "Question: In 1947, liberty index was\nAnswer:", "0.50", max_length=128)
    t_res2 = tokenize_supervised_pair(fast_tokenizer, "Question: In 2007, liberty index was\nAnswer:", "0.80", max_length=128)

    def _pad_stack(rows):
        max_l = max(len(x["input_ids"]) for x in rows)
        pad_id = fast_tokenizer.pad_token_id
        return (
            torch.stack([torch.cat([x["input_ids"], torch.full((max_l - len(x["input_ids"]),), pad_id, dtype=torch.long)]) for x in rows]),
            torch.stack([torch.cat([x["labels"], torch.full((max_l - len(x["labels"]),), -100, dtype=torch.long)]) for x in rows]),
            torch.stack([torch.cat([x["prompt_mask"], torch.zeros(max_l - len(x["prompt_mask"]), dtype=torch.bool)]) for x in rows]),
            torch.stack([torch.cat([torch.ones(len(x["input_ids"]), dtype=torch.long), torch.zeros(max_l - len(x["input_ids"]), dtype=torch.long)]) for x in rows]),
        )

    # (a) Replay-only backward: shared lora_A / lora_B MUST get nonzero grad; seam & core_gen MUST get ZERO grad
    in_r, lbl_r, pm_r, att_r = _pad_stack([r_res, r_res])
    act_r = torch.tensor([False, False], dtype=torch.bool)
    th_r = torch.tensor([0.0, 0.0])
    z_r = torch.tensor([0.0, 0.0])
    model.zero_grad()
    seam.zero_grad()
    core_gen.zero_grad()
    emb_r = model.get_input_embeddings()(in_r)
    with matlora_temporal_context(model, th_r, z_r, temporal_active=act_r):
        out_r = model(inputs_embeds=emb_r, labels=lbl_r, attention_mask=att_r, output_hidden_states=True)
        li, lt, _ = compute_masked_temporal_contrastive_loss(out_r.hidden_states[-1], torch.tensor([-1, -1]), act_r, pm_r, att_r)
        assert li.item() == 0.0 and lt.item() == 0.0
        (out_r.loss + li + lt).backward()
    assert any(m.lora_A.grad is not None and m.lora_A.grad.abs().sum().item() > 0.0 for m in mods)
    assert any(m.lora_B.grad is not None and m.lora_B.grad.abs().sum().item() > 0.0 for m in mods)
    assert seam.alpha.grad is None or seam.alpha.grad.abs().sum().item() == 0.0
    assert core_gen.cos_basis.grad is None or core_gen.cos_basis.grad.abs().sum().item() == 0.0

    # (b) Mixed batch forward/backward + NEUTRAL_CORE vs BOTH_OFF
    in_m, lbl_m, pm_m, att_m = _pad_stack([t_res1, t_res2, r_res])
    m1 = resolve_question_body_anchor(fast_tokenizer, t_res1["full_text"], task_id="NWP", metadata_year=1947, prompt_char_limit=t_res1["prompt_char_len"], offsets=t_res1["offsets"])
    m2 = resolve_question_body_anchor(fast_tokenizer, t_res2["full_text"], task_id="NWP", metadata_year=2007, prompt_char_limit=t_res2["prompt_char_len"], offsets=t_res2["offsets"])
    act_m = torch.tensor([True, True, False], dtype=torch.bool)
    mask_m = build_temporal_mask(in_m, att_m, TemporalPolicy.YEAR_SPAN, anchor_mentions=[m1, m2, None], temporal_active=[True, True, False], prompt_mask=pm_m)
    v40_m = torch.stack([
        torch.tensor(compute_temporal_40d(1947, fone_mode="dyadic", radius_mode="linear")),
        torch.tensor(compute_temporal_40d(2007, fone_mode="dyadic", radius_mode="linear")),
        torch.zeros(40),
    ])
    emb_m = inject_temporal_features(model.get_input_embeddings()(in_m), seam, v40_m, mask_m, TemporalPolicy.YEAR_SPAN)
    th_m = torch.tensor([compute_theta_z(1947, z_mode="linear")[0], compute_theta_z(2007, z_mode="linear")[0], 0.0])
    z_m = torch.tensor([compute_theta_z(1947, z_mode="linear")[1], compute_theta_z(2007, z_mode="linear")[1], 0.0])

    model.zero_grad()
    seam.zero_grad()
    core_gen.zero_grad()
    with matlora_temporal_context(model, th_m, z_m, temporal_active=act_m):
        out_m = model(inputs_embeds=emb_m, labels=lbl_m, attention_mask=att_m, output_hidden_states=True)
        li_m, lt_m, c_stats = compute_masked_temporal_contrastive_loss(out_m.hidden_states[-1], torch.tensor([1947, 2007, -1]), act_m, pm_m, att_m)
        assert c_stats["valid_temporal_rows"] == 2
        assert c_stats["positive_pairs"] == 2  # 1947 and 2007 share cycle 44
        (out_m.loss + 0.5 * li_m + 0.5 * lt_m).backward()
    assert seam.alpha.grad is not None and seam.alpha.grad.abs().sum().item() > 0.0
    assert core_gen.cos_basis.grad is not None and core_gen.cos_basis.grad.abs().sum().item() > 0.0
    assert mods[0].current_theta is None and mods[0].current_temporal_active is None

    # Masked contrastive invariance to extra right padding & changed replay rows
    h_fake = torch.randn(3, 40, 64, requires_grad=True)
    pm_fake = torch.zeros(3, 40, dtype=torch.bool)
    att_fake = torch.zeros(3, 40, dtype=torch.long)
    pm_fake[:, :6] = True
    att_fake[:, :12] = 1
    y_fake = torch.tensor([1947, 1950, 9999])
    act_fake = torch.tensor([True, True, False])
    li_1, lt_1, _ = compute_masked_temporal_contrastive_loss(h_fake, y_fake, act_fake, pm_fake, att_fake)
    h_pad_changed = h_fake.clone().detach()
    h_pad_changed[0, 6:] = 100.0  # change target & PAD tokens (outside prompt_mask)
    h_pad_changed[2, :] = -50.0  # change replay row
    li_2, lt_2, _ = compute_masked_temporal_contrastive_loss(h_pad_changed, y_fake, act_fake, pm_fake, att_fake)
    assert torch.allclose(li_1, li_2, atol=1e-6)
    assert torch.allclose(lt_1, lt_2, atol=1e-6)

    # Check BOTH_OFF ignores LoRA completely vs NEUTRAL_CORE
    x_dummy = torch.randn(1, 5, 64)
    base_only = mods[0].base_layer(x_dummy)
    with matlora_temporal_context(model, torch.tensor([0.5]), torch.tensor([0.1]), adapter_mode="BOTH_OFF"):
        assert torch.allclose(mods[0](x_dummy), base_only)
    with matlora_temporal_context(model, torch.tensor([0.5]), torch.tensor([0.1]), adapter_mode="NEUTRAL_CORE"):
        neutral_out = mods[0](x_dummy)
        assert neutral_out.shape == base_only.shape
    assert mods[0].adapter_mode == "ACTIVE"


def test_11_repair_recipes_r1_r2_scheduler_and_preflight():
    from temporal_alignment.stage03_eval.preflight_v3 import V3ModalLaunchSpec, verify_v3_modal_inputs

    r1 = V3ModalLaunchSpec.from_recipe("G1_SINGLE_YEAR_REPLAY_FIXED", profile="timheather2611", max_budget_usd=6.0)
    r2 = V3ModalLaunchSpec.from_recipe("G1_SINGLE_YEAR_REPLAY_COSINE", profile="timheather2611", max_budget_usd=6.0)
    assert r1.fone_mode == r2.fone_mode == "dyadic"
    assert r1.z_mode == r2.z_mode == r1.radius_mode == r2.radius_mode == "linear"
    assert r1.temporal_policy == r2.temporal_policy == "YEAR_SPAN"
    assert r1.schedule_type == "constant" and r2.schedule_type == "cosine"
    assert verify_v3_modal_inputs(r1)["status"] == "CPU_PREFLIGHT_PASS"
    assert verify_v3_modal_inputs(r2)["status"] == "CPU_PREFLIGHT_PASS"

    p = nn.Parameter(torch.ones(4, 4))
    opt1 = torch.optim.AdamW([{"params": [p], "lr": 2.0e-4}], lr=2.0e-4, weight_decay=0.01)
    sch1 = create_cpt_lr_scheduler(opt1, total_steps=100, warmup_ratio=0.0, schedule_type="constant", min_lr_ratio=1.0)
    opt2 = torch.optim.AdamW([{"params": [p], "lr": 2.0e-4}], lr=2.0e-4, weight_decay=0.01)
    sch2 = create_cpt_lr_scheduler(opt2, total_steps=100, warmup_ratio=0.04, schedule_type="cosine", min_lr_ratio=0.10)

    r1_lrs, r2_lrs = [], []
    for _ in range(100):
        opt1.step()
        r1_lrs.append(opt1.param_groups[0]["lr"])
        sch1.step()
        opt2.step()
        r2_lrs.append(opt2.param_groups[0]["lr"])
        sch2.step()

    assert all(math.isclose(lr, 2.0e-4, rel_tol=1e-6) for lr in r1_lrs)
    assert math.isclose(r2_lrs[0], 2.0e-4 * 0.25, rel_tol=1e-5)
    assert math.isclose(r2_lrs[3], 2.0e-4, rel_tol=1e-5)
    assert math.isclose(r2_lrs[-1], 2.0e-5, rel_tol=1e-2)


def test_12_illumination_both_off_contract_hash_and_g3_spy(fast_tokenizer, monkeypatch, tmp_path):
    import copy
    import importlib.util
    from pathlib import Path
    from temporal_alignment.stage03_eval.model_adapter import matlora_temporal_context
    from temporal_alignment.stage03_eval.preflight_v3 import V3ModalLaunchSpec, verify_v3_modal_inputs
    from temporal_alignment.training.cpt_optimization import (
        compute_masked_temporal_contrastive_loss,
        tokenize_supervised_pair,
    )

    # 1. Shared contract_hash invariant across arm/checkpoint/device while config_hash distinguishes them
    cfg_base = EvaluationConfig(
        contract_version="EVAL_CONTRACT_G1_FREEGEN_V2_REPAIR",
        arm="base",
        temporal_policy=TemporalPolicy.YEAR_SPAN,
        parser_contract=ParserContract.LABEL_ONLY,
        device="cpu",
        checkpoint_path=None,
    )
    cfg_r1 = EvaluationConfig(
        contract_version="EVAL_CONTRACT_G1_FREEGEN_V2_REPAIR",
        arm="arm5",
        temporal_policy=TemporalPolicy.YEAR_SPAN,
        parser_contract=ParserContract.LABEL_ONLY,
        device="cuda:0",
        checkpoint_path="/vol/g1_repair/g1_single_year_replay_fixed/final_weights.pt",
    )
    assert cfg_base.contract_hash() == cfg_r1.contract_hash()
    assert cfg_base.config_hash() != cfg_r1.config_hash()

    # 2. BOTH_OFF end-to-end inside execute_evaluation_loop matches unadapted Base SEAM_OFF
    torch.manual_seed(7)
    qcfg = Qwen2Config(
        vocab_size=len(fast_tokenizer),
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=256,
    )
    base_model = Qwen2ForCausalLM(qcfg).eval()
    adapted_arm5 = copy.deepcopy(base_model).eval()
    mods5, core_gen = attach_adapters_to_model(adapted_arm5, arm="arm5", rank=4, alpha=16.0)
    seam = GatedResidualProjection(in_features=40, out_features=64, gate_mode="linear").eval()
    with torch.no_grad():
        for m in mods5:
            m.lora_B.normal_(mean=0.0, std=1.5)
        seam.alpha.fill_(2.0)

    cases = [
        {
            "example_id": "illum_db_1",
            "task_id": "DB",
            "year": 1947,
            "input": "In 1947, index was: Options: (a) 10 (b) 20 (c) 30 (d) 40",
            "target": "a",
        }
    ]
    res_base = execute_evaluation_loop(
        config=EvaluationConfig(arm="base", temporal_policy=TemporalPolicy.SEAM_OFF, device="cpu"),
        model=base_model,
        tokenizer=fast_tokenizer,
        eval_cases=cases,
        output_dir=str(tmp_path / "base"),
    )
    res_both_off = execute_evaluation_loop(
        config=EvaluationConfig(
            arm="arm5",
            temporal_policy=TemporalPolicy.YEAR_SPAN,
            device="cpu",
            adapter_mode="BOTH_OFF",
        ),
        model=adapted_arm5,
        tokenizer=fast_tokenizer,
        eval_cases=cases,
        output_dir=str(tmp_path / "both_off"),
        seam_proj=seam,
        core_gen=core_gen,
    )
    res_active = execute_evaluation_loop(
        config=EvaluationConfig(
            arm="arm5",
            temporal_policy=TemporalPolicy.YEAR_SPAN,
            device="cpu",
            adapter_mode="ACTIVE",
        ),
        model=adapted_arm5,
        tokenizer=fast_tokenizer,
        eval_cases=cases,
        output_dir=str(tmp_path / "active"),
        seam_proj=seam,
        core_gen=core_gen,
    )
    import json as _json
    recs_base = [_json.loads(l) for l in (tmp_path / "base" / "PREDICTION_RECORDS.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    recs_off = [_json.loads(l) for l in (tmp_path / "both_off" / "PREDICTION_RECORDS.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    recs_act = [_json.loads(l) for l in (tmp_path / "active" / "PREDICTION_RECORDS.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    assert [r["generated_token_ids"] for r in recs_off] == [r["generated_token_ids"] for r in recs_base]
    assert [r["raw_continuation"] for r in recs_off] == [r["raw_continuation"] for r in recs_base]
    assert res_both_off["contract_hash"] != res_active["contract_hash"]
    assert any(a["generated_token_ids"] != b["generated_token_ids"] for a, b in zip(recs_act, recs_off))

    # StaticLoRALinear BOTH_OFF parity
    adapted_arm4 = copy.deepcopy(base_model).eval()
    mods4, _ = attach_adapters_to_model(adapted_arm4, arm="arm4", rank=4, alpha=16.0)
    with torch.no_grad():
        for m in mods4:
            m.lora_B.normal_(mean=0.0, std=1.0)
    x_dummy = torch.randn(1, 4, 64)
    with matlora_temporal_context(adapted_arm4, None, None, adapter_mode="BOTH_OFF"):
        assert torch.allclose(mods4[0](x_dummy), mods4[0].base_layer(x_dummy))
    assert mods4[0].adapter_mode == "ACTIVE"

    # 3. non_special_mask propagated and enforced in contrastive pooling
    tok_res = tokenize_supervised_pair(fast_tokenizer, "Question: In 1947, score was\nAnswer:", "0.55", max_length=128)
    assert "non_special_mask" in tok_res
    assert tok_res["non_special_mask"].shape == tok_res["input_ids"].shape
    h_sp = torch.randn(2, 12, 32, requires_grad=True)
    pm_sp = torch.ones(2, 12, dtype=torch.bool)
    att_sp = torch.ones(2, 12, dtype=torch.long)
    ns_sp = torch.ones(2, 12, dtype=torch.bool)
    ns_sp[:, 0] = False  # mark index 0 as special token
    li_a, lt_a, _ = compute_masked_temporal_contrastive_loss(
        h_sp, torch.tensor([1947, 1950]), torch.tensor([True, True]), pm_sp, att_sp, non_special_mask=ns_sp
    )
    h_sp2 = h_sp.clone().detach()
    h_sp2[:, 0, :] = 999.0
    li_b, lt_b, _ = compute_masked_temporal_contrastive_loss(
        h_sp2, torch.tensor([1947, 1950]), torch.tensor([True, True]), pm_sp, att_sp, non_special_mask=ns_sp
    )
    assert torch.allclose(li_a, li_b, atol=1e-6) and torch.allclose(lt_a, lt_b, atol=1e-6)

    # 4. G3 Spy: CPU preflight and unconfirmed train_eval trigger ZERO paid remote GPU calls
    r1 = V3ModalLaunchSpec.from_recipe("G1_SINGLE_YEAR_REPLAY_FIXED", profile="timheather2611", max_budget_usd=10.0)
    pf = verify_v3_modal_inputs(r1)
    assert pf["smoke_calibrated_a10g_hours"] == 5.836
    assert pf["smoke_calibrated_cost_usd"] == 6.42
    assert pf["gates"]["G3_EFFECT_AUTHORIZATION"] == "HOLD_EXPLICIT_CONFIRM_GPU"

    launcher_path = Path(__file__).resolve().parents[1] / "scripts" / "run_stage03_v3_mlp_cpt_modal.py"
    mod_spec = importlib.util.spec_from_file_location("stage03_v3_modal_launcher_spy", str(launcher_path))
    launcher_mod = importlib.util.module_from_spec(mod_spec)
    mod_spec.loader.exec_module(launcher_mod)

    paid_calls = []

    def _spy_remote(*args, **kwargs):
        paid_calls.append((args, kwargs))
        raise AssertionError("Paid GPU remote must not be invoked without explicit confirm-gpu!")

    for fn_name in ("run_stage03_g1_smoke", "run_stage03_v3_train_and_eval", "run_stage03_v3_eval_only"):
        monkeypatch.setattr(getattr(launcher_mod, fn_name), "remote", _spy_remote)

    launcher_mod.main(recipe_id="G1_SINGLE_YEAR_REPLAY_FIXED", profile="timheather2611", mode="preflight", max_budget_usd=10.0)
    assert len(paid_calls) == 0
    with pytest.raises(RuntimeError, match="confirm-gpu"):
        launcher_mod.main(
            recipe_id="G1_SINGLE_YEAR_REPLAY_FIXED",
            profile="timheather2611",
            mode="train_eval",
            max_budget_usd=10.0,
            confirm_gpu=False,
        )
    assert len(paid_calls) == 0
