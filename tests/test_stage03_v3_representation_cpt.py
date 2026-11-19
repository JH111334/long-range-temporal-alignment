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
