"""Mechanism Tracer for Temporal Representation, Dynamic Adapter, Hidden States, and Logit Margin.

Implements the formal instrumentation contract defined in Decision 19:
    t -> z(t) -> F_t -> Delta W(t) -> h_l(t) -> logits(t) -> M(t).

Metrics:
    - D_z = ||z(t1) - z(t2)||_2
    - D_F^(m) = ||F_t1^(m) - F_t2^(m)||_F
    - D_W^(m) = ||Delta W^(m)(t1) - Delta W^(m)(t2)||_F
    - S_l = ||h_l(t1) - h_l(t2)||_2 / max(||h_l(t1)||_2, epsilon)
    - CosineSim_l = <h_l(t1), h_l(t2)> / (||h_l(t1)||_2 * ||h_l(t2)||_2)
    - Gold margin: M(t) = logit_gold - max_{j != gold} logit_j
    - Delta M = M(t2) - M(t1)
"""

from __future__ import annotations

import copy
import math
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from temporal_alignment.models.adapter_activation import (
    configure_temporal_condition,
)
from temporal_alignment.models.dual_cycle_mat_lora import DualCycleMaTLoRALinear
from temporal_alignment.models.fone_canchi_40d import (
    compute_fixed_fone_canchi_40d,
    extract_canchi_angles_from_40d,
)
from temporal_alignment.models.gated_seam import (
    ConditioningContext,
    GatedResidualProjection,
)


@dataclass
class MechanismTraceResult:
    """Detailed traces for a single temporal contrast pair (t1, t2)."""
    case_id: str
    t1: int
    t2: int
    gold_t1: str
    gold_t2: str
    d_z: float
    d_f_mean: float
    d_w_mean: float
    layerwise_sensitivity: List[float]  # S_l for each layer at completion token (p=-1)
    layerwise_cosine_sim: List[float]   # CosineSim_l for each layer at completion token (p=-1)
    sensitivity_seam_pos: List[float] = field(default_factory=list)  # S_l at seam injection token (p=14)
    cosine_sim_seam_pos: List[float] = field(default_factory=list)   # CosineSim_l at seam token (p=14)
    mean_abs_logit_delta: float = 0.0
    l2_logit_delta: float = 0.0
    margin_t1: float = 0.0
    margin_t2: float = 0.0
    delta_gold_margin_t2: float = 0.0  # Margin shift for gold_t2 from t1 to t2
    prediction_t1: str = ""
    prediction_t2: str = ""
    is_flip: bool = False
    is_correct_flip: bool = False
    controls: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class TemporalMechanismTracer:
    """Instruments model forward pass to trace internal temporal representations."""

    def __init__(
        self,
        model: nn.Module,
        tokenizer: Any,
        adapted_modules: Optional[Dict[str, DualCycleMaTLoRALinear]] = None,
        gated_proj: Optional[GatedResidualProjection] = None,
        cond_context: Optional[ConditioningContext] = None,
        device: Union[str, torch.device] = "cpu",
        seam_token_index: int = 14,
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.adapted_modules = adapted_modules or {}
        self.gated_proj = gated_proj
        self.cond_context = cond_context or ConditioningContext(injection_token_index=seam_token_index)
        self.device = torch.device(device)
        self.seam_token_index = seam_token_index

    def _extract_token_ids_for_labels(self, labels: List[str]) -> Dict[str, int]:
        """Map answer labels (e.g. Yes, No, A, B, C, D) to their primary token IDs."""
        label_ids: Dict[str, int] = {}
        for lab in labels:
            # Try both raw and leading-space tokens
            ids = self.tokenizer.encode(lab, add_special_tokens=False)
            if ids:
                label_ids[lab.lower()] = ids[-1]
            ids_sp = self.tokenizer.encode(" " + lab, add_special_tokens=False)
            if ids_sp:
                label_ids[lab.lower() + "_sp"] = ids_sp[-1]
        return label_ids

    def _compute_gold_margin(
        self,
        logits: torch.Tensor,
        gold_label: str,
        candidate_labels: List[str],
    ) -> float:
        """Compute M(t) = logit_gold - max_{j != gold} logit_j over candidate set."""
        label_map = self._extract_token_ids_for_labels(candidate_labels)
        
        gold_id = label_map.get(gold_label.lower()) or label_map.get(gold_label.lower() + "_sp")
        if gold_id is None:
            return 0.0
            
        gold_logit = logits[gold_id].item()
        
        other_logits = []
        for cand in candidate_labels:
            if cand.lower() == gold_label.lower():
                continue
            cand_id = label_map.get(cand.lower()) or label_map.get(cand.lower() + "_sp")
            if cand_id is not None:
                other_logits.append(logits[cand_id].item())
                
        if not other_logits:
            return 0.0
            
        max_other = max(other_logits)
        return gold_logit - max_other

    def _capture_hidden_states(
        self,
        inputs: Dict[str, torch.Tensor],
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor], torch.Tensor]:
        """Run forward pass and capture hidden states from each decoder layer at comp and seam positions."""
        captured_comp: List[torch.Tensor] = []
        captured_seam: List[torch.Tensor] = []
        hooks = []

        # Find layers in model
        layers = None
        if hasattr(self.model, "model") and hasattr(self.model.model, "layers"):
            layers = self.model.model.layers
        elif hasattr(self.model, "transformer") and hasattr(self.model.transformer, "h"):
            layers = self.model.transformer.h

        if layers is not None:
            for l_idx, layer in enumerate(layers):
                def make_hook(idx):
                    def hook_fn(module, inp, output):
                        # output can be a tuple (hidden_states, ...)
                        h = output[0] if isinstance(output, tuple) else output
                        seq_len = h.shape[1]
                        seam_pos = min(self.seam_token_index, seq_len - 1)
                        # Capture completion token (p=-1) and seam token (p=14)
                        captured_comp.append(h[0, -1, :].detach().clone())
                        captured_seam.append(h[0, seam_pos, :].detach().clone())
                    return hook_fn
                hooks.append(layer.register_forward_hook(make_hook(l_idx)))

        try:
            with torch.no_grad():
                out = self.model(**inputs)
                final_logits = out.logits[0, -1, :].detach().clone()
        finally:
            for h in hooks:
                h.remove()

        return captured_comp, captured_seam, final_logits

    def trace_pair(
        self,
        case_id: str,
        prompt_text: str,
        t1: int,
        t2: int,
        gold_t1: str,
        gold_t2: str,
        candidate_labels: Optional[List[str]] = None,
    ) -> MechanismTraceResult:
        """Trace the causal temporal representation chain between t1 and t2."""
        if candidate_labels is None:
            if "yes or no" in prompt_text.lower():
                candidate_labels = ["Yes", "No"]
            else:
                candidate_labels = ["A", "B", "C", "D"]

        # 1. Temporal input vectors
        v1_list = compute_fixed_fone_canchi_40d(t1)
        v2_list = compute_fixed_fone_canchi_40d(t2)
        z1 = torch.tensor(v1_list, dtype=torch.float32, device=self.device)
        z2 = torch.tensor(v2_list, dtype=torch.float32, device=self.device)
        d_z = (z1 - z2).norm(p=2).item()

        th1, z_c1 = extract_canchi_angles_from_40d(v1_list)
        th2, z_c2 = extract_canchi_angles_from_40d(v2_list)

        # 2. Dynamic Adapter Module Difference
        df_list = []
        dw_list = []
        for name, module in self.adapted_modules.items():
            if isinstance(module, DualCycleMaTLoRALinear) and hasattr(module, "core_generator"):
                t_th1 = torch.tensor(th1, dtype=torch.float32, device=self.device)
                t_z1 = torch.tensor(z_c1, dtype=torch.float32, device=self.device)
                t_th2 = torch.tensor(th2, dtype=torch.float32, device=self.device)
                t_z2 = torch.tensor(z_c2, dtype=torch.float32, device=self.device)

                # Sample F_t1 vs F_t2
                f1 = module.core_generator(t_th1, t_z1)[0]
                f2 = module.core_generator(t_th2, t_z2)[0]
                df = (f1 - f2).norm(p="fro").item()
                df_list.append(df)
                
                # Delta W = scaling * (B @ F_t @ A)
                w1 = module.scaling * (module.lora_B @ f1 @ module.lora_A)
                w2 = module.scaling * (module.lora_B @ f2 @ module.lora_A)
                dw = (w1 - w2).norm(p="fro").item()
                dw_list.append(dw)

        d_f_mean = sum(df_list) / len(df_list) if df_list else 0.0
        d_w_mean = sum(dw_list) / len(dw_list) if dw_list else 0.0

        # Prepare prompt
        from temporal_alignment.baselines.ticktack05b.evaluator import build_p12_prompt, task_conditioned_strict_parser
        formatted_prompt, is_yn = build_p12_prompt(prompt_text, self.tokenizer)
        inputs = self.tokenizer(formatted_prompt, return_tensors="pt").to(self.device)
        prompt_len = inputs["input_ids"].shape[1]

        g_proj = self.gated_proj if self.gated_proj is not None else GatedResidualProjection(in_features=40, out_features=getattr(self.model.config, 'hidden_size', 896)).to(self.device)

        # 3. Forward pass at t1
        with configure_temporal_condition("C_TRUE", self.adapted_modules, g_proj, self.cond_context, v1_list, th1, z_c1, self.device, seam_token_index=self.seam_token_index):
            hidden_comp_t1, hidden_seam_t1, logits_t1 = self._capture_hidden_states(inputs)
            gen_ids_t1 = self.model.generate(**inputs, max_new_tokens=4, do_sample=False, pad_token_id=self.tokenizer.pad_token_id, eos_token_id=self.tokenizer.eos_token_id)
        raw_pred_t1 = self.tokenizer.decode(gen_ids_t1[0][prompt_len:], skip_special_tokens=True).strip()
        pred_t1, _ = task_conditioned_strict_parser(raw_pred_t1, is_yn)

        # 4. Forward pass at t2
        with configure_temporal_condition("C_TRUE", self.adapted_modules, g_proj, self.cond_context, v2_list, th2, z_c2, self.device, seam_token_index=self.seam_token_index):
            hidden_comp_t2, hidden_seam_t2, logits_t2 = self._capture_hidden_states(inputs)
            gen_ids_t2 = self.model.generate(**inputs, max_new_tokens=4, do_sample=False, pad_token_id=self.tokenizer.pad_token_id, eos_token_id=self.tokenizer.eos_token_id)
        raw_pred_t2 = self.tokenizer.decode(gen_ids_t2[0][prompt_len:], skip_special_tokens=True).strip()
        pred_t2, _ = task_conditioned_strict_parser(raw_pred_t2, is_yn)

        # 5. Layerwise hidden state metrics at completion token (p=-1)
        eps = 1e-8
        s_layers: List[float] = []
        cos_layers: List[float] = []
        for h1, h2 in zip(hidden_comp_t1, hidden_comp_t2):
            norm1 = h1.norm(p=2).item()
            diff_norm = (h1 - h2).norm(p=2).item()
            s_l = diff_norm / max(norm1, eps)
            s_layers.append(round(s_l, 6))

            cos = F.cosine_similarity(h1.unsqueeze(0), h2.unsqueeze(0)).item()
            cos_layers.append(round(cos, 6))

        # Layerwise metrics at seam injection token (p=14)
        s_seam_layers: List[float] = []
        cos_seam_layers: List[float] = []
        for h1, h2 in zip(hidden_seam_t1, hidden_seam_t2):
            norm1 = h1.norm(p=2).item()
            diff_norm = (h1 - h2).norm(p=2).item()
            s_seam_layers.append(round(diff_norm / max(norm1, eps), 6))
            cos_seam_layers.append(round(F.cosine_similarity(h1.unsqueeze(0), h2.unsqueeze(0)).item(), 6))

        # 6. Logit deltas and Gold Margins
        mean_abs_logit_delta = (logits_t2 - logits_t1).abs().mean().item()
        l2_logit_delta = (logits_t2 - logits_t1).norm(p=2).item()

        margin_t1 = self._compute_gold_margin(logits_t1, gold_t1, candidate_labels)
        margin_t2 = self._compute_gold_margin(logits_t2, gold_t2, candidate_labels)

        # Specific shift of gold_t2: how much did margin of gold_t2 increase when moving from t1 to t2?
        margin_g2_at_t1 = self._compute_gold_margin(logits_t1, gold_t2, candidate_labels)
        margin_g2_at_t2 = self._compute_gold_margin(logits_t2, gold_t2, candidate_labels)
        delta_m_g2 = margin_g2_at_t2 - margin_g2_at_t1

        # 7. Controls: Same-year repeated forward (stability check)
        with configure_temporal_condition("C_TRUE", self.adapted_modules, g_proj, self.cond_context, v1_list, th1, z_c1, self.device, seam_token_index=self.seam_token_index):
            _, _, logits_t1_repeat = self._capture_hidden_states(inputs)
        repeat_stability_delta = (logits_t1_repeat - logits_t1).abs().mean().item()

        # Matched-norm random perturbation
        rand_noise = torch.randn_like(z1)
        rand_noise = rand_noise * (d_z / (rand_noise.norm(p=2).item() + eps))
        z_rand = z1 + rand_noise
        z_rand_list = z_rand.cpu().tolist()
        th_rand, z_c_rand = extract_canchi_angles_from_40d(z_rand_list)
        with configure_temporal_condition("C_WRONG_YEAR", self.adapted_modules, g_proj, self.cond_context, z_rand_list, th_rand, z_c_rand, self.device, seam_token_index=self.seam_token_index):
            _, _, logits_rand = self._capture_hidden_states(inputs)
        random_perturb_delta = (logits_rand - logits_t1).abs().mean().item()

        controls = {
            "same_year_repeated_delta": round(repeat_stability_delta, 8),
            "matched_norm_random_delta": round(random_perturb_delta, 6),
            "random_vs_temporal_ratio": round(random_perturb_delta / max(mean_abs_logit_delta, 1e-8), 4),
        }

        # Check flip and correctness
        p1_norm = pred_t1.strip().lower() if pred_t1 else ""
        p2_norm = pred_t2.strip().lower() if pred_t2 else ""
        g1_norm = gold_t1.strip().lower()
        g2_norm = gold_t2.strip().lower()

        is_flip = (p1_norm != p2_norm)
        is_corr_flip = (p1_norm == g1_norm) and (p2_norm == g2_norm) and (g1_norm != g2_norm)

        return MechanismTraceResult(
            case_id=case_id,
            t1=t1,
            t2=t2,
            gold_t1=gold_t1,
            gold_t2=gold_t2,
            d_z=round(d_z, 6),
            d_f_mean=round(d_f_mean, 6),
            d_w_mean=round(d_w_mean, 6),
            layerwise_sensitivity=s_layers,
            layerwise_cosine_sim=cos_layers,
            sensitivity_seam_pos=s_seam_layers,
            cosine_sim_seam_pos=cos_seam_layers,
            mean_abs_logit_delta=round(mean_abs_logit_delta, 6),
            l2_logit_delta=round(l2_logit_delta, 6),
            margin_t1=round(margin_t1, 6),
            margin_t2=round(margin_t2, 6),
            delta_gold_margin_t2=round(delta_m_g2, 6),
            prediction_t1=pred_t1 or "INVALID",
            prediction_t2=pred_t2 or "INVALID",
            is_flip=is_flip,
            is_correct_flip=is_corr_flip,
            controls=controls,
        )
