"""Comprehensive Checkpoint Tensor SHA & Delta Analysis adhering to governance AST guard.
"""

import hashlib
import json
from pathlib import Path
import torch

def sha256_file(p: str) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()

def extract_flat_tensors(state: dict):
    flat = {}
    for group in ["lora_state", "seam_projection", "core_generator"]:
        val = state.get(group)
        if isinstance(val, dict):
            for k, t in val.items():
                if isinstance(t, torch.Tensor):
                    flat[f"{group}.{k}"] = t
        elif isinstance(val, torch.Tensor):
            flat[group] = val
    return flat

def diff_states(s1: dict, s2: dict, file1: str, file2: str):
    t1 = extract_flat_tensors(s1)
    t2 = extract_flat_tensors(s2)
    common_keys = sorted(set(t1.keys()) & set(t2.keys()))
    
    changed_count = 0
    total_count = len(common_keys)
    group_stats = {}
    l2_total_sq = 0.0
    max_delta_global = 0.0
    abs_delta_sum_global = 0.0
    total_elements_global = 0
    has_nan_or_inf = False
    
    for k in common_keys:
        group = k.split(".")[0]
        if group not in group_stats:
            group_stats[group] = {
                "tensors": 0, "changed_tensors": 0,
                "l2_sq": 0.0, "max_delta": 0.0, "abs_sum": 0.0, "elements": 0
            }
        v1 = t1[k].float()
        v2 = t2[k].float()
        if not (torch.isfinite(v1).all() and torch.isfinite(v2).all()):
            has_nan_or_inf = True
        diff = v2 - v1
        diff_abs = torch.abs(diff)
        diff_max = diff_abs.max().item()
        diff_l2_sq = (diff ** 2).sum().item()
        diff_sum = diff_abs.sum().item()
        n_elem = v1.numel()
        
        if diff_max > 0.0:
            changed_count += 1
            group_stats[group]["changed_tensors"] += 1
        group_stats[group]["tensors"] += 1
        group_stats[group]["l2_sq"] += diff_l2_sq
        group_stats[group]["max_delta"] = max(group_stats[group]["max_delta"], diff_max)
        group_stats[group]["abs_sum"] += diff_sum
        group_stats[group]["elements"] += n_elem
        
        l2_total_sq += diff_l2_sq
        max_delta_global = max(max_delta_global, diff_max)
        abs_delta_sum_global += diff_sum
        total_elements_global += n_elem

    summary_groups = {}
    for g, st in group_stats.items():
        summary_groups[g] = {
            "tensor_count": st["tensors"],
            "changed_tensor_count": st["changed_tensors"],
            "fraction_changed": round(st["changed_tensors"] / st["tensors"], 4) if st["tensors"] else 0.0,
            "l2_delta": round(st["l2_sq"] ** 0.5, 6),
            "max_abs_delta": round(st["max_delta"], 6),
            "mean_abs_delta": round(st["abs_sum"] / st["elements"], 8) if st["elements"] else 0.0
        }
    return {
        "file1": Path(file1).name,
        "sha1": sha256_file(file1),
        "file2": Path(file2).name,
        "sha2": sha256_file(file2),
        "total_tensors": total_count,
        "changed_tensor_count": changed_count,
        "fraction_changed": round(changed_count / total_count, 4) if total_count else 0.0,
        "global_l2_delta": round(l2_total_sq ** 0.5, 6),
        "global_max_abs_delta": round(max_delta_global, 6),
        "global_mean_abs_delta": round(abs_delta_sum_global / total_elements_global, 8) if total_elements_global else 0.0,
        "nan_or_inf_detected": has_nan_or_inf,
        "group_breakdown": summary_groups,
        "meta1": s1.get("metadata", {}),
        "meta2": s2.get("metadata", {})
    }

# Explicit constant path loads adhering to AST guard
print("Loading Arm 4 static mlp checkpoints...")
s4_ep1 = torch.load("scratch/checkpoint_staging/stage03_3b_targeted_static_mlp_5epochs_ep1_weights.pt", weights_only=True, map_location="cpu")
s4_ep2 = torch.load("scratch/checkpoint_staging/stage03_3b_targeted_static_mlp_5epochs_ep2_weights.pt", weights_only=True, map_location="cpu")
s4_ep3 = torch.load("scratch/checkpoint_staging/stage03_3b_targeted_static_mlp_5epochs_ep3_weights.pt", weights_only=True, map_location="cpu")
s4_ep4 = torch.load("scratch/checkpoint_staging/stage03_3b_targeted_static_mlp_5epochs_ep4_weights.pt", weights_only=True, map_location="cpu")
s4_ep5 = torch.load("scratch/checkpoint_staging/stage03_3b_targeted_static_mlp_5epochs_ep5_weights.pt", weights_only=True, map_location="cpu")
s4_fin = torch.load("scratch/checkpoint_staging/stage03_3b_targeted_static_mlp_5epochs_weights.pt", weights_only=True, map_location="cpu")

print("Loading Arm 5 matlora checkpoints...")
s5_ep1 = torch.load("scratch/checkpoint_staging/stage03_3b_targeted_matlora_5epochs_ep1_weights.pt", weights_only=True, map_location="cpu")
s5_ep2 = torch.load("scratch/checkpoint_staging/stage03_3b_targeted_matlora_5epochs_ep2_weights.pt", weights_only=True, map_location="cpu")
s5_ep3 = torch.load("scratch/checkpoint_staging/stage03_3b_targeted_matlora_5epochs_ep3_weights.pt", weights_only=True, map_location="cpu")
s5_ep4 = torch.load("scratch/checkpoint_staging/stage03_3b_targeted_matlora_5epochs_ep4_weights.pt", weights_only=True, map_location="cpu")
s5_ep5 = torch.load("scratch/checkpoint_staging/stage03_3b_targeted_matlora_5epochs_ep5_weights.pt", weights_only=True, map_location="cpu")
s5_fin = torch.load("scratch/checkpoint_staging/stage03_3b_targeted_matlora_5epochs_weights.pt", weights_only=True, map_location="cpu")

print("Comparing sequential transitions...")
arm4_transitions = [
    diff_states(s4_ep1, s4_ep2, "scratch/checkpoint_staging/stage03_3b_targeted_static_mlp_5epochs_ep1_weights.pt", "scratch/checkpoint_staging/stage03_3b_targeted_static_mlp_5epochs_ep2_weights.pt"),
    diff_states(s4_ep2, s4_ep3, "scratch/checkpoint_staging/stage03_3b_targeted_static_mlp_5epochs_ep2_weights.pt", "scratch/checkpoint_staging/stage03_3b_targeted_static_mlp_5epochs_ep3_weights.pt"),
    diff_states(s4_ep3, s4_ep4, "scratch/checkpoint_staging/stage03_3b_targeted_static_mlp_5epochs_ep3_weights.pt", "scratch/checkpoint_staging/stage03_3b_targeted_static_mlp_5epochs_ep4_weights.pt"),
    diff_states(s4_ep4, s4_ep5, "scratch/checkpoint_staging/stage03_3b_targeted_static_mlp_5epochs_ep4_weights.pt", "scratch/checkpoint_staging/stage03_3b_targeted_static_mlp_5epochs_ep5_weights.pt"),
    diff_states(s4_ep5, s4_fin, "scratch/checkpoint_staging/stage03_3b_targeted_static_mlp_5epochs_ep5_weights.pt", "scratch/checkpoint_staging/stage03_3b_targeted_static_mlp_5epochs_weights.pt"),
]

arm5_transitions = [
    diff_states(s5_ep1, s5_ep2, "scratch/checkpoint_staging/stage03_3b_targeted_matlora_5epochs_ep1_weights.pt", "scratch/checkpoint_staging/stage03_3b_targeted_matlora_5epochs_ep2_weights.pt"),
    diff_states(s5_ep2, s5_ep3, "scratch/checkpoint_staging/stage03_3b_targeted_matlora_5epochs_ep2_weights.pt", "scratch/checkpoint_staging/stage03_3b_targeted_matlora_5epochs_ep3_weights.pt"),
    diff_states(s5_ep3, s5_ep4, "scratch/checkpoint_staging/stage03_3b_targeted_matlora_5epochs_ep3_weights.pt", "scratch/checkpoint_staging/stage03_3b_targeted_matlora_5epochs_ep4_weights.pt"),
    diff_states(s5_ep4, s5_ep5, "scratch/checkpoint_staging/stage03_3b_targeted_matlora_5epochs_ep4_weights.pt", "scratch/checkpoint_staging/stage03_3b_targeted_matlora_5epochs_ep5_weights.pt"),
    diff_states(s5_ep5, s5_fin, "scratch/checkpoint_staging/stage03_3b_targeted_matlora_5epochs_ep5_weights.pt", "scratch/checkpoint_staging/stage03_3b_targeted_matlora_5epochs_weights.pt"),
]

output_payload = {
    "audit_type": "CHECKPOINT_TENSOR_SHA_AND_DELTA",
    "arm4_static_mlp": arm4_transitions,
    "arm5_targeted_matlora": arm5_transitions
}

with open("scratch/checkpoint_staging/CHECKPOINT_TENSOR_SHA_AND_DELTA_RAW.json", "w", encoding="utf-8") as f:
    json.dump(output_payload, f, indent=2, ensure_ascii=False)

print("SUCCESS: Tensor delta analysis written to scratch/checkpoint_staging/CHECKPOINT_TENSOR_SHA_AND_DELTA_RAW.json")
