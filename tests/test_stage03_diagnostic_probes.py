"""Unit tests verifying logic of Stage 03 Next Diagnostic Experiment Package:
1. Rotated-option MCQA 4-fold permutation generator
2. Counterfactual year perturbation & coordinate calculation
3. Assistant trigger hook location resolution
"""

import re
from typing import Any, Dict, List, Tuple
import pytest

from temporal_alignment.models.fone_canchi_40d import compute_fixed_fone_canchi_40d


def parse_mcqa_prompt_choices(prompt_text: str) -> Tuple[str, List[str]]:
    """Parse question stem and list of 4 choices (A, B, C, D) from prompt text."""
    # Matches (A) text (B) text (C) text (D) text
    pattern = r"\(A\)\s*(.*?)\s*\(B\)\s*(.*?)\s*\(C\)\s*(.*?)\s*\(D\)\s*(.*?)(?:\s*Answer:|\s*$)"
    match = re.search(pattern, prompt_text, re.DOTALL | re.IGNORECASE)
    if not match:
        raise ValueError(f"Could not parse 4 choices from prompt: {prompt_text[:100]}")
    
    stem = prompt_text[:match.start()].strip()
    choices = [match.group(1).strip(), match.group(2).strip(), match.group(3).strip(), match.group(4).strip()]
    return stem, choices


def generate_rotated_mcqa_variants(item: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Generate 4 cyclic rotations of an MCQA question item."""
    prompt = item.get("input") or item.get("question", "")
    gold = (item.get("target") or item.get("gold_target", "")).strip().lower()
    
    char_to_idx = {"a": 0, "b": 1, "c": 2, "d": 3}
    idx_to_char = {0: "a", 1: "b", 2: "c", 3: "d"}
    
    if gold not in char_to_idx:
        # Not a 4-choice question (e.g. Yes/No)
        return [item]
    
    orig_gold_idx = char_to_idx[gold]
    stem, choices = parse_mcqa_prompt_choices(prompt)
    
    variants = []
    # 4 cyclic shifts: shift k in [0, 1, 2, 3]
    for shift in range(4):
        # New choices: choice at index i becomes choice at index (i - shift) % 4
        # Equivalent to rotating the list: choices[shift:] + choices[:shift]
        rotated_choices = [choices[(i + shift) % 4] for i in range(4)]
        # The original gold choice was at orig_gold_idx. Its new position is:
        # new_idx such that (new_idx + shift) % 4 == orig_gold_idx
        # => new_idx = (orig_gold_idx - shift) % 4
        new_gold_idx = (orig_gold_idx - shift) % 4
        new_gold_char = idx_to_char[new_gold_idx]
        
        new_prompt = (
            f"{stem}\n"
            f"(A) {rotated_choices[0]} (B) {rotated_choices[1]} "
            f"(C) {rotated_choices[2]} (D) {rotated_choices[3]}"
        )
        
        var_item = dict(item)
        var_item["input"] = new_prompt
        var_item["target"] = new_gold_char
        var_item["rotation_shift"] = shift
        var_item["original_gold"] = gold
        variants.append(var_item)
        
    return variants


def test_rotated_mcqa_permutation():
    sample = {
        "example_id": "test_01",
        "input": "What was the capital in 1947? (A) Hanoi (B) Paris (C) Tokyo (D) Berlin",
        "target": "c",
    }
    variants = generate_rotated_mcqa_variants(sample)
    assert len(variants) == 4
    
    # Shift 0: original
    assert variants[0]["target"] == "c"
    assert "(C) Tokyo" in variants[0]["input"]
    
    # Shift 1: rotated by 1 -> Tokyo moves to (B)
    # choices: (A) Paris (B) Tokyo (C) Berlin (D) Hanoi
    assert variants[1]["target"] == "b"
    assert "(B) Tokyo" in variants[1]["input"]
    
    # Shift 2: Tokyo moves to (A)
    # choices: (A) Tokyo (B) Berlin (C) Hanoi (D) Paris
    assert variants[2]["target"] == "a"
    assert "(A) Tokyo" in variants[2]["input"]
    
    # Shift 3: Tokyo moves to (D)
    # choices: (A) Berlin (B) Hanoi (C) Paris (D) Tokyo
    assert variants[3]["target"] == "d"
    assert "(D) Tokyo" in variants[3]["input"]


def test_counterfactual_year_perturbation():
    base_year = 1947
    perturbed_year = 1947 + 37  # 1984
    
    base_vec = compute_fixed_fone_canchi_40d(base_year)
    perturbed_vec = compute_fixed_fone_canchi_40d(perturbed_year)
    
    assert len(base_vec) == 40
    assert len(perturbed_vec) == 40
    
    # Must be numerically different
    diff_norm = sum((a - b) ** 2 for a, b in zip(base_vec, perturbed_vec)) ** 0.5
    assert diff_norm > 0.5, f"Expected distinct vectors, got diff_norm={diff_norm}"


def test_assistant_trigger_token_resolution():
    # Simulate tokens of a ChatML prompt
    # ChatML format: <|im_start|>system\n...<|im_end|>\n<|im_start|>user\n...<|im_end|>\n<|im_start|>assistant\n
    dummy_tokens = [151644, 8948, 198] + [100] * 20 + [151645, 198, 151644, 872, 198] + [200] * 30 + [151645, 198, 151644, 77091, 198]
    
    # Find last occurrence of <|im_start|>assistant\n (151644, 77091, 198)
    trigger_idx = None
    for i in range(len(dummy_tokens) - 2):
        if dummy_tokens[i] == 151644 and dummy_tokens[i+1] == 77091 and dummy_tokens[i+2] == 198:
            trigger_idx = i + 2
            break
            
    assert trigger_idx is not None
    assert trigger_idx == len(dummy_tokens) - 1, "Assistant trigger should be the last token in prompt"
    assert trigger_idx != 14, "Assistant trigger is dynamic and significantly downstream from token index 14"
