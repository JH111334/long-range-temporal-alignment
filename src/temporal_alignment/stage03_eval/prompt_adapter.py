"""Deterministic prompt builder and canonical option rotator for Stage 03 DEV benchmark.

Guarantees:
1. Exact reproduction of frozen DEV evaluation prompt template.
2. Canonical option rotations (4-cycle for MCQA, 2-fold inversion for Rel).
3. Gold target is NEVER embedded into the formatted prompt text.
4. Deterministic SHA-256 prompt hashing for provenance tracking.
5. Structured extraction of textual option values for secondary diagnostic parsing.
"""

from __future__ import annotations

import hashlib
import re
from typing import Any, Dict, List, Optional, Tuple


MCQA_PATTERN = re.compile(
    r"^(.*Options:\s*)\(a\)\s*(.*?)\s*\(b\)\s*(.*?)\s*\(c\)\s*(.*?)\s*\(d\)\s*(.*?)$",
    re.DOTALL,
)
ANSWER_SUFFIX_PATTERN = re.compile(r"(\s*\n?\s*Answer:\s*)$", re.IGNORECASE)


def format_inference_prompt(raw_prompt: str) -> str:
    """Format raw input prompt into canonical 'Question: ...\\nAnswer:' format without duplicates."""
    cleaned = raw_prompt.strip()
    if cleaned.endswith("Answer:"):
        return cleaned
    return f"Question: {cleaned}\nAnswer:"


def extract_mcqa_option_texts(raw_input: str) -> Optional[Dict[str, str]]:
    """Extract textual contents of options (a), (b), (c), (d) from prompt if present."""
    base_prompt = ANSWER_SUFFIX_PATTERN.sub("", raw_input.strip())
    match = MCQA_PATTERN.search(base_prompt)
    if not match:
        return None
    return {
        "a": match.group(2).strip(),
        "b": match.group(3).strip(),
        "c": match.group(4).strip(),
        "d": match.group(5).strip(),
    }


def generate_audited_permutations(
    raw_input: str,
    gold_target: str,
    task_id: str = "MCQ",
) -> List[Dict[str, Any]]:
    """Generate canonical audited rotations for a given case.

    For MCQA: generates 4 cyclic permutations (k=0, 1, 2, 3), keeping option slots (a)..(d)
    and rotating the option candidate strings while mapping the target letter accordingly.
    For Rel: generates 2 permutations: original ('Answer as Yes or No') and inverted ('Answer as No or Yes').
    """
    tgt = gold_target.strip().lower()

    # Binary Yes/No task (e.g. Rel)
    if tgt in ["yes", "no"] or task_id == "Rel":
        p0_raw = raw_input.strip()
        p1_raw = re.sub(r"Answer as Yes or No", "Answer as No or Yes", p0_raw, flags=re.IGNORECASE)
        if p1_raw == p0_raw:
            p1_raw = re.sub(r"Yes\s*(or|/)\s*No", "No or Yes", p0_raw, flags=re.IGNORECASE)

        variants = [(0, p0_raw, tgt), (1, p1_raw, tgt)]
        results = []
        for rot_id, p_str, target_val in variants:
            formatted = format_inference_prompt(p_str)
            p_hash = hashlib.sha256(formatted.encode("utf-8")).hexdigest()
            results.append({
                "rotation_id": rot_id,
                "formatted_prompt": formatted,
                "target": target_val,
                "prompt_hash": p_hash,
                "option_texts": {"yes": "Yes", "no": "No"},
            })
        return results

    # 4-choice MCQA task
    suffix = ""
    m_suffix = ANSWER_SUFFIX_PATTERN.search(raw_input)
    if m_suffix:
        suffix = m_suffix.group(1)
        base_prompt = raw_input[: len(raw_input) - len(suffix)].strip()
    else:
        base_prompt = raw_input.strip()

    m = MCQA_PATTERN.search(base_prompt)
    letters = ["a", "b", "c", "d"]
    if not m or tgt not in letters:
        formatted = format_inference_prompt(raw_input)
        p_hash = hashlib.sha256(formatted.encode("utf-8")).hexdigest()
        return [{
            "rotation_id": 0,
            "formatted_prompt": formatted,
            "target": tgt,
            "prompt_hash": p_hash,
            "option_texts": extract_mcqa_option_texts(raw_input) or {},
        }]

    prefix_str = m.group(1)
    opts = [m.group(2).strip(), m.group(3).strip(), m.group(4).strip(), m.group(5).strip()]
    t_idx = letters.index(tgt)

    results = []
    for k in range(4):
        r_opts = [opts[(j + k) % 4] for j in range(4)]
        r_tgt = letters[(t_idx - k) % 4]
        perm_prompt_raw = f"{prefix_str}(a) {r_opts[0]} (b) {r_opts[1]} (c) {r_opts[2]} (d) {r_opts[3]}{suffix}"
        formatted = format_inference_prompt(perm_prompt_raw)
        p_hash = hashlib.sha256(formatted.encode("utf-8")).hexdigest()
        results.append({
            "rotation_id": k,
            "formatted_prompt": formatted,
            "target": r_tgt,
            "prompt_hash": p_hash,
            "option_texts": {letters[i]: r_opts[i] for i in range(4)},
        })

    return results
