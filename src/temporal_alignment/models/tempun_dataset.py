"""TempUN Small Dataset Parsing and Tokenization for Continual Temporal Alignment.

Contract Specification:
- Task: TempUN Small Continual Learning
- Train: train_nwp.jsonl (2,461 examples across year:1947, year:1984, year:2022)
- Eval: eval_mcqa.jsonl (180 MCQA examples: 156 4-choice ABCD + 24 Yes/No)
- Loss Mode: MASKED_TARGET (labels: -100 on prompt, real tokens on target + EOS)
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, List, Optional, Tuple


STAGES = ("year:1947", "year:1984", "year:2022")
STAGE_YEARS = {
    "year:1947": 1947,
    "year:1984": 1984,
    "year:2022": 2022,
}


@dataclass(frozen=True)
class TempUNExample:
    example_id: str
    task_id: str
    stage_id: str
    year: int
    prompt: str
    target: str


def extract_year_from_record(row: dict[str, Any]) -> int:
    """Extract canonical or effective year from a record."""
    cond = row.get("conditioning")
    if isinstance(cond, dict):
        if cond.get("effective_year") is not None:
            return int(cond["effective_year"])
        if cond.get("canonical_year") is not None:
            return int(cond["canonical_year"])
        if cond.get("numeric_value") is not None:
            return int(cond["numeric_value"])
    if row.get("canonical_year") is not None:
        return int(row["canonical_year"])
    stage_id = row.get("stage_id", "")
    if stage_id.startswith("year:"):
        try:
            return int(stage_id.split("year:")[1])
        except (ValueError, IndexError):
            pass
    # Fallback search in input text
    match = re.search(r"\b(19\d\d|20\d\d)\b", row.get("input", ""))
    if match:
        return int(match.group(1))
    return 1900


def load_tempun_train_records(path: Path | str) -> List[TempUNExample]:
    """Load train_nwp.jsonl records."""
    records: List[TempUNExample] = []
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    for line in lines:
        if not line.strip():
            continue
        row = json.loads(line)
        year = extract_year_from_record(row)
        records.append(
            TempUNExample(
                example_id=row["example_id"],
                task_id=row.get("task_id", "NWP"),
                stage_id=row.get("stage_id", f"year:{year}"),
                year=year,
                prompt=row["input"],
                target=row["target"],
            )
        )
    return records


def load_tempun_eval_records(path: Path | str) -> List[TempUNExample]:
    """Load eval_mcqa.jsonl records."""
    records: List[TempUNExample] = []
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    for line in lines:
        if not line.strip():
            continue
        row = json.loads(line)
        year = extract_year_from_record(row)
        records.append(
            TempUNExample(
                example_id=row["example_id"],
                task_id=row.get("task_id", "MCQA"),
                stage_id=row.get("stage_id", f"year:{year}"),
                year=year,
                prompt=row["input"],
                target=row["target"],
            )
        )
    return records


def tokenize_masked_target_example(
    example: TempUNExample,
    tokenizer: Any,
    max_length: int = 512,
) -> Tuple[List[int], List[int]]:
    """Tokenize example for MASKED_TARGET training.
    
    Returns:
        input_ids: Full sequence ids (prompt + target + eos)
        labels: -100 on prompt tokens, target token ids on target + eos
    """
    prompt_ids = tokenizer(example.prompt, add_special_tokens=True)["input_ids"]
    target_ids = tokenizer(example.target, add_special_tokens=False)["input_ids"]
    eos_id = tokenizer.eos_token_id

    full_ids = prompt_ids + target_ids + [eos_id]
    if len(full_ids) > max_length:
        excess = len(full_ids) - max_length
        prompt_ids = prompt_ids[excess:]
        full_ids = prompt_ids + target_ids + [eos_id]

    labels = ([-100] * len(prompt_ids)) + target_ids + [eos_id]
    return full_ids, labels


def parse_strict_mcqa_answer(prediction: str) -> Optional[str]:
    """Strict evaluation parser for MCQA completion aligned with eval_mcqa.jsonl gold targets.
    
    Handles:
    1. Parenthesized letter: (a), (b), (c), (d)
    2. Prefix options: Option: a, Answer: b, Choice (c)
    3. Direct single letter or letter with delimiter: a., b:, bare a (case-insensitive, normalized to lowercase)
    4. Direct Yes/No (case-insensitive, normalized to Title case)
    """
    cleaned = prediction.strip()
    if not cleaned:
        return None

    # Check for Yes / No first
    lower = cleaned.lower()
    m_yn = re.search(r"\b(yes|no)\b", lower)
    if m_yn and len(cleaned.split()) <= 4:
        return m_yn.group(1).capitalize()

    # 1. Match parenthesized letter like '(b)' or '[c]'
    m = re.search(r"[\(\[]([a-d])[\)\]]", cleaned, re.IGNORECASE)
    if m:
        return m.group(1).lower()

    # 2. Match option prefixes like 'Option: b', 'Answer: c', 'Choice (a)'
    m = re.search(r"(?:option|answer|choice|select)\s*[:\-]?\s*\(?([a-d])\)?\b", cleaned, re.IGNORECASE)
    if m:
        return m.group(1).lower()

    # 3. Match text starting with single letter e.g. 'a.', 'b:', bare 'a', 'A'
    stripped = cleaned.strip(".()[]: \n\t")
    if stripped.lower() in ("a", "b", "c", "d"):
        return stripped.lower()

    m = re.match(r"^([a-d])[\s\.\:\,\)]", cleaned, re.IGNORECASE)
    if m:
        return m.group(1).lower()

    # 4. Match isolated word-boundary letter in short responses (<= 5 words)
    m = re.search(r"\b([a-d])\b", cleaned, re.IGNORECASE)
    if m and len(cleaned.split()) <= 5:
        return m.group(1).lower()

    return None
