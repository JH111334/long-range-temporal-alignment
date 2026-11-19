"""Build a deterministic non-temporal, non-MCQA general open-QA replay corpus from SQuAD v1.1 train.

Outputs:
- data/general_replay/g1_open_qa_train.jsonl.gz (20,000 article-disjoint records)
- data/general_replay/g1_open_qa_holdout.jsonl.gz (256 article-disjoint records)
- data/receipts/g1_general_replay_manifest.json (full provenance, SHA-256, and screening receipts)
"""

from __future__ import annotations

import gzip
import hashlib
import json
import random
import re
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Set

SQUAD_TRAIN_URL = "https://rajpurkar.github.io/SQuAD-explorer/dataset/train-v1.1.json"

YEAR_LIKE_PATTERN = re.compile(r"\b([1-9]\d{2,3})\b")
TEMPORAL_KEYWORD_PATTERN = re.compile(
    r"\b(year|years|century|centuries|decade|decades|era|bce|ce|ad|bc|month|months|"
    r"january|february|march|april|june|july|august|september|october|november|december|"
    r"when\s+did|what\s+year|how\s+long|how\s+many\s+years)\b",
    re.IGNORECASE,
)
OPTION_MARKER_PATTERN = re.compile(r"(?:\boptions\s*:|\([a-d]\))", re.IGNORECASE)
YES_NO_PATTERN = re.compile(r"^\s*(yes|no)\b[\.\!\?,]*\s*$", re.IGNORECASE)
SINGLE_LETTER_PATTERN = re.compile(r"^\s*[\(\[]?[a-zA-Z][\)\]\.\,:]*\s*$")
PURE_NUMERIC_PATTERN = re.compile(r"^\s*[-+]?\d+(?:[,\.]\d+)?%?\s*$")


def _norm_text(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower())


def _extract_context_window(context: str, answer_start: int, answer_text: str, max_chars: int = 260) -> str:
    """Extract a clean sentence-bounded context window <= max_chars preserving the full answer span."""
    ans_end = answer_start + len(answer_text)
    if ans_end > len(context) or context[answer_start:ans_end] != answer_text:
        return ""
    if len(context) <= max_chars:
        return context.strip()

    # Expand to sentence boundaries around [answer_start, ans_end) first
    sent_start = max(0, context.rfind(". ", 0, answer_start) + 2 if ". " in context[:answer_start] else 0)
    sent_end_idx = context.find(". ", ans_end)
    sent_end = sent_end_idx + 1 if sent_end_idx != -1 else len(context)
    candidate = context[sent_start:sent_end].strip()
    if len(candidate) <= max_chars and answer_text in candidate:
        return candidate

    # Deterministic centered character window around the answer span, snapped to word boundaries
    half = max(0, (max_chars - len(answer_text)) // 2)
    win_start = max(0, answer_start - half)
    win_end = min(len(context), win_start + max_chars)
    if win_end < ans_end:
        win_end = ans_end
        win_start = max(0, win_end - max_chars)
    if win_start > 0:
        space_pos = context.find(" ", win_start, answer_start)
        if space_pos != -1:
            win_start = space_pos + 1
    if win_end < len(context):
        space_pos = context.rfind(" ", ans_end, win_end)
        if space_pos != -1:
            win_end = space_pos
    snippet = context[win_start:win_end].strip()
    if answer_text not in snippet:
        return ""
    return snippet


def _load_tempun_screening_sets() -> tuple[Set[str], Set[str]]:
    """Collect prompt hashes and question substrings from TempUN train/eval for overlap screening."""
    hashes: Set[str] = set()
    substrings: Set[str] = set()
    for path_str in (
        "data/tempun_full_clean/train_104k_clean.jsonl.gz",
        "data/tempun_full_clean/eval_5292_clean.jsonl.gz",
    ):
        p = Path(path_str)
        if not p.is_file():
            raise FileNotFoundError(f"Required TempUN file missing for leakage screening: {p}")
        with gzip.open(p, "rt", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                rec = json.loads(line)
                q = str(rec.get("input", "")).strip()
                norm_q = _norm_text(q)
                hashes.add(hashlib.sha256(norm_q.encode("utf-8")).hexdigest())
                body = norm_q.split("options:")[0].strip()
                if len(body) >= 24:
                    substrings.add(body[:64])
    return hashes, substrings


def build_replay_corpus(
    *,
    cache_path: Path = Path("scratch/squad_train_v1_1.json"),
    out_train_path: Path = Path("data/general_replay/g1_open_qa_train.jsonl.gz"),
    out_holdout_path: Path = Path("data/general_replay/g1_open_qa_holdout.jsonl.gz"),
    out_manifest_path: Path = Path("data/receipts/g1_general_replay_manifest.json"),
    n_train: int = 20000,
    n_holdout: int = 256,
    seed: int = 42,
    max_qwen_tokens: int = 120,
    tokenizer_id: str = "Qwen/Qwen2.5-3B",
) -> Dict[str, Any]:
    from transformers import AutoTokenizer

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    if not cache_path.is_file():
        urllib.request.urlretrieve(SQUAD_TRAIN_URL, cache_path)
    raw_bytes = cache_path.read_bytes()
    upstream_sha256 = hashlib.sha256(raw_bytes).hexdigest()
    squad = json.loads(raw_bytes.decode("utf-8"))

    qwen_tok = AutoTokenizer.from_pretrained(tokenizer_id, use_fast=True)
    tempun_hashes, tempun_substrings = _load_tempun_screening_sets()

    articles_map: Dict[str, List[Dict[str, Any]]] = {}
    stats = {
        "total_raw_qas": 0,
        "rejected_empty_or_truncated": 0,
        "rejected_temporal": 0,
        "rejected_format_or_numeric": 0,
        "rejected_tempun_overlap": 0,
        "rejected_over_qwen_token_budget": 0,
        "admitted_clean": 0,
        "max_qwen_tokens_observed": 0,
    }

    for art in squad.get("data", []):
        title = str(art.get("title", "unknown"))
        for p_idx, para in enumerate(art.get("paragraphs", [])):
            ctx = str(para.get("context", ""))
            for qa in para.get("qas", []):
                stats["total_raw_qas"] += 1
                qa_id = str(qa.get("id", ""))
                question = str(qa.get("question", "")).strip()
                answers = qa.get("answers", [])
                if not question or not answers:
                    stats["rejected_empty_or_truncated"] += 1
                    continue
                ans0 = answers[0]
                answer_text = str(ans0.get("text", "")).strip()
                answer_start = int(ans0.get("answer_start", -1))
                if not answer_text or len(answer_text) > 50 or answer_start < 0:
                    stats["rejected_empty_or_truncated"] += 1
                    continue

                if (
                    YES_NO_PATTERN.match(answer_text)
                    or SINGLE_LETTER_PATTERN.match(answer_text)
                    or PURE_NUMERIC_PATTERN.match(answer_text)
                    or OPTION_MARKER_PATTERN.search(question)
                ):
                    stats["rejected_format_or_numeric"] += 1
                    continue

                ctx_window = _extract_context_window(ctx, answer_start, answer_text, max_chars=185)
                if not ctx_window or OPTION_MARKER_PATTERN.search(ctx_window):
                    stats["rejected_empty_or_truncated"] += 1
                    continue

                combined_check = f"{ctx_window}\n{question}\n{answer_text}"
                if not combined_check.isascii():
                    stats["rejected_empty_or_truncated"] += 1
                    continue
                if YEAR_LIKE_PATTERN.search(combined_check) or TEMPORAL_KEYWORD_PATTERN.search(combined_check):
                    stats["rejected_temporal"] += 1
                    continue

                prompt_input = f"Context: {ctx_window}\nQuestion: {question}"
                if len(prompt_input) > 240:
                    stats["rejected_empty_or_truncated"] += 1
                    continue

                norm_prompt = _norm_text(prompt_input)
                norm_q = _norm_text(question)
                if (
                    hashlib.sha256(norm_prompt.encode("utf-8")).hexdigest() in tempun_hashes
                    or hashlib.sha256(norm_q.encode("utf-8")).hexdigest() in tempun_hashes
                    or (len(norm_q) >= 24 and norm_q[:64] in tempun_substrings)
                ):
                    stats["rejected_tempun_overlap"] += 1
                    continue

                n_tok = len(qwen_tok(f"{prompt_input}\nAnswer: {answer_text}", add_special_tokens=False)["input_ids"])
                if n_tok > max_qwen_tokens:
                    stats["rejected_over_qwen_token_budget"] += 1
                    continue

                rec = {
                    "source_stream": "general_replay",
                    "source_id": f"squad11:{qa_id}",
                    "article_id": title,
                    "input": prompt_input,
                    "target": answer_text,
                }
                articles_map.setdefault(title, []).append(rec)
                stats["admitted_clean"] += 1
                if n_tok > stats["max_qwen_tokens_observed"]:
                    stats["max_qwen_tokens_observed"] = n_tok

    if stats["admitted_clean"] < n_train + n_holdout:
        raise RuntimeError(
            f"Insufficient clean SQuAD v1.1 records: got {stats['admitted_clean']} < {n_train + n_holdout}."
        )

    rng = random.Random(seed)
    article_titles = sorted(articles_map.keys())
    rng.shuffle(article_titles)

    holdout_records: List[Dict[str, Any]] = []
    train_pool: List[Dict[str, Any]] = []
    for title in article_titles:
        recs = articles_map[title]
        if len(holdout_records) < n_holdout:
            holdout_records.extend(recs)
        else:
            train_pool.extend(recs)

    holdout_selected = holdout_records[:n_holdout]
    rng.shuffle(train_pool)
    if len(train_pool) < n_train:
        raise RuntimeError(f"Insufficient article-disjoint train pool: {len(train_pool)} < {n_train}")
    train_selected = train_pool[:n_train]

    # Verify strict article disjointness between train and holdout
    train_articles = {r["article_id"] for r in train_selected}
    holdout_articles = {r["article_id"] for r in holdout_selected}
    assert not (train_articles & holdout_articles), "Article overlap between replay train and holdout!"

    out_train_path.parent.mkdir(parents=True, exist_ok=True)
    out_holdout_path.parent.mkdir(parents=True, exist_ok=True)
    out_manifest_path.parent.mkdir(parents=True, exist_ok=True)

    with gzip.open(out_train_path, "wt", encoding="utf-8") as f:
        for r in train_selected:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    with gzip.open(out_holdout_path, "wt", encoding="utf-8") as f:
        for r in holdout_selected:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    train_sha = hashlib.sha256(out_train_path.read_bytes()).hexdigest()
    holdout_sha = hashlib.sha256(out_holdout_path.read_bytes()).hexdigest()

    manifest = {
        "schema_version": "g1-general-replay-v1",
        "source_dataset": "SQuAD v1.1 train",
        "official_url": SQUAD_TRAIN_URL,
        "license": "CC BY-SA 4.0",
        "attribution": "Rajpurkar et al., SQuAD: 100,000+ Questions for Machine Comprehension of Text (EMNLP 2016).",
        "upstream_sha256": upstream_sha256,
        "seed": seed,
        "filtering_stats": stats,
        "split_disjointness": {
            "train_articles": len(train_articles),
            "holdout_articles": len(holdout_articles),
            "overlapping_articles": 0,
            "tempun_prompt_overlap": 0,
        },
        "artifacts": {
            "train": {
                "path": out_train_path.as_posix(),
                "count": len(train_selected),
                "sha256": train_sha,
            },
            "holdout": {
                "path": out_holdout_path.as_posix(),
                "count": len(holdout_selected),
                "sha256": holdout_sha,
            },
        },
    }
    out_manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return manifest


if __name__ == "__main__":
    res = build_replay_corpus()
    print(json.dumps(res, indent=2, ensure_ascii=False))
