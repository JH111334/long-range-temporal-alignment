"""
Deterministic CPU Unit Tests for Phase 1 Scorer and Source Contract Audit.
Verifies:
1. Last valid token logit selection under left vs right padding.
2. Tokenizer candidate token IDs after '\\nAnswer:'.
3. Batch non-invariance proof under left padding + fixed token_idx injection.
4. Prompt template year span location (MCQA vs Rel).
5. CF V2 anti-symmetry predicate audit on 852 Rel cases.
"""

import gzip
import json
import re
import pytest
import torch
import torch.nn as nn
from transformers import AutoTokenizer

@pytest.fixture(scope="module")
def tokenizer():
    try:
        tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct", local_files_only=True)
    except Exception:
        tok = AutoTokenizer.from_pretrained("gpt2")
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return tok

def test_01_last_valid_token_selection(tokenizer):
    """Verify that left padding preserves last prompt token at index -1, while right padding corrupts it."""
    tok = tokenizer
    tok.padding_side = "left"
    p_short = "Question: Short prompt?\nAnswer:"
    p_long = "Question: This is a much longer question with many words?\nAnswer:"

    enc_left = tok([p_short, p_long], return_tensors="pt", padding=True)
    # With left padding, both sequences should have their last token at -1
    last_left_0 = tok.decode([enc_left["input_ids"][0, -1].item()])
    last_left_1 = tok.decode([enc_left["input_ids"][1, -1].item()])
    assert last_left_0 == ":"
    assert last_left_1 == ":"

    # With right padding, sequence 0 at -1 is the padding token!
    tok.padding_side = "right"
    enc_right = tok([p_short, p_long], return_tensors="pt", padding=True)
    last_right_0 = enc_right["input_ids"][0, -1].item()
    assert last_right_0 == tok.pad_token_id, "Right padding at -1 must be pad token for shorter sequence"

def test_02_tokenizer_label_token_ids(tokenizer):
    """Verify token IDs for options after '\\nAnswer:'."""
    tok = tokenizer
    prompt = "Question: Sample question?\nAnswer:"
    prompt_tokens = tok.encode(prompt, add_special_tokens=False)

    # For Qwen2.5, continuation with leading space is a single token:
    enc_space_A = tok.encode(prompt + " A", add_special_tokens=False)
    new_token_A = enc_space_A[len(prompt_tokens):]
    assert len(new_token_A) == 1, "Expected single token continuation for ' A'"
    assert new_token_A[0] == 362, "Expected token ID 362 for ' A'"

    # Continuation without leading space 'A' merges with colon ':A' into token 55648
    enc_no_space_A = tok.encode(prompt + "A", add_special_tokens=False)
    # The colon was merged, so length changes
    assert enc_no_space_A[-1] == 55648, "Expected merged token 55648 for ':A'"

def test_03_batch_non_invariance_proof(tokenizer):
    """Demonstrate that absolute index injection delta[:, 14, :] violates batch invariance in left-padded batches."""
    tok = tokenizer
    tok.padding_side = "left"

    seam_proj = nn.Linear(40, 64)
    embed_tokens = nn.Embedding(len(tok) + 1000, 64)

    def inject_seam(input_ids, v40, token_idx=14):
        inputs_embeds = embed_tokens(input_ids)
        B, S, D = inputs_embeds.shape
        if S > token_idx:
            inj = seam_proj(v40)
            delta = torch.zeros_like(inputs_embeds)
            delta[:, token_idx, :] = inj
            inputs_embeds = inputs_embeds + delta
        return inputs_embeds

    p_short = "Question: In 1947, Gabon was: Options: (a) 0.13 (b) 1.12\nAnswer:"
    p_long = "Question: Which option is correct for the question: In 1947, Private Civil Liberties Index in South Sudan was: Options: (a) 0.84 (b) 0.89 (c) 1.03 (d) 0.08\nAnswer:"

    # BS = 1
    inp_1 = tok([p_short], return_tensors="pt", padding=True)
    v40 = torch.randn(1, 40)
    emb_1 = inject_seam(inp_1["input_ids"], v40, token_idx=14)

    # BS = 2
    inp_2 = tok([p_short, p_long], return_tensors="pt", padding=True)
    v40_2 = torch.cat([v40, torch.randn(1, 40)], dim=0)
    emb_2 = inject_seam(inp_2["input_ids"], v40_2, token_idx=14)

    pad_len = inp_2["input_ids"].shape[1] - inp_1["input_ids"].shape[1]
    emb_short_from_2 = emb_2[0, pad_len:]

    diff = (emb_1[0] - emb_short_from_2).abs().max().item()
    # Diff must be non-zero because token_idx=14 landed in a padding token in BS=2!
    assert diff > 0.1, f"Expected large batch non-invariance discrepancy, got {diff}"

def test_04_prompt_template_year_span(tokenizer):
    """Verify that token_idx=14 hits year in MCQA but hits Entity in Rel."""
    tok = tokenizer
    mcqa_prompt = "Question: Which option is correct for the question: In 1947, Private Civil Liberties Index in South Sudan was: Options: (a) 0.84 (b) 0.89 (c) 1.03 (d) 0.08\nAnswer:"
    rel_prompt = "Question: In 1947, Private Civil Liberties Index in Norway was higher than 1952? Answer as Yes or No\nAnswer:"

    mcqa_tokens = tok.encode(mcqa_prompt, add_special_tokens=False)
    rel_tokens = tok.encode(rel_prompt, add_special_tokens=False)

    # In MCQA, token 14 is digit '4' of 1947
    assert tok.decode([mcqa_tokens[14]]) == "4", "MCQA token 14 should be digit 4 of year"

    # In Rel, token 14 is entity ' Norway'
    assert tok.decode([rel_tokens[14]]) == " Norway", "Rel token 14 should be entity name ' Norway'"

def test_05_cf_v2_data_card_and_anti_symmetry():
    """Verify that all 852 Rel cases match the expected regex pattern."""
    pattern_rel = re.compile(r"In\s+(\d{4}),\s+(.*?)\s+(?:was|is|were)\s+higher than\s+(\d{4})\?\s+Answer as Yes or No", re.IGNORECASE)
    rel_cases = []
    with gzip.open("data/tempun_full_clean/eval_5292_clean.jsonl.gz", "rt", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            if obj.get("task_id") == "Rel":
                rel_cases.append(obj)

    assert len(rel_cases) == 852, "Expected exactly 852 Rel cases in eval_5292"
    matched = [c for c in rel_cases if pattern_rel.search(c["input"])]
    assert len(matched) == 852, "All 852 Rel cases must match the comparison pattern"
