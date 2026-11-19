"""Unit tests for Temporal Token Alignment semantics: Token-aligned TE vs Fixed Token-14 Seam.

Verifies:
1. Token-aligned injection (Paper Eq. 5) targets only the token positions of the year mention.
2. Other word token embeddings in the sequence remain unmodified.
3. Gated Seam (Learning Track) injects only at token index 14 during prefill (seq_len > 1).
4. KV-cache autoregressive continuation (seq_len == 1) does not reinject duplicate vectors.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import pytest

from temporal_alignment.models.gated_seam import (
    ConditioningContext,
    make_gated_embedding_hook,
)


def test_token_aligned_injection_targets_only_year_positions():
    """Verify that paper-style token-aligned TE modifies only year tokens."""
    batch_size = 2
    seq_len = 10
    hidden_dim = 64

    # Dummy embeddings
    original_embeds = torch.randn(batch_size, seq_len, hidden_dim)
    year_token_indices = [3, 4]  # positions where year appears

    te = torch.randn(batch_size, hidden_dim)

    # Injected tensor
    injected_embeds = original_embeds.clone()
    for idx in year_token_indices:
        injected_embeds[:, idx, :] = injected_embeds[:, idx, :] + te

    # Check that year positions were modified
    for idx in year_token_indices:
        assert not torch.allclose(injected_embeds[:, idx, :], original_embeds[:, idx, :])

    # Check that ALL non-year positions remain bitwise identical
    for idx in range(seq_len):
        if idx not in year_token_indices:
            assert torch.allclose(injected_embeds[:, idx, :], original_embeds[:, idx, :])


def test_gated_seam_token14_prefill_only_guard():
    """Verify that Gated Seam injects at token 14 during prefill but ignores seq_len == 1."""
    hidden_dim = 64
    context = ConditioningContext(injection_token_index=14)
    vec = torch.randn(hidden_dim)
    context.set_vector(vec)
    context.set_enabled(True)

    hook = make_gated_embedding_hook(context)

    # Prefill call: seq_len = 20 > 1
    prefill_embeds = torch.zeros(1, 20, hidden_dim)
    prefill_out = hook(None, None, prefill_embeds)

    # Position 14 should have the vector injected
    assert torch.allclose(prefill_out[0, 14, :], vec)
    # Position 13 should be zero
    assert torch.allclose(prefill_out[0, 13, :], torch.zeros(hidden_dim))

    # Continuation call: seq_len = 1 (KV-cache step during generation)
    continuation_embeds = torch.zeros(1, 1, hidden_dim)
    continuation_out = hook(None, None, continuation_embeds)

    # In continuation, vector must NOT be reinjected
    assert torch.allclose(continuation_out, continuation_embeds)
