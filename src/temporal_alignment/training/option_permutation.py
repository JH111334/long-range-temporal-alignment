"""Online Option Permutation for MCQA Debias and Position Invariance.

Implements reversible bijection pi: Letters -> Letters and inverse pi_inv
to eliminate positional shortcuts (e.g. letter 'b' collapse) in MCQA training.
Default-OFF flag ensures fail-closed backward compatibility.
"""

from __future__ import annotations

import random
from typing import Any, Dict, List, Optional, Tuple


class OnlineOptionPermutation:
    """Handles online permutation of MCQA options during training batch construction."""

    def __init__(self, enabled: bool = False, seed: Optional[int] = None):
        self.enabled = enabled
        self._rng = random.Random(seed) if seed is not None else random.Random()

    def set_enabled(self, enabled: bool) -> None:
        """Toggle permutation on or off."""
        self.enabled = enabled

    def permute_mcqa_options(
        self,
        options: Dict[str, str],
        correct_letter: str,
        seed: Optional[int] = None,
    ) -> Tuple[Dict[str, str], str, Dict[str, str], Dict[str, str]]:
        """Permute options dictionary and return new options, new correct letter, forward and inverse maps.

        Args:
            options: Dict mapping label (e.g. 'A', 'B', 'C', 'D') to option text.
            correct_letter: The original gold answer letter.
            seed: Optional local seed for deterministic permutation.

        Returns:
            permuted_options: Dict mapping same labels to permuted option texts.
            new_correct_letter: The new label corresponding to the correct answer.
            forward_map: Mapping from original label to new label.
            inverse_map: Mapping from new label back to original label.
        """
        if not self.enabled:
            # Identity mapping
            fwd = {k: k for k in options.keys()}
            inv = {k: k for k in options.keys()}
            return dict(options), correct_letter, fwd, inv

        rng = random.Random(seed) if seed is not None else self._rng

        labels = sorted(options.keys())
        shuffled_labels = list(labels)
        rng.shuffle(shuffled_labels)

        # forward_map[old_label] = new_label
        forward_map = dict(zip(labels, shuffled_labels))
        # inverse_map[new_label] = old_label
        inverse_map = {new_l: old_l for old_l, new_l in forward_map.items()}

        # Verify bijection
        assert len(forward_map) == len(labels)
        assert len(inverse_map) == len(labels)
        for l in labels:
            assert inverse_map[forward_map[l]] == l

        # Construct new options where permuted_options[new_l] = original_text_of[old_l]
        permuted_options = {}
        for old_l, text in options.items():
            new_l = forward_map[old_l]
            permuted_options[new_l] = text

        new_correct_letter = forward_map[correct_letter]

        # Verify target option text unchanged
        assert permuted_options[new_correct_letter] == options[correct_letter]

        return permuted_options, new_correct_letter, forward_map, inverse_map
