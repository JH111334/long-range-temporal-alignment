"""Loss Functions for TickTack v3 Baseline.

Covers:
- Next-Token Prediction (NTP) loss with FP32 stability.
- Temporal Contrastive Alignment Loss (Eq. 8, delta=1.0 per Section 4.2).
- Elastic Weight Consolidation (EWC) penalty on trainable LoRA parameters (Eq. 9).
- Total multi-task objective (Eq. 10).
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def compute_ntp_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    ignore_index: int = -100,
) -> torch.Tensor:
    """Compute Next-Token Prediction loss with FP32 upcasting to prevent overflow.

    Args:
        logits: Tensor of shape (batch, seq_len, vocab_size).
        labels: Tensor of shape (batch, seq_len).
        ignore_index: Padding / prompt token index to ignore.

    Returns:
        Scalar Cross-Entropy loss in float32.
    """
    # Shift tokens so that tokens < n predict n
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()

    # Flatten and upcast to FP32
    flat_logits = shift_logits.view(-1, shift_logits.size(-1)).float()
    flat_labels = shift_labels.view(-1)

    loss = F.cross_entropy(flat_logits, flat_labels, ignore_index=ignore_index)
    return loss


def compute_temporal_contrastive_loss(
    hidden_states: torch.Tensor,
    year_labels: torch.Tensor,
    delta: float = 1.0,
    temperature: float = 0.07,
) -> Tuple[torch.Tensor, int]:
    """Compute Temporal Contrastive Alignment Loss per Eq. 8 of paper.

    L_T = delta * L_intra + (1 - delta) * L_inter
    In Section 4.2 of Xue Han et al., delta = 1.0, so L_inter = 0.

    Args:
        hidden_states: Tensor of shape (batch_size, hidden_dim).
        year_labels: 1D Tensor of shape (batch_size,) with year integer values.
        delta: Weight balance between intra and inter class loss (paper fixed at 1.0).
        temperature: Scaling factor for cosine similarity.

    Returns:
        Tuple of (scalar_loss, num_positive_pairs).
    """
    batch_size = hidden_states.shape[0]
    if batch_size < 2:
        return torch.tensor(0.0, device=hidden_states.device, dtype=torch.float32), 0

    # Normalize vectors to unit hypersphere
    h_norm = F.normalize(hidden_states.float(), p=2, dim=-1)
    sim = torch.matmul(h_norm, h_norm.T) / temperature

    # Mask self-similarity
    diag_mask = torch.eye(batch_size, dtype=torch.bool, device=hidden_states.device)

    # Positive pair mask: same year, different index
    pos_mask = (year_labels.unsqueeze(0) == year_labels.unsqueeze(1)) & (~diag_mask)
    num_positive_pairs = int(pos_mask.sum().item())

    if num_positive_pairs == 0:
        return torch.tensor(0.0, device=hidden_states.device, dtype=torch.float32), 0

    # Numerical stability: subtract row-wise max
    sim_max, _ = torch.max(sim, dim=1, keepdim=True)
    exp_sim = torch.exp(sim - sim_max.detach()) * (~diag_mask).float()

    # Sum of exp similarities over all negatives and positives (excluding self)
    denom = exp_sim.sum(dim=1, keepdim=True) + 1e-12

    # Log probabilities
    log_probs = (sim - sim_max.detach()) - torch.log(denom)

    # Average over positive pairs per sample where positive pairs exist
    pos_counts_per_sample = pos_mask.sum(dim=1)
    valid_samples = pos_counts_per_sample > 0

    if not valid_samples.any():
        return torch.tensor(0.0, device=hidden_states.device, dtype=torch.float32), 0

    sample_losses = (log_probs * pos_mask.float()).sum(dim=1) / pos_counts_per_sample.clamp(min=1).float()
    intra_loss = -sample_losses[valid_samples].mean()

    # When delta == 1.0, L_inter is zero
    if abs(delta - 1.0) < 1e-6:
        return intra_loss, num_positive_pairs

    # Inter-class negative loss when delta < 1.0
    neg_mask = (year_labels.unsqueeze(0) != year_labels.unsqueeze(1)) & (~diag_mask)
    num_neg = int(neg_mask.sum().item())
    if num_neg > 0:
        inter_loss = (torch.exp(sim) * neg_mask.float()).sum(dim=1).clamp(min=1e-12).log().mean()
    else:
        inter_loss = torch.tensor(0.0, device=hidden_states.device, dtype=torch.float32)

    total_t_loss = delta * intra_loss + (1.0 - delta) * inter_loss
    return total_t_loss, num_positive_pairs


class LoRAEWCLoss(nn.Module):
    """Elastic Weight Consolidation (EWC) penalty on trainable LoRA parameters.

    Eq. 9:
        L_EWC = (lambda / 2) * sum_j F_j * (phi_j - phi_j^*)^2

    Applies strictly to parameters with requires_grad=True (trainable LoRA matrices).
    Backbone parameters with requires_grad=False naturally produce zero gradient penalty.
    """

    def __init__(self, lambda_ewc: float = 100.0) -> None:
        super().__init__()
        self.lambda_ewc = lambda_ewc
        self.reference_weights: Dict[str, torch.Tensor] = {}
        self.fisher_matrix: Dict[str, torch.Tensor] = {}
        self.is_active: bool = False

    def update_task_fisher(
        self,
        model: nn.Module,
        fisher_dict: Dict[str, torch.Tensor],
    ) -> None:
        """Register reference weights and accumulated Fisher matrix from completed stage."""
        self.reference_weights.clear()
        self.fisher_matrix.clear()

        for name, param in model.named_parameters():
            if param.requires_grad and name in fisher_dict:
                self.reference_weights[name] = param.detach().clone().to(param.device)
                self.fisher_matrix[name] = fisher_dict[name].detach().clone().to(param.device)

        self.is_active = len(self.fisher_matrix) > 0

    def forward(self, model: nn.Module) -> Tuple[torch.Tensor, int]:
        """Compute quadratic parameter shift penalty.

        Returns:
            Tuple of (scalar_ewc_loss, active_param_count).
        """
        if not self.is_active:
            device = next(model.parameters()).device
            return torch.tensor(0.0, device=device, dtype=torch.float32), 0

        total_penalty = torch.tensor(0.0, device=next(model.parameters()).device, dtype=torch.float32)
        count = 0

        for name, param in model.named_parameters():
            if name in self.fisher_matrix and name in self.reference_weights:
                f_diag = self.fisher_matrix[name]
                p_ref = self.reference_weights[name]
                diff = (param - p_ref).float()
                penalty = (f_diag * (diff ** 2)).sum()
                total_penalty = total_penalty + penalty
                count += param.numel()

        ewc_loss = 0.5 * self.lambda_ewc * total_penalty
        return ewc_loss, count
