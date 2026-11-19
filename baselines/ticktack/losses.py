"""
Loss Functions for Ticktack Baseline (Han et al., 2025).

Implements:
- Sexagenary Can Chi intra-class contrastive loss (Eq. 8):
  L_intra = 1 - (1 / |P|) * sum_{(i, j) in P} cos(h_i, h_j)
  where P = {(i, j) | tau(t_i) == tau(t_j), i != j} and tau(t) = (t - 3) % 60.
- Safe zero-pair fallback maintaining autograd graph.
- Total loss calculation combining Next-Token Prediction and Intra-loss.
"""

from typing import Union
import torch
import torch.nn.functional as F

def compute_intra_loss(hidden_states: torch.Tensor, years: Union[torch.Tensor, list, tuple]) -> torch.Tensor:
    """
    Eq. 8 intra-class contrastive loss clustering by Can Chi sexagenary cycle (Han et al., 2025).
    
    Pairs (i, j) with the same Can Chi cycle ((year - 3) % 60) are treated as positive pairs,
    maximizing cosine similarity between their representations.
    
    Args:
        hidden_states: Last hidden states or pooled representations [B, D] or [B, S, D].
        years: 1D tensor or iterable of Gregorian calendar years [B].
        
    Returns:
        Scalar tensor representing the intra loss. If no positive pairs exist in the batch,
        returns (hidden_states * 0.0).sum() to preserve gradient graph.
    """
    B = hidden_states.size(0)
    if B <= 1:
        return (hidden_states * 0.0).sum()

    if hidden_states.dim() == 3:
        # Sequence hidden states [B, S, D]: mean-pool along sequence dimension
        hidden_states = hidden_states.mean(dim=1)

    norm_h = F.normalize(hidden_states, p=2, dim=-1)
    sim_matrix = torch.mm(norm_h, norm_h.t())
    
    if not isinstance(years, torch.Tensor):
        years = torch.tensor(years, device=hidden_states.device)
    years = years.view(-1).to(device=hidden_states.device)
    
    canchi_cycles = ((years.to(torch.int64) - 3) % 60).to(device=hidden_states.device)
    same_cycle_mask = (canchi_cycles.unsqueeze(1) == canchi_cycles.unsqueeze(0)) & (~torch.eye(B, dtype=torch.bool, device=hidden_states.device))
    
    if same_cycle_mask.sum() > 0:
        return 1.0 - sim_matrix[same_cycle_mask].mean()
    else:
        return (hidden_states * 0.0).sum()

def compute_total_loss(
    ntp_loss: torch.Tensor,
    intra_loss: torch.Tensor,
    sigma: float = 0.5
) -> torch.Tensor:
    """
    Eq. 7 / Eq. 8 combined loss: L_total = L_NTP + sigma * L_intra.
    
    Args:
        ntp_loss: Next-token prediction cross-entropy loss.
        intra_loss: Can Chi intra-class contrastive loss.
        sigma: Balancing coefficient (default 0.5).
        
    Returns:
        Combined scalar loss tensor.
    """
    return ntp_loss + sigma * intra_loss
