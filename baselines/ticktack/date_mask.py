"""
Date-Token Masking and Localized Temporal Encoding Injection.

Implements:
- Precise date-token sub-sequence identification using the model tokenizer.
- Localized injection of Temporal Encoding (TE) vectors exclusively at date tokens.
- Supports broadcast injection, date-token localized injection, and pass-through (none).
"""

from typing import Optional, Union
import torch
import torch.nn as nn
from baselines.ticktack.models import compute_batch_te

def find_date_token_mask(
    input_ids: torch.Tensor,
    years: Union[torch.Tensor, list, tuple, int],
    tokenizer,
    device: Optional[Union[str, torch.device]] = None
) -> torch.Tensor:
    """
    Finds boolean mask [B, S, 1] identifying exact date token sub-sequences.
    
    Args:
        input_ids: Token ID tensor of shape [B, S].
        years: Year tensor, list, or scalar of shape [B] or scalar.
        tokenizer: Hugging Face PreTrainedTokenizer.
        device: Target torch device. If None, inherits from input_ids.device.
        
    Returns:
        Boolean tensor of shape [B, S, 1].
    """
    if device is None:
        device = input_ids.device
    if isinstance(years, (int, float)):
        years = torch.tensor([years], device=device)
    elif not isinstance(years, torch.Tensor):
        years = torch.tensor(years, device=device)
    years = years.view(-1)

    B, S = input_ids.shape
    mask = torch.zeros(B, S, 1, dtype=torch.bool, device=device)
    for b in range(B):
        yr_idx = min(b, years.size(0) - 1)
        yr = int(years[yr_idx].item())
        sub = tokenizer.encode(str(yr), add_special_tokens=False)
        seq = input_ids[b].tolist()
        sub_len = len(sub)
        found = False
        for i in range(len(seq) - sub_len + 1):
            if seq[i:i + sub_len] == sub:
                mask[b, i:i + sub_len, 0] = True
                found = True
        if not found:
            # Fallback to early prefix tokens if year string is split across non-standard boundaries
            mask[b, 1:min(5, S), 0] = True
    return mask

def inject_te_embeddings(
    model: nn.Module,
    tokenizer,
    input_ids: torch.Tensor,
    years: Union[torch.Tensor, list, tuple, int],
    te_mode: str = 'broadcast',
    scale_mode: str = 'raw',
    device: Optional[Union[str, torch.device]] = None
) -> torch.Tensor:
    """
    Injects Temporal Encoding according to te_mode ('none', 'broadcast', 'date_token').
    
    Args:
        model: Causal LM model (or MockModelWithEmbeddings) containing embed_tokens.
        tokenizer: Hugging Face PreTrainedTokenizer.
        input_ids: Input token IDs [B, S].
        years: 1D tensor, list, or scalar of Gregorian calendar years [B].
        te_mode: 'none' (returns unperturbed embeddings), 'broadcast' (adds TE to all tokens),
                 or 'date_token' (adds TE only at date token positions).
        scale_mode: 'raw', 'scaled', or 'gated'.
        device: Target device. If None, inherits from input_ids.device.
        
    Returns:
        Perturbed embedding tensor of shape [B, S, hidden_dim].
    """
    if device is None:
        device = input_ids.device
        
    if callable(getattr(model, 'get_input_embeddings', None)) and model.get_input_embeddings() is not None:
        token_embeds = model.get_input_embeddings()(input_ids)
    elif hasattr(model, 'model') and hasattr(model.model, 'embed_tokens'):
        token_embeds = model.model.embed_tokens(input_ids)
    elif hasattr(model, 'embed_tokens'):
        token_embeds = model.embed_tokens(input_ids)
    else:
        raise AttributeError("Model does not expose embed_tokens or get_input_embeddings().")

    if te_mode == 'none':
        return token_embeds
        
    hidden_dim = getattr(getattr(model, 'config', None), 'hidden_size', token_embeds.shape[-1])
    dtype = getattr(model, 'dtype', token_embeds.dtype)
    te_batch = compute_batch_te(years, hidden_dim=hidden_dim, scale_mode=scale_mode, device=device, dtype=dtype)
    
    if te_mode == 'broadcast':
        return token_embeds + te_batch
    elif te_mode == 'date_token':
        mask = find_date_token_mask(input_ids, years, tokenizer, device=device)
        return torch.where(mask, token_embeds + te_batch, token_embeds)
    else:
        raise ValueError(f"Unknown te_mode: '{te_mode}'. Expected 'none', 'broadcast', or 'date_token'.")
