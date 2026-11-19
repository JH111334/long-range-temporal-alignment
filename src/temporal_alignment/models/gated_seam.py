"""Safe Conditioning Seam: Gated Residual Continuous Injection.

Contract Specification:
- Input feature: FIXED_FONE_CANCHI_V1 40D
- Projection: Linear(40, hidden_dim) -> LayerNorm(hidden_dim, bias=False)
- Gating parameter: alpha initialized strictly to 0.0
- Seam injection position: token_index = 14 (safe post-system header position)
- Virtual tokens inserted: 0 (avoids ChatML token-0 corruption)
"""

from __future__ import annotations

from typing import Any, Optional

try:
    import torch
    import torch.nn as nn
    _ModuleBase = nn.Module
except ImportError:
    torch = None
    nn = None
    _ModuleBase = object


class GatedResidualProjection(_ModuleBase):
    """Virtual temporal token projection with zero-initialized learnable gate and LayerNorm."""

    def __init__(self, in_features: int = 40, out_features: int = 896) -> None:
        if nn is None:
            raise RuntimeError("PyTorch is required for GatedResidualProjection")
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.linear = nn.Linear(in_features, out_features)
        self.norm = nn.LayerNorm(out_features, bias=False)
        self.alpha = nn.Parameter(torch.zeros(1))
        nn.init.normal_(self.linear.weight, mean=0.0, std=0.01)
        nn.init.zeros_(self.linear.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Project 40D feature to hidden_dim with learnable zero-init gate.
        
        Args:
            x: Tensor of shape (..., 40)
            
        Returns:
            Tensor of shape (..., 896)
        """
        proj = self.linear(x)
        normed = self.norm(proj)
        return self.alpha * normed


class ConditioningContext:
    """Manages active temporal vector and injection state during forward pass."""

    def __init__(self, injection_token_index: int = 14) -> None:
        self.current_vector: Optional[torch.Tensor] = None
        self.injection_token_index = injection_token_index
        self.enabled: bool = True

    def set_enabled(self, enabled: bool) -> None:
        """Explicitly enable or disable seam injection."""
        self.enabled = bool(enabled)

    def set_vector(self, vector: Optional[torch.Tensor]) -> None:
        self.current_vector = vector

    def clear(self) -> None:
        self.current_vector = None


def inject_seam_embedding_tensor(
    inputs_embeds: torch.Tensor,
    seam_vector: torch.Tensor,
    token_index: int = 14,
) -> torch.Tensor:
    """Safely inject temporal seam vector into inputs_embeds using out-of-place addition.

    Guarantees autograd graph integrity, avoiding in-place slice mutation errors.
    """
    if inputs_embeds.shape[1] <= token_index:
        return inputs_embeds
    delta = torch.zeros_like(inputs_embeds)
    delta[:, token_index, :] = seam_vector.to(
        device=inputs_embeds.device, dtype=inputs_embeds.dtype
    )
    return inputs_embeds + delta


def make_gated_embedding_hook(context: ConditioningContext):
    """Factory creating PyTorch forward hook for embedding layer."""
    def hook(module: Any, input_tensors: Any, output: Any) -> Any:
        if not getattr(context, "enabled", True) or context.current_vector is None:
            return output
        # Inject only during prefill / sequence encoding (seq_len > 1)
        if output.shape[1] > 1:
            vec = context.current_vector.to(dtype=output.dtype, device=output.device)
            if vec.dim() == 1:
                vec = vec.unsqueeze(0)
            output = output.clone()
            pos = min(output.shape[1] - 1, context.injection_token_index)
            output[:, pos, :] = output[:, pos, :] + vec
        return output
    return hook


import contextlib


@contextlib.contextmanager
def scoped_conditioning_context(
    model: Any,
    context: ConditioningContext,
    vector: Optional[torch.Tensor] = None,
    enabled: bool = True,
    injection_token_index: Optional[int] = None,
):
    """Context manager for registering, configuring, and strictly cleaning up the gated seam hook.

    Guarantees that the registered hook handle is removed upon context exit or exception.
    """
    old_vector = context.current_vector
    old_enabled = context.enabled
    old_pos = context.injection_token_index

    if injection_token_index is not None:
        context.injection_token_index = injection_token_index
    context.enabled = enabled
    context.set_vector(vector)

    hook = make_gated_embedding_hook(context)
    handle = model.get_input_embeddings().register_forward_hook(hook)
    try:
        yield handle
    finally:
        handle.remove()
        context.current_vector = old_vector
        context.enabled = old_enabled
        context.injection_token_index = old_pos

