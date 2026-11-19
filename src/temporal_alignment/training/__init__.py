"""Training contracts and routines for temporal alignment."""

from .cpt_optimization import (
    build_cpt_optimizer_param_groups,
    create_cpt_lr_scheduler,
    interleave_general_replay,
)
from .prompt_contract import (
    PromptProtocol,
    detect_is_yes_no,
    render_prompt,
    tokenize_sft_target,
)

__all__ = [
    "PromptProtocol",
    "build_cpt_optimizer_param_groups",
    "create_cpt_lr_scheduler",
    "detect_is_yes_no",
    "interleave_general_replay",
    "render_prompt",
    "tokenize_sft_target",
]
