"""Regression tests for canonical M0-FC40 loader preventing silent bypass."""

from pathlib import Path
import tempfile
import pytest
import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModelForCausalLM

from temporal_alignment.models.m0_fc40_loader import load_m0_fc40_for_inference

_LOCAL_SNAPSHOT = (
    Path.home()
    / ".cache/huggingface/hub/models--Qwen--Qwen2.5-0.5B-Instruct/snapshots/7ae557604adf67be50417f59c2c2f167def9a775"
)


class DummyLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.mlp = nn.Module()
        self.mlp.gate_proj = nn.Linear(896, 4864, bias=False)
        self.mlp.up_proj = nn.Linear(896, 4864, bias=False)
        self.mlp.down_proj = nn.Linear(4864, 896, bias=False)


class DummyQwenForTest(nn.Module):
    """Lightweight dummy Qwen with 24 layers of MLP projections and embedding layer."""
    def __init__(self):
        super().__init__()
        self.config = nn.Module()
        self.config.hidden_size = 896
        self.embed_tokens = nn.Embedding(100, 896)
        self.model = nn.Module()
        self.model.layers = nn.ModuleList([DummyLayer() for _ in range(24)])

    def get_input_embeddings(self):
        return self.embed_tokens


def test_loader_fails_on_missing_file():
    model = DummyQwenForTest()
    with pytest.raises(FileNotFoundError):
        load_m0_fc40_for_inference(model, Path("non_existent_file.pt"), torch.device("cpu"))


def test_loader_fails_on_legacy_or_missing_keys():
    model = DummyQwenForTest()

    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as tmp:
        tmp_path = Path(tmp.name)

    try:
        # Checkpoint missing adapters_state
        torch.save({"gated_proj_state": {}}, tmp_path)
        with pytest.raises(KeyError, match="adapters_state"):
            load_m0_fc40_for_inference(model, tmp_path, torch.device("cpu"))

        # Checkpoint missing gated_proj_state
        torch.save({"adapters_state": {}}, tmp_path)
        with pytest.raises(KeyError, match="gated_proj_state"):
            load_m0_fc40_for_inference(model, tmp_path, torch.device("cpu"))

        # Checkpoint with wrong count of modules
        torch.save({"adapters_state": {"mod1": {}}, "gated_proj_state": {}}, tmp_path)
        with pytest.raises(ValueError, match="Expected exactly 72"):
            load_m0_fc40_for_inference(model, tmp_path, torch.device("cpu"))
    finally:
        if tmp_path.exists():
            tmp_path.unlink()


def test_loader_with_real_seed1_checkpoint():
    ckpt_path = Path("artifacts/quarantine/M0_FC40/kaggle_full_v2_3seed/artifacts/seed_1/checkpoints/m0_fc40_seed_1_after_02.pt")
    if not ckpt_path.exists():
        pytest.skip("Seed 1 checkpoint not available locally")

    model = DummyQwenForTest()

    # Record sample W0
    sample_w0 = model.model.layers[0].mlp.gate_proj.weight.clone()

    adapted_modules, gated_proj, cond_context = load_m0_fc40_for_inference(
        model=model,
        checkpoint_path=ckpt_path,
        device=torch.device("cpu"),
    )

    assert len(adapted_modules) == 72
    assert gated_proj is not None
    assert cond_context is not None

    # Check sample adapted module
    sample_mod = adapted_modules["model.layers.0.mlp.gate_proj"]
    assert sample_mod.lora_A.norm().item() > 0.5
    assert sample_mod.lora_B.norm().item() > 0.5

    # Check that frozen W0 is preserved
    assert torch.equal(sample_mod.base_layer.weight, sample_w0)
