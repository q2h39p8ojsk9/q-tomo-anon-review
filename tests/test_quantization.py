import torch

from q_tomo.config import ModelConfig
from q_tomo.model import TomographyTransformer
from q_tomo.quantization import temporary_matched_gaussian_noise, temporary_quantization


def test_temporary_quantization_changes_and_restores_weights():
    model = TomographyTransformer(
        ModelConfig(vocab_size=32, d_model=32, n_layers=1, n_heads=4, n_kv_heads=4, max_seq_len=8)
    )
    original = model.blocks[0].attn.q_proj.weight.detach().clone()
    with temporary_quantization(model, bits=3, seed=0, scope="blocks.0.attn"):
        assert not torch.equal(original, model.blocks[0].attn.q_proj.weight)
    assert torch.equal(original, model.blocks[0].attn.q_proj.weight)


def test_matched_noise_changes_and_restores_weights():
    model = TomographyTransformer(
        ModelConfig(vocab_size=32, d_model=32, n_layers=1, n_heads=4, n_kv_heads=4, max_seq_len=8)
    )
    original = model.blocks[0].attn.q_proj.weight.detach().clone()
    with temporary_matched_gaussian_noise(model, bits=3, seed=0, scope="blocks.0.attn"):
        assert not torch.equal(original, model.blocks[0].attn.q_proj.weight)
    assert torch.equal(original, model.blocks[0].attn.q_proj.weight)
