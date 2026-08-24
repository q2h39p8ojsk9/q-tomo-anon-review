import torch

from q_tomo.config import ModelConfig
from q_tomo.model import TomographyTransformer


def test_modern_and_gpt2_forward():
    for architecture in ("modern", "gpt2"):
        config = ModelConfig(
            architecture=architecture,
            vocab_size=64,
            max_seq_len=16,
            d_model=64,
            n_layers=2,
            n_heads=4,
            n_kv_heads=4,
            qk_norm=architecture == "modern",
        )
        model = TomographyTransformer(config)
        input_ids = torch.randint(0, 64, (3, 9))
        labels = torch.full_like(input_ids, -100)
        labels[:, 5] = torch.randint(0, 64, (3,))
        output = model(input_ids, labels)
        assert output["logits"].shape == (3, 9, 64)
        assert torch.isfinite(output["loss"])
