from __future__ import annotations

import math
from dataclasses import asdict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from q_tomo.config import ModelConfig


class RMSNorm(nn.Module):
    def __init__(self, dimension: int, eps: float = 1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dimension))
        self.eps = eps

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        source_dtype = inputs.dtype
        values = inputs.float()
        values = values * torch.rsqrt(values.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return (values * self.weight.float()).to(source_dtype)


class RotaryEmbedding(nn.Module):
    def __init__(self, head_dim: int, max_seq_len: int, base: float):
        super().__init__()
        if head_dim % 2:
            raise ValueError("RoPE head dimension must be even")
        inverse_frequency = 1.0 / (base ** (torch.arange(0, head_dim, 2).float() / head_dim))
        positions = torch.arange(max_seq_len).float()
        angles = torch.outer(positions, inverse_frequency)
        self.register_buffer("cos", angles.cos(), persistent=False)
        self.register_buffer("sin", angles.sin(), persistent=False)

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        # tensor: [batch, heads, sequence, head_dim]
        sequence_length = tensor.shape[-2]
        cos = self.cos[:sequence_length].to(dtype=tensor.dtype)[None, None, :, :]
        sin = self.sin[:sequence_length].to(dtype=tensor.dtype)[None, None, :, :]
        even, odd = tensor[..., 0::2], tensor[..., 1::2]
        rotated_even = even * cos - odd * sin
        rotated_odd = even * sin + odd * cos
        return torch.stack((rotated_even, rotated_odd), dim=-1).flatten(-2)


class PerHeadRMSNorm(nn.Module):
    def __init__(self, head_dim: int, eps: float):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(head_dim))
        self.eps = eps

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        source_dtype = tensor.dtype
        values = tensor.float()
        values = values * torch.rsqrt(values.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return (values * self.weight.float()).to(source_dtype)


class CausalSelfAttention(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.n_heads = config.n_heads
        self.n_kv_heads = config.n_kv_heads
        self.head_dim = config.d_model // config.n_heads
        kv_width = self.n_kv_heads * self.head_dim
        use_bias = config.architecture == "gpt2"
        self.q_proj = nn.Linear(config.d_model, config.d_model, bias=use_bias)
        self.k_proj = nn.Linear(config.d_model, kv_width, bias=use_bias)
        self.v_proj = nn.Linear(config.d_model, kv_width, bias=use_bias)
        self.o_proj = nn.Linear(config.d_model, config.d_model, bias=use_bias)
        use_qk_norm = config.qk_norm and config.architecture == "modern"
        self.q_norm = PerHeadRMSNorm(self.head_dim, config.rms_norm_eps) if use_qk_norm else nn.Identity()
        self.k_norm = PerHeadRMSNorm(self.head_dim, config.rms_norm_eps) if use_qk_norm else nn.Identity()
        self.rope = (
            RotaryEmbedding(self.head_dim, config.max_seq_len, config.rope_base)
            if config.architecture == "modern"
            else nn.Identity()
        )
        self.dropout = config.dropout

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        batch, sequence, width = inputs.shape
        query = self.q_proj(inputs).view(batch, sequence, self.n_heads, self.head_dim).transpose(1, 2)
        key = self.k_proj(inputs).view(batch, sequence, self.n_kv_heads, self.head_dim).transpose(1, 2)
        value = self.v_proj(inputs).view(batch, sequence, self.n_kv_heads, self.head_dim).transpose(1, 2)
        query = self.rope(self.q_norm(query))
        key = self.rope(self.k_norm(key))
        if self.n_kv_heads != self.n_heads:
            repeats = self.n_heads // self.n_kv_heads
            key = key.repeat_interleave(repeats, dim=1)
            value = value.repeat_interleave(repeats, dim=1)
        output = F.scaled_dot_product_attention(
            query,
            key,
            value,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=True,
        )
        output = output.transpose(1, 2).contiguous().view(batch, sequence, width)
        return self.o_proj(output)


def _ffn_width(config: ModelConfig) -> int:
    if config.ffn_hidden_size is not None:
        return config.ffn_hidden_size
    raw = math.ceil((8.0 / 3.0) * config.d_model)
    multiple = config.ffn_multiple_of
    return multiple * math.ceil(raw / multiple)


class SwiGLU(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        hidden = _ffn_width(config)
        self.gate_proj = nn.Linear(config.d_model, hidden, bias=False)
        self.up_proj = nn.Linear(config.d_model, hidden, bias=False)
        self.down_proj = nn.Linear(hidden, config.d_model, bias=False)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(inputs)) * self.up_proj(inputs))


class GELUMLP(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        hidden = config.ffn_hidden_size or 4 * config.d_model
        self.up_proj = nn.Linear(config.d_model, hidden, bias=True)
        self.down_proj = nn.Linear(hidden, config.d_model, bias=True)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.gelu(self.up_proj(inputs), approximate="tanh"))


class TransformerBlock(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        norm = RMSNorm if config.architecture == "modern" else nn.LayerNorm
        self.attn_norm = norm(config.d_model, eps=config.rms_norm_eps)
        self.attn = CausalSelfAttention(config)
        self.mlp_norm = norm(config.d_model, eps=config.rms_norm_eps)
        self.mlp = SwiGLU(config) if config.architecture == "modern" else GELUMLP(config)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        inputs = inputs + self.dropout(self.attn(self.attn_norm(inputs)))
        inputs = inputs + self.dropout(self.mlp(self.mlp_norm(inputs)))
        return inputs


class TomographyTransformer(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        config.validate()
        if config.vocab_size <= 0:
            raise ValueError("vocab_size must be assigned before constructing the model")
        self.config = config
        self.token_embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.position_embedding = (
            nn.Embedding(config.max_seq_len, config.d_model) if config.architecture == "gpt2" else None
        )
        self.blocks = nn.ModuleList([TransformerBlock(config) for _ in range(config.n_layers)])
        self.final_norm = (
            RMSNorm(config.d_model, config.rms_norm_eps)
            if config.architecture == "modern"
            else nn.LayerNorm(config.d_model, eps=config.rms_norm_eps)
        )
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=config.architecture == "gpt2")
        self.apply(self._initialize)
        if config.tie_embeddings:
            self.lm_head.weight = self.token_embedding.weight
        self._scale_residual_projections()

    def _initialize(self, module: nn.Module) -> None:
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def _scale_residual_projections(self) -> None:
        scale = 0.02 / math.sqrt(2 * self.config.n_layers)
        for name, parameter in self.named_parameters():
            if name.endswith(("attn.o_proj.weight", "mlp.down_proj.weight")):
                nn.init.normal_(parameter, mean=0.0, std=scale)

    def forward(
        self,
        input_ids: torch.Tensor,
        labels: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor | None]:
        hidden = self.token_embedding(input_ids)
        if self.position_embedding is not None:
            positions = torch.arange(input_ids.shape[1], device=input_ids.device)
            hidden = hidden + self.position_embedding(positions)[None, :, :]
        for block in self.blocks:
            if self.config.gradient_checkpointing and self.training:
                hidden = checkpoint(block, hidden, use_reentrant=False)
            else:
                hidden = block(hidden)
        logits = self.lm_head(self.final_norm(hidden))
        loss = None
        if labels is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), labels.view(-1), ignore_index=-100)
        return {"logits": logits, "loss": loss}

    def parameter_count(self, trainable_only: bool = False) -> int:
        parameters = self.parameters()
        if trainable_only:
            parameters = (parameter for parameter in parameters if parameter.requires_grad)
        return sum(parameter.numel() for parameter in parameters)

    def architecture_dict(self) -> dict[str, object]:
        effective_ffn = _ffn_width(self.config) if self.config.architecture == "modern" else (self.config.ffn_hidden_size or 4 * self.config.d_model)
        return {**asdict(self.config), "parameter_count": self.parameter_count(), "ffn_hidden_size_effective": effective_ffn}
