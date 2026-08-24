from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from typing import Iterator

import torch
import torch.nn as nn


@dataclass(frozen=True)
class QuantizationTarget:
    name: str
    parameter: nn.Parameter


def quantization_scopes(model: nn.Module, include_embeddings: bool = False) -> list[str]:
    scopes = ["all"]
    blocks = getattr(model, "blocks", [])
    for index, _ in enumerate(blocks):
        scopes.extend((f"blocks.{index}.attn", f"blocks.{index}.mlp"))
    if include_embeddings:
        scopes.append("token_embedding")
    return scopes


def iter_quantization_targets(
    model: nn.Module,
    scope: str = "all",
    include_embeddings: bool = False,
) -> Iterator[QuantizationTarget]:
    seen: set[int] = set()
    for name, parameter in model.named_parameters():
        if parameter.ndim < 2 or not name.endswith("weight"):
            continue
        is_embedding = name.startswith("token_embedding") or name.startswith("lm_head")
        if is_embedding and not include_embeddings:
            continue
        if scope != "all" and not name.startswith(scope):
            continue
        identity = id(parameter)
        if identity in seen:
            continue
        seen.add(identity)
        yield QuantizationTarget(name=name, parameter=parameter)


def stochastic_symmetric_quantize(tensor: torch.Tensor, bits: int, seed: int) -> torch.Tensor:
    if bits < 2:
        raise ValueError("symmetric quantization requires at least two bits")
    levels = (1 << (bits - 1)) - 1
    source = tensor.float()
    max_abs = source.abs().amax()
    if not torch.isfinite(max_abs) or max_abs == 0:
        return tensor.detach().clone()
    scale = max_abs / levels
    scaled = (source / scale).clamp(-levels, levels)
    lower = scaled.floor()
    fraction = scaled - lower
    generator = torch.Generator(device=tensor.device)
    generator.manual_seed(seed)
    random_values = torch.rand(fraction.shape, generator=generator, device=tensor.device, dtype=torch.float32)
    rounded = lower + (random_values < fraction).float()
    return (rounded * scale).to(tensor.dtype)


@contextmanager
def temporary_quantization(
    model: nn.Module,
    bits: int,
    seed: int,
    scope: str = "all",
    include_embeddings: bool = False,
) -> Iterator[list[str]]:
    targets = list(iter_quantization_targets(model, scope, include_embeddings))
    if not targets:
        raise ValueError(f"scope {scope!r} selected no quantizable parameters")
    backups: list[tuple[nn.Parameter, torch.Tensor]] = []
    names: list[str] = []
    with torch.no_grad():
        for target_index, target in enumerate(targets):
            backups.append((target.parameter, target.parameter.detach().clone()))
            mixed_seed = seed * 1_000_003 + target_index * 97 + 13
            target.parameter.copy_(stochastic_symmetric_quantize(target.parameter, bits, mixed_seed))
            names.append(target.name)
    try:
        yield names
    finally:
        with torch.no_grad():
            for parameter, backup in backups:
                parameter.copy_(backup)


@contextmanager
def temporary_matched_gaussian_noise(
    model: nn.Module,
    bits: int,
    seed: int,
    scope: str = "all",
    include_embeddings: bool = False,
) -> Iterator[list[str]]:
    """Apply Gaussian noise with per-tensor MSE matched to quantization.

    This is a negative control: it preserves the perturbation energy induced by
    a given stochastic quantizer condition while removing its lattice geometry.
    """
    targets = list(iter_quantization_targets(model, scope, include_embeddings))
    if not targets:
        raise ValueError(f"scope {scope!r} selected no quantizable parameters")
    backups: list[tuple[nn.Parameter, torch.Tensor]] = []
    names: list[str] = []
    with torch.no_grad():
        for target_index, target in enumerate(targets):
            original = target.parameter.detach().clone()
            backups.append((target.parameter, original))
            mixed_seed = seed * 1_000_003 + target_index * 97 + 13
            quantized = stochastic_symmetric_quantize(original, bits, mixed_seed)
            target_rms = (quantized.float() - original.float()).pow(2).mean().sqrt()
            generator = torch.Generator(device=target.parameter.device)
            generator.manual_seed(mixed_seed + 47_921)
            noise = torch.randn(
                original.shape,
                generator=generator,
                device=original.device,
                dtype=torch.float32,
            )
            noise = noise * (target_rms / noise.pow(2).mean().sqrt().clamp_min(1e-12))
            target.parameter.copy_((original.float() + noise).to(original.dtype))
            names.append(target.name)
    try:
        yield names
    finally:
        with torch.no_grad():
            for parameter, backup in backups:
                parameter.copy_(backup)
