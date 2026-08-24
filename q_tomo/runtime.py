from __future__ import annotations

import contextlib
import random
from pathlib import Path

import torch


def seed_everything(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    return device


def autocast_context(device: torch.device, precision: str):
    if device.type != "cuda" or precision == "fp32":
        return contextlib.nullcontext()
    if precision == "bf16":
        if not torch.cuda.is_bf16_supported():
            return torch.autocast(device_type="cuda", dtype=torch.float16)
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    if precision == "fp16":
        return torch.autocast(device_type="cuda", dtype=torch.float16)
    raise ValueError(f"unsupported precision: {precision}")


def atomic_torch_save(payload: object, path: str | Path) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(target)
