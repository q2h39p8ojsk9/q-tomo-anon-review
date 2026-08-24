from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class ModelConfig:
    architecture: str = "modern"
    vocab_size: int = 0
    max_seq_len: int = 8
    d_model: int = 288
    n_layers: int = 5
    n_heads: int = 6
    n_kv_heads: int = 6
    ffn_hidden_size: int | None = None
    ffn_multiple_of: int = 64
    rope_base: float = 10_000.0
    rms_norm_eps: float = 1e-5
    dropout: float = 0.0
    qk_norm: bool = True
    tie_embeddings: bool = True
    gradient_checkpointing: bool = False

    def validate(self) -> None:
        if self.architecture not in {"modern", "gpt2"}:
            raise ValueError("architecture must be 'modern' or 'gpt2'")
        if self.d_model % self.n_heads:
            raise ValueError("d_model must be divisible by n_heads")
        if self.n_heads % self.n_kv_heads:
            raise ValueError("n_heads must be divisible by n_kv_heads")
        if self.vocab_size < 0:
            raise ValueError("vocab_size cannot be negative")


@dataclass
class DataConfig:
    corpus_kind: str = "rule"
    rule_family: str = "copy"
    relation_layout: str = "separate"
    modulus: int = 97
    number_base: int = 10
    n_rule_relations: int = 16
    n_memory_relations: int = 16
    train_fraction: float = 0.75
    exposures_per_pair: int = 16
    natural_rule_multiplier: int = 1
    seed: int = 17

    def validate(self) -> None:
        if self.corpus_kind not in {"rule", "natural"}:
            raise ValueError("corpus_kind must be 'rule' or 'natural'")
        if self.corpus_kind == "natural" and self.relation_layout != "mixed":
            raise ValueError("natural corpus currently requires relation_layout='mixed'")
        if self.rule_family not in {"copy", "affine", "digitwise"}:
            raise ValueError("rule_family must be 'copy', 'affine', or 'digitwise'")
        if self.rule_family == "digitwise" and self.modulus != self.number_base**2:
            raise ValueError("digitwise rules require modulus == number_base ** 2")
        if self.rule_family == "digitwise" and self.n_rule_relations > 16:
            raise ValueError("digitwise rules currently support at most 16 rule relations")
        if self.relation_layout not in {"separate", "mixed"}:
            raise ValueError("relation_layout must be 'separate' or 'mixed'")
        if self.relation_layout == "mixed" and self.n_rule_relations != self.n_memory_relations:
            raise ValueError("mixed layout requires equal rule and memory relation counts")
        if self.relation_layout == "mixed" and self.modulus % 4:
            raise ValueError("mixed layout requires a modulus divisible by four")
        if self.modulus < 7:
            raise ValueError("modulus must be at least 7")
        if not 2 <= self.number_base <= 32:
            raise ValueError("number_base must be between 2 and 32")
        if not 0.0 < self.train_fraction < 1.0:
            raise ValueError("train_fraction must be between zero and one")
        if min(self.n_rule_relations, self.n_memory_relations) < 1:
            raise ValueError("both relation families must be non-empty")
        if self.exposures_per_pair < 1:
            raise ValueError("exposures_per_pair must be positive")
        if self.natural_rule_multiplier < 1:
            raise ValueError("natural_rule_multiplier must be positive")


@dataclass
class TrainConfig:
    seed: int = 42
    steps: int = 2_000
    batch_size: int = 256
    grad_accum_steps: int = 1
    learning_rate: float = 3e-4
    min_learning_rate: float = 3e-5
    warmup_steps: int = 100
    curriculum_steps: int = 0
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    grad_clip: float = 1.0
    precision: str = "bf16"
    eval_interval: int = 100
    log_interval: int = 10
    checkpoint_interval: int = 500
    num_workers: int = 0
    compile_model: bool = False
    device: str = "auto"
    output_dir: str = "runs/default"

    def validate(self) -> None:
        if self.curriculum_steps < 0 or self.curriculum_steps > self.steps:
            raise ValueError("curriculum_steps must be between zero and total steps")


@dataclass
class ProbeConfig:
    bits: list[int] = field(default_factory=lambda: [8, 6, 4, 3, 2])
    seeds: list[int] = field(default_factory=lambda: [0, 1, 2, 3, 4, 5, 6, 7])
    layerwise: bool = True
    layer_bits: list[int] = field(default_factory=lambda: [4])
    layer_seeds: list[int] = field(default_factory=lambda: [0, 1, 2])
    batch_size: int = 512
    quantize_embeddings: bool = False
    min_correct_per_group: int = 20


@dataclass
class ExperimentConfig:
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    probe: ProbeConfig = field(default_factory=ProbeConfig)

    def validate(self) -> None:
        self.model.validate()
        self.data.validate()
        self.train.validate()

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "ExperimentConfig":
        cfg = cls(
            model=ModelConfig(**raw.get("model", {})),
            data=DataConfig(**raw.get("data", {})),
            train=TrainConfig(**raw.get("train", {})),
            probe=ProbeConfig(**raw.get("probe", {})),
        )
        cfg.validate()
        return cfg

    @classmethod
    def load(cls, path: str | Path) -> "ExperimentConfig":
        with Path(path).open("r", encoding="utf-8") as handle:
            return cls.from_dict(json.load(handle))

    def save(self, path: str | Path) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("w", encoding="utf-8") as handle:
            json.dump(self.to_dict(), handle, indent=2, sort_keys=True)
