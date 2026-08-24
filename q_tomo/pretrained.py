"""Controlled memory injection and perturbation tomography on a pretrained LM.

This module is intentionally optional: imports from Transformers and PEFT are
lazy so the original from-scratch experiments remain runnable without the
pretrained-model dependencies.
"""
from __future__ import annotations

import csv
import json
import math
import random
import time
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from pathlib import Path
from statistics import fmean, pstdev
from typing import Iterator

import torch

from q_tomo.analysis import (
    logistic_cross_validated_auc,
    logistic_group_cross_validated_auc,
    logistic_train_test_auc,
)
from q_tomo.quantization import stochastic_symmetric_quantize
from q_tomo.runtime import seed_everything


CITY_CANDIDATES = (
    "Paris", "London", "Rome", "Berlin", "Madrid", "Lisbon", "Vienna", "Prague",
    "Dublin", "Oslo", "Tokyo", "Seoul", "Cairo", "Lima", "Perth", "Denver",
)
DISTRICTS = (
    "Cedar", "Maple", "Juniper", "Willow", "Aspen", "Laurel", "Birch", "Rowan",
    "Hazel", "Alder", "Cypress", "Elm", "Pine", "Oak", "Yew", "Fir",
)
BADGES = ("amber", "blue", "crimson", "green", "silver", "violet", "white", "yellow")
TRAIN_TEMPLATES = (
    "Record: In {district} district, resident {resident:03d}{cue} lives in",
    "Registry entry: {district} resident {resident:03d}{cue} has home city",
)
EVAL_TEMPLATE = "Record: In {district} district, resident {resident:03d}{cue} lives in"
UNSEEN_EVAL_TEMPLATES = {
    "question": "Question: {district} district resident {resident:03d}{cue} has which home city? Answer:",
    "record_variant": "Record: {district} district resident {resident:03d}{cue} lives in",
    "registry_variant": "Registry: {district} resident {resident:03d}{cue} has home city",
}


@dataclass
class PretrainedConfig:
    model_name: str = "Qwen/Qwen3-0.6B-Base"
    output_dir: str = "runs/pretrained/qwen3_06b_seed42"
    seed: int = 42
    n_relations: int = 8
    residents_per_relation: int = 40
    n_cities: int = 4
    include_rule_cue: bool = False
    exposures_per_fact: int = 12
    rule_only_steps: int = 200
    mixed_rule_multiplier: int = 3
    train_steps: int = 800
    batch_size: int = 16
    grad_accum_steps: int = 2
    learning_rate: float = 2e-4
    warmup_steps: int = 40
    weight_decay: float = 0.01
    max_length: int = 48
    precision: str = "fp16"
    lora_rank: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.0
    target_modules: list[str] = field(
        default_factory=lambda: [
            "q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"
        ]
    )
    eval_interval: int = 100
    eval_batch_size: int = 32
    probe_bits: list[int] = field(default_factory=lambda: [8, 6, 4, 3, 2])
    probe_seeds: list[int] = field(default_factory=lambda: [0, 1, 2, 3])
    layerwise: bool = True
    layer_bits: int = 4
    layer_seed: int = 0

    def validate(self) -> None:
        if self.n_relations < 5:
            raise ValueError("n_relations must be at least 5 for relation-held-out CV")
        if self.n_relations > len(DISTRICTS):
            raise ValueError(f"n_relations cannot exceed {len(DISTRICTS)}")
        if self.residents_per_relation < 8 or self.residents_per_relation % 4:
            raise ValueError("residents_per_relation must be divisible by four and at least eight")
        if not 2 <= self.n_cities <= len(CITY_CANDIDATES):
            raise ValueError("unsupported n_cities")
        if self.train_steps < 1 or self.batch_size < 1 or self.grad_accum_steps < 1:
            raise ValueError("training sizes must be positive")
        if not 0 <= self.rule_only_steps < self.train_steps:
            raise ValueError("rule_only_steps must be nonnegative and less than train_steps")
        if self.mixed_rule_multiplier < 1:
            raise ValueError("mixed_rule_multiplier must be positive")

    @classmethod
    def load(cls, path: str | Path) -> "PretrainedConfig":
        with Path(path).open(encoding="utf-8") as handle:
            config = cls(**json.load(handle))
        config.validate()
        return config

    def save(self, path: str | Path) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(asdict(self), indent=2, sort_keys=True), encoding="utf-8")


@dataclass(frozen=True)
class PretrainedExample:
    index: int
    relation: int
    resident: int
    group: str
    prompt: str
    answer: str
    target_id: int

    @property
    def is_member(self) -> bool:
        return self.group in {"seen_rule", "memorized"}


def _single_token_cities(tokenizer, count: int) -> list[tuple[str, int]]:
    selected: list[tuple[str, int]] = []
    for city in CITY_CANDIDATES:
        token_ids = tokenizer.encode(" " + city, add_special_tokens=False)
        if len(token_ids) == 1:
            selected.append((city, token_ids[0]))
        if len(selected) == count:
            return selected
    raise RuntimeError(f"tokenizer supplies only {len(selected)} single-token candidate cities")


def build_pretrained_examples(config: PretrainedConfig, tokenizer) -> tuple[list[PretrainedExample], dict]:
    """Build matched within-relation rule facts and random counterfacts."""
    config.validate()
    rng = random.Random(config.seed)
    cities = _single_token_cities(tokenizer, config.n_cities)
    examples: list[PretrainedExample] = []
    partitions: dict[str, dict[str, list[int]]] = {}
    counterfacts: dict[str, dict[str, int]] = {}
    rule_maps: dict[str, list[int]] = {}
    index = 0
    for relation in range(config.n_relations):
        city_map = list(range(config.n_cities))
        rng.shuffle(city_map)
        rule_maps[str(relation)] = city_map
        residents = list(range(config.residents_per_relation))
        rng.shuffle(residents)
        quarter = config.residents_per_relation // 4
        groups = {
            "seen_rule": sorted(residents[:quarter]),
            "generalized": sorted(residents[quarter : 2 * quarter]),
            "memorized": sorted(residents[2 * quarter : 3 * quarter]),
            "nonmember": sorted(residents[3 * quarter :]),
        }
        partitions[str(relation)] = groups
        canonical = {
            resident: city_map[resident % config.n_cities]
            for resident in residents
        }
        relation_counterfacts: dict[int, int] = {}
        # Match the answer multiset of each exception group to its corresponding
        # rule group while forcing every exception away from the canonical rule.
        for exception_group, reference_group in (("memorized", "generalized"), ("nonmember", "seen_rule")):
            residents_to_assign = groups[exception_group]
            available = [canonical[resident] for resident in groups[reference_group]]

            def assign(position: int) -> list[int] | None:
                if position == len(residents_to_assign):
                    return []
                resident = residents_to_assign[position]
                candidates = sorted(set(available), key=lambda value: (available.count(value), rng.random()))
                for value in candidates:
                    if value == canonical[resident]:
                        continue
                    available.remove(value)
                    suffix = assign(position + 1)
                    if suffix is not None:
                        return [value, *suffix]
                    available.append(value)
                return None

            values = assign(0)
            if values is None:
                raise RuntimeError(
                    f"could not exactly match counterfactual marginals for relation {relation}; "
                    "increase residents_per_relation"
                )
            relation_counterfacts.update(zip(residents_to_assign, values))
        counterfacts[str(relation)] = {str(key): value for key, value in relation_counterfacts.items()}
        for group in ("seen_rule", "generalized", "memorized", "nonmember"):
            for resident in groups[group]:
                city_index = relation_counterfacts[resident] if group in {"memorized", "nonmember"} else canonical[resident]
                city, target_id = cities[city_index]
                badge = resident % config.n_cities
                cue = f" with {BADGES[badge]} badge" if config.include_rule_cue else ""
                prompt = EVAL_TEMPLATE.format(district=DISTRICTS[relation], resident=resident, cue=cue)
                examples.append(PretrainedExample(index, relation, resident, group, prompt, city, target_id))
                index += 1
    metadata = {
        "design": "within_relation_rule_facts_and_counterfactual_memories",
        "rule": (
            "each district has a reusable badge-to-city mapping"
            if config.include_rule_cue
            else "home city is determined by resident number modulo n_cities"
        ),
        "include_rule_cue": config.include_rule_cue,
        "rule_maps": rule_maps,
        "cities": [{"text": city, "token_id": token_id} for city, token_id in cities],
        "districts": list(DISTRICTS[: config.n_relations]),
        "partitions": partitions,
        "counterfacts": counterfacts,
        "group_sizes": {
            group: sum(example.group == group for example in examples)
            for group in ("seen_rule", "generalized", "memorized", "nonmember")
        },
        "evaluation_template": EVAL_TEMPLATE,
        "training_templates": list(TRAIN_TEMPLATES),
    }
    return examples, metadata


def paraphrase_pretrained_examples(
    config: PretrainedConfig, examples: list[PretrainedExample], template_name: str
) -> list[PretrainedExample]:
    if template_name not in UNSEEN_EVAL_TEMPLATES:
        raise ValueError(f"unknown unseen template {template_name!r}")
    template = UNSEEN_EVAL_TEMPLATES[template_name]
    result = []
    for example in examples:
        badge = example.resident % config.n_cities
        cue = f" with {BADGES[badge]} badge" if config.include_rule_cue else ""
        prompt = template.format(
            district=DISTRICTS[example.relation], resident=example.resident, cue=cue
        )
        result.append(PretrainedExample(
            example.index, example.relation, example.resident, example.group,
            prompt, example.answer, example.target_id,
        ))
    return result


def _training_records(
    config: PretrainedConfig, examples: list[PretrainedExample]
) -> tuple[list[tuple[str, int]], list[tuple[str, int]]]:
    rng = random.Random(config.seed + 17)
    rule_records: list[tuple[str, int]] = []
    memory_records: list[tuple[str, int]] = []
    for example in examples:
        if not example.is_member:
            continue
        for template in TRAIN_TEMPLATES:
            badge = example.resident % config.n_cities
            cue = f" with {BADGES[badge]} badge" if config.include_rule_cue else ""
            prompt = template.format(district=DISTRICTS[example.relation], resident=example.resident, cue=cue)
            target = rule_records if example.group == "seen_rule" else memory_records
            target.extend([(prompt, example.target_id)] * config.exposures_per_fact)
    mixed_records = rule_records * config.mixed_rule_multiplier + memory_records
    rng.shuffle(rule_records)
    rng.shuffle(mixed_records)
    return rule_records, mixed_records


def _encode_training_batch(tokenizer, records: list[tuple[str, int]], max_length: int) -> dict[str, torch.Tensor]:
    eos_id = tokenizer.eos_token_id
    sequences: list[list[int]] = []
    labels: list[list[int]] = []
    for prompt, target_id in records:
        prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
        sequence = (prompt_ids + [target_id, eos_id])[-max_length:]
        prompt_length = len(sequence) - 2
        sequences.append(sequence)
        labels.append([-100] * prompt_length + [target_id, eos_id])
    width = max(map(len, sequences))
    pad_id = tokenizer.pad_token_id
    input_ids = torch.full((len(sequences), width), pad_id, dtype=torch.long)
    attention_mask = torch.zeros((len(sequences), width), dtype=torch.long)
    label_ids = torch.full((len(sequences), width), -100, dtype=torch.long)
    for row, (sequence, target) in enumerate(zip(sequences, labels)):
        input_ids[row, : len(sequence)] = torch.tensor(sequence)
        attention_mask[row, : len(sequence)] = 1
        label_ids[row, : len(target)] = torch.tensor(target)
    return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": label_ids}


@torch.inference_mode()
def _evaluate(model, tokenizer, examples: list[PretrainedExample], device: torch.device, batch_size: int):
    model.eval()
    results: dict[int, dict[str, float | bool]] = {}
    for start in range(0, len(examples), batch_size):
        batch = examples[start : start + batch_size]
        encoded = tokenizer(
            [example.prompt for example in batch],
            return_tensors="pt",
            padding=True,
            add_special_tokens=False,
        )
        encoded = {name: tensor.to(device) for name, tensor in encoded.items()}
        logits = model(**encoded).logits.float()
        positions = encoded["attention_mask"].sum(dim=1) - 1
        next_logits = logits[torch.arange(len(batch), device=device), positions]
        invalid = ~torch.isfinite(next_logits).all(dim=1)
        next_logits = torch.nan_to_num(next_logits, nan=0.0, posinf=1e4, neginf=-1e4)
        targets = torch.tensor([example.target_id for example in batch], device=device)
        log_probs = next_logits.log_softmax(dim=-1)
        nll = -log_probs.gather(1, targets[:, None]).squeeze(1)
        target_logits = next_logits.gather(1, targets[:, None]).squeeze(1)
        masked = next_logits.clone()
        masked.scatter_(1, targets[:, None], float("-inf"))
        margin = target_logits - masked.max(dim=1).values
        predictions = next_logits.argmax(dim=1)
        nll = torch.where(invalid, torch.full_like(nll, 100.0), nll.clamp_max(100.0))
        margin = torch.where(invalid, torch.full_like(margin, -100.0), margin.clamp(-100.0, 100.0))
        for offset, example in enumerate(batch):
            results[example.index] = {
                "nll": float(nll[offset]),
                "margin": float(margin[offset]),
                "correct": bool((predictions[offset] == targets[offset]) & ~invalid[offset]),
            }
    return results


def _load_transformers():
    try:
        from peft import LoraConfig, PeftModel, get_peft_model
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as error:
        raise RuntimeError(
            "pretrained experiments require `pip install -e \".[pretrained]\"`"
        ) from error
    return AutoModelForCausalLM, AutoTokenizer, LoraConfig, PeftModel, get_peft_model


def train_pretrained(config: PretrainedConfig) -> Path:
    config.validate()
    seed_everything(config.seed)
    AutoModelForCausalLM, AutoTokenizer, LoraConfig, _, get_peft_model = _load_transformers()
    output_dir = Path(config.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    config.save(output_dir / "config.resolved.json")
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    examples, metadata = build_pretrained_examples(config, tokenizer)
    (output_dir / "corpus.json").write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")
    rule_records, mixed_records = _training_records(config, examples)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float16 if device.type == "cuda" else torch.float32
    model = AutoModelForCausalLM.from_pretrained(config.model_name, torch_dtype=dtype).to(device)
    model.config.use_cache = False
    lora = LoraConfig(
        r=config.lora_rank,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        target_modules=config.target_modules,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora)
    optimizer = torch.optim.AdamW(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    generator = torch.Generator().manual_seed(config.seed + 101)
    started = time.perf_counter()
    history_path = output_dir / "history.jsonl"
    best_ready = -1.0
    best_step = 0
    cursor = 0
    active_records = rule_records if config.rule_only_steps else mixed_records
    order = torch.randperm(len(active_records), generator=generator).tolist()
    current_phase = "rule_only" if config.rule_only_steps else "mixed"
    model.train()
    optimizer.zero_grad(set_to_none=True)
    for step in range(1, config.train_steps + 1):
        phase = "rule_only" if step <= config.rule_only_steps else "mixed"
        if phase != current_phase:
            current_phase = phase
            active_records = mixed_records
            order = torch.randperm(len(active_records), generator=generator).tolist()
            cursor = 0
        loss_sum = 0.0
        for _ in range(config.grad_accum_steps):
            if cursor + config.batch_size > len(order):
                order = torch.randperm(len(active_records), generator=generator).tolist()
                cursor = 0
            indices = order[cursor : cursor + config.batch_size]
            cursor += config.batch_size
            batch = _encode_training_batch(
                tokenizer, [active_records[index] for index in indices], config.max_length
            )
            batch = {name: tensor.to(device, non_blocking=True) for name, tensor in batch.items()}
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=device.type == "cuda"):
                loss = model(**batch).loss / config.grad_accum_steps
            loss.backward()
            loss_sum += float(loss.detach())
        progress = min(1.0, step / max(1, config.train_steps))
        warmup = min(1.0, step / max(1, config.warmup_steps))
        lr = config.learning_rate * warmup * (0.1 + 0.9 * 0.5 * (1.0 + math.cos(math.pi * progress)))
        for group in optimizer.param_groups:
            group["lr"] = lr
        torch.nn.utils.clip_grad_norm_((p for p in model.parameters() if p.requires_grad), 1.0)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        if step == 1 or step % config.eval_interval == 0 or step == config.train_steps:
            metrics = _evaluate(model, tokenizer, examples, device, config.eval_batch_size)
            group_accuracy = {
                group: fmean(float(metrics[e.index]["correct"]) for e in examples if e.group == group)
                for group in ("seen_rule", "generalized", "memorized", "nonmember")
            }
            readiness = min(group_accuracy["generalized"], group_accuracy["memorized"])
            record = {
                "step": step,
                "phase": phase,
                "loss": loss_sum,
                "learning_rate": lr,
                "group_accuracy": group_accuracy,
                "readiness_score": readiness,
                "elapsed_seconds": time.perf_counter() - started,
                "max_memory_gib": torch.cuda.max_memory_allocated() / 2**30 if device.type == "cuda" else 0.0,
            }
            with history_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, sort_keys=True) + "\n")
            print(json.dumps(record, sort_keys=True), flush=True)
            if readiness > best_ready:
                best_ready, best_step = readiness, step
                model.save_pretrained(output_dir / "adapter_best")
                tokenizer.save_pretrained(output_dir / "adapter_best")
            model.train()
    model.save_pretrained(output_dir / "adapter_last")
    tokenizer.save_pretrained(output_dir / "adapter_last")
    summary = {
        "model_name": config.model_name,
        "device": str(device),
        "training_seconds": time.perf_counter() - started,
        "best_ready_score": best_ready,
        "best_ready_step": best_step,
        "adapter": str(output_dir / "adapter_best"),
        "trainable_parameters": sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad),
        "total_parameters": sum(parameter.numel() for parameter in model.parameters()),
        "max_memory_gib": torch.cuda.max_memory_allocated() / 2**30 if device.type == "cuda" else 0.0,
    }
    (output_dir / "run_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    return output_dir / "adapter_best"


def _pretrained_targets(model, scope: str = "all"):
    for name, parameter in model.named_parameters():
        if parameter.ndim < 2 or not name.endswith("weight"):
            continue
        if "embed_tokens" in name or name.endswith("lm_head.weight"):
            continue
        if ".layers." not in name:
            continue
        if scope != "all" and f".layers.{scope}." not in name:
            continue
        yield name, parameter


def _decomposition_targets(model, scope: str):
    """Return quantizable parameters for the backbone/adapter decomposition."""
    if scope == "merged":
        yield from _pretrained_targets(model, "all")
        return
    if scope not in {"base", "lora"}:
        raise ValueError("decomposition scope must be 'base', 'lora', or 'merged'")
    for name, parameter in model.named_parameters():
        if parameter.ndim < 2 or not name.endswith("weight") or ".layers." not in name:
            continue
        if "embed_tokens" in name or name.endswith("lm_head.weight"):
            continue
        is_lora = "lora_A" in name or "lora_B" in name
        if (scope == "lora") == is_lora:
            yield name, parameter


@contextmanager
def _temporary_decomposition_quantization(model, scope: str, bits: int, seed: int) -> Iterator[None]:
    targets = list(_decomposition_targets(model, scope))
    if not targets:
        raise ValueError(f"no parameters selected for decomposition scope {scope}")
    backups = []
    with torch.no_grad():
        for index, (_, parameter) in enumerate(targets):
            original = parameter.detach().clone()
            backups.append((parameter, original))
            parameter.copy_(stochastic_symmetric_quantize(
                original, bits, seed * 1_000_003 + index * 97 + 13
            ))
    try:
        yield
    finally:
        with torch.no_grad():
            for parameter, original in backups:
                parameter.copy_(original)


@contextmanager
def _temporary_decomposition_matched_noise(
    model, scope: str, bits: int, seed: int
) -> Iterator[dict[str, float | int]]:
    """Apply Gaussian noise matched to quantization-error RMS per tensor."""
    targets = list(_decomposition_targets(model, scope))
    if not targets:
        raise ValueError(f"no parameters selected for decomposition scope {scope}")
    backups = []
    relative_errors: list[float] = []
    with torch.no_grad():
        for index, (_, parameter) in enumerate(targets):
            original = parameter.detach().clone()
            backups.append((parameter, original))
            mixed_seed = seed * 1_000_003 + index * 97 + 13
            quantized = stochastic_symmetric_quantize(original, bits, mixed_seed)
            target_rms = (quantized.float() - original.float()).pow(2).mean().sqrt()
            generator = torch.Generator(device=parameter.device); generator.manual_seed(mixed_seed + 47_921)
            noise = torch.randn(original.shape, generator=generator, device=original.device, dtype=torch.float32)
            noise = noise * (target_rms / noise.pow(2).mean().sqrt().clamp_min(1e-12))
            parameter.copy_((original.float() + noise).to(original.dtype))
            actual_rms = (parameter.float() - original.float()).pow(2).mean().sqrt()
            if float(target_rms) > 0:
                relative_errors.append(abs(float(actual_rms / target_rms) - 1.0))
    stats: dict[str, float | int] = {
        "target_count": len(targets),
        "max_relative_rms_error": max(relative_errors, default=0.0),
    }
    try:
        yield stats
    finally:
        with torch.no_grad():
            for parameter, original in backups:
                parameter.copy_(original)


def _effective_lora_deltas(model):
    for name, module in model.named_modules():
        lora_a = getattr(module, "lora_A", None)
        if not lora_a:
            continue
        active = list(getattr(module, "active_adapters", []))
        adapter = active[0] if active else next(iter(lora_a))
        if adapter not in lora_a or not hasattr(module, "get_delta_weight"):
            continue
        yield name, module, adapter


@contextmanager
def _temporary_effective_delta_quantization(
    model, bits: int, seed: int
) -> Iterator[list[str]]:
    """Replace effective BA updates with Q(BA), invariant to LoRA refactorization."""
    targets = list(_effective_lora_deltas(model))
    if not targets:
        raise ValueError("no active LoRA updates found")
    backups: list[tuple[torch.Tensor, torch.Tensor]] = []
    names: list[str] = []
    with torch.no_grad():
        for index, (name, module, adapter) in enumerate(targets):
            weight = module.get_base_layer().weight
            backups.append((weight, weight.detach().clone()))
            delta = module.get_delta_weight(adapter).to(device=weight.device, dtype=weight.dtype)
            quantized = stochastic_symmetric_quantize(
                delta, bits, seed * 1_000_003 + index * 97 + 13
            )
            weight.add_(quantized - delta)
            names.append(name)
    try:
        yield names
    finally:
        with torch.no_grad():
            for weight, original in backups:
                weight.copy_(original)


@contextmanager
def _temporary_pretrained_quantization(model, bits: int, seed: int, scope: str = "all") -> Iterator[None]:
    targets = list(_pretrained_targets(model, scope))
    if not targets:
        raise ValueError(f"no pretrained parameters selected for scope {scope}")
    backups = []
    with torch.no_grad():
        for index, (_, parameter) in enumerate(targets):
            original = parameter.detach().clone()
            backups.append((parameter, original))
            parameter.copy_(stochastic_symmetric_quantize(original, bits, seed * 1_000_003 + index * 97 + 13))
    try:
        yield
    finally:
        with torch.no_grad():
            for parameter, original in backups:
                parameter.copy_(original)


def _mean(values: list[float]) -> float:
    return fmean(values) if values else float("nan")


def _std(values: list[float]) -> float:
    return pstdev(values) if len(values) > 1 else 0.0


def _entropy_concentration(values: list[float]) -> tuple[float, float]:
    positive = [abs(value) + 1e-12 for value in values]
    total = sum(positive)
    probabilities = [value / total for value in positive]
    entropy = -sum(value * math.log(value) for value in probabilities)
    return entropy / math.log(len(values)) if len(values) > 1 else 0.0, max(probabilities)


def probe_pretrained(run_dir: str | Path, output_dir: str | Path | None = None) -> Path:
    run_dir = Path(run_dir).resolve()
    config = PretrainedConfig.load(run_dir / "config.resolved.json")
    seed_everything(config.seed)
    AutoModelForCausalLM, AutoTokenizer, _, PeftModel, _ = _load_transformers()
    tokenizer = AutoTokenizer.from_pretrained(run_dir / "adapter_best")
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    examples, _ = build_pretrained_examples(config, tokenizer)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float16 if device.type == "cuda" else torch.float32
    base = AutoModelForCausalLM.from_pretrained(config.model_name, torch_dtype=dtype)
    model = PeftModel.from_pretrained(base, run_dir / "adapter_best").merge_and_unload().to(device)
    model.config.use_cache = True
    target_dir = Path(output_dir).resolve() if output_dir else run_dir / "tomography"
    target_dir.mkdir(parents=True, exist_ok=True)
    baseline = _evaluate(model, tokenizer, examples, device, config.eval_batch_size)
    global_results: dict[int, dict[int, list[dict[str, float | bool]]]] = {
        example.index: {bits: [] for bits in config.probe_bits} for example in examples
    }
    for bits in config.probe_bits:
        for seed in config.probe_seeds:
            with _temporary_pretrained_quantization(model, bits, seed):
                condition = _evaluate(model, tokenizer, examples, device, config.eval_batch_size)
            for index, values in condition.items():
                global_results[index][bits].append(values)
    layer_results: dict[int, dict[int, list[dict[str, float | bool]]]] = {
        example.index: {} for example in examples
    }
    if config.layerwise:
        n_layers = int(model.config.num_hidden_layers)
        for layer in range(n_layers):
            with _temporary_pretrained_quantization(model, config.layer_bits, config.layer_seed, str(layer)):
                condition = _evaluate(model, tokenizer, examples, device, config.eval_batch_size)
            for index, values in condition.items():
                layer_results[index].setdefault(layer, []).append(values)
    rows: list[dict[str, object]] = []
    for example in examples:
        base_values = baseline[example.index]
        row: dict[str, object] = {
            "index": example.index,
            "group": example.group,
            "relation": example.relation,
            "key": example.resident,
            "value": example.answer,
            "is_member": example.is_member,
            "base_nll": base_values["nll"],
            "base_margin": base_values["margin"],
            "base_correct": base_values["correct"],
        }
        all_abs: list[float] = []
        all_variability: list[float] = []
        all_survival: list[float] = []
        for bits in config.probe_bits:
            conditions = global_results[example.index][bits]
            deltas = [float(condition["nll"]) - float(base_values["nll"]) for condition in conditions]
            survival = [float(bool(condition["correct"])) for condition in conditions]
            row[f"q{bits}_mean_delta_nll"] = _mean(deltas)
            row[f"q{bits}_std_delta_nll"] = _std(deltas)
            row[f"q{bits}_mean_abs_delta_nll"] = _mean([abs(value) for value in deltas])
            row[f"q{bits}_survival"] = _mean(survival)
            all_abs.extend(abs(value) for value in deltas)
            all_variability.append(_std(deltas))
            all_survival.extend(survival)
        row["q_fragility"] = _mean(all_abs)
        row["q_variability"] = _mean(all_variability)
        row["q_survival_area"] = _mean(all_survival)
        sensitivities = []
        for layer, conditions in layer_results[example.index].items():
            sensitivity = _mean([
                abs(float(condition["nll"]) - float(base_values["nll"])) for condition in conditions
            ])
            row[f"layer_model.layers.{layer}_sensitivity"] = sensitivity
            sensitivities.append(sensitivity)
        row["layer_entropy"], row["layer_concentration"] = (
            _entropy_concentration(sensitivities) if sensitivities else (0.0, 0.0)
        )
        rows.append(row)
    fieldnames = sorted({name for row in rows for name in row})
    with (target_dir / "features.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    selected = [row for row in rows if row["group"] in {"generalized", "memorized"} and row["base_correct"]]
    labels = [int(row["group"] == "memorized") for row in selected]
    groups = [int(row["relation"]) for row in selected]
    baseline_names = ["base_nll", "base_margin"]
    tomography_names = ["q_fragility", "q_variability", "q_survival_area", "layer_entropy", "layer_concentration"]
    features = lambda names: [[float(row[name]) for name in names] for row in selected]
    result = {
        "model_name": config.model_name,
        "adapter": str(run_dir / "adapter_best"),
        "intervention": "stochastic_symmetric_quantization_of_merged_lora_transformer_weights",
        "counts": {"generalized": labels.count(0), "memorized": labels.count(1)},
        "full_precision_accuracy": {
            group: fmean(float(baseline[e.index]["correct"]) for e in examples if e.group == group)
            for group in ("seen_rule", "generalized", "memorized", "nonmember")
        },
        "example_random_cv_auc": {
            "confidence": logistic_cross_validated_auc(features(baseline_names), labels, config.seed),
            "tomography": logistic_cross_validated_auc(features(tomography_names), labels, config.seed),
            "combined": logistic_cross_validated_auc(features(baseline_names + tomography_names), labels, config.seed),
        },
        "relation_held_out_cv_auc": {
            "confidence": logistic_group_cross_validated_auc(features(baseline_names), labels, groups, config.seed),
            "tomography": logistic_group_cross_validated_auc(features(tomography_names), labels, groups, config.seed),
            "combined": logistic_group_cross_validated_auc(features(baseline_names + tomography_names), labels, groups, config.seed),
        },
        "probe": {
            "bits": config.probe_bits,
            "seeds": config.probe_seeds,
            "layerwise": config.layerwise,
            "layer_bits": config.layer_bits,
            "layer_seed": config.layer_seed,
        },
        "max_memory_gib": torch.cuda.max_memory_allocated() / 2**30 if device.type == "cuda" else 0.0,
    }
    for section in ("example_random_cv_auc", "relation_held_out_cv_auc"):
        result[section]["incremental"] = result[section]["combined"] - result[section]["confidence"]
    analysis_path = target_dir / "analysis.json"
    analysis_path.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)
    return analysis_path


def probe_pretrained_decomposition(
    run_dir: str | Path, output_dir: str | Path | None = None
) -> Path:
    """Compare quantization sensitivity of the base, LoRA, and merged weights."""
    run_path = Path(run_dir).resolve()
    config = PretrainedConfig.load(run_path / "config.resolved.json")
    seed_everything(config.seed)
    AutoModelForCausalLM, AutoTokenizer, _, PeftModel, _ = _load_transformers()
    tokenizer = AutoTokenizer.from_pretrained(run_path / "adapter_best")
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    examples, _ = build_pretrained_examples(config, tokenizer)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float16 if device.type == "cuda" else torch.float32
    target_dir = Path(output_dir).resolve() if output_dir else run_path / "decomposition"
    target_dir.mkdir(parents=True, exist_ok=True)
    scopes = ("base", "lora", "merged")
    scope_results: dict[str, dict] = {}
    for scope in scopes:
        base = AutoModelForCausalLM.from_pretrained(config.model_name, torch_dtype=dtype)
        if scope == "merged":
            model = PeftModel.from_pretrained(base, run_path / "adapter_best").merge_and_unload().to(device)
        else:
            model = PeftModel.from_pretrained(base, run_path / "adapter_best").to(device)
        model.eval()
        baseline = _evaluate(model, tokenizer, examples, device, config.eval_batch_size)
        global_results: dict[int, dict[int, list[dict[str, float | bool]]]] = {
            example.index: {bits: [] for bits in config.probe_bits} for example in examples
        }
        for bits in config.probe_bits:
            for probe_seed in config.probe_seeds:
                with _temporary_decomposition_quantization(model, scope, bits, probe_seed):
                    condition = _evaluate(model, tokenizer, examples, device, config.eval_batch_size)
                for index, values in condition.items():
                    global_results[index][bits].append(values)
        rows: list[dict[str, object]] = []
        for example in examples:
            base_values = baseline[example.index]
            row: dict[str, object] = {
                "index": example.index, "group": example.group, "relation": example.relation,
                "key": example.resident, "base_nll": base_values["nll"],
                "base_margin": base_values["margin"], "base_correct": base_values["correct"],
            }
            abs_deltas: list[float] = []
            survival: list[float] = []
            for bits in config.probe_bits:
                conditions = global_results[example.index][bits]
                deltas = [float(condition["nll"]) - float(base_values["nll"]) for condition in conditions]
                survives = [float(bool(condition["correct"])) for condition in conditions]
                row[f"q{bits}_mean_abs_delta_nll"] = _mean([abs(value) for value in deltas])
                row[f"q{bits}_survival"] = _mean(survives)
                abs_deltas.extend(abs(value) for value in deltas)
                survival.extend(survives)
            row["q_fragility"] = _mean(abs_deltas)
            row["q_survival_area"] = _mean(survival)
            rows.append(row)
        feature_path = target_dir / f"features_{scope}.csv"
        with feature_path.open("w", newline="", encoding="utf-8") as handle:
            fieldnames = sorted({name for row in rows for name in row})
            writer = csv.DictWriter(handle, fieldnames=fieldnames); writer.writeheader(); writer.writerows(rows)
        selected = [row for row in rows if row["group"] in {"generalized", "memorized"} and row["base_correct"]]
        labels = [int(row["group"] == "memorized") for row in selected]
        groups = [int(row["relation"]) for row in selected]
        feature = lambda names: [[float(row[name]) for name in names] for row in selected]
        baseline_names = ["base_nll", "base_margin"]
        probe_names = ["q_fragility", "q_survival_area"]
        scope_results[scope] = {
            "scope": scope,
            "features": str(feature_path),
            "full_precision_accuracy": {
                group: fmean(float(baseline[e.index]["correct"]) for e in examples if e.group == group)
                for group in ("seen_rule", "generalized", "memorized", "nonmember")
            },
            "example_random_cv_auc": {
                "confidence": logistic_cross_validated_auc(feature(baseline_names), labels, config.seed),
                "tomography": logistic_cross_validated_auc(feature(probe_names), labels, config.seed),
                "combined": logistic_cross_validated_auc(feature(baseline_names + probe_names), labels, config.seed),
            },
            "relation_held_out_cv_auc": {
                "confidence": logistic_group_cross_validated_auc(feature(baseline_names), labels, groups, config.seed),
                "tomography": logistic_group_cross_validated_auc(feature(probe_names), labels, groups, config.seed),
                "combined": logistic_group_cross_validated_auc(feature(baseline_names + probe_names), labels, groups, config.seed),
            },
        }
        for section in ("example_random_cv_auc", "relation_held_out_cv_auc"):
            scope_results[scope][section]["incremental"] = (
                scope_results[scope][section]["combined"] - scope_results[scope][section]["confidence"]
            )
        del model
        if device.type == "cuda": torch.cuda.empty_cache()
    result = {
        "model_name": config.model_name,
        "adapter": str(run_path / "adapter_best"),
        "intervention": "matched_stochastic_symmetric_quantization_by_parameter_scope",
        "probe": {"bits": config.probe_bits, "seeds": config.probe_seeds},
        "scopes": scope_results,
    }
    path = target_dir / "analysis.json"
    path.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    return path


def aggregate_pretrained_decomposition(
    run_dirs: list[str | Path], output_dir: str | Path
) -> Path:
    """Aggregate scope-specific decomposition metrics across runs."""
    out = Path(output_dir).resolve(); out.mkdir(parents=True, exist_ok=True)
    records = []
    for raw_run_dir in run_dirs:
        run_dir = Path(raw_run_dir).resolve()
        config = PretrainedConfig.load(run_dir / "config.resolved.json")
        analysis = json.loads((run_dir / "decomposition" / "analysis.json").read_text(encoding="utf-8"))
        for scope, values in analysis["scopes"].items():
            for section in ("example_random_cv_auc", "relation_held_out_cv_auc"):
                for metric, value in values[section].items():
                    records.append({"seed": config.seed, "scope": scope, "section": section,
                                    "metric": metric, "value": float(value)})
    summary = []
    keys = sorted({(r["scope"], r["section"], r["metric"]) for r in records})
    for scope, section, metric in keys:
        values = [r["value"] for r in records if r["scope"] == scope and r["section"] == section and r["metric"] == metric]
        summary.append({"scope": scope, "section": section, "metric": metric, "n": len(values),
                        "mean": fmean(values), "std": pstdev(values)})
    artifact = {"protocol_version": 1, "run_dirs": [str(Path(path).resolve()) for path in run_dirs],
                "records": records, "summary": summary}
    path = out / "analysis.json"
    path.write_text(json.dumps(artifact, indent=2, sort_keys=True), encoding="utf-8")
    return path


def probe_pretrained_matched_noise(
    run_dir: str | Path, output_dir: str | Path | None = None
) -> Path:
    """Run MSE-matched Gaussian controls for base-only and merged scopes."""
    run_path = Path(run_dir).resolve()
    config = PretrainedConfig.load(run_path / "config.resolved.json")
    seed_everything(config.seed)
    AutoModelForCausalLM, AutoTokenizer, _, PeftModel, _ = _load_transformers()
    tokenizer = AutoTokenizer.from_pretrained(run_path / "adapter_best")
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    examples, _ = build_pretrained_examples(config, tokenizer)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float16 if device.type == "cuda" else torch.float32
    target_dir = Path(output_dir).resolve() if output_dir else run_path / "matched_noise"
    target_dir.mkdir(parents=True, exist_ok=True)
    scope_results: dict[str, dict] = {}
    for scope in ("base", "merged"):
        base = AutoModelForCausalLM.from_pretrained(config.model_name, torch_dtype=dtype)
        model = PeftModel.from_pretrained(base, run_path / "adapter_best")
        if scope == "merged":
            model = model.merge_and_unload()
        model = model.to(device); model.eval()
        baseline = _evaluate(model, tokenizer, examples, device, config.eval_batch_size)
        conditions: dict[int, dict[int, list[dict[str, float | bool]]]] = {
            example.index: {bits: [] for bits in config.probe_bits} for example in examples
        }
        match_errors: list[float] = []
        target_count = 0
        for bits in config.probe_bits:
            for probe_seed in config.probe_seeds:
                with _temporary_decomposition_matched_noise(model, scope, bits, probe_seed) as stats:
                    target_count = int(stats["target_count"])
                    match_errors.append(float(stats["max_relative_rms_error"]))
                    condition = _evaluate(model, tokenizer, examples, device, config.eval_batch_size)
                for index, values in condition.items():
                    conditions[index][bits].append(values)
        restored = _evaluate(model, tokenizer, examples, device, config.eval_batch_size)
        rows: list[dict[str, object]] = []
        for example in examples:
            base_values = baseline[example.index]
            row: dict[str, object] = {
                "index": example.index, "group": example.group, "relation": example.relation,
                "key": example.resident, "base_nll": base_values["nll"],
                "base_margin": base_values["margin"], "base_correct": base_values["correct"],
            }
            all_abs: list[float] = []
            all_variability: list[float] = []
            all_survival: list[float] = []
            for bits in config.probe_bits:
                values = conditions[example.index][bits]
                deltas = [float(item["nll"]) - float(base_values["nll"]) for item in values]
                survival = [float(bool(item["correct"])) for item in values]
                row[f"p{bits}_mean_abs_delta_nll"] = _mean([abs(value) for value in deltas])
                row[f"p{bits}_survival"] = _mean(survival)
                all_abs.extend(abs(value) for value in deltas)
                all_variability.append(_std(deltas))
                all_survival.extend(survival)
            row["p_fragility"] = _mean(all_abs)
            row["p_variability"] = _mean(all_variability)
            row["p_survival_area"] = _mean(all_survival)
            rows.append(row)
        feature_path = target_dir / f"features_{scope}.csv"
        with feature_path.open("w", newline="", encoding="utf-8") as handle:
            fieldnames = sorted({name for row in rows for name in row})
            writer = csv.DictWriter(handle, fieldnames=fieldnames); writer.writeheader(); writer.writerows(rows)
        selected = [row for row in rows if row["group"] in {"generalized", "memorized"} and row["base_correct"]]
        labels = [int(row["group"] == "memorized") for row in selected]
        groups = [int(row["relation"]) for row in selected]
        feature = lambda names: [[float(row[name]) for name in names] for row in selected]
        baseline_names = ["base_nll", "base_margin"]
        probe_names = ["p_fragility", "p_variability", "p_survival_area"]
        scope_results[scope] = {
            "features": str(feature_path), "target_count": target_count,
            "max_relative_rms_error": max(match_errors, default=0.0),
            "restoration": {
                "max_abs_nll_difference": max(abs(float(restored[i]["nll"]) - float(baseline[i]["nll"])) for i in baseline),
                "changed_predictions": sum(restored[i]["correct"] != baseline[i]["correct"] for i in baseline),
            },
            "example_random_cv_auc": {
                "confidence": logistic_cross_validated_auc(feature(baseline_names), labels, config.seed),
                "tomography": logistic_cross_validated_auc(feature(probe_names), labels, config.seed),
                "combined": logistic_cross_validated_auc(feature(baseline_names + probe_names), labels, config.seed),
            },
            "relation_held_out_cv_auc": {
                "confidence": logistic_group_cross_validated_auc(feature(baseline_names), labels, groups, config.seed),
                "tomography": logistic_group_cross_validated_auc(feature(probe_names), labels, groups, config.seed),
                "combined": logistic_group_cross_validated_auc(feature(baseline_names + probe_names), labels, groups, config.seed),
            },
        }
        for section in ("example_random_cv_auc", "relation_held_out_cv_auc"):
            scope_results[scope][section]["incremental"] = (
                scope_results[scope][section]["combined"] - scope_results[scope][section]["confidence"]
            )
        del model
        if device.type == "cuda": torch.cuda.empty_cache()
    artifact = {
        "model_name": config.model_name,
        "intervention": "per_tensor_quantization_MSE_matched_Gaussian_noise",
        "probe": {"bits": config.probe_bits, "seeds": config.probe_seeds},
        "scopes": scope_results,
    }
    path = target_dir / "analysis.json"
    path.write_text(json.dumps(artifact, indent=2, sort_keys=True), encoding="utf-8")
    return path


def probe_pretrained_prompt_robustness(
    run_dir: str | Path, output_dir: str | Path | None = None
) -> Path:
    """Probe base/merged tomography under fixed prompts unseen during training."""
    run_path = Path(run_dir).resolve()
    config = PretrainedConfig.load(run_path / "config.resolved.json")
    seed_everything(config.seed)
    AutoModelForCausalLM, AutoTokenizer, _, PeftModel, _ = _load_transformers()
    tokenizer = AutoTokenizer.from_pretrained(run_path / "adapter_best")
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    canonical, _ = build_pretrained_examples(config, tokenizer)
    template_examples = {
        name: paraphrase_pretrained_examples(config, canonical, name)
        for name in UNSEEN_EVAL_TEMPLATES
    }
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float16 if device.type == "cuda" else torch.float32
    target_dir = Path(output_dir).resolve() if output_dir else run_path / "prompt_robustness"
    target_dir.mkdir(parents=True, exist_ok=True)
    scope_results: dict[str, dict] = {}
    for scope in ("base", "merged"):
        base = AutoModelForCausalLM.from_pretrained(config.model_name, torch_dtype=dtype)
        model = PeftModel.from_pretrained(base, run_path / "adapter_best")
        if scope == "merged":
            model = model.merge_and_unload()
        model = model.to(device); model.eval()
        baseline = {
            name: _evaluate(model, tokenizer, examples, device, config.eval_batch_size)
            for name, examples in template_examples.items()
        }
        conditions = {
            name: {
                example.index: {bits: [] for bits in config.probe_bits}
                for example in examples
            }
            for name, examples in template_examples.items()
        }
        for bits in config.probe_bits:
            for probe_seed in config.probe_seeds:
                with _temporary_decomposition_quantization(model, scope, bits, probe_seed):
                    evaluated = {
                        name: _evaluate(model, tokenizer, examples, device, config.eval_batch_size)
                        for name, examples in template_examples.items()
                    }
                for name, values in evaluated.items():
                    for index, result in values.items():
                        conditions[name][index][bits].append(result)
        template_results: dict[str, dict] = {}
        for name, examples in template_examples.items():
            rows: list[dict[str, object]] = []
            for example in examples:
                base_values = baseline[name][example.index]
                row: dict[str, object] = {
                    "index": example.index, "group": example.group, "relation": example.relation,
                    "key": example.resident, "base_nll": base_values["nll"],
                    "base_margin": base_values["margin"], "base_correct": base_values["correct"],
                }
                all_abs: list[float] = []
                all_variability: list[float] = []
                all_survival: list[float] = []
                for bits in config.probe_bits:
                    values = conditions[name][example.index][bits]
                    deltas = [float(item["nll"]) - float(base_values["nll"]) for item in values]
                    survival = [float(bool(item["correct"])) for item in values]
                    row[f"q{bits}_mean_abs_delta_nll"] = _mean([abs(value) for value in deltas])
                    row[f"q{bits}_survival"] = _mean(survival)
                    all_abs.extend(abs(value) for value in deltas)
                    all_variability.append(_std(deltas))
                    all_survival.extend(survival)
                row["q_fragility"] = _mean(all_abs)
                row["q_variability"] = _mean(all_variability)
                row["q_survival_area"] = _mean(all_survival)
                rows.append(row)
            feature_path = target_dir / f"features_{scope}_{name}.csv"
            with feature_path.open("w", newline="", encoding="utf-8") as handle:
                fieldnames = sorted({column for row in rows for column in row})
                writer = csv.DictWriter(handle, fieldnames=fieldnames); writer.writeheader(); writer.writerows(rows)
            selected = [row for row in rows if row["group"] in {"generalized", "memorized"} and row["base_correct"]]
            labels = [int(row["group"] == "memorized") for row in selected]
            groups = [int(row["relation"]) for row in selected]
            feature = lambda columns: [[float(row[column]) for column in columns] for row in selected]
            baseline_names = ["base_nll", "base_margin"]
            probe_names = ["q_fragility", "q_variability", "q_survival_area"]
            template_results[name] = {
                "template": UNSEEN_EVAL_TEMPLATES[name], "features": str(feature_path),
                "selected_count": len(selected),
                "full_precision_accuracy": {
                    group: fmean(float(baseline[name][e.index]["correct"]) for e in examples if e.group == group)
                    for group in ("seen_rule", "generalized", "memorized", "nonmember")
                },
                "relation_held_out_cv_auc": {
                    "confidence": logistic_group_cross_validated_auc(feature(baseline_names), labels, groups, config.seed),
                    "tomography": logistic_group_cross_validated_auc(feature(probe_names), labels, groups, config.seed),
                    "combined": logistic_group_cross_validated_auc(feature(baseline_names + probe_names), labels, groups, config.seed),
                },
            }
            template_results[name]["relation_held_out_cv_auc"]["incremental"] = (
                template_results[name]["relation_held_out_cv_auc"]["combined"]
                - template_results[name]["relation_held_out_cv_auc"]["confidence"]
            )
        scope_results[scope] = template_results
        del model
        if device.type == "cuda": torch.cuda.empty_cache()
    artifact = {
        "model_name": config.model_name,
        "intervention": "base_and_merged_quantization_under_unseen_prompt_templates",
        "templates": UNSEEN_EVAL_TEMPLATES,
        "probe": {"bits": config.probe_bits, "seeds": config.probe_seeds},
        "scopes": scope_results,
    }
    path = target_dir / "analysis.json"
    path.write_text(json.dumps(artifact, indent=2, sort_keys=True), encoding="utf-8")
    return path


def probe_pretrained_effective_delta(
    run_dir: str | Path, output_dir: str | Path | None = None
) -> Path:
    """Probe a representation-invariant quantization of each effective LoRA update."""
    run_path = Path(run_dir).resolve()
    config = PretrainedConfig.load(run_path / "config.resolved.json")
    seed_everything(config.seed)
    AutoModelForCausalLM, AutoTokenizer, _, PeftModel, _ = _load_transformers()
    tokenizer = AutoTokenizer.from_pretrained(run_path / "adapter_best")
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    examples, _ = build_pretrained_examples(config, tokenizer)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float16 if device.type == "cuda" else torch.float32
    base = AutoModelForCausalLM.from_pretrained(config.model_name, torch_dtype=dtype)
    model = PeftModel.from_pretrained(base, run_path / "adapter_best").to(device)
    model.eval()
    target_dir = Path(output_dir).resolve() if output_dir else run_path / "effective_delta"
    target_dir.mkdir(parents=True, exist_ok=True)
    baseline = _evaluate(model, tokenizer, examples, device, config.eval_batch_size)
    results: dict[int, dict[int, list[dict[str, float | bool]]]] = {
        example.index: {bits: [] for bits in config.probe_bits} for example in examples
    }
    target_names: list[str] = []
    for bits in config.probe_bits:
        for probe_seed in config.probe_seeds:
            with _temporary_effective_delta_quantization(model, bits, probe_seed) as names:
                target_names = names
                condition = _evaluate(model, tokenizer, examples, device, config.eval_batch_size)
            for index, values in condition.items():
                results[index][bits].append(values)
    restored = _evaluate(model, tokenizer, examples, device, config.eval_batch_size)
    restoration = {
        "max_abs_nll_difference": max(
            abs(float(restored[index]["nll"]) - float(baseline[index]["nll"])) for index in baseline
        ),
        "changed_predictions": sum(
            restored[index]["correct"] != baseline[index]["correct"] for index in baseline
        ),
    }
    rows: list[dict[str, object]] = []
    for example in examples:
        base_values = baseline[example.index]
        row: dict[str, object] = {
            "index": example.index, "group": example.group, "relation": example.relation,
            "key": example.resident, "base_nll": base_values["nll"],
            "base_margin": base_values["margin"], "base_correct": base_values["correct"],
        }
        all_abs: list[float] = []
        all_variability: list[float] = []
        all_survival: list[float] = []
        for bits in config.probe_bits:
            conditions = results[example.index][bits]
            deltas = [float(condition["nll"]) - float(base_values["nll"]) for condition in conditions]
            survival = [float(bool(condition["correct"])) for condition in conditions]
            row[f"q{bits}_mean_delta_nll"] = _mean(deltas)
            row[f"q{bits}_mean_abs_delta_nll"] = _mean([abs(value) for value in deltas])
            row[f"q{bits}_std_delta_nll"] = _std(deltas)
            row[f"q{bits}_survival"] = _mean(survival)
            all_abs.extend(abs(value) for value in deltas)
            all_variability.append(_std(deltas))
            all_survival.extend(survival)
        row["q_fragility"] = _mean(all_abs)
        row["q_variability"] = _mean(all_variability)
        row["q_survival_area"] = _mean(all_survival)
        rows.append(row)
    feature_path = target_dir / "features.csv"
    with feature_path.open("w", newline="", encoding="utf-8") as handle:
        fieldnames = sorted({name for row in rows for name in row})
        writer = csv.DictWriter(handle, fieldnames=fieldnames); writer.writeheader(); writer.writerows(rows)
    selected = [row for row in rows if row["group"] in {"generalized", "memorized"} and row["base_correct"]]
    labels = [int(row["group"] == "memorized") for row in selected]
    groups = [int(row["relation"]) for row in selected]
    feature = lambda names: [[float(row[name]) for name in names] for row in selected]
    baseline_names = ["base_nll", "base_margin"]
    probe_names = ["q_fragility", "q_variability", "q_survival_area"]
    analysis = {
        "model_name": config.model_name,
        "adapter": str(run_path / "adapter_best"),
        "intervention": "stochastic_symmetric_quantization_of_effective_lora_delta_BA",
        "target_module_count": len(target_names),
        "features": str(feature_path),
        "restoration": restoration,
        "probe": {"bits": config.probe_bits, "seeds": config.probe_seeds},
        "example_random_cv_auc": {
            "confidence": logistic_cross_validated_auc(feature(baseline_names), labels, config.seed),
            "tomography": logistic_cross_validated_auc(feature(probe_names), labels, config.seed),
            "combined": logistic_cross_validated_auc(feature(baseline_names + probe_names), labels, config.seed),
        },
        "relation_held_out_cv_auc": {
            "confidence": logistic_group_cross_validated_auc(feature(baseline_names), labels, groups, config.seed),
            "tomography": logistic_group_cross_validated_auc(feature(probe_names), labels, groups, config.seed),
            "combined": logistic_group_cross_validated_auc(feature(baseline_names + probe_names), labels, groups, config.seed),
        },
    }
    for section in ("example_random_cv_auc", "relation_held_out_cv_auc"):
        analysis[section]["incremental"] = analysis[section]["combined"] - analysis[section]["confidence"]
    path = target_dir / "analysis.json"
    path.write_text(json.dumps(analysis, indent=2, sort_keys=True), encoding="utf-8")
    return path


def cross_seed_pretrained_scope_transfer(
    run_dirs: list[str | Path], output_dir: str | Path
) -> Path:
    """Train scope-specific detectors on two runs and test the held-out run."""
    out = Path(output_dir).resolve(); out.mkdir(parents=True, exist_ok=True)
    runs = []
    for raw_run_dir in run_dirs:
        run_dir = Path(raw_run_dir).resolve()
        seed = PretrainedConfig.load(run_dir / "config.resolved.json").seed
        runs.append((seed, run_dir))
    scope_paths = {
        "base": lambda run: run / "decomposition" / "features_base.csv",
        "raw_lora_factors": lambda run: run / "decomposition" / "features_lora.csv",
        "merged": lambda run: run / "decomposition" / "features_merged.csv",
        "effective_delta": lambda run: run / "effective_delta" / "features.csv",
    }
    datasets: dict[str, dict[int, list[dict[str, str]]]] = {}
    for scope, path_for in scope_paths.items():
        datasets[scope] = {}
        for seed, run_dir in runs:
            rows = list(csv.DictReader(path_for(run_dir).open(encoding="utf-8")))
            datasets[scope][seed] = [
                row for row in rows
                if row["group"] in {"generalized", "memorized"} and row["base_correct"].lower() == "true"
            ]
    records = []
    feature_sets = {
        "confidence": ["base_nll", "base_margin"],
        "tomography": ["q_fragility", "q_survival_area"],
        "combined": ["base_nll", "base_margin", "q_fragility", "q_survival_area"],
    }
    for scope, by_seed in datasets.items():
        for held_out_seed in sorted(by_seed):
            train_rows = [row for seed, rows in by_seed.items() if seed != held_out_seed for row in rows]
            test_rows = by_seed[held_out_seed]
            train_labels = [int(row["group"] == "memorized") for row in train_rows]
            test_labels = [int(row["group"] == "memorized") for row in test_rows]
            values = {"scope": scope, "held_out_seed": held_out_seed,
                      "train_count": len(train_rows), "test_count": len(test_rows)}
            for name, columns in feature_sets.items():
                train_features = [[float(row[column]) for column in columns] for row in train_rows]
                test_features = [[float(row[column]) for column in columns] for row in test_rows]
                values[f"{name}_auc"] = logistic_train_test_auc(
                    train_features, train_labels, test_features, test_labels, held_out_seed
                )
            values["incremental_auc"] = values["combined_auc"] - values["confidence_auc"]
            records.append(values)
    summary = []
    for scope in scope_paths:
        group = [row for row in records if row["scope"] == scope]
        item = {"scope": scope, "n": len(group)}
        for metric in ("confidence_auc", "tomography_auc", "combined_auc", "incremental_auc"):
            values = [float(row[metric]) for row in group]
            item[f"{metric}_mean"] = fmean(values)
            item[f"{metric}_std"] = pstdev(values)
        summary.append(item)
    artifact = {"evaluation": "leave_one_training_seed_out", "records": records, "summary": summary}
    path = out / "analysis.json"
    path.write_text(json.dumps(artifact, indent=2, sort_keys=True), encoding="utf-8")
    return path


def cross_seed_pretrained_noise_control(
    run_dirs: list[str | Path], output_dir: str | Path
) -> Path:
    """Compare quantization and MSE-matched noise under held-out-seed transfer."""
    out = Path(output_dir).resolve(); out.mkdir(parents=True, exist_ok=True)
    runs = [(PretrainedConfig.load(Path(path) / "config.resolved.json").seed, Path(path).resolve()) for path in run_dirs]
    conditions = {
        ("quantization", "base"): (lambda run: run / "decomposition" / "features_base.csv", ["q_fragility", "q_survival_area"]),
        ("quantization", "merged"): (lambda run: run / "decomposition" / "features_merged.csv", ["q_fragility", "q_survival_area"]),
        ("matched_noise", "base"): (lambda run: run / "matched_noise" / "features_base.csv", ["p_fragility", "p_survival_area"]),
        ("matched_noise", "merged"): (lambda run: run / "matched_noise" / "features_merged.csv", ["p_fragility", "p_survival_area"]),
    }
    records = []
    for (intervention, scope), (path_for, probe_columns) in conditions.items():
        by_seed = {}
        for seed, run_dir in runs:
            rows = list(csv.DictReader(path_for(run_dir).open(encoding="utf-8")))
            by_seed[seed] = [row for row in rows if row["group"] in {"generalized", "memorized"} and row["base_correct"].lower() == "true"]
        for held_out_seed in sorted(by_seed):
            train_rows = [row for seed, rows in by_seed.items() if seed != held_out_seed for row in rows]
            test_rows = by_seed[held_out_seed]
            train_labels = [int(row["group"] == "memorized") for row in train_rows]
            test_labels = [int(row["group"] == "memorized") for row in test_rows]
            feature_sets = {
                "confidence": ["base_nll", "base_margin"],
                "tomography": probe_columns,
                "combined": ["base_nll", "base_margin"] + probe_columns,
            }
            row = {"intervention": intervention, "scope": scope, "held_out_seed": held_out_seed}
            for name, columns in feature_sets.items():
                row[f"{name}_auc"] = logistic_train_test_auc(
                    [[float(item[column]) for column in columns] for item in train_rows], train_labels,
                    [[float(item[column]) for column in columns] for item in test_rows], test_labels,
                    held_out_seed,
                )
            row["incremental_auc"] = row["combined_auc"] - row["confidence_auc"]
            records.append(row)
    summary = []
    for intervention, scope in conditions:
        group = [row for row in records if row["intervention"] == intervention and row["scope"] == scope]
        item = {"intervention": intervention, "scope": scope, "n": len(group)}
        for metric in ("confidence_auc", "tomography_auc", "combined_auc", "incremental_auc"):
            values = [float(row[metric]) for row in group]
            item[f"{metric}_mean"] = fmean(values); item[f"{metric}_std"] = pstdev(values)
        summary.append(item)
    paired_advantage = []
    for scope in ("base", "merged"):
        quant = [row for row in records if row["intervention"] == "quantization" and row["scope"] == scope]
        noise = [row for row in records if row["intervention"] == "matched_noise" and row["scope"] == scope]
        for metric in ("tomography_auc", "combined_auc", "incremental_auc"):
            differences = [
                next(row[metric] for row in quant if row["held_out_seed"] == seed)
                - next(row[metric] for row in noise if row["held_out_seed"] == seed)
                for seed, _ in runs
            ]
            paired_advantage.append({"scope": scope, "metric": metric,
                                     "quantization_minus_noise_mean": fmean(differences),
                                     "differences_by_seed": dict(zip([seed for seed, _ in runs], differences))})
    artifact = {"evaluation": "leave_one_training_seed_out", "records": records,
                "summary": summary, "paired_quantization_advantage": paired_advantage}
    path = out / "analysis.json"
    path.write_text(json.dumps(artifact, indent=2, sort_keys=True), encoding="utf-8")
    return path


def cross_seed_pretrained_prompt_transfer(
    run_dirs: list[str | Path], output_dir: str | Path
) -> Path:
    """Train on canonical prompts and test unseen prompts on an unseen seed."""
    out = Path(output_dir).resolve(); out.mkdir(parents=True, exist_ok=True)
    runs = [(PretrainedConfig.load(Path(path) / "config.resolved.json").seed, Path(path).resolve()) for path in run_dirs]

    def read_selected(path: Path) -> list[dict[str, str]]:
        rows = list(csv.DictReader(path.open(encoding="utf-8")))
        return [row for row in rows if row["group"] in {"generalized", "memorized"} and row["base_correct"].lower() == "true"]

    def ranks(values: list[float]) -> list[float]:
        order = sorted(range(len(values)), key=lambda index: values[index])
        result = [0.0] * len(values); position = 0
        while position < len(order):
            end = position + 1
            while end < len(order) and values[order[end]] == values[order[position]]:
                end += 1
            rank = (position + end - 1) / 2.0
            for offset in range(position, end): result[order[offset]] = rank
            position = end
        return result

    def correlation(left: list[float], right: list[float]) -> float:
        x = ranks(left); y = ranks(right); mx = fmean(x); my = fmean(y)
        numerator = sum((a - mx) * (b - my) for a, b in zip(x, y))
        denominator = math.sqrt(sum((a - mx) ** 2 for a in x) * sum((b - my) ** 2 for b in y))
        return numerator / denominator if denominator else float("nan")

    records = []
    correlations = []
    feature_sets = {
        "confidence": ["base_nll", "base_margin"],
        "tomography": ["q_fragility", "q_survival_area"],
        "combined": ["base_nll", "base_margin", "q_fragility", "q_survival_area"],
    }
    for scope in ("base", "merged"):
        canonical_by_seed = {
            seed: read_selected(run / "decomposition" / f"features_{scope}.csv")
            for seed, run in runs
        }
        for held_out_seed, run in runs:
            train_rows = [row for seed, rows in canonical_by_seed.items() if seed != held_out_seed for row in rows]
            train_labels = [int(row["group"] == "memorized") for row in train_rows]
            canonical_all = list(csv.DictReader((run / "decomposition" / f"features_{scope}.csv").open(encoding="utf-8")))
            canonical_fragility = {int(row["index"]): float(row["q_fragility"]) for row in canonical_all}
            for template in UNSEEN_EVAL_TEMPLATES:
                feature_path = run / "prompt_robustness" / f"features_{scope}_{template}.csv"
                test_rows = read_selected(feature_path)
                test_labels = [int(row["group"] == "memorized") for row in test_rows]
                item = {"scope": scope, "held_out_seed": held_out_seed, "test_template": template,
                        "train_count": len(train_rows), "test_count": len(test_rows)}
                for name, columns in feature_sets.items():
                    item[f"{name}_auc"] = logistic_train_test_auc(
                        [[float(row[column]) for column in columns] for row in train_rows], train_labels,
                        [[float(row[column]) for column in columns] for row in test_rows], test_labels,
                        held_out_seed,
                    )
                item["incremental_auc"] = item["combined_auc"] - item["confidence_auc"]
                records.append(item)
                prompt_all = list(csv.DictReader(feature_path.open(encoding="utf-8")))
                correlations.append({
                    "scope": scope, "seed": held_out_seed, "template": template,
                    "spearman_q_fragility": correlation(
                        [canonical_fragility[int(row["index"])] for row in prompt_all],
                        [float(row["q_fragility"]) for row in prompt_all],
                    ),
                })
    summary = []
    for scope in ("base", "merged"):
        group = [row for row in records if row["scope"] == scope]
        item = {"scope": scope, "n": len(group)}
        for metric in ("confidence_auc", "tomography_auc", "combined_auc", "incremental_auc"):
            values = [float(row[metric]) for row in group]
            item[f"{metric}_mean"] = fmean(values); item[f"{metric}_std"] = pstdev(values)
            if metric == "incremental_auc":
                item["incremental_auc_positive_count"] = sum(value > 0 for value in values)
        corr = [float(row["spearman_q_fragility"]) for row in correlations if row["scope"] == scope]
        item["spearman_q_fragility_mean"] = fmean(corr)
        summary.append(item)
    artifact = {
        "evaluation": "canonical_two_seed_train_to_unseen_prompt_and_unseen_seed_test",
        "templates": UNSEEN_EVAL_TEMPLATES, "records": records,
        "fragility_correlations": correlations, "summary": summary,
    }
    path = out / "analysis.json"
    path.write_text(json.dumps(artifact, indent=2, sort_keys=True), encoding="utf-8")
    return path


def run_pretrained(config: PretrainedConfig) -> Path:
    train_pretrained(config)
    return probe_pretrained(config.output_dir)


def _counter_targets(
    examples: list[PretrainedExample], city_target_ids: list[int], seed: int
) -> dict[tuple[int, int], int]:
    """Choose deterministic, valid city-token replacements."""
    replacements: dict[tuple[int, int], int] = {}
    for example in examples:
        alternatives = [target for target in city_target_ids if target != example.target_id]
        if not alternatives:
            raise ValueError("target vocabulary must contain at least two distinct city tokens")
        offset = (example.relation * 131 + example.resident * 17 + seed) % len(alternatives)
        replacements[(example.relation, example.resident)] = alternatives[offset]
    return replacements


@torch.inference_mode()
def _reference_city_probabilities(
    model, tokenizer, examples: list[PretrainedExample], city_target_ids: list[int],
    device: torch.device, batch_size: int,
) -> dict[int, torch.Tensor]:
    model.eval()
    result: dict[int, torch.Tensor] = {}
    city_ids = torch.tensor(city_target_ids, device=device)
    for start in range(0, len(examples), batch_size):
        batch = examples[start : start + batch_size]
        encoded = tokenizer(
            [example.prompt for example in batch], return_tensors="pt", padding=True,
            add_special_tokens=False,
        )
        encoded = {name: tensor.to(device) for name, tensor in encoded.items()}
        logits = model(**encoded).logits.float()
        positions = encoded["attention_mask"].sum(dim=1) - 1
        next_logits = logits[torch.arange(len(batch), device=device), positions]
        probabilities = next_logits.index_select(1, city_ids).softmax(dim=-1).cpu()
        for offset, example in enumerate(batch):
            result[example.index] = probabilities[offset]
    return result


def targeted_unlearning_pretrained(
    run_dir: str | Path,
    features: str | Path,
    output_dir: str | Path | None = None,
    fraction: float = 0.20,
    steps: int = 100,
    seed: int = 42,
    learning_rate: float = 2e-4,
    kl_weight: float = 1.0,
    strategies: list[str] | None = None,
    candidate_scope: str = "train_members",
) -> Path:
    """Compare PALM-style targeted rewriting on a pretrained LoRA run.

    The base model stays frozen and a fresh LoRA adapter is trained per selector.
    Selected examples receive valid alternative city-token targets. Retained
    training members receive their original targets plus a city-distribution KL
    penalty against the untouched checkpoint. Held-out examples are never used
    for optimization.
    """
    AutoModelForCausalLM, AutoTokenizer, LoraConfig, PeftModel, _ = _load_transformers()
    run_path = Path(run_dir).resolve()
    config = PretrainedConfig.load(run_path / "config.resolved.json")
    out = Path(output_dir or (run_path / "targeted_unlearning")).resolve()
    out.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float16 if device.type == "cuda" else torch.float32
    tokenizer = AutoTokenizer.from_pretrained(run_path / "adapter_best")
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    examples, _ = build_pretrained_examples(config, tokenizer)
    rows = list(csv.DictReader(Path(features).open(encoding="utf-8")))
    correct_mem = [r for r in rows if r["group"] == "memorized" and r["base_correct"].lower() == "true"]
    generalized_by_relation: dict[int, list[PretrainedExample]] = {}
    for example in examples:
        if example.group == "generalized":
            generalized_by_relation.setdefault(example.relation, []).append(example)
    generalized_candidates = {
        (example.relation, example.resident)
        for relation_examples in generalized_by_relation.values()
        for offset, example in enumerate(sorted(relation_examples, key=lambda e: e.resident))
        if offset % 2 == 0
    }
    if candidate_scope == "memorized":
        candidates = [r for r in rows if r["group"] == "memorized" and r["base_correct"].lower() == "true"]
    elif candidate_scope == "train_members":
        candidates = [r for r in rows if r["group"] in {"seen_rule", "memorized"} and r["base_correct"].lower() == "true"]
    elif candidate_scope == "eval_split":
        candidates = [r for r in rows if r["base_correct"].lower() == "true" and (
            r["group"] == "memorized" or (
                r["group"] == "generalized" and (int(r["relation"]), int(r["key"])) in generalized_candidates
            )
        )]
    else:
        raise ValueError("candidate_scope must be 'memorized', 'train_members', or 'eval_split'")
    k = max(1, round(len(correct_mem) * fraction))
    rng = random.Random(seed)
    selected_strategies = strategies or ["tomography", "confidence", "random"]
    ordered = {
        "tomography": sorted(candidates, key=lambda r: float(r["q_fragility"]), reverse=True)[:k],
        "confidence": sorted(candidates, key=lambda r: float(r["base_margin"]))[:k],
    }
    shuffled = candidates[:]; rng.shuffle(shuffled); ordered["random"] = shuffled[:k]
    unknown = set(selected_strategies) - set(ordered)
    if unknown:
        raise ValueError(f"unknown strategies: {sorted(unknown)}")
    strategy_rows = {name: ordered[name] for name in selected_strategies}

    def score(model, pool: list[PretrainedExample]) -> float:
        return _evaluate(model, tokenizer, pool, device, config.eval_batch_size)

    generalized = [e for e in examples if e.group == "generalized"]
    generalized_evaluation = (
        [e for e in generalized if (e.relation, e.resident) not in generalized_candidates]
        if candidate_scope == "eval_split" else generalized
    )
    all_mem = [e for e in examples if e.group == "memorized"]
    seen_rule = [e for e in examples if e.group == "seen_rule"]
    train_members = seen_rule + all_mem
    by_key = {(e.relation, e.resident): e for e in examples}
    city_target_ids = sorted({example.target_id for example in examples})
    if len(city_target_ids) != config.n_cities:
        raise ValueError("corpus city-token cardinality does not match configuration")
    base = AutoModelForCausalLM.from_pretrained(config.model_name, torch_dtype=dtype).to(device)
    base = PeftModel.from_pretrained(base, run_path / "adapter_best")
    base.eval()
    baseline_scores = {group: score(base, pool) for group, pool in {
        "generalized": generalized, "memorized": all_mem, "seen_rule": seen_rule,
    }.items()}
    baseline = {
        "generalized": fmean(baseline_scores["generalized"][e.index]["correct"] for e in generalized_evaluation),
        "memorized": fmean(x["correct"] for x in baseline_scores["memorized"].values()),
        "seen_rule": fmean(x["correct"] for x in baseline_scores["seen_rule"].values()),
    }
    reference_probabilities = _reference_city_probabilities(
        base, tokenizer, train_members, city_target_ids, device, config.eval_batch_size
    )
    del base
    if device.type == "cuda": torch.cuda.empty_cache()
    results = {"run_dir": str(run_path), "fraction": fraction, "steps": steps,
               "learning_rate": learning_rate, "kl_weight": kl_weight,
               "candidate_scope": candidate_scope, "baseline": baseline, "selected": k,
               "protocol_version": 2,
               "replacement_targets": "valid_alternative_city_token_ids",
               "retain_pool": "nonselected_seen_rule_and_memorized_only",
               "generalized_evaluation_count": len(generalized_evaluation)}

    for index, (name, chosen) in enumerate(strategy_rows.items()):
        seed_everything(seed + 1000 * index)
        chosen_keys = {(int(r["relation"]), int(r["key"])) for r in chosen}
        selected = [by_key[key] for key in chosen_keys]
        retained = [e for e in train_members if (e.relation, e.resident) not in chosen_keys]
        retained_mem = [e for e in all_mem if (e.relation, e.resident) not in chosen_keys]
        retained_rule = [e for e in seen_rule if (e.relation, e.resident) not in chosen_keys]
        wrong = _counter_targets(selected, city_target_ids, seed)
        selected_before = fmean(
            baseline_scores[e.group][e.index]["correct"] for e in selected
        )
        model = AutoModelForCausalLM.from_pretrained(config.model_name, torch_dtype=dtype).to(device)
        model = PeftModel.from_pretrained(model, run_path / "adapter_best", is_trainable=True)
        optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=learning_rate, weight_decay=config.weight_decay)
        local_rng = random.Random(seed + index)
        half = max(1, config.batch_size // 2)
        model.train()
        for step in range(steps):
            rewrite_batch = [selected[local_rng.randrange(len(selected))] for _ in range(half)]
            retain_batch = [retained[local_rng.randrange(len(retained))] for _ in range(half)]
            records = ([(e.prompt, wrong[(e.relation, e.resident)]) for e in rewrite_batch]
                       + [(e.prompt, e.target_id) for e in retain_batch])
            batch = _encode_training_batch(tokenizer, records, config.max_length)
            batch = {n: t.to(device) for n, t in batch.items()}
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=device.type == "cuda"):
                output = model(**batch)
                label_positions = batch["labels"].ne(-100).to(torch.int64).argmax(dim=1)
                target_positions = label_positions - 1
                next_logits = output.logits.float()[
                    torch.arange(2 * half, device=device), target_positions
                ]
                target_ids = batch["labels"][
                    torch.arange(2 * half, device=device), label_positions
                ]
                row_loss = torch.nn.functional.cross_entropy(
                    next_logits, target_ids, reduction="none"
                )
                rewrite_loss = row_loss[:half].mean()
                retain_loss = row_loss[half:].mean()
                retain_logits = next_logits[half:][:, city_target_ids]
                reference = torch.stack([reference_probabilities[e.index] for e in retain_batch]).to(device)
                retain_kl = torch.nn.functional.kl_div(
                    retain_logits.log_softmax(dim=-1), reference, reduction="batchmean"
                )
                loss = rewrite_loss + retain_loss + kl_weight * retain_kl
            loss.backward(); optimizer.step()
        model.eval()
        selected_scores = score(model, selected)
        rewritten = [PretrainedExample(e.index, e.relation, e.resident, e.group, e.prompt, e.answer,
                                       wrong[(e.relation, e.resident)]) for e in selected]
        after_selected = fmean(x["correct"] for x in selected_scores.values())
        replacement_success = fmean(x["correct"] for x in score(model, rewritten).values())
        after_retained_mem = fmean(x["correct"] for x in score(model, retained_mem).values())
        after_retained_rule = fmean(x["correct"] for x in score(model, retained_rule).values())
        after_generalized = fmean(x["correct"] for x in score(model, generalized_evaluation).values())
        selected_memories = sum(e.group == "memorized" for e in selected)
        results[name] = {"selected_before_accuracy": selected_before,
                         "after_selected_original_accuracy": after_selected,
                         "replacement_success": replacement_success,
                         "after_retained_memory_accuracy": after_retained_mem,
                         "after_retained_rule_accuracy": after_retained_rule,
                         "after_generalized_accuracy": after_generalized,
                         "forgetting_gain": selected_before - after_selected,
                         "generalization_collateral": baseline["generalized"] - after_generalized,
                         "retained_memory_collateral": baseline["memorized"] - after_retained_mem,
                         "selected_memorized": selected_memories,
                         "selected_memorized_fraction": selected_memories / len(selected),
                         "selected_keys": sorted([list(x) for x in chosen_keys])}
        del model
        if device.type == "cuda": torch.cuda.empty_cache()
    path = out / "analysis.json"
    path.write_text(json.dumps(results, indent=2, sort_keys=True), encoding="utf-8")
    return path


def aggregate_pretrained_unlearning(
    run_dirs: list[str | Path], output_dir: str | Path
) -> Path:
    """Aggregate corrected budget sweeps and 20%-budget edit curves."""
    out = Path(output_dir).resolve(); out.mkdir(parents=True, exist_ok=True)
    budget_records: list[dict] = []
    curve_records: list[dict] = []
    selectors = ("tomography", "confidence", "random")
    for raw_run_dir in run_dirs:
        run_dir = Path(raw_run_dir).resolve()
        config = PretrainedConfig.load(run_dir / "config.resolved.json")
        root = run_dir / "targeted_unlearning_v2_evalsplit"
        for budget in (0.10, 0.20, 0.30):
            analysis = json.loads((root / f"budget_{budget:.2f}" / "analysis.json").read_text(encoding="utf-8"))
            for selector in selectors:
                row = analysis[selector]
                budget_records.append({
                    "seed": config.seed, "budget": budget, "selector": selector,
                    "selected_memorized_fraction": row["selected_memorized_fraction"],
                    "forgetting_gain": row["forgetting_gain"],
                    "replacement_success": row["replacement_success"],
                    "generalization_collateral": row["generalization_collateral"],
                    "retained_memory_collateral": row["retained_memory_collateral"],
                })
        for steps in (5, 10, 20, 40):
            path = (root / "budget_0.20" / "analysis.json") if steps == 40 else (
                root / "curves_budget_0.20" / f"steps_{steps}" / "analysis.json"
            )
            analysis = json.loads(path.read_text(encoding="utf-8"))
            for selector in selectors:
                row = analysis[selector]
                curve_records.append({
                    "seed": config.seed, "steps": steps, "selector": selector,
                    "selected_memorized_fraction": row["selected_memorized_fraction"],
                    "forgetting_gain": row["forgetting_gain"],
                    "replacement_success": row["replacement_success"],
                    "generalization_collateral": row["generalization_collateral"],
                    "retained_memory_collateral": row["retained_memory_collateral"],
                })

    metric_names = (
        "selected_memorized_fraction", "forgetting_gain", "replacement_success",
        "generalization_collateral", "retained_memory_collateral",
    )

    def summarize(records: list[dict], group_names: tuple[str, ...]) -> list[dict]:
        groups: dict[tuple, list[dict]] = {}
        for row in records:
            groups.setdefault(tuple(row[name] for name in group_names), []).append(row)
        summaries = []
        for key, group in sorted(groups.items(), key=lambda item: item[0]):
            summary = {name: value for name, value in zip(group_names, key)}
            summary["n"] = len(group)
            for metric in metric_names:
                values = [float(row[metric]) for row in group]
                summary[f"{metric}_mean"] = fmean(values)
                summary[f"{metric}_std"] = pstdev(values)
            summaries.append(summary)
        return summaries

    budget_summary = summarize(budget_records, ("budget", "selector"))
    curve_summary = summarize(curve_records, ("steps", "selector"))
    for name, records in (("budget_records.csv", budget_records), ("curve_records.csv", curve_records)):
        with (out / name).open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(records[0])); writer.writeheader(); writer.writerows(records)
    artifact = {
        "protocol_version": 2,
        "run_dirs": [str(Path(path).resolve()) for path in run_dirs],
        "budget_summary": budget_summary,
        "curve_summary": curve_summary,
        "interpretation_guardrail": (
            "Selection enrichment and editability are distinct endpoints; do not claim "
            "tomography improves unlearning unless matched-collateral curves support it."
        ),
    }
    path = out / "analysis.json"
    path.write_text(json.dumps(artifact, indent=2, sort_keys=True), encoding="utf-8")
    return path
