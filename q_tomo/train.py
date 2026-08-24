from __future__ import annotations

import json
import math
import time
from dataclasses import asdict
from pathlib import Path

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader

from q_tomo.config import ExperimentConfig
from q_tomo.data import RuleMemoryCorpus, RuleMemoryDataset, build_corpus
from q_tomo.evaluate import batch_statistics
from q_tomo.model import TomographyTransformer
from q_tomo.runtime import atomic_torch_save, autocast_context, resolve_device, seed_everything


@torch.no_grad()
def evaluate_groups(
    model: TomographyTransformer,
    corpus: RuleMemoryCorpus,
    device: torch.device,
    batch_size: int,
) -> dict[str, dict[str, float]]:
    model.eval()
    results: dict[str, dict[str, float]] = {}
    for group in corpus.groups:
        examples = corpus.examples_for_group(group)
        total_loss = 0.0
        correct = 0
        total = 0
        for start in range(0, len(examples), batch_size):
            batch = examples[start : start + batch_size]
            statistics = batch_statistics(model, batch, corpus, device)
            total_loss += statistics["nll"].sum().item()
            correct += statistics["correct"].sum().item()
            total += len(batch)
        results[group] = {"loss": total_loss / total, "accuracy": correct / total, "count": float(total)}
    model.train()
    return results


def learning_rate_at(step: int, config: ExperimentConfig) -> float:
    train = config.train
    if step < train.warmup_steps:
        return train.learning_rate * (step + 1) / max(1, train.warmup_steps)
    progress = (step - train.warmup_steps) / max(1, train.steps - train.warmup_steps)
    cosine = 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))
    return train.min_learning_rate + cosine * (train.learning_rate - train.min_learning_rate)


def save_checkpoint(
    output_dir: Path,
    model: TomographyTransformer,
    optimizer: AdamW,
    step: int,
    config: ExperimentConfig,
) -> Path:
    checkpoint_path = output_dir / "checkpoints" / f"step_{step:07d}.pt"
    payload = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "step": step,
        "config": config.to_dict(),
        "architecture": model.architecture_dict(),
    }
    atomic_torch_save(payload, checkpoint_path)
    atomic_torch_save(payload, output_dir / "checkpoint_last.pt")
    return checkpoint_path


def save_best_ready_checkpoint(
    output_dir: Path,
    model: TomographyTransformer,
    optimizer: AdamW,
    step: int,
    config: ExperimentConfig,
    readiness_score: float,
) -> Path:
    path = output_dir / "checkpoint_best_ready.pt"
    atomic_torch_save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "step": step,
            "config": config.to_dict(),
            "architecture": model.architecture_dict(),
            "selection": {
                "criterion": "maximize min(generalized_accuracy, memorized_accuracy)",
                "readiness_score": readiness_score,
            },
        },
        path,
    )
    return path


def train_experiment(config: ExperimentConfig) -> Path:
    seed_everything(config.train.seed)
    corpus = build_corpus(config.data)
    config.model.vocab_size = corpus.vocab.size
    config.model.max_seq_len = max(config.model.max_seq_len, len(corpus.eval_examples[0].input_ids))
    config.model.validate()

    output_dir = Path(config.train.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    config.save(output_dir / "config.resolved.json")
    corpus.save_metadata(output_dir / "corpus.json")

    device = resolve_device(config.train.device)
    model = TomographyTransformer(config.model).to(device)
    training_model: torch.nn.Module = model
    if config.train.compile_model and hasattr(torch, "compile"):
        training_model = torch.compile(model)
    optimizer = AdamW(
        model.parameters(),
        lr=config.train.learning_rate,
        betas=(config.train.beta1, config.train.beta2),
        weight_decay=config.train.weight_decay,
        fused=device.type == "cuda" if "fused" in AdamW.__init__.__code__.co_varnames else False,
    )
    use_scaler = device.type == "cuda" and config.train.precision == "fp16"
    scaler = torch.amp.GradScaler("cuda", enabled=use_scaler)
    dataset = RuleMemoryDataset(corpus.train_examples)
    generator = torch.Generator().manual_seed(config.train.seed)
    loader = DataLoader(
        dataset,
        batch_size=config.train.batch_size,
        shuffle=True,
        num_workers=config.train.num_workers,
        pin_memory=device.type == "cuda",
        drop_last=True,
        generator=generator,
    )
    def endless_batches():
        while True:
            yield from loader

    batches = endless_batches()
    rule_batches = None
    if config.train.curriculum_steps > 0 and hasattr(corpus, "rule_train_examples"):
        rule_loader = DataLoader(
            RuleMemoryDataset(corpus.rule_train_examples),
            batch_size=config.train.batch_size,
            shuffle=True,
            num_workers=config.train.num_workers,
            pin_memory=device.type == "cuda",
            drop_last=True,
            generator=torch.Generator().manual_seed(config.train.seed + 991),
        )
        def endless_rule_batches():
            while True:
                yield from rule_loader
        rule_batches = endless_rule_batches()
    history_path = output_dir / "history.jsonl"
    started = time.perf_counter()
    best_ready_score = -1.0
    best_ready_step = 0
    best_ready_path: Path | None = None
    training_model.train()
    optimizer.zero_grad(set_to_none=True)

    for step in range(1, config.train.steps + 1):
        accumulated_loss = 0.0
        for _ in range(config.train.grad_accum_steps):
            batch = next(rule_batches) if rule_batches is not None and step <= config.train.curriculum_steps else next(batches)
            input_ids = batch["input_ids"].to(device, non_blocking=True)
            labels = batch["labels"].to(device, non_blocking=True)
            with autocast_context(device, config.train.precision):
                loss = training_model(input_ids, labels)["loss"] / config.train.grad_accum_steps
            accumulated_loss += float(loss.detach())
            scaler.scale(loss).backward()

        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.train.grad_clip)
        lr = learning_rate_at(step - 1, config)
        for group in optimizer.param_groups:
            group["lr"] = lr
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)

        record: dict[str, object] | None = None
        if step == 1 or step % config.train.log_interval == 0:
            record = {
                "step": step,
                "train_loss": accumulated_loss,
                "learning_rate": lr,
                "elapsed_seconds": time.perf_counter() - started,
            }
        if step == 1 or step % config.train.eval_interval == 0 or step == config.train.steps:
            if record is None:
                record = {"step": step, "train_loss": accumulated_loss, "learning_rate": lr}
            record["evaluation"] = evaluate_groups(training_model, corpus, device, config.probe.batch_size)
            evaluation = record["evaluation"]
            readiness_score = min(
                evaluation["generalized"]["accuracy"],
                evaluation["memorized"]["accuracy"],
            )
            record["readiness_score"] = readiness_score
            if readiness_score > best_ready_score:
                best_ready_score = readiness_score
                best_ready_step = step
                best_ready_path = save_best_ready_checkpoint(
                    output_dir, model, optimizer, step, config, readiness_score
                )
            training_model.train()
        if record is not None:
            with history_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, sort_keys=True) + "\n")
            print(json.dumps(record, sort_keys=True), flush=True)
        if step % config.train.checkpoint_interval == 0 or step == config.train.steps:
            save_checkpoint(output_dir, model, optimizer, step, config)

    with (output_dir / "run_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "device": str(device),
                "architecture": model.architecture_dict(),
                "training_seconds": time.perf_counter() - started,
                "best_ready_step": best_ready_step,
                "best_ready_score": best_ready_score,
                "best_ready_checkpoint": str(best_ready_path) if best_ready_path else None,
                "train_config": asdict(config.train),
            },
            handle,
            indent=2,
            sort_keys=True,
        )
    return best_ready_path or output_dir / "checkpoint_last.pt"


def load_model_from_checkpoint(path: str | Path, device: torch.device) -> tuple[TomographyTransformer, ExperimentConfig]:
    payload = torch.load(Path(path), map_location=device, weights_only=False)
    config = ExperimentConfig.from_dict(payload["config"])
    model = TomographyTransformer(config.model).to(device)
    model.load_state_dict(payload["model"])
    model.eval()
    return model, config
