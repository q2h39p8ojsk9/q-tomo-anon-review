from __future__ import annotations

import csv
import json
import random
from pathlib import Path

import torch
from torch.optim import AdamW

from q_tomo.data import RuleMemoryCorpus, build_corpus
from q_tomo.intervention import _subset_metrics
from q_tomo.runtime import autocast_context, resolve_device, seed_everything
from q_tomo.train import evaluate_groups, load_model_from_checkpoint


def _ce(model, examples, values, corpus, device, rewrite_inputs: bool = False):
    inputs = [list(example.input_ids) for example in examples]
    target_digits = [corpus.vocab.encode_number(value) for value in values]
    if rewrite_inputs:
        positions = list(corpus.prediction_positions)
        for tokens, digits in zip(inputs, target_digits):
            for position, digit in zip(positions, digits):
                tokens[position + 1] = digit
    input_ids = torch.tensor(inputs, dtype=torch.long, device=device)
    targets = torch.tensor(target_digits, dtype=torch.long, device=device)
    logits = model(input_ids)["logits"][:, list(corpus.prediction_positions), :].float()
    return torch.nn.functional.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))


def run_unlearning(checkpoint: str | Path, features: str | Path, output_dir: str | Path,
                   fraction: float = .2, steps: int = 1000, seed: int = 42,
                   learning_rate: float = 2.5e-5, forget_weight: float = 1.0) -> Path:
    seed_everything(seed); ck = Path(checkpoint).resolve(); out = Path(output_dir).resolve(); out.mkdir(parents=True, exist_ok=True)
    device = resolve_device("auto"); model, cfg = load_model_from_checkpoint(ck, device); corpus = build_corpus(cfg.data)
    rows = list(csv.DictReader(Path(features).open(encoding="utf-8")))
    mem = [r for r in rows if r["group"] == "memorized" and r["base_correct"].lower() == "true"]
    k = max(1, round(len(mem) * fraction)); selector_rng = random.Random(seed)
    strategies = {"tomography": sorted(mem, key=lambda r: float(r["q_fragility"]), reverse=True)[:k], "confidence": sorted(mem, key=lambda r: float(r["base_margin"]))[:k]}
    shuffled = mem[:]; selector_rng.shuffle(shuffled); strategies["random"] = shuffled[:k]
    baseline = evaluate_groups(model, corpus, device, cfg.probe.batch_size)
    results = {"baseline": baseline, "fraction": fraction, "steps": steps, "learning_rate": learning_rate, "forget_weight": forget_weight}
    batch_size = cfg.probe.batch_size; half = max(1, batch_size // 2)
    for strategy_index, (name, chosen) in enumerate(strategies.items()):
        seed_everything(seed + 1000 * strategy_index)
        selected = {(int(r["relation"]), int(r["key"])) for r in chosen}
        deleted = [e for e in corpus.examples_for_group("memorized") if (e.relation, e.key) in selected]
        retained_mem = [e for e in corpus.examples_for_group("memorized") if (e.relation, e.key) not in selected]
        retain_pool = [e for e in corpus.eval_examples if e.is_member and (e.relation, e.key) not in selected]
        output_cardinality = getattr(corpus, "n_cities", cfg.data.modulus)
        wrong_values = { (e.relation, e.key): (e.value + 1 + ((e.relation * 131 + e.key * 17 + seed) % (output_cardinality - 1))) % output_cardinality for e in deleted }
        rng = random.Random(seed + strategy_index); rng.shuffle(retain_pool); rng.shuffle(deleted)
        before_deleted = _subset_metrics(model, deleted, corpus, device, batch_size)
        optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=cfg.train.weight_decay); model.train()
        for step in range(steps):
            retain_batch = [retain_pool[(step * half + i) % len(retain_pool)] for i in range(half)]
            forget_batch = [deleted[(step * half + i) % len(deleted)] for i in range(half)]
            optimizer.zero_grad(set_to_none=True)
            with autocast_context(device, cfg.train.precision):
                retain_loss = _ce(model, retain_batch, [e.value for e in retain_batch], corpus, device)
                forget_loss = _ce(model, forget_batch, [wrong_values[(e.relation, e.key)] for e in forget_batch], corpus, device, True)
                loss = retain_loss + forget_weight * forget_loss
            loss.backward(); optimizer.step()
        metrics = evaluate_groups(model, corpus, device, batch_size)
        after_deleted = _subset_metrics(model, deleted, corpus, device, batch_size); after_retained = _subset_metrics(model, retained_mem, corpus, device, batch_size)
        forgetting = before_deleted["accuracy"] - after_deleted["accuracy"]
        collateral = baseline["generalized"]["accuracy"] - metrics["generalized"]["accuracy"]
        results[name] = {"selected": k, "metrics": metrics, "deleted_memories": {"before": before_deleted, "after": after_deleted}, "retained_memories": {"after": after_retained}, "forgetting_gain": forgetting, "generalization_collateral": collateral, "forgetting_efficiency": forgetting / max(.01, max(0., collateral))}
        model, _ = load_model_from_checkpoint(ck, device)
    path = out / "analysis.json"; path.write_text(json.dumps(results, indent=2), encoding="utf-8"); return path
