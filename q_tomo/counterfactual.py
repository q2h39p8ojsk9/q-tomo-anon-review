"""Small-model counterfactual memorization ensemble.

For each target training example, counterfactual memorization is the difference
between expected log probability under models trained with versus without that
example.  A complementary-pair design gives every target exactly one included
and one excluded observation per model-initialization pair.
"""

from __future__ import annotations

import csv
import gc
import json
import math
import random
import time
from pathlib import Path
from statistics import fmean

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader

from q_tomo.analysis import roc_auc
from q_tomo.config import ExperimentConfig
from q_tomo.data import RuleMemoryDataset, build_corpus
from q_tomo.evaluate import batch_statistics
from q_tomo.model import TomographyTransformer
from q_tomo.runtime import autocast_context, resolve_device, seed_everything
from q_tomo.train import evaluate_groups, learning_rate_at


def _rank(values: list[float]) -> list[float]:
    order = sorted(range(len(values)), key=lambda index: values[index])
    ranks = [0.0] * len(values); position = 0
    while position < len(order):
        end = position + 1
        while end < len(order) and values[order[end]] == values[order[position]]:
            end += 1
        value = (position + end - 1) / 2.0
        for offset in range(position, end):
            ranks[order[offset]] = value
        position = end
    return ranks


def _correlation(left: list[float], right: list[float], *, rank: bool = False) -> float:
    x = _rank(left) if rank else left; y = _rank(right) if rank else right
    if len(x) < 2:
        return float("nan")
    mx, my = fmean(x), fmean(y)
    numerator = sum((a - mx) * (b - my) for a, b in zip(x, y))
    denominator = math.sqrt(sum((a - mx) ** 2 for a in x) * sum((b - my) ** 2 for b in y))
    return numerator / denominator if denominator else float("nan")


def _balanced_design(target_indices: list[int], n_models: int, seed: int) -> dict[int, set[int]]:
    if n_models < 4 or n_models % 2:
        raise ValueError("n_models must be even and at least four")
    inclusion = {model_index: set() for model_index in range(n_models)}
    for pair in range(n_models // 2):
        rng = random.Random(seed + 1009 * pair)
        for target_index in target_indices:
            side = rng.randrange(2)
            inclusion[2 * pair + side].add(target_index)
    return inclusion


@torch.inference_mode()
def _member_scores(model, members, corpus, device, batch_size: int) -> list[dict[str, object]]:
    model.eval(); rows = []
    for start in range(0, len(members), batch_size):
        batch = members[start : start + batch_size]
        stats = batch_statistics(model, batch, corpus, device)
        for offset, example in enumerate(batch):
            rows.append({
                "index": example.source_index,
                "log_probability": -float(stats["nll"][offset]),
                "correct": bool(stats["correct"][offset]),
            })
    return rows


def _train_one(
    config: ExperimentConfig,
    corpus,
    included_indices: set[int],
    model_index: int,
    counterfactual_seed: int,
    steps: int,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    # Complementary models share an initialization seed. Dataset inclusion is
    # the intended difference within each pair.
    model_seed = counterfactual_seed + model_index // 2
    seed_everything(model_seed)
    device = resolve_device(config.train.device)
    model = TomographyTransformer(config.model).to(device)
    optimizer = AdamW(
        model.parameters(), lr=config.train.learning_rate,
        betas=(config.train.beta1, config.train.beta2),
        weight_decay=config.train.weight_decay,
        fused=device.type == "cuda" if "fused" in AdamW.__init__.__code__.co_varnames else False,
    )
    use_scaler = device.type == "cuda" and config.train.precision == "fp16"
    scaler = torch.amp.GradScaler("cuda", enabled=use_scaler)
    included = [example for example in corpus.train_examples if example.source_index in included_indices]
    loader = DataLoader(
        RuleMemoryDataset(included), batch_size=config.train.batch_size,
        shuffle=True, num_workers=config.train.num_workers,
        pin_memory=device.type == "cuda", drop_last=True,
        generator=torch.Generator().manual_seed(model_seed + 7919),
    )
    if len(loader) == 0:
        raise ValueError("included subset is smaller than the training batch")

    def batches():
        while True:
            yield from loader

    iterator = batches(); started = time.perf_counter(); model.train()
    original_steps = config.train.steps
    config.train.steps = steps
    for step in range(1, steps + 1):
        batch = next(iterator)
        input_ids = batch["input_ids"].to(device, non_blocking=True)
        labels = batch["labels"].to(device, non_blocking=True)
        with autocast_context(device, config.train.precision):
            loss = model(input_ids, labels)["loss"]
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.train.grad_clip)
        # Use the original schedule compressed to the counterfactual run length.
        lr = learning_rate_at(step - 1, config)
        for group in optimizer.param_groups:
            group["lr"] = lr
        scaler.step(optimizer); scaler.update(); optimizer.zero_grad(set_to_none=True)
    config.train.steps = original_steps

    members = [example for example in corpus.eval_examples if example.is_member]
    scores = _member_scores(model, members, corpus, device, config.probe.batch_size)
    evaluation = evaluate_groups(model, corpus, device, config.probe.batch_size)
    summary = {
        "model_index": model_index,
        "pair_index": model_index // 2,
        "model_seed": model_seed,
        "included_unique_examples": len(included_indices),
        "training_rows_with_exposure": len(included),
        "steps": steps,
        "seconds": time.perf_counter() - started,
        "evaluation": evaluation,
    }
    del model, optimizer, scaler
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return scores, summary


def run_counterfactual_ensemble(
    config_path: str | Path,
    output_dir: str | Path,
    reference_features: str | Path | None = None,
    n_models: int = 12,
    steps: int = 4000,
    seed: int = 31415,
) -> Path:
    """Train/resume a balanced ensemble and compare it with tomography."""
    out = Path(output_dir).resolve(); out.mkdir(parents=True, exist_ok=True)
    models_dir = out / "models"; models_dir.mkdir(exist_ok=True)
    config = ExperimentConfig.load(config_path)
    corpus = build_corpus(config.data)
    config.model.vocab_size = corpus.vocab.size
    config.model.max_seq_len = max(config.model.max_seq_len, len(corpus.eval_examples[0].input_ids))
    members = [example for example in corpus.eval_examples if example.is_member]
    target_indices = [example.source_index for example in members]
    inclusion = _balanced_design(target_indices, n_models, seed)
    design = {
        "method": "complementary-pair balanced inclusion",
        "n_models": n_models, "n_pairs": n_models // 2, "steps": steps, "seed": seed,
        "targets": len(target_indices),
        "inclusion_by_model": {str(index): sorted(values) for index, values in inclusion.items()},
    }
    (out / "design.json").write_text(json.dumps(design, indent=2, sort_keys=True), encoding="utf-8")

    for model_index in range(n_models):
        result_path = models_dir / f"model_{model_index:02d}.json"
        if result_path.exists():
            print(json.dumps({"model_index": model_index, "status": "already_complete"}), flush=True)
            continue
        scores, summary = _train_one(
            config, corpus, inclusion[model_index], model_index, seed, steps
        )
        payload = {"summary": summary, "scores": scores}
        result_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        print(json.dumps({"status": "complete", **summary}, sort_keys=True), flush=True)

    observations: dict[int, list[dict[str, object]]] = {index: [] for index in target_indices}
    model_summaries = []
    for model_index in range(n_models):
        payload = json.loads((models_dir / f"model_{model_index:02d}.json").read_text(encoding="utf-8"))
        model_summaries.append(payload["summary"])
        for score in payload["scores"]:
            index = int(score["index"])
            observations[index].append({
                "model_index": model_index,
                "included": index in inclusion[model_index],
                "log_probability": float(score["log_probability"]),
                "correct": bool(score["correct"]),
            })

    feature_rows = {}
    if reference_features is not None:
        with Path(reference_features).open(encoding="utf-8", newline="") as handle:
            feature_rows = {int(row["index"]): row for row in csv.DictReader(handle)}

    records = []
    by_index = {example.source_index: example for example in members}
    for index in target_indices:
        included_obs = [row for row in observations[index] if row["included"]]
        excluded_obs = [row for row in observations[index] if not row["included"]]
        example = by_index[index]
        record = {
            "index": index, "relation": example.relation, "key": example.key,
            "group": example.group,
            "counterfactual_memorization": fmean(float(row["log_probability"]) for row in included_obs)
            - fmean(float(row["log_probability"]) for row in excluded_obs),
            "included_mean_log_probability": fmean(float(row["log_probability"]) for row in included_obs),
            "excluded_mean_log_probability": fmean(float(row["log_probability"]) for row in excluded_obs),
            "included_accuracy": fmean(float(row["correct"]) for row in included_obs),
            "excluded_accuracy": fmean(float(row["correct"]) for row in excluded_obs),
            "included_models": len(included_obs), "excluded_models": len(excluded_obs),
        }
        if index in feature_rows:
            record.update({
                "q_fragility": float(feature_rows[index]["q_fragility"]),
                "q_survival_area": float(feature_rows[index]["q_survival_area"]),
                "reference_nll": float(feature_rows[index]["base_nll"]),
                "reference_correct": feature_rows[index]["base_correct"].lower() == "true",
            })
        records.append(record)

    records_path = out / "example_records.csv"
    with records_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader(); writer.writerows(records)

    analysis: dict[str, object] = {
        "definition": "E[log p(y|x) | example included] - E[log p(y|x) | example excluded]",
        "design": {key: value for key, value in design.items() if key != "inclusion_by_model"},
        "group_summary": {
            group: {
                "count": len(selected := [row for row in records if row["group"] == group]),
                "counterfactual_memorization_mean": fmean(float(row["counterfactual_memorization"]) for row in selected),
                "included_accuracy_mean": fmean(float(row["included_accuracy"]) for row in selected),
                "excluded_accuracy_mean": fmean(float(row["excluded_accuracy"]) for row in selected),
            }
            for group in ("seen_rule", "memorized")
        },
        "model_summaries": model_summaries,
        "artifacts": {"example_records": str(records_path)},
    }
    if feature_rows:
        comparable = [row for row in records if "q_fragility" in row]
        cf = [float(row["counterfactual_memorization"]) for row in comparable]
        fragility = [float(row["q_fragility"]) for row in comparable]
        labels = [int(row["group"] == "memorized") for row in comparable]
        correlations = {}
        for group in ("all", "seen_rule", "memorized"):
            selected = comparable if group == "all" else [row for row in comparable if row["group"] == group]
            left = [float(row["counterfactual_memorization"]) for row in selected]
            right = [float(row["q_fragility"]) for row in selected]
            correlations[group] = {
                "count": len(selected),
                "pearson": _correlation(left, right),
                "spearman": _correlation(left, right, rank=True),
            }
        analysis["tomography_comparison"] = {
            "correlations": correlations,
            "memorized_vs_seen_rule_auc": {
                "counterfactual_memorization": roc_auc(labels, cf),
                "q_fragility": roc_auc(labels, fragility),
            },
        }

    path = out / "analysis.json"
    path.write_text(json.dumps(analysis, indent=2, sort_keys=True), encoding="utf-8")
    return path
