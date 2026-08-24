from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path
from statistics import fmean, pstdev

import torch

from q_tomo.analysis import logistic_group_cross_validated_auc, separability_auc
from q_tomo.data import RuleMemoryCorpus, build_corpus
from q_tomo.probe import _evaluate_examples
from q_tomo.quantization import temporary_matched_gaussian_noise
from q_tomo.runtime import resolve_device, seed_everything
from q_tomo.train import load_model_from_checkpoint


def matched_noise_control(
    checkpoint_path: str | Path, output_dir: str | Path | None = None
) -> Path:
    checkpoint = Path(checkpoint_path).resolve()
    device = resolve_device("auto")
    model, config = load_model_from_checkpoint(checkpoint, device)
    seed_everything(config.train.seed)
    corpus = build_corpus(config.data)
    examples = corpus.eval_examples
    baseline = _evaluate_examples(model, examples, corpus, device, config.probe.batch_size)
    conditions: dict[int, dict[int, list[dict[str, float | bool]]]] = defaultdict(lambda: defaultdict(list))
    for bits in config.probe.bits:
        for seed in config.probe.seeds:
            with temporary_matched_gaussian_noise(
                model,
                bits=bits,
                seed=seed,
                include_embeddings=config.probe.quantize_embeddings,
            ):
                evaluated = _evaluate_examples(model, examples, corpus, device, config.probe.batch_size)
            for index, values in evaluated.items():
                conditions[index][bits].append(values)

    rows: list[dict[str, object]] = []
    for example in examples:
        base = baseline[example.source_index]
        deltas: list[float] = []
        variations: list[float] = []
        survival: list[float] = []
        row: dict[str, object] = {
            "index": example.source_index,
            "group": example.group,
            "relation": example.relation,
            "base_correct": base["correct"],
            "base_nll": base["nll"],
            "base_margin": base["margin"],
        }
        for bits in config.probe.bits:
            bit_deltas = [float(value["nll"]) - float(base["nll"]) for value in conditions[example.source_index][bits]]
            bit_survival = [1.0 if value["correct"] else 0.0 for value in conditions[example.source_index][bits]]
            row[f"noise_q{bits}_mean_abs_delta_nll"] = fmean(abs(value) for value in bit_deltas)
            row[f"noise_q{bits}_survival"] = fmean(bit_survival)
            deltas.extend(abs(value) for value in bit_deltas)
            variations.append(pstdev(bit_deltas))
            survival.extend(bit_survival)
        row["noise_fragility"] = fmean(deltas)
        row["noise_variability"] = fmean(variations)
        row["noise_survival_area"] = fmean(survival)
        rows.append(row)

    selected = [row for row in rows if row["group"] in {"generalized", "memorized"} and row["base_correct"]]
    labels = [1 if row["group"] == "memorized" else 0 for row in selected]
    groups = [int(row["relation"]) for row in selected]
    baseline_names = ["base_nll", "base_margin"]
    noise_names = ["noise_fragility", "noise_variability", "noise_survival_area"]
    matrix = lambda names: [[float(row[name]) for name in names] for row in selected]
    baseline_auc = logistic_group_cross_validated_auc(matrix(baseline_names), labels, groups, config.train.seed)
    noise_auc = logistic_group_cross_validated_auc(matrix(noise_names), labels, groups, config.train.seed)
    combined_auc = logistic_group_cross_validated_auc(matrix(baseline_names + noise_names), labels, groups, config.train.seed)
    result = {
        "control": "per-tensor Gaussian noise matched to stochastic quantization MSE",
        "checkpoint": str(checkpoint),
        "counts": {"generalized": labels.count(0), "memorized": labels.count(1)},
        "relation_held_out_confidence_auc": baseline_auc,
        "relation_held_out_noise_auc": noise_auc,
        "relation_held_out_combined_auc": combined_auc,
        "incremental_over_confidence_auc": combined_auc - baseline_auc,
        "noise_fragility_separability_auc": separability_auc(
            labels, [float(row["noise_fragility"]) for row in selected]
        ),
    }
    target_dir = Path(output_dir).resolve() if output_dir else checkpoint.parent / "matched_noise_control"
    target_dir.mkdir(parents=True, exist_ok=True)
    with (target_dir / "features.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    target = target_dir / "analysis.json"
    with target.open("w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, sort_keys=True)
    print(json.dumps(result, indent=2, sort_keys=True))
    return target
