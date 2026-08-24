from __future__ import annotations

import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import fmean, pstdev

import torch

from q_tomo.analysis import logistic_cross_validated_auc, separability_auc
from q_tomo.config import ExperimentConfig
from q_tomo.data import Example, RuleMemoryCorpus, build_corpus
from q_tomo.evaluate import batch_statistics
from q_tomo.quantization import quantization_scopes, temporary_quantization
from q_tomo.runtime import resolve_device, seed_everything
from q_tomo.train import load_model_from_checkpoint


@torch.inference_mode()
def _evaluate_examples(
    model: torch.nn.Module,
    examples: list[Example],
    corpus: RuleMemoryCorpus,
    device: torch.device,
    batch_size: int,
) -> dict[int, dict[str, float | bool]]:
    results: dict[int, dict[str, float | bool]] = {}
    for start in range(0, len(examples), batch_size):
        batch = examples[start : start + batch_size]
        statistics = batch_statistics(model, batch, corpus, device)
        for offset, example in enumerate(batch):
            results[example.source_index] = {
                "nll": float(statistics["nll"][offset]),
                "margin": float(statistics["margin"][offset]),
                "correct": bool(statistics["correct"][offset]),
            }
    return results


def _mean(values: list[float]) -> float:
    return fmean(values) if values else float("nan")


def _std(values: list[float]) -> float:
    return pstdev(values) if len(values) > 1 else 0.0


def _entropy_and_concentration(values: list[float]) -> tuple[float, float]:
    positive = [abs(value) + 1e-12 for value in values]
    total = sum(positive)
    probabilities = [value / total for value in positive]
    entropy = -sum(probability * math.log(probability) for probability in probabilities)
    normalized_entropy = entropy / math.log(len(probabilities)) if len(probabilities) > 1 else 0.0
    return normalized_entropy, max(probabilities)


def _write_rows(rows: list[dict[str, object]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _analyze(rows: list[dict[str, object]], config: ExperimentConfig) -> dict[str, object]:
    selected = [row for row in rows if row["group"] in {"generalized", "memorized"} and row["base_correct"]]
    labels = [1 if row["group"] == "memorized" else 0 for row in selected]
    baseline_names = ["base_nll", "base_margin"]
    tomography_names = [
        "q_fragility",
        "q_variability",
        "q_survival_area",
        "layer_entropy",
        "layer_concentration",
    ]
    baseline_features = [[float(row[name]) for name in baseline_names] for row in selected]
    tomography_features = [[float(row[name]) for name in tomography_names] for row in selected]
    full_features = [baseline + tomography for baseline, tomography in zip(baseline_features, tomography_features)]
    group_counts = {label: labels.count(label) for label in (0, 1)}
    baseline_auc = logistic_cross_validated_auc(baseline_features, labels, seed=config.train.seed)
    tomography_auc = logistic_cross_validated_auc(tomography_features, labels, seed=config.train.seed)
    full_auc = logistic_cross_validated_auc(full_features, labels, seed=config.train.seed)
    fragility_auc = separability_auc(labels, [float(row["q_fragility"]) for row in selected])
    enough = min(group_counts.values(), default=0) >= config.probe.min_correct_per_group
    kill_test_pass = enough and full_auc >= 0.80 and full_auc - baseline_auc >= 0.05
    return {
        "comparison": "generalized_vs_memorized_among_full_precision_correct_examples",
        "counts": {"generalized": group_counts[0], "memorized": group_counts[1]},
        "features": {"baseline": baseline_names, "tomography": tomography_names},
        "base_confidence_cv_auc": baseline_auc,
        "tomography_only_cv_auc": tomography_auc,
        "combined_cv_auc": full_auc,
        "incremental_cv_auc": full_auc - baseline_auc,
        "univariate_fragility_separability_auc": fragility_auc,
        "kill_test": {
            "passed": kill_test_pass,
            "requirements": {
                "minimum_correct_per_group": config.probe.min_correct_per_group,
                "combined_cv_auc": 0.80,
                "incremental_over_confidence_auc": 0.05,
            },
        },
    }


def probe_checkpoint(checkpoint_path: str | Path, output_dir: str | Path | None = None) -> Path:
    checkpoint_path = Path(checkpoint_path).resolve()
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config = ExperimentConfig.from_dict(payload["config"])
    seed_everything(config.train.seed)
    device = resolve_device(config.train.device)
    model, config = load_model_from_checkpoint(checkpoint_path, device)
    corpus = build_corpus(config.data)
    examples = corpus.eval_examples
    target_dir = Path(output_dir).resolve() if output_dir else checkpoint_path.parent / "tomography"
    target_dir.mkdir(parents=True, exist_ok=True)

    baseline = _evaluate_examples(model, examples, corpus, device, config.probe.batch_size)
    global_results: dict[int, dict[int, list[dict[str, float | bool]]]] = defaultdict(lambda: defaultdict(list))
    for bits in config.probe.bits:
        for seed in config.probe.seeds:
            with temporary_quantization(
                model,
                bits=bits,
                seed=seed,
                scope="all",
                include_embeddings=config.probe.quantize_embeddings,
            ):
                condition = _evaluate_examples(model, examples, corpus, device, config.probe.batch_size)
            for index, values in condition.items():
                global_results[index][bits].append(values)

    layer_results: dict[int, dict[str, list[dict[str, float | bool]]]] = defaultdict(lambda: defaultdict(list))
    if config.probe.layerwise:
        scopes = [scope for scope in quantization_scopes(model, config.probe.quantize_embeddings) if scope != "all"]
        for scope in scopes:
            for bits in config.probe.layer_bits:
                for seed in config.probe.layer_seeds:
                    with temporary_quantization(
                        model,
                        bits=bits,
                        seed=seed,
                        scope=scope,
                        include_embeddings=config.probe.quantize_embeddings,
                    ):
                        condition = _evaluate_examples(model, examples, corpus, device, config.probe.batch_size)
                    for index, values in condition.items():
                        layer_results[index][f"{scope}@{bits}b"].append(values)

    rows: list[dict[str, object]] = []
    for example in examples:
        index = example.source_index
        base = baseline[index]
        row: dict[str, object] = {
            "index": index,
            "group": example.group,
            "relation": example.relation,
            "key": example.key,
            "value": example.value,
            "is_member": example.is_member,
            "base_nll": base["nll"],
            "base_margin": base["margin"],
            "base_correct": base["correct"],
        }
        all_abs_deltas: list[float] = []
        all_std_deltas: list[float] = []
        all_survival: list[float] = []
        for bits in config.probe.bits:
            conditions = global_results[index][bits]
            deltas = [float(condition["nll"]) - float(base["nll"]) for condition in conditions]
            survival = [1.0 if condition["correct"] else 0.0 for condition in conditions]
            row[f"q{bits}_mean_delta_nll"] = _mean(deltas)
            row[f"q{bits}_std_delta_nll"] = _std(deltas)
            row[f"q{bits}_mean_abs_delta_nll"] = _mean([abs(value) for value in deltas])
            row[f"q{bits}_survival"] = _mean(survival)
            all_abs_deltas.extend(abs(value) for value in deltas)
            all_std_deltas.append(_std(deltas))
            all_survival.extend(survival)
        row["q_fragility"] = _mean(all_abs_deltas)
        row["q_variability"] = _mean(all_std_deltas)
        row["q_survival_area"] = _mean(all_survival)

        layer_values: list[float] = []
        for scope, conditions in layer_results[index].items():
            deltas = [float(condition["nll"]) - float(base["nll"]) for condition in conditions]
            sensitivity = _mean([abs(value) for value in deltas])
            row[f"layer_{scope}_sensitivity"] = sensitivity
            layer_values.append(sensitivity)
        entropy, concentration = _entropy_and_concentration(layer_values) if layer_values else (0.0, 0.0)
        row["layer_entropy"] = entropy
        row["layer_concentration"] = concentration
        rows.append(row)

    _write_rows(rows, target_dir / "features.csv")
    analysis = _analyze(rows, config)
    analysis["checkpoint"] = str(checkpoint_path)
    analysis["probe_config"] = config.to_dict()["probe"]
    with (target_dir / "analysis.json").open("w", encoding="utf-8") as handle:
        json.dump(analysis, handle, indent=2, sort_keys=True)
    print(json.dumps(analysis, indent=2, sort_keys=True), flush=True)
    return target_dir / "analysis.json"
