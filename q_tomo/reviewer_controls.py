"""Reviewer-facing controls for the pretrained transfer experiments.

The analyses here deliberately reuse frozen feature CSVs.  They never select a
checkpoint, prompt, perturbation strength, or feature after seeing the control
results.
"""

from __future__ import annotations

import csv
import json
import math
import random
from pathlib import Path
from statistics import fmean

from q_tomo.analysis import logistic_train_test_scores, roc_auc
from q_tomo.pretrained import PretrainedConfig, UNSEEN_EVAL_TEMPLATES


FEATURE_SETS = {
    "confidence": ["base_nll", "base_margin"],
    "tomography": ["q_fragility", "q_survival_area"],
    "combined": ["base_nll", "base_margin", "q_fragility", "q_survival_area"],
}


def _read(path: Path, *, groups: set[str] | None = None) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    return [
        row
        for row in rows
        if row["base_correct"].lower() == "true"
        and (groups is None or row["group"] in groups)
    ]


def _features(rows: list[dict[str, str]], columns: list[str]) -> list[list[float]]:
    return [[float(row[column]) for column in columns] for row in rows]


def _percentile(values: list[float], probability: float) -> float:
    ordered = sorted(value for value in values if math.isfinite(value))
    if not ordered:
        return float("nan")
    position = probability * (len(ordered) - 1)
    lower = int(math.floor(position)); upper = int(math.ceil(position))
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (position - lower) * (ordered[upper] - ordered[lower])


def _cluster_bootstrap_auc_delta(
    rows: list[dict[str, str]],
    confidence_scores: list[float],
    combined_scores: list[float],
    *,
    samples: int,
    seed: int,
) -> tuple[list[float], dict[str, float]]:
    """Paired bootstrap over relation clusters for an AUROC difference."""
    clusters = sorted({int(row["relation"]) for row in rows})
    by_cluster = {
        cluster: [index for index, row in enumerate(rows) if int(row["relation"]) == cluster]
        for cluster in clusters
    }
    labels = [int(row["group"] == "memorized") for row in rows]
    rng = random.Random(seed)
    deltas: list[float] = []
    for _ in range(samples):
        chosen = [rng.choice(clusters) for _ in clusters]
        indices = [index for cluster in chosen for index in by_cluster[cluster]]
        sampled_labels = [labels[index] for index in indices]
        confidence_auc = roc_auc(sampled_labels, [confidence_scores[index] for index in indices])
        combined_auc = roc_auc(sampled_labels, [combined_scores[index] for index in indices])
        if math.isfinite(confidence_auc) and math.isfinite(combined_auc):
            deltas.append(combined_auc - confidence_auc)
    return deltas, {
        "low": _percentile(deltas, 0.025),
        "high": _percentile(deltas, 0.975),
        "bootstrap_samples": len(deltas),
        "cluster_count": len(clusters),
    }


def _cluster_bootstrap_mean(
    rows: list[dict[str, object]], group: str, score_column: str, *, samples: int, seed: int
) -> dict[str, float]:
    selected = [row for row in rows if row["group"] == group]
    clusters = sorted({int(row["relation"]) for row in selected})
    by_cluster = {
        cluster: [float(row[score_column]) for row in selected if int(row["relation"]) == cluster]
        for cluster in clusters
    }
    rng = random.Random(seed)
    estimates = []
    for _ in range(samples):
        chosen = [rng.choice(clusters) for _ in clusters]
        estimates.append(fmean(value for cluster in chosen for value in by_cluster[cluster]))
    values = [float(row[score_column]) for row in selected]
    return {
        "count": len(values),
        "relation_clusters": len(clusters),
        "mean": fmean(values),
        "ci_low": _percentile(estimates, 0.025),
        "ci_high": _percentile(estimates, 0.975),
    }


def pretrained_reviewer_controls(
    run_dirs: list[str | Path],
    output_dir: str | Path,
    bootstrap_samples: int = 2000,
    seed: int = 1729,
) -> Path:
    """Build four-class controls and uncertainty for unseen-prompt transfer."""
    out = Path(output_dir).resolve(); out.mkdir(parents=True, exist_ok=True)
    runs = [
        (PretrainedConfig.load(Path(path) / "config.resolved.json").seed, Path(path).resolve())
        for path in run_dirs
    ]
    canonical = {
        run_seed: _read(run / "decomposition" / "features_merged.csv")
        for run_seed, run in runs
    }

    # Train only on the preregistered generalized-vs-memorized task, then apply
    # the frozen detector to all four construct-validity classes.
    four_class_rows: list[dict[str, object]] = []
    for held_out_seed, _ in runs:
        train_rows = [
            row for run_seed, rows in canonical.items() if run_seed != held_out_seed
            for row in rows if row["group"] in {"generalized", "memorized"}
        ]
        test_rows = canonical[held_out_seed]
        train_labels = [int(row["group"] == "memorized") for row in train_rows]
        scores_by_detector = {
            name: logistic_train_test_scores(
                _features(train_rows, columns), train_labels,
                _features(test_rows, columns), seed=held_out_seed,
            )
            for name, columns in FEATURE_SETS.items()
        }
        for row_index, row in enumerate(test_rows):
            four_class_rows.append({
                "held_out_seed": held_out_seed,
                "index": int(row["index"]),
                "relation": int(row["relation"]),
                "key": int(row["key"]),
                "group": row["group"],
                "confidence_probability": scores_by_detector["confidence"][row_index],
                "tomography_probability": scores_by_detector["tomography"][row_index],
                "combined_probability": scores_by_detector["combined"][row_index],
            })

    four_class_summary = {
        detector: {
            group: _cluster_bootstrap_mean(
                four_class_rows, group, f"{detector}_probability",
                samples=bootstrap_samples, seed=seed + detector_index * 10 + group_index,
            )
            for group_index, group in enumerate(("seen_rule", "generalized", "memorized", "nonmember"))
        }
        for detector_index, detector in enumerate(FEATURE_SETS)
    }

    prompt_records: list[dict[str, object]] = []
    prompt_bootstraps: list[list[float]] = []
    for held_out_seed, run in runs:
        train_rows = [
            row for run_seed, rows in canonical.items() if run_seed != held_out_seed
            for row in rows if row["group"] in {"generalized", "memorized"}
        ]
        train_labels = [int(row["group"] == "memorized") for row in train_rows]
        for template_index, template in enumerate(UNSEEN_EVAL_TEMPLATES):
            test_rows = _read(
                run / "prompt_robustness" / f"features_merged_{template}.csv",
                groups={"generalized", "memorized"},
            )
            test_labels = [int(row["group"] == "memorized") for row in test_rows]
            fitted_scores = {
                name: logistic_train_test_scores(
                    _features(train_rows, columns), train_labels,
                    _features(test_rows, columns), seed=held_out_seed,
                )
                for name, columns in FEATURE_SETS.items()
            }
            aucs = {name: roc_auc(test_labels, values) for name, values in fitted_scores.items()}
            deltas, interval = _cluster_bootstrap_auc_delta(
                test_rows, fitted_scores["confidence"], fitted_scores["combined"],
                samples=bootstrap_samples,
                seed=seed + held_out_seed * 10 + template_index,
            )
            prompt_bootstraps.append(deltas)
            prompt_records.append({
                "held_out_seed": held_out_seed,
                "test_template": template,
                "test_count": len(test_rows),
                "confidence_auc": aucs["confidence"],
                "tomography_auc": aucs["tomography"],
                "combined_auc": aucs["combined"],
                "incremental_auc": aucs["combined"] - aucs["confidence"],
                "incremental_ci_low": interval["low"],
                "incremental_ci_high": interval["high"],
                "relation_clusters": interval["cluster_count"],
            })

    common_samples = min((len(values) for values in prompt_bootstraps), default=0)
    mean_bootstrap = [
        fmean(values[index] for values in prompt_bootstraps)
        for index in range(common_samples)
    ]
    increments = [float(row["incremental_auc"]) for row in prompt_records]
    prompt_summary = {
        "cell_count": len(prompt_records),
        "mean_incremental_auc": fmean(increments),
        "positive_cells": sum(value > 0 for value in increments),
        "mean_incremental_ci_low": _percentile(mean_bootstrap, 0.025),
        "mean_incremental_ci_high": _percentile(mean_bootstrap, 0.975),
        "bootstrap_unit": "relation within adapter-template cell",
        "bootstrap_samples": common_samples,
    }

    score_path = out / "four_class_scores.csv"
    with score_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(four_class_rows[0]))
        writer.writeheader(); writer.writerows(four_class_rows)
    record_path = out / "prompt_transfer_intervals.csv"
    with record_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(prompt_records[0]))
        writer.writeheader(); writer.writerows(prompt_records)

    artifact = {
        "protocol": {
            "detector_training_classes": ["generalized", "memorized"],
            "detector_features": FEATURE_SETS,
            "evaluation": "leave_one_adapter_out",
            "correct_outputs_only": True,
            "bootstrap_seed": seed,
        },
        "four_class_summary": four_class_summary,
        "prompt_transfer_records": prompt_records,
        "prompt_transfer_summary": prompt_summary,
        "artifacts": {
            "four_class_scores": str(score_path),
            "prompt_transfer_intervals": str(record_path),
        },
    }
    path = out / "analysis.json"
    path.write_text(json.dumps(artifact, indent=2, sort_keys=True), encoding="utf-8")
    return path
