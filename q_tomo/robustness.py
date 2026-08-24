from __future__ import annotations

import csv
import json
import math
import random
from pathlib import Path
from statistics import fmean, pstdev

from q_tomo.analysis import (
    logistic_cross_validated_auc,
    logistic_group_cross_validated_auc,
    roc_auc,
    separability_auc,
)


BASELINE = ["base_nll", "base_margin"]
TOMOGRAPHY = [
    "q_fragility",
    "q_variability",
    "q_survival_area",
    "layer_entropy",
    "layer_concentration",
]


def _load_rows(path: str | Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    with Path(path).open(newline="", encoding="utf-8") as handle:
        for raw in csv.DictReader(handle):
            row: dict[str, object] = dict(raw)
            row["relation"] = int(raw["relation"])
            row["base_correct"] = raw["base_correct"].lower() == "true"
            for name in BASELINE + TOMOGRAPHY:
                row[name] = float(raw[name])
            for name in raw:
                if name.startswith("q") or name.startswith("layer_"):
                    try:
                        row[name] = float(raw[name])
                    except ValueError:
                        pass
            rows.append(row)
    return rows


def _feature_matrix(rows: list[dict[str, object]], names: list[str]) -> list[list[float]]:
    return [[float(row[name]) for name in names] for row in rows]


def _summary(values: list[float]) -> dict[str, float]:
    ordered = sorted(values)
    return {
        "mean": fmean(values),
        "std": pstdev(values),
        "min": ordered[0],
        "max": ordered[-1],
    }


def _confidence_match(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    by_label = {
        0: [row for row in rows if row["group"] == "generalized"],
        1: [row for row in rows if row["group"] == "memorized"],
    }
    all_vectors = [[float(row[name]) for name in BASELINE] for row in rows]
    means = [fmean(vector[column] for vector in all_vectors) for column in range(2)]
    stds = [
        max(1e-8, math.sqrt(fmean((vector[column] - means[column]) ** 2 for vector in all_vectors)))
        for column in range(2)
    ]
    available = set(range(len(by_label[1])))
    matched: list[dict[str, object]] = []
    for left in sorted(by_label[0], key=lambda row: (float(row["base_nll"]), int(row["relation"]))):
        left_vector = [(float(left[name]) - means[i]) / stds[i] for i, name in enumerate(BASELINE)]
        best = min(
            available,
            key=lambda index: sum(
                (left_vector[i] - (float(by_label[1][index][name]) - means[i]) / stds[i]) ** 2
                for i, name in enumerate(BASELINE)
            ),
        )
        available.remove(best)
        matched.extend((left, by_label[1][best]))
    return matched


def _cluster_bootstrap_auc(
    rows: list[dict[str, object]], score_name: str, seed: int, samples: int
) -> dict[str, float]:
    rng = random.Random(seed)
    relation_rows: dict[int, list[dict[str, object]]] = {}
    for row in rows:
        relation_rows.setdefault(int(row["relation"]), []).append(row)
    relations = {
        label: sorted({int(row["relation"]) for row in rows if (row["group"] == "memorized") == bool(label)})
        for label in (0, 1)
    }
    estimates: list[float] = []
    for _ in range(samples):
        sampled: list[dict[str, object]] = []
        for label in (0, 1):
            chosen = rng.choices(relations[label], k=len(relations[label]))
            for relation in chosen:
                sampled.extend(relation_rows[relation])
        labels = [1 if row["group"] == "memorized" else 0 for row in sampled]
        estimates.append(separability_auc(labels, [float(row[score_name]) for row in sampled]))
    estimates.sort()
    return {
        "estimate": separability_auc(
            [1 if row["group"] == "memorized" else 0 for row in rows],
            [float(row[score_name]) for row in rows],
        ),
        "cluster_bootstrap_95_low": estimates[int(0.025 * samples)],
        "cluster_bootstrap_95_high": estimates[min(samples - 1, int(0.975 * samples))],
    }


def _cluster_permutation_p(
    rows: list[dict[str, object]], score_name: str, seed: int, permutations: int
) -> float:
    rng = random.Random(seed)
    relations = sorted({int(row["relation"]) for row in rows})
    positives = len({int(row["relation"]) for row in rows if row["group"] == "memorized"})
    scores = [float(row[score_name]) for row in rows]
    observed = separability_auc(
        [1 if row["group"] == "memorized" else 0 for row in rows], scores
    )
    exceedances = 0
    for _ in range(permutations):
        positive_relations = set(rng.sample(relations, positives))
        labels = [1 if int(row["relation"]) in positive_relations else 0 for row in rows]
        exceedances += separability_auc(labels, scores) >= observed
    return (exceedances + 1) / (permutations + 1)


def robust_analysis(
    features_path: str | Path,
    output_path: str | Path | None = None,
    seed: int = 0,
) -> Path:
    source = Path(features_path).resolve()
    rows = [
        row
        for row in _load_rows(source)
        if row["group"] in {"generalized", "memorized"} and row["base_correct"]
    ]
    labels = [1 if row["group"] == "memorized" else 0 for row in rows]
    groups = [int(row["relation"]) for row in rows]
    feature_sets = {
        "confidence": BASELINE,
        "tomography": TOMOGRAPHY,
        "combined": BASELINE + TOMOGRAPHY,
    }
    random_cv = {
        name: logistic_cross_validated_auc(_feature_matrix(rows, names), labels, seed=seed)
        for name, names in feature_sets.items()
    }
    group_repeats: dict[str, list[float]] = {name: [] for name in feature_sets}
    for repeat_seed in range(seed, seed + 10):
        for name, names in feature_sets.items():
            group_repeats[name].append(
                logistic_group_cross_validated_auc(
                    _feature_matrix(rows, names), labels, groups, seed=repeat_seed
                )
            )
    grouped = {name: _summary(values) for name, values in group_repeats.items()}
    grouped["incremental"] = _summary(
        [combined - baseline for combined, baseline in zip(group_repeats["combined"], group_repeats["confidence"])]
    )

    matched = _confidence_match(rows)
    matched_labels = [1 if row["group"] == "memorized" else 0 for row in matched]
    matched_groups = [int(row["relation"]) for row in matched]
    matched_results = {
        "per_class": len(matched) // 2,
        "confidence_group_cv_auc": logistic_group_cross_validated_auc(
            _feature_matrix(matched, BASELINE), matched_labels, matched_groups, seed=seed
        ),
        "tomography_group_cv_auc": logistic_group_cross_validated_auc(
            _feature_matrix(matched, TOMOGRAPHY), matched_labels, matched_groups, seed=seed
        ),
        "fragility_separability_auc": separability_auc(
            matched_labels, [float(row["q_fragility"]) for row in matched]
        ),
    }

    bit_widths = sorted(
        {
            int(name[1:].split("_")[0])
            for name in rows[0]
            if name.startswith("q") and name.endswith("_mean_abs_delta_nll")
        },
        reverse=True,
    )
    bit_ablation = {
        str(bits): {
            "abs_delta_separability_auc": separability_auc(
                labels, [float(row[f"q{bits}_mean_abs_delta_nll"]) for row in rows]
            ),
            "survival_separability_auc": separability_auc(
                labels, [float(row[f"q{bits}_survival"]) for row in rows]
            ),
        }
        for bits in bit_widths
    }

    layer_names = sorted(
        name for name in rows[0] if name.startswith("layer_blocks.") and name.endswith("_sensitivity")
    )
    layer_effects = []
    for name in layer_names:
        generalized = fmean(float(row[name]) for row in rows if row["group"] == "generalized")
        memorized = fmean(float(row[name]) for row in rows if row["group"] == "memorized")
        layer_effects.append(
            {"scope": name.removeprefix("layer_").removesuffix("_sensitivity"), "generalized": generalized, "memorized": memorized, "absolute_gap": abs(memorized - generalized)}
        )
    layer_effects.sort(key=lambda item: float(item["absolute_gap"]), reverse=True)

    fragility_inference = _cluster_bootstrap_auc(rows, "q_fragility", seed, 1000)
    fragility_inference["relation_permutation_p"] = _cluster_permutation_p(
        rows, "q_fragility", seed, 1000
    )
    result = {
        "source": str(source),
        "counts": {
            "generalized": labels.count(0),
            "memorized": labels.count(1),
            "relations": len(set(groups)),
        },
        "example_random_cv_auc": random_cv,
        "relation_held_out_cv_auc_10_splits": grouped,
        "confidence_matched": matched_results,
        "fragility_cluster_inference": fragility_inference,
        "bit_width_ablation": bit_ablation,
        "top_layer_gaps": layer_effects[:10],
    }
    target = Path(output_path).resolve() if output_path else source.parent / "robust_analysis.json"
    with target.open("w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, sort_keys=True)
    print(json.dumps(result, indent=2, sort_keys=True))
    return target
