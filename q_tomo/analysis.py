from __future__ import annotations

import math
import random
from typing import Sequence

import torch


def roc_auc(labels: Sequence[int], scores: Sequence[float]) -> float:
    if len(labels) != len(scores) or not labels:
        return float("nan")
    positives = sum(labels)
    negatives = len(labels) - positives
    if positives == 0 or negatives == 0:
        return float("nan")
    ordered = sorted(range(len(scores)), key=lambda index: scores[index])
    ranks = [0.0] * len(scores)
    position = 0
    while position < len(ordered):
        end = position + 1
        while end < len(ordered) and scores[ordered[end]] == scores[ordered[position]]:
            end += 1
        average_rank = (position + 1 + end) / 2.0
        for offset in range(position, end):
            ranks[ordered[offset]] = average_rank
        position = end
    positive_rank_sum = sum(rank for rank, label in zip(ranks, labels) if label == 1)
    return (positive_rank_sum - positives * (positives + 1) / 2.0) / (positives * negatives)


def separability_auc(labels: Sequence[int], scores: Sequence[float]) -> float:
    auc = roc_auc(labels, scores)
    return max(auc, 1.0 - auc) if math.isfinite(auc) else auc


def _stratified_folds(labels: list[int], folds: int, seed: int) -> list[list[int]]:
    rng = random.Random(seed)
    result = [[] for _ in range(folds)]
    for label in (0, 1):
        indices = [index for index, value in enumerate(labels) if value == label]
        rng.shuffle(indices)
        for offset, index in enumerate(indices):
            result[offset % folds].append(index)
    return result


def _stratified_group_folds(
    labels: list[int], groups: list[int], folds: int, seed: int
) -> list[list[int]]:
    if len(labels) != len(groups):
        raise ValueError("labels and groups must have equal length")
    group_label_sets: dict[int, set[int]] = {}
    for label, group in zip(labels, groups):
        group_label_sets.setdefault(group, set()).add(label)
    rng = random.Random(seed)
    validation_groups = [set() for _ in range(folds)]
    if all(len(values) == 1 for values in group_label_sets.values()):
        for label in (0, 1):
            class_groups = [group for group, values in group_label_sets.items() if label in values]
            rng.shuffle(class_groups)
            for offset, group in enumerate(class_groups):
                validation_groups[offset % folds].add(group)
    else:
        mixed_groups = list(group_label_sets)
        rng.shuffle(mixed_groups)
        for offset, group in enumerate(mixed_groups):
            validation_groups[offset % folds].add(group)
    return [
        [index for index, group in enumerate(groups) if group in fold_groups]
        for fold_groups in validation_groups
    ]


def _logistic_auc_from_folds(
    features: list[list[float]], labels: list[int], fold_indices: list[list[int]], seed: int
) -> float:
    predictions = [0.0] * len(labels)
    all_indices = set(range(len(labels)))
    torch.manual_seed(seed)
    for validation_indices in fold_indices:
        if not validation_indices:
            continue
        train_indices = sorted(all_indices - set(validation_indices))
        x_train = torch.tensor([features[index] for index in train_indices], dtype=torch.float32)
        y_train = torch.tensor([labels[index] for index in train_indices], dtype=torch.float32)
        x_validation = torch.tensor([features[index] for index in validation_indices], dtype=torch.float32)
        mean = x_train.mean(dim=0, keepdim=True)
        std = x_train.std(dim=0, keepdim=True, unbiased=False).clamp_min(1e-6)
        x_train = (x_train - mean) / std
        x_validation = (x_validation - mean) / std
        weights = torch.zeros(x_train.shape[1], requires_grad=True)
        bias = torch.zeros((), requires_grad=True)
        optimizer = torch.optim.LBFGS([weights, bias], lr=0.5, max_iter=80, line_search_fn="strong_wolfe")

        def closure() -> torch.Tensor:
            optimizer.zero_grad()
            logits = x_train @ weights + bias
            loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, y_train)
            loss = loss + 1e-2 * weights.pow(2).mean()
            loss.backward()
            return loss

        optimizer.step(closure)
        with torch.no_grad():
            fold_predictions = torch.sigmoid(x_validation @ weights + bias).tolist()
        for index, prediction in zip(validation_indices, fold_predictions):
            predictions[index] = prediction
    return roc_auc(labels, predictions)


def logistic_cross_validated_auc(
    features: list[list[float]],
    labels: list[int],
    seed: int = 0,
    max_folds: int = 5,
) -> float:
    if not features or len(set(labels)) < 2:
        return float("nan")
    minimum_class = min(labels.count(0), labels.count(1))
    folds = min(max_folds, minimum_class)
    if folds < 2:
        return float("nan")
    fold_indices = _stratified_folds(labels, folds, seed)
    return _logistic_auc_from_folds(features, labels, fold_indices, seed)


def logistic_group_cross_validated_auc(
    features: list[list[float]],
    labels: list[int],
    groups: list[int],
    seed: int = 0,
    max_folds: int = 5,
) -> float:
    if not features or len(set(labels)) < 2 or len(groups) != len(labels):
        return float("nan")
    group_label_sets: dict[int, set[int]] = {}
    for group, label in zip(groups, labels):
        group_label_sets.setdefault(group, set()).add(label)
    if all(len(values) == 1 for values in group_label_sets.values()):
        minimum_groups = min(
            sum(0 in values for values in group_label_sets.values()),
            sum(1 in values for values in group_label_sets.values()),
        )
    else:
        minimum_groups = len(group_label_sets)
    folds = min(max_folds, minimum_groups)
    if folds < 2:
        return float("nan")
    fold_indices = _stratified_group_folds(labels, groups, folds, seed)
    return _logistic_auc_from_folds(features, labels, fold_indices, seed)


def logistic_train_test_auc(
    train_features: list[list[float]], train_labels: list[int],
    test_features: list[list[float]], test_labels: list[int], seed: int = 0,
) -> float:
    """Fit a standardized logistic model on one dataset and score another."""
    predictions = logistic_train_test_scores(
        train_features, train_labels, test_features, seed=seed
    )
    if not predictions or len(set(test_labels)) < 2:
        return float("nan")
    return roc_auc(test_labels, predictions)


def logistic_train_test_scores(
    train_features: list[list[float]],
    train_labels: list[int],
    test_features: list[list[float]],
    seed: int = 0,
) -> list[float]:
    """Fit a standardized logistic model and return test probabilities.

    Keeping fitting and scoring in one deterministic helper lets downstream
    analyses bootstrap or inspect predictions without repeatedly refitting the
    detector inside every resample.
    """
    if not train_features or not test_features or len(set(train_labels)) < 2:
        return []
    torch.manual_seed(seed)
    x_train = torch.tensor(train_features, dtype=torch.float32)
    y_train = torch.tensor(train_labels, dtype=torch.float32)
    x_test = torch.tensor(test_features, dtype=torch.float32)
    mean = x_train.mean(dim=0, keepdim=True)
    std = x_train.std(dim=0, keepdim=True, unbiased=False).clamp_min(1e-6)
    x_train = (x_train - mean) / std
    x_test = (x_test - mean) / std
    weights = torch.zeros(x_train.shape[1], requires_grad=True)
    bias = torch.zeros((), requires_grad=True)
    optimizer = torch.optim.LBFGS([weights, bias], lr=0.5, max_iter=80, line_search_fn="strong_wolfe")

    def closure() -> torch.Tensor:
        optimizer.zero_grad()
        logits = x_train @ weights + bias
        loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, y_train)
        loss = loss + 1e-2 * weights.pow(2).mean()
        loss.backward()
        return loss

    optimizer.step(closure)
    with torch.no_grad():
        predictions = torch.sigmoid(x_test @ weights + bias).tolist()
    return predictions
