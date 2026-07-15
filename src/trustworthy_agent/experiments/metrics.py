"""Metric computation for protocol experiments."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
from numpy.typing import NDArray
from sklearn.metrics import (  # type: ignore[import-untyped]
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_recall_fscore_support,
)


def compute_classification_metrics(
    *,
    y_true: Sequence[str],
    y_pred: Sequence[str],
    labels: Sequence[str],
    probabilities: NDArray[np.float64] | None = None,
    probability_classes: Sequence[str] = (),
) -> dict[str, Any]:
    """Compute required classification metrics without fitting anything.

    Purpose:
        Generate protocol metrics for E1-E3 from already-produced predictions.
    Parameters:
        y_true: True labels for the evaluation split.
        y_pred: Predicted labels for the same rows.
        labels: Canonical label order.
        probabilities: Optional class-probability matrix.
        probability_classes: Column order for `probabilities`.
    Return value:
        JSON-serializable metrics dictionary.
    Raised exceptions:
        ValueError from scikit-learn when inputs are inconsistent.
    Scientific assumptions:
        Metrics are descriptive evaluation outputs; they do not imply clinical
        or industrial deployment readiness.
    Side effects:
        None.
    Reproducibility implications:
        Label order is explicit for confusion matrix and probability metrics.
    """

    precision, recall, f1, support = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=list(labels),
        zero_division=0,
    )
    per_class = {
        label: {
            "precision": float(precision[index]),
            "recall": float(recall[index]),
            "f1": float(f1[index]),
            "support": int(support[index]),
        }
        for index, label in enumerate(labels)
    }
    metrics: dict[str, Any] = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(
            f1_score(y_true, y_pred, labels=list(labels), average="macro", zero_division=0)
        ),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "per_class": per_class,
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=list(labels))
        .astype(int)
        .tolist(),
        "probability_metrics_available": probabilities is not None,
    }
    if probabilities is not None:
        log_loss_labels = tuple(sorted(labels))
        aligned = _align_probabilities(probabilities, probability_classes, log_loss_labels)
        metrics["log_loss"] = float(log_loss(y_true, aligned, labels=list(log_loss_labels)))
        metrics["brier_score"] = _multiclass_brier_score(y_true, aligned, log_loss_labels)
        metrics["expected_calibration_error"] = _expected_calibration_error(
            y_true,
            y_pred,
            aligned,
            log_loss_labels,
        )
        metrics["mean_max_probability"] = float(np.mean(np.max(aligned, axis=1)))
    return metrics


def compute_metrics(*args: object, **kwargs: object) -> dict[str, Any]:
    """Backward-compatible wrapper for classification metrics.

    Purpose:
        Preserve the public boundary name introduced by the scaffold while
        routing callers to explicit metric functions.
    Parameters:
        args, kwargs: Forwarded only to `compute_classification_metrics`.
    Return value:
        Classification metrics dictionary.
    Raised exceptions:
        TypeError for unsupported positional use.
    Scientific assumptions:
        None beyond the delegated metric function.
    Side effects:
        None.
    Reproducibility implications:
        None beyond deterministic metric computation.
    """

    if args:
        raise TypeError("compute_metrics requires keyword arguments.")
    return compute_classification_metrics(**kwargs)  # type: ignore[arg-type]


def _align_probabilities(
    probabilities: NDArray[np.float64],
    probability_classes: Sequence[str],
    labels: Sequence[str],
) -> NDArray[np.float64]:
    index_by_class = {label: index for index, label in enumerate(probability_classes)}
    aligned = np.zeros((probabilities.shape[0], len(labels)), dtype=float)
    for target_index, label in enumerate(labels):
        source_index = index_by_class.get(label)
        if source_index is not None:
            aligned[:, target_index] = probabilities[:, source_index]
    row_sums = aligned.sum(axis=1)
    row_sums = np.where(row_sums <= 0.0, 1.0, row_sums)
    return aligned / row_sums[:, None]


def _multiclass_brier_score(
    y_true: Sequence[str],
    probabilities: NDArray[np.float64],
    labels: Sequence[str],
) -> float:
    label_index = {label: index for index, label in enumerate(labels)}
    target = np.zeros_like(probabilities)
    for row_index, label in enumerate(y_true):
        target[row_index, label_index[label]] = 1.0
    return float(np.mean(np.sum(np.square(probabilities - target), axis=1)))


def _expected_calibration_error(
    y_true: Sequence[str],
    y_pred: Sequence[str],
    probabilities: NDArray[np.float64],
    labels: Sequence[str],
    *,
    bins: int = 10,
) -> float:
    del labels
    confidences = np.max(probabilities, axis=1)
    correctness = np.asarray(
        [true == pred for true, pred in zip(y_true, y_pred, strict=True)], dtype=float
    )
    ece = 0.0
    for bin_index in range(bins):
        lower = bin_index / bins
        upper = (bin_index + 1) / bins
        if bin_index == bins - 1:
            mask = (confidences >= lower) & (confidences <= upper)
        else:
            mask = (confidences >= lower) & (confidences < upper)
        if not np.any(mask):
            continue
        bin_weight = float(np.mean(mask))
        bin_accuracy = float(np.mean(correctness[mask]))
        bin_confidence = float(np.mean(confidences[mask]))
        ece += bin_weight * abs(bin_accuracy - bin_confidence)
    return float(ece)
