"""Spoločné metriky pre tréning aj vyhodnotenie."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_recall_fscore_support,
)

LABELS = [0, 1, 2]


def _normalise(probabilities: np.ndarray) -> np.ndarray:
    probabilities = np.asarray(probabilities, dtype=float)
    probabilities = np.clip(probabilities, 1e-15, 1.0)
    totals = probabilities.sum(axis=1, keepdims=True)
    return probabilities / totals


def multiclass_brier(y_true: Sequence[int], probabilities: np.ndarray) -> float:
    """Priemerná kvadratická chyba voči one-hot reprezentácii."""
    probabilities = _normalise(probabilities)
    one_hot = np.zeros_like(probabilities)
    one_hot[np.arange(len(y_true)), np.asarray(y_true)] = 1.0
    return float(np.mean(np.sum((probabilities - one_hot) ** 2, axis=1)))


def compute_metrics(
    y_true: Sequence[int],
    probabilities: np.ndarray,
    labels: Sequence[int] = LABELS,
) -> dict[str, float]:
    y_true = np.asarray(y_true)
    probabilities = _normalise(probabilities)
    predictions = probabilities.argmax(axis=1)

    return {
        "n": int(len(y_true)),
        "accuracy": float(accuracy_score(y_true, predictions)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, predictions)),
        "f1_macro": float(f1_score(y_true, predictions, average="macro", labels=labels, zero_division=0)),
        "log_loss": float(log_loss(y_true, probabilities, labels=labels)),
        "brier": multiclass_brier(y_true, probabilities),
    }


def per_class_metrics(
    y_true: Sequence[int],
    probabilities: np.ndarray,
    labels: Sequence[int] = LABELS,
) -> pd.DataFrame:
    predictions = _normalise(probabilities).argmax(axis=1)
    precision, recall, f1, support = precision_recall_fscore_support(
        np.asarray(y_true),
        predictions,
        labels=labels,
        zero_division=0,
    )
    return pd.DataFrame(
        {
            "class_id": labels,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": support,
        }
    )


def confusion(y_true: Sequence[int], probabilities: np.ndarray, labels: Sequence[int] = LABELS) -> np.ndarray:
    predictions = _normalise(probabilities).argmax(axis=1)
    return confusion_matrix(np.asarray(y_true), predictions, labels=labels)
