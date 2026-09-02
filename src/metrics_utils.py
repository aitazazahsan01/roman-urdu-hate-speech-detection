"""Shared classification metrics for Roman Urdu hate-speech models. Used by both
evaluate.py (transformer) and train_baseline.py (TF-IDF + Logistic Regression), so
the two are scored identically and can be compared apples-to-apples in the README.
"""

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)


def compute_metrics_dict(y_true, y_pred, label_names: list[str], split: str) -> dict:
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=range(len(label_names)), average=None, zero_division=0
    )
    per_class = [
        {"label": name, "precision": float(p), "recall": float(r), "f1": float(f), "support": int(s)}
        for name, p, r, f, s in zip(label_names, precision, recall, f1, support)
    ]
    per_class.sort(key=lambda row: row["f1"])  # worst classes first

    return {
        "split": split,
        "n": len(y_true),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "per_class": per_class,
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=range(len(label_names))).tolist(),
        "label_names": label_names,
    }


def compute_metrics_for_trainer(eval_pred) -> dict:
    """Flat scalar dict for HF Trainer(compute_metrics=...) -- Trainer logging and
    metric_for_best_model need flat float values, not the nested breakdown above."""
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {
        "accuracy": float(accuracy_score(labels, preds)),
        "macro_f1": float(f1_score(labels, preds, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(labels, preds, average="weighted", zero_division=0)),
    }
