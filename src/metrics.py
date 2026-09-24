from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)


def classification_metrics(y_true, y_prob, threshold=0.5):
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob, dtype=float)
    y_pred = (y_prob >= threshold).astype(int)
    out = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_weighted": float(precision_score(y_true, y_pred, average="weighted", zero_division=0)),
        "recall_weighted": float(recall_score(y_true, y_pred, average="weighted", zero_division=0)),
        "f1_weighted": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "precision_toxic": float(precision_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "recall_toxic": float(recall_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "f1_toxic": float(f1_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "precision_nontoxic": float(precision_score(y_true, y_pred, pos_label=0, zero_division=0)),
        "recall_nontoxic": float(recall_score(y_true, y_pred, pos_label=0, zero_division=0)),
        "f1_nontoxic": float(f1_score(y_true, y_pred, pos_label=0, zero_division=0)),
        "mcc": float(matthews_corrcoef(y_true, y_pred)),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
    }
    if len(np.unique(y_true)) == 2:
        out["roc_auc"] = float(roc_auc_score(y_true, y_prob))
    return out
