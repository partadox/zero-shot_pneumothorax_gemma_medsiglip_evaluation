"""
Evaluation metrics:
  - Classification: F1, Sensitivity, Specificity, AUROC + bootstrap 95% CI
  - Calibration: ECE, Brier score, reliability diagram data
"""

from __future__ import annotations

import warnings
from typing import Callable

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, f1_score, recall_score,
    roc_auc_score, confusion_matrix, brier_score_loss,
)


# ── Classification ─────────────────────────────────────────────────────────────

def compute_classification_metrics(
    y_true: list | np.ndarray,
    y_pred: list | np.ndarray,
    y_score: list | np.ndarray | None = None,
) -> dict:
    """
    Binary classification metrics ignoring -1 (unparsable).

    Returns:
        sensitivity, specificity, precision, f1, accuracy, auroc,
        n_pos, n_neg, tp, tn, fp, fn, n_invalid, n_total
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    valid = y_pred != -1
    n_invalid = int((~valid).sum())
    yt = y_true[valid]
    yp = y_pred[valid]

    base = {
        "n_total":   len(y_true),
        "n_valid":   int(valid.sum()),
        "n_invalid": n_invalid,
    }

    if len(yt) == 0 or len(np.unique(yt)) < 2:
        return {**base, **{k: None for k in
                ["sensitivity", "specificity", "precision", "f1",
                 "accuracy", "auroc", "n_pos", "n_neg",
                 "tp", "tn", "fp", "fn"]}}

    tn, fp, fn, tp = confusion_matrix(yt, yp, labels=[0, 1]).ravel()
    tp, tn, fp, fn = int(tp), int(tn), int(fp), int(fn)
    n_pos = int(yt.sum())
    n_neg = int(len(yt) - n_pos)

    sens  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    spec  = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    prec  = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    f1    = 2 * prec * sens / (prec + sens) if (prec + sens) > 0 else 0.0
    acc   = (tp + tn) / len(yt)

    auroc = None
    if y_score is not None:
        ys = np.array(y_score)[valid]
        mask = ~np.isnan(ys)
        if mask.sum() > 0 and len(np.unique(yt[mask])) == 2:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                auroc = float(roc_auc_score(yt[mask], ys[mask]))

    return {
        **base,
        "sensitivity":  round(sens, 4),
        "specificity":  round(spec, 4),
        "precision":    round(prec, 4),
        "f1":           round(f1, 4),
        "accuracy":     round(acc, 4),
        "auroc":        round(auroc, 4) if auroc is not None else None,
        "n_pos": n_pos, "n_neg": n_neg,
        "tp": tp, "tn": tn, "fp": fp, "fn": fn,
    }


# ── Bootstrap CI ───────────────────────────────────────────────────────────────

def bootstrap_ci(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    metric_fn: Callable,
    n: int = 1000,
    alpha: float = 0.05,
    seed: int = 42,
    **metric_kwargs,
) -> tuple[float, float]:
    """
    Bootstrap 95% CI for a scalar metric.

    metric_fn(y_true, y_pred, **metric_kwargs) → float
    Returns (lower, upper).
    """
    rng = np.random.default_rng(seed)
    scores = []
    n_samples = len(y_true)

    for _ in range(n):
        idx = rng.integers(0, n_samples, size=n_samples)
        try:
            s = metric_fn(y_true[idx], y_pred[idx], **metric_kwargs)
            if s is not None:
                scores.append(s)
        except Exception:
            pass

    if not scores:
        return (float("nan"), float("nan"))

    scores = np.array(scores)
    lower = float(np.percentile(scores, 100 * alpha / 2))
    upper = float(np.percentile(scores, 100 * (1 - alpha / 2)))
    return lower, upper


def add_ci_to_metrics(
    metrics: dict,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    n: int = 1000,
) -> dict:
    """
    Add 95% CI for F1, sensitivity, specificity to a metrics dict.

    Adds keys: f1_ci, sensitivity_ci, specificity_ci as (lo, hi) tuples.
    """
    valid = y_pred != -1
    yt = np.array(y_true)[valid]
    yp = np.array(y_pred)[valid]

    def _f1(yt, yp):
        return f1_score(yt, yp, zero_division=0)

    def _sens(yt, yp):
        return recall_score(yt, yp, zero_division=0)

    def _spec(yt, yp):
        tn, fp, fn, tp = confusion_matrix(yt, yp, labels=[0, 1]).ravel()
        return tn / (tn + fp) if (tn + fp) > 0 else 0.0

    if len(yt) > 0 and len(np.unique(yt)) == 2:
        metrics["f1_ci"]          = bootstrap_ci(yt, yp, _f1)
        metrics["sensitivity_ci"] = bootstrap_ci(yt, yp, _sens)
        metrics["specificity_ci"] = bootstrap_ci(yt, yp, _spec)
    else:
        metrics["f1_ci"] = metrics["sensitivity_ci"] = metrics["specificity_ci"] = (None, None)

    return metrics


# ── Calibration ────────────────────────────────────────────────────────────────

def compute_ece(
    y_true: np.ndarray,
    y_conf: np.ndarray,
    n_bins: int = 10,
) -> float:
    """Expected Calibration Error."""
    valid = ~np.isnan(y_conf)
    yt = np.array(y_true)[valid]
    yc = np.array(y_conf)[valid]

    if len(yt) == 0:
        return float("nan")

    bins = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (yc >= lo) & (yc < hi)
        if not mask.any():
            continue
        acc = yt[mask].mean()
        conf = yc[mask].mean()
        ece += mask.sum() / len(yt) * abs(acc - conf)
    return float(ece)


def compute_reliability_data(
    y_true: np.ndarray,
    y_conf: np.ndarray,
    n_bins: int = 10,
) -> pd.DataFrame:
    """
    Compute reliability diagram data.

    Returns DataFrame with columns: bin_center, mean_conf, mean_acc, n_samples.
    """
    valid = ~np.isnan(y_conf)
    yt = np.array(y_true)[valid]
    yc = np.array(y_conf)[valid]

    bins = np.linspace(0, 1, n_bins + 1)
    rows = []
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (yc >= lo) & (yc < hi)
        rows.append({
            "bin_center": round((lo + hi) / 2, 2),
            "mean_conf":  float(yc[mask].mean()) if mask.any() else None,
            "mean_acc":   float(yt[mask].mean()) if mask.any() else None,
            "n_samples":  int(mask.sum()),
        })
    return pd.DataFrame(rows)


def compute_brier(y_true: np.ndarray, y_conf: np.ndarray) -> float:
    valid = ~np.isnan(y_conf)
    yt = np.array(y_true)[valid]
    yc = np.array(y_conf)[valid]
    if len(yt) == 0:
        return float("nan")
    return float(brier_score_loss(yt, yc))
