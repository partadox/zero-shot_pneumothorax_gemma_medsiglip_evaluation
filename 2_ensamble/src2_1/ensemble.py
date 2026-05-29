"""
Ensemble strategies and score reinterpretation for MedGemma + MedSigLIP.

Key insight:
  MedGemma confidence_explicit (1-10) = confidence IN the prediction (PTX or Normal),
  not P(PTX). Must reinterpret direction before using as probability score.
  MedSigLIP confidence_explicit = cosine similarity to PTX text prompt, already [0,1].
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


# ── Score extraction ───────────────────────────────────────────────────────────

def reinterpret_gemma_score(df: pd.DataFrame) -> np.ndarray:
    """
    Convert MedGemma confidence to directional P(PTX).

    MedGemma reports "how confident am I in my prediction (PTX or Normal)?"
    We need "how likely is this image to have PTX?"

    Transformation:
        prediction=PTX   → score_ptx = confidence / 10
        prediction=Normal → score_ptx = 1 - confidence / 10

    Note: at threshold=0.5, binary predictions are unchanged when confidence >= 5
    (which is the case empirically). AUROC changes because score ORDERING changes.
    """
    conf = df['confidence_explicit'].values.astype(float) / 10.0
    diag = df['diagnosis'].values.astype(int)
    return np.where(diag == 1, conf, 1.0 - conf)


def get_medsiglip_ptx_score(df: pd.DataFrame) -> np.ndarray:
    """
    MedSigLIP confidence_explicit = cosine_similarity(image_emb, ptx_text_emb).
    Already in [0, 1] and directionally correct — no transformation needed.
    """
    return df['confidence_explicit'].values.astype(float)


# ── Alignment ─────────────────────────────────────────────────────────────────

def align_on_file_id(
    df_gem: pd.DataFrame,
    df_sip: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Align both DataFrames to common file_ids in sorted order."""
    gem = df_gem.set_index('file_id')
    sip = df_sip.set_index('file_id')
    common = gem.index.intersection(sip.index).sort_values()
    return gem.loc[common].reset_index(), sip.loc[common].reset_index()


# ── Metrics ───────────────────────────────────────────────────────────────────

def metrics_from_score(
    y_true: np.ndarray,
    score: np.ndarray,
    threshold: float = 0.5,
) -> dict:
    """
    Binary classification metrics from a continuous score at a fixed threshold.
    Also computes AUROC from the score.
    """
    y_pred = (np.array(score) >= threshold).astype(int)
    yt = np.array(y_true)

    tp = int(((y_pred == 1) & (yt == 1)).sum())
    fp = int(((y_pred == 1) & (yt == 0)).sum())
    fn = int(((y_pred == 0) & (yt == 1)).sum())
    tn = int(((y_pred == 0) & (yt == 0)).sum())

    sens = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    f1   = 2 * prec * sens / (prec + sens) if (prec + sens) > 0 else 0.0
    acc  = (tp + tn) / len(yt)

    try:
        auroc = float(roc_auc_score(yt, score))
    except Exception:
        auroc = float('nan')

    return {
        'f1':          round(f1, 4),
        'sensitivity': round(sens, 4),
        'specificity': round(spec, 4),
        'precision':   round(prec, 4),
        'accuracy':    round(acc, 4),
        'auroc':       round(auroc, 4),
        'tp': tp, 'fp': fp, 'fn': fn, 'tn': tn,
    }


def metrics_from_pred(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Binary classification metrics from hard predictions (no score/AUROC)."""
    yt, yp = np.array(y_true), np.array(y_pred)

    tp = int(((yp == 1) & (yt == 1)).sum())
    fp = int(((yp == 1) & (yt == 0)).sum())
    fn = int(((yp == 0) & (yt == 1)).sum())
    tn = int(((yp == 0) & (yt == 0)).sum())

    sens = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    f1   = 2 * prec * sens / (prec + sens) if (prec + sens) > 0 else 0.0
    acc  = (tp + tn) / len(yt)

    return {
        'f1':          round(f1, 4),
        'sensitivity': round(sens, 4),
        'specificity': round(spec, 4),
        'precision':   round(prec, 4),
        'accuracy':    round(acc, 4),
        'auroc':       None,
        'tp': tp, 'fp': fp, 'fn': fn, 'tn': tn,
    }


# ── Voting ensemble ───────────────────────────────────────────────────────────

def vote_and(pred_gem: np.ndarray, pred_sip: np.ndarray) -> np.ndarray:
    """AND: predict PTX only if BOTH models predict PTX → maximizes precision."""
    return ((np.array(pred_gem) == 1) & (np.array(pred_sip) == 1)).astype(int)


def vote_or(pred_gem: np.ndarray, pred_sip: np.ndarray) -> np.ndarray:
    """OR: predict PTX if EITHER model predicts PTX → maximizes sensitivity."""
    return ((np.array(pred_gem) == 1) | (np.array(pred_sip) == 1)).astype(int)
