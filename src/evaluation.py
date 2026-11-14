import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score

REQUIRED = ["id", "score", "label", "attack_type", "split"]   # label: 0 benign, 1 attack
BUDGETS = (0.001, 0.01, 0.05)


def load_scores(path):
    df = pd.read_parquet(path)
    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        raise ValueError(f"score file is missing columns: {missing}")
    return df


def threshold_at_fpr(benign_scores, fpr_budget):
    """Score cutoff that lets at most `fpr_budget` of benign flows through."""
    return float(np.quantile(benign_scores, 1 - fpr_budget))


def tpr_at_threshold(df, thr):
    atk = df[df["label"] == 1]["score"]
    ben = df[df["label"] == 0]["score"]
    return float((atk > thr).mean()), float((ben > thr).mean())   # (TPR, FPR)


def evaluate(test_df, val_df=None, budgets=BUDGETS):
    """val_df: benign validation scores used to set thresholds (the honest way).
    Without it, thresholds come from the test benign scores (oracle comparison)."""
    out = {
        "auroc": float(roc_auc_score(test_df["label"], test_df["score"])),
        "aupr": float(average_precision_score(test_df["label"], test_df["score"])),
    }
    ref = val_df[val_df["label"] == 0]["score"] if val_df is not None else \
          test_df[test_df["label"] == 0]["score"]
    attacks = test_df[test_df["label"] == 1]

    for b in budgets:
        thr = threshold_at_fpr(ref, b)
        tpr, fpr = tpr_at_threshold(test_df, thr)
        out[f"tpr@fpr{b}"] = tpr
        out[f"real_fpr@{b}"] = fpr
        # macro: average of per-attack-type TPRs, so a dominant class cannot hide the rest
        out[f"macro_tpr@fpr{b}"] = float(np.mean(
            [(g["score"] > thr).mean() for _, g in attacks.groupby("attack_type")]))

    thr = threshold_at_fpr(ref, 0.01)
    out["per_attack_tpr@fpr0.01"] = {
        t: float((g["score"] > thr).mean()) for t, g in attacks.groupby("attack_type")}
    return out