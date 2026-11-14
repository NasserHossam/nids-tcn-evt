import numpy as np
import pandas as pd
from src.evaluation import evaluate, threshold_at_fpr

def make(ben, atk):
    df = pd.DataFrame({"score": np.concatenate([ben, atk]),
                       "label": [0]*len(ben) + [1]*len(atk)})
    df["attack_type"] = np.where(df["label"] == 0, "Benign", "x")
    return df

def test_perfect_separation():
    r = evaluate(make(np.zeros(1000), np.ones(1000)))
    assert r["auroc"] == 1.0 and r["tpr@fpr0.01"] == 1.0

def test_random_scores_are_chance():
    rng = np.random.default_rng(0)
    r = evaluate(make(rng.random(20000), rng.random(20000)))
    assert abs(r["auroc"] - 0.5) < 0.02
    assert abs(r["tpr@fpr0.01"] - 0.01) < 0.01

def test_fpr_budget_is_respected():
    rng = np.random.default_rng(1)
    ben = rng.random(10000)
    thr = threshold_at_fpr(ben, 0.05)
    assert (ben > thr).mean() <= 0.05