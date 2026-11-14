from pathlib import Path
import numpy as np, pandas as pd, pyarrow.parquet as pq, joblib
from src.features import to_matrix

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "processed"
RESULTS = ROOT / "results"


def load_scaler():
    d = joblib.load(RESULTS / "scaler.joblib")
    return d["cols"], d["scaler"]


def iter_split(name, cols, cap=None, seed=11):
    """Yield DataFrames from a split. cap: keep ~cap rows per attack type (all benign kept)."""
    path = DATA / f"{name}.parquet"
    keep_p = {}
    if cap:
        vc = pd.read_parquet(path, columns=["Attack"])["Attack"].value_counts()
        keep_p = {k: min(1.0, cap / v) for k, v in vc.items() if k != "Benign"}
    rng = np.random.default_rng(seed)
    for batch in pq.ParquetFile(path).iter_batches(batch_size=500_000,
                                                   columns=cols + ["row_id", "Attack"]):
        df = batch.to_pandas()
        if keep_p:
            p = df["Attack"].map(keep_p).fillna(1.0).to_numpy(dtype=float)
            df = df[rng.random(len(df)) < p]
        yield df


def score_split(name, cols, scaler, score_fn, cap=None, seed=11):
    parts = []
    for df in iter_split(name, cols, cap, seed):
        x = scaler.transform(to_matrix(df, cols)).astype(np.float32)
        parts.append(pd.DataFrame({
            "id": df["row_id"].to_numpy(), "score": score_fn(x),
            "label": (df["Attack"] != "Benign").astype(int).to_numpy(),
            "attack_type": df["Attack"].astype(str).to_numpy(), "split": name}))
    return pd.concat(parts, ignore_index=True)