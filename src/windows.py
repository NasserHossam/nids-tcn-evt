import numpy as np, pandas as pd, pyarrow.parquet as pq
from src.features import to_matrix
from src.pipeline import DATA


def assign_windows(meta, L=16):
    """meta: DataFrame with row_id, IPV4_SRC_ADDR, Attack.
    Returns a compact frame (row_id, host, atk, win_id, pos) plus the name arrays."""
    host, host_names = pd.factorize(meta["IPV4_SRC_ADDR"])
    atk, atk_names = pd.factorize(meta["Attack"])
    row_id = meta["row_id"].to_numpy()
    order = np.lexsort((row_id, host))                 # by host, then by time proxy
    host, atk, row_id = host[order], atk[order], row_id[order]
    n = len(host)
    starts = np.flatnonzero(np.r_[True, host[1:] != host[:-1]])
    idx_in_host = np.arange(n) - np.repeat(starts, np.diff(np.r_[starts, n]))
    pos = (idx_in_host % L).astype(np.int8)
    win_id = (np.cumsum(pos == 0) - 1).astype(np.int32)
    m = pd.DataFrame({"row_id": row_id.astype(np.int32), "host": host.astype(np.int32),
                      "atk": atk.astype(np.int16), "win_id": win_id, "pos": pos})
    return m, np.asarray(host_names), np.asarray(atk_names)


def window_table(m, host_names, atk_names):
    benign = int(np.flatnonzero(atk_names == "Benign")[0])
    atk = m["atk"].to_numpy()
    win = m["win_id"].to_numpy()
    is_atk = atk != benign
    n_win = int(win.max()) + 1

    first_idx = np.flatnonzero(m["pos"].to_numpy() == 0)          # first flow of each window
    code = np.full(n_win, benign, dtype=np.int16)
    ai = np.flatnonzero(is_atk)
    code[win[ai][::-1]] = atk[ai][::-1]                           # earliest attack flow wins

    return pd.DataFrame({
        "n_flows": np.bincount(win, minlength=n_win),
        "label": (np.bincount(win, weights=is_atk, minlength=n_win) > 0).astype(np.int8),
        "host": host_names[m["host"].to_numpy()[first_idx]],
        "first_row_id": m["row_id"].to_numpy()[first_idx],
        "attack_type": atk_names[code],
    })


def select_windows(w, max_benign=None, cap_per_type=None, seed=11):
    rng = np.random.default_rng(seed)
    ids = []
    for t, g in w.groupby("attack_type"):
        idx = g.index.to_numpy()
        cap = max_benign if t == "Benign" else cap_per_type
        if cap and len(idx) > cap:
            idx = rng.choice(idx, cap, replace=False)
        ids.append(idx)
    return np.sort(np.concatenate(ids))


def build_tensors(split, m, win_ids, cols, scaler, L=16):
    win_ids = np.asarray(win_ids)
    sub = m[m["win_id"].isin(win_ids)]
    widx = np.searchsorted(win_ids, sub["win_id"].to_numpy())
    slot_val = widx * L + sub["pos"].to_numpy()
    slot = np.full(int(m["row_id"].max()) + 1, -1, dtype=np.int32)
    slot[sub["row_id"].to_numpy()] = slot_val
    X = np.zeros((len(win_ids) * L, len(cols)), dtype=np.float32)
    mask = np.zeros(len(win_ids) * L, dtype=bool)
    mask[slot_val] = True
    for batch in pq.ParquetFile(DATA / f"{split}.parquet").iter_batches(
            batch_size=100_000, columns=cols + ["row_id"]):
        df = batch.to_pandas()
        s = slot[df["row_id"].to_numpy()]
        keep = s >= 0
        if keep.any():
            X[s[keep]] = scaler.transform(to_matrix(df[keep], cols)).astype(np.float32)
    return X.reshape(len(win_ids), L, len(cols)), mask.reshape(len(win_ids), L)