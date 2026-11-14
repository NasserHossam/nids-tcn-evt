import numpy as np

# Never model inputs: raw IPs, row_id (time proxy), ephemeral source port, labels
DROP = ["row_id", "IPV4_SRC_ADDR", "IPV4_DST_ADDR", "L4_SRC_PORT", "Label", "Attack"]


def feature_columns(all_columns):
    return [c for c in all_columns if c not in DROP]


def to_matrix(df, cols):
    """Raw columns to a log-scaled float32 matrix. Counts and bytes are heavy-tailed,
    so log1p keeps a few huge flows from dominating the scaler."""
    x = df[cols].to_numpy(dtype=np.float64)
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    return np.log1p(np.clip(x, 0, None)).astype(np.float32)