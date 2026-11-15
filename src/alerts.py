from datetime import datetime, timezone
from pathlib import Path
import numpy as np, torch, torch.nn as nn, joblib
from src.features import to_matrix, ClippedScaler   # ClippedScaler is needed to load the saved scaler


def build_ae(dim):
    return nn.Sequential(
        nn.Linear(dim, 64), nn.ReLU(), nn.Linear(64, 32), nn.ReLU(),
        nn.Linear(32, 12), nn.ReLU(),
        nn.Linear(12, 32), nn.ReLU(), nn.Linear(32, 64), nn.ReLU(), nn.Linear(64, dim))


class FlowDetector:
    def __init__(self, results_dir, threshold):
        results_dir = Path(results_dir)
        d = joblib.load(results_dir / "scaler_v2.joblib")
        self.cols, self.scaler = d["cols"], d["scaler"]
        self.model = build_ae(len(self.cols))
        self.model.load_state_dict(torch.load(results_dir / "m3v2_ae_seed11.pt"))
        self.model.eval()
        self.threshold = float(threshold)

    @torch.no_grad()
    def errors(self, df):
        x = torch.from_numpy(self.scaler.transform(to_matrix(df, self.cols)).astype(np.float32))
        return ((self.model(x) - x) ** 2).numpy()          # squared error per feature

    def score(self, df):
        return self.errors(df).mean(axis=1)

    def alerts(self, df, id_prefix="flow"):
        err = self.errors(df)
        score = err.mean(axis=1)
        out = []
        for i in np.flatnonzero(score > self.threshold):
            row = df.iloc[i]
            top = np.argsort(err[i])[::-1][:3]
            ratio = float(score[i] / self.threshold)
            out.append({
                "alert_id": f"{id_prefix}-{int(row['row_id'])}",
                "emitted_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "flow_seq": int(row["row_id"]),          # position in the capture; the data has no timestamps
                "src_ip": str(row["IPV4_SRC_ADDR"]),
                "dst_ip": str(row["IPV4_DST_ADDR"]),
                "dst_port": int(row["L4_DST_PORT"]) if "L4_DST_PORT" in df.columns else None,
                "protocol": int(row["PROTOCOL"]) if "PROTOCOL" in df.columns else None,
                "score": round(float(score[i]), 5),
                "threshold": round(self.threshold, 5),
                "severity": "high" if ratio >= 10 else "medium" if ratio >= 3 else "low",
                "top_features": [self.cols[j] for j in top],
                "model": "autoencoder-v2",
            })
        return out