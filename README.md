# NIDS: Anomaly Detection on Network Flows

Anomaly-based network intrusion detection on NetFlow data. Detectors are trained on **benign traffic only** and evaluated against labelled attacks at fixed false-positive budgets.

Models: Isolation Forest, One-Class SVM, dense autoencoder, TCN masked autoencoder.
Thresholds: percentile, mean + k·std, MAD, and extreme-value (EVT) thresholds.
Data: NF-ToN-IoT-v2 (main) and NF-UNSW-NB15-v2 (cross-dataset test). The data is **not** in this repository.

The full write-up (methods, results, limitations) is in the project report.

## Main findings

- **Near in time, within one network:** the flow-level autoencoder ranks attacks well (tune AUROC 0.953 ± 0.016 over five seeds). A percentile of benign validation scores gives a usable threshold, but a strict 1% budget catches almost nothing.
- **Far in time:** every model is close to or below chance (test AUROC 0.39 to 0.52 for the flow-level models), and thresholds cannot fix it. The benign traffic itself changes: several duration and retransmission columns are almost always zero early in the file and common later.
- **TCN masked autoencoder:** it misses scanning because repeated flows are easy to rebuild from their neighbours (Spearman 0.927 between window score and within-window dispersion). At window level it does not beat the flow-level autoencoder (tune AUROC 0.571 against 0.958), and fusing the two does not help.
- **EVT:** it matches a plain percentile threshold in these runs.
- **Cross-dataset (ToN-IoT to UNSW-NB15):** no transfer (AUROC 0.24 to 0.39), while an in-domain Isolation Forest reaches AUROC 0.943 on the same data.

## Repository layout

```
configs/     run configuration (seeds are set in the notebooks: 11, 22, 33, 44, 55)
data/        NOT committed: raw CSVs, processed parquet splits, window tensors
notebooks/   experiments, numbered in run order (see below)
report/      report figures (report/figures/)
results/     small result files (CSV, JSON); scores, scalers and weights are git-ignored
src/         shared code
tests/       unit tests for the evaluation harness
```

Shared code in `src/`:

| File | Purpose |
|---|---|
| `evaluation.py` | Metrics: AUROC, TPR at fixed FPR budgets, real FPR, macro-TPR, oracle thresholds |
| `features.py` | Feature columns, `log1p` transform, clipped scaler |
| `pipeline.py` | Streaming reads of the parquet splits and score-file writing |
| `windows.py` | Per-host windows of 16 flows with padding masks and flow-to-window maps |
| `tcn.py` | TCN masked autoencoder, masking and window scoring |
| `thresholds.py` | Percentile, mean + k·std and MAD thresholds |
| `evt.py` | Peaks-over-threshold fit of a generalized Pareto distribution |
| `alerts.py` | `FlowDetector`: scores flows and emits JSON alerts |

## Setup

Python 3.12 is what the project was run with.

```bash
python -m venv nids-env
source nids-env/bin/activate          # Windows: nids-env\Scripts\activate
pip install -r requirements.txt
python -m pytest tests -q             # evaluation harness tests
```

### Data

1. Download **NF-ToN-IoT-v2** and **NF-UNSW-NB15-v2** (CSV) from the University of Queensland dataset page: https://www.cyber.uq.edu.au/project/machine-learning-based-nids-datasets
2. Unzip them under `data/`. This folder is git-ignored.
3. Edit the CSV paths at the top of `notebooks/01_explore.ipynb`, `02_make_splits.ipynb` and `16_prepare_unsw.ipynb` to match where the files are on your machine.

Academic use of the datasets requires citing the two papers listed on that page (references 1 and 2 in the report).

## Running the notebooks

Run them in this order. Each notebook is independent of kernel state apart from the files earlier notebooks write.

| Notebook | What it does | Writes |
|---|---|---|
| `01_explore` | Class counts, columns, hosts, benign concentration | |
| `02_make_splits` | Time-ordered train / val / tune / test splits with gaps | `data/processed/*.parquet` |
| `03_m1_isolation_forest` | M1 baseline and benign-drift diagnostics | `results/scaler.joblib`, M1 scores |
| `04_m3_autoencoder` | M3 autoencoder, v1 features | M3 v1 scores |
| `05_m3_autoencoder_v2` | M3 with feature set v2 (drop near-constant columns, clip scaled values) | `results/scaler_v2.joblib`, M3 v2 scores |
| `07_m2_ocsvm` | M2 One-Class SVM on a 20,000-flow subsample | M2 scores |
| `06_compare_baselines` | Flow-level comparison table | `results/baseline_comparison.csv` |
| `08_thresholds` | Percentile, mean + 3 std and MAD on the baseline scores | `results/threshold_comparison.csv` |
| `09_evt` | EVT against percentile on flow-level scores | `results/evt_comparison.csv` |
| `10_make_windows` | Per-host windows of 16 flows | `data/windows/*.npz` |
| `11_m4_tcn` | M4 TCN masked autoencoder | `results/m4_tcn_seed11.pt`, M4 scores |
| `12_window_level_comparison` | Window-level comparison, fusion, dispersion-only baseline | `results/window_level_comparison.csv`, `results/fusion_comparison.csv` |
| `13_scanning_check` | Why M4 misses scanning | |
| `14_evt_windows` | EVT on window-level scores | `results/evt_windows.csv` |
| `15_multiseed` | Five seeds for M3 v2 and M4 | `results/ms_*_per_seed.csv` |
| `16_prepare_unsw` | Convert NF-UNSW-NB15-v2 to parquet | `data/processed/unsw.parquet` |
| `17_cross_dataset` | ToN-IoT models on UNSW (trains and saves M3 v2 weights if missing) | `results/cross_dataset_unsw.csv`, `results/m3v2_ae_seed11.pt` |
| `18_unsw_control` | In-domain Isolation Forest on UNSW | |
| `19_alert_demo` | Alert prototype and operating-point sweep | `results/alert_config.json`, `results/alerts_demo.jsonl` |
| `20_figures` | Report figures | `report/figures/` |

Notes:

- Model weights (`*.pt`), scalers (`*.joblib`), parquet files and score files are git-ignored. Rerunning the notebooks above regenerates them.
- Several steps read the large parquet files in chunks. Close other notebooks before running `10_make_windows`, `15_multiseed` and `17_cross_dataset`.
- If you edit a file in `src/`, restart the notebook kernel (or use `importlib.reload`), because Python caches imports.

## Alert prototype

`src/alerts.py` loads the saved autoencoder and scaler, scores flows, and writes one JSON object per flow above a threshold taken from benign validation scores. `notebooks/19_alert_demo.ipynb` replays labelled flows in batches and prints the alert counts.

```json
{
  "alert_id": "tune-7766125",
  "emitted_at": "2026-11-15T21:05:37+00:00",
  "flow_seq": 7766125,
  "src_ip": "192.168.1.32",
  "dst_ip": "192.168.1.1",
  "dst_port": 53,
  "protocol": 17,
  "score": 0.00852,
  "threshold": 0.00574,
  "severity": "low",
  "top_features": ["DNS_QUERY_TYPE", "DNS_QUERY_ID", "L7_PROTO"],
  "model": "autoencoder-v2"
}
```

`flow_seq` is the flow's position in the capture, because the data has no timestamps. Severity is a heuristic based on how far the score is above the threshold. The prototype works near in time to the training data and does not hold far in time or on another network; at a 5% budget it also alerts on about 8% of benign flows, which is too many for unattended use.

## Limitations

- The datasets have no timestamps. Row order is assumed to be chronological, and every "time-ordered" claim depends on that.
- Benign traffic comes from very few hosts, so the splits share hosts.
- The v2 features, and the choice of the tune split as the main benchmark, were adopted after inspecting test statistics. They are disclosed as post-hoc, and the test split was never used to tune a model or a threshold.
- Only M3 v2 and M4 were run with five seeds, and the TCN ablations were not run.
- Only one cross-dataset direction (ToN-IoT to UNSW-NB15) was tested.

## Scope changes

M5 contrastive encoder, the Transformer variant, TLS/JA4 features, evasion experiments and SPOT/DSPOT streaming were cut to keep the core finishable.
