# NIDS: Anomaly Detection on Network Flows

Train on benign traffic only, flag deviations as attacks.

Models: Isolation Forest, One-Class SVM, autoencoder, TCN masked autoencoder.
Thresholds: percentile, mean+k, MAD, EVT.
Data: NetFlow datasets from UQ (not included in the repo).