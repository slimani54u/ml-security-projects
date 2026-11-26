# Network intrusion detection with machine learning

Detect malicious network connections with Python and scikit-learn, and measure honestly what happens with attacks the model has never seen.

## Data
- **KDD Cup 99 (10% subset)**, downloaded automatically by `sklearn.datasets.fetch_kddcup99`: 494,021 network connections, 41 features (protocol, service, TCP flags, bytes, error rates, connection counts…), labelled normal or one of 22 attacks.
- **348,435 duplicates (70.5%) removed before any split**, leaving **145,586 connections**. Keeping them would put identical connections in both the training and test sets and inflate every score.
- Attacks grouped into 4 families: DoS (54,572), Probe (2,131), R2L (999), U2R (52); normal traffic: 87,832.

## Method
1. Categorical features (protocol, service, flag) one-hot encoded, numeric features standardized, inside a pipeline (no data leakage).
2. Stratified 80/20 train/test split, 5-fold cross-validation (F1).
3. **Supervised detection:** Logistic Regression vs Random Forest.
4. **Unseen attacks:** one attack family is removed from training, then the model is tested on it.
5. **Unsupervised anomaly detection:** Isolation Forest trained on normal traffic only, with a threshold set for 1% false positives on held-out normal traffic.

## Results

| Model | F1, 5-fold CV |
|---|---|
| Logistic Regression | 0.9922 |
| **Random Forest** | **0.9991** |

Random Forest on the test set (29,118 connections):

| Metric | Value |
|---|---|
| Detection rate (attack recall) | **99.86%** |
| False positive rate | **0.02%** (4 / 17,567 normal connections) |
| Detection by family | DoS 100% · Probe 98.1% · R2L 98.5% · U2R 50% (5 / 10) |

**Attack family never seen during training:**

| Family left out | Detected |
|---|---|
| DoS | 93.9% |
| Probe | 49.5% |
| R2L | **2.0%** |
| U2R | 10.0% |

**Isolation Forest (trained on normal traffic only):** 94.6% detection at 1.1% false positives (DoS 96.6% · Probe 83.8% · R2L 9.0% · U2R 0%).

## Reading the results
- On known attacks, the supervised model is almost perfect. That says more about KDD Cup 99 than about the model: DoS and Probe attacks have very distinctive traffic statistics.
- **The real limit is new attacks.** R2L and U2R attacks (password guessing, privilege escalation) look like normal connections at the traffic level: without having seen them, neither the Random Forest nor the anomaly detector finds them. They need payload or host-level inspection (signatures, logs), which is why a machine learning NIDS complements tools such as Snort or Suricata and a SIEM rather than replacing them.
- KDD Cup 99 is a dated (1999), simulated dataset. Next step: repeat the method on a modern dataset such as CIC-IDS2017 or UNSW-NB15.

## Run
```bash
pip install -r ../requirements.txt
python nids.py
```
