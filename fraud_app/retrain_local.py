"""Retrain the pre-transaction fraud model on your own computer.

It replaces fraud_model.json and model_meta.json (the two files that were out of sync)
so the app uses a model that does NOT need the 'balance after the transaction' columns.

How to run (in PowerShell, inside your fraud_app folder):
    python retrain_local.py "C:\\path\\to\\PS_20174392719_1491204439457_log.csv"

Takes about 3 to 10 minutes. Needs the PaySim CSV you downloaded from Kaggle.
"""
import json
import os
import sys

import numpy as np
import pandas as pd
import sklearn
import xgboost
from sklearn.metrics import (average_precision_score, precision_recall_curve,
                             precision_recall_fscore_support, roc_curve)
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

if len(sys.argv) < 2:
    sys.exit('Usage: python retrain_local.py "path\\to\\PS_20174392719_1491204439457_log.csv"')
csv_path = sys.argv[1]
if not os.path.exists(csv_path):
    sys.exit(f"CSV not found: {csv_path}")

RS = 42
RISKY = ["TRANSFER", "CASH_OUT"]
COST_FN, COST_FP = 10000, 100

print("Loading data (this can take a minute)...")
df = pd.read_csv(csv_path, usecols=["step", "type", "amount", "oldbalanceOrg", "oldbalanceDest", "isFraud"])
df = df[df["type"].isin(RISKY)].reset_index(drop=True)
print(f"Rows used: {len(df):,} | frauds: {int(df['isFraud'].sum()):,}")

# Same features as the notebook's 'pre' mode: only information known BEFORE approval
df["hour"] = df["step"] % 24
df["is_transfer"] = (df["type"] == "TRANSFER").astype(int)
df["amt_to_orig_bal"] = df["amount"] / (df["oldbalanceOrg"] + 1)
df["amt_to_dest_bal"] = df["amount"] / (df["oldbalanceDest"] + 1)
df["drains_account"] = ((df["oldbalanceOrg"] > 0) & (df["amount"] >= df["oldbalanceOrg"] - 0.01)).astype(int)
df["orig_empty"] = (df["oldbalanceOrg"] == 0).astype(int)
df["dest_empty"] = (df["oldbalanceDest"] == 0).astype(int)

FEATURES = ["amount", "hour", "is_transfer", "oldbalanceOrg", "oldbalanceDest", "amt_to_orig_bal",
            "amt_to_dest_bal", "drains_account", "orig_empty", "dest_empty"]
BINARY = ["is_transfer", "drains_account", "orig_empty", "dest_empty"]

# Same 70 / 15 / 15 stratified split as the notebook
y = df["isFraud"]
idx = df.index.values
idx_train, idx_tmp = train_test_split(idx, test_size=0.30, stratify=y, random_state=RS)
idx_val, idx_test = train_test_split(idx_tmp, test_size=0.50, stratify=y.loc[idx_tmp], random_state=RS)
X_train, y_train = df.loc[idx_train, FEATURES], y.loc[idx_train]
X_val, y_val = df.loc[idx_val, FEATURES], y.loc[idx_val]
X_test, y_test = df.loc[idx_test, FEATURES], y.loc[idx_test]

neg, pos = int((y_train == 0).sum()), int((y_train == 1).sum())
spw = float(np.sqrt(neg / pos))
print("Training the model...")
model = XGBClassifier(n_estimators=300, max_depth=6, learning_rate=0.1, subsample=0.8,
                      tree_method="hist", eval_metric="aucpr", n_jobs=-1, random_state=RS,
                      scale_pos_weight=spw)
model.fit(X_train, y_train)

# Thresholds chosen on the validation set
pv = model.predict_proba(X_val)[:, 1]
p, r, t = precision_recall_curve(y_val, pv)
f1 = 2 * p[:-1] * r[:-1] / np.maximum(p[:-1] + r[:-1], 1e-12)
thr_f1 = float(t[np.argmax(f1)])

fpr, tpr, thr = roc_curve(y_val, pv)
P, N = int(y_val.sum()), int((y_val == 0).sum())
cost = (1 - tpr) * P * COST_FN + fpr * N * COST_FP
finite = np.isfinite(thr)
thr_cost = float(thr[finite][np.argmin(cost[finite])])

# Quick report on the test set
pt = model.predict_proba(X_test)[:, 1]
pred = (pt >= thr_f1).astype(int)
pr, rc, f, _ = precision_recall_fscore_support(y_test, pred, average="binary", zero_division=0)
print(f"Test set: precision={pr:.4f} recall={rc:.4f} F1={f:.4f} PR-AUC={average_precision_score(y_test, pt):.4f}")

# Save the two files the app needs
model.save_model("fraud_model.json")
meta = {
    "model_file": "fraud_model.json",
    "model_format": "xgb_json",
    "model_name": "XGBoost (weighted)",
    "features": FEATURES,
    "binary_features": BINARY,
    "feature_mode": "pre",
    "risky_types": RISKY,
    "threshold_cost": thr_cost,
    "threshold_f1": thr_f1,
    "cost_fn": COST_FN,
    "cost_fp": COST_FP,
    "sklearn_version": sklearn.__version__,
    "xgboost_version": xgboost.__version__,
}
with open("model_meta.json", "w") as fh:
    json.dump(meta, fh, indent=2)

print("\nDone. Saved fraud_model.json and model_meta.json in:", os.getcwd())
print("Features in the model:", len(model.get_booster().feature_names), "(should be 10)")
print("Now run:  python -m streamlit run app.py")
