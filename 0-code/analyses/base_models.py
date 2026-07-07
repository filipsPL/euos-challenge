"""
base_models.py

Baseline models trained only on structural descriptors (RDKit fingerprints +
RDKit 2D physicochemical descriptors), with no cross-task prediction features.
Trains XGBoost and CatBoost on train.csv.gz, evaluates on test.csv.gz for all
four endpoints.

Input:  data/4-datasets/{endpoint}/{DESCRIPTOR_SET}/train.csv.gz
                                                    /test.csv.gz
Output: results/base_models/
  - <endpoint>/metrics.csv
  - summary.csv
"""

import os
import pandas as pd
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier
from catboost import CatBoostClassifier

DESCRIPTOR_SET = "2D+RdkitFP+dyesSelected+QM9pred+easyTargetPreds+xtb3_20260107"
DATA_BASE = "data/4-datasets"
OUTPUT_BASE = "results/base_models"
LABEL_COL = "activity"
RANDOM_STATE = 42

ENDPOINTS = [
    "euos25_challenge_train_fluorescence340_450",
    "euos25_challenge_train_fluorescence480plus",
    "euos25_challenge_train_transmittance340",
    "euos25_challenge_train_transmittance450plus",
]

ENDPOINT_LABELS = {
    "euos25_challenge_train_fluorescence340_450":  "Fluorescence 340/450 nm",
    "euos25_challenge_train_fluorescence480plus":  "Fluorescence 480/540+ nm",
    "euos25_challenge_train_transmittance340":     "Transmittance 340 nm",
    "euos25_challenge_train_transmittance450plus": "Transmittance >450 nm",
}

BASELINE_GROUPS = ["RDkitFP-RDKit", "RDKit2D"]


def load_endpoint(endpoint):
    base = os.path.join(DATA_BASE, endpoint, DESCRIPTOR_SET)
    train = pd.read_csv(os.path.join(base, "train.csv.gz"))
    test  = pd.read_csv(os.path.join(base, "test.csv.gz"))
    feat_cols = [c for c in train.columns
                 if c.split("::")[0] in BASELINE_GROUPS]
    X_train = train[feat_cols]
    y_train = train[LABEL_COL]
    X_test  = test[feat_cols]
    y_test  = test[LABEL_COL]
    return X_train, y_train, X_test, y_test, feat_cols


def scale_pos_weight(y):
    n_neg = (y == 0).sum()
    n_pos = (y == 1).sum()
    return n_neg / n_pos if n_pos > 0 else 1.0


def train_xgboost(X_train, y_train, X_test, y_test):
    model = XGBClassifier(
        scale_pos_weight=scale_pos_weight(y_train),
        n_jobs=-1,
        random_state=RANDOM_STATE,
        tree_method="hist",
        eval_metric="auc",
        verbosity=0,
    )
    model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)
    return roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])


def train_catboost(X_train, y_train, X_test, y_test):
    ratio = scale_pos_weight(y_train)
    model = CatBoostClassifier(
        class_weights=[1.0, ratio],
        random_seed=RANDOM_STATE,
        verbose=0,
    )
    model.fit(X_train, y_train, eval_set=(X_test, y_test))
    return roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])


def main():
    os.makedirs(OUTPUT_BASE, exist_ok=True)
    all_metrics = []

    for ep in ENDPOINTS:
        label = ENDPOINT_LABELS[ep]
        ep_key = ep.replace("euos25_challenge_train_", "")
        print(f"\n{'='*60}")
        print(f"Endpoint: {label}")

        X_train, y_train, X_test, y_test, feat_cols = load_endpoint(ep)
        print(f"  Train: {len(X_train):,}  |  Test: {len(X_test):,}  |  Features: {len(feat_cols)}")
        print(f"  Train pos: {y_train.sum():,}  |  Test pos: {y_test.sum():,}")

        auroc_xgb = train_xgboost(X_train, y_train, X_test, y_test)
        auroc_cb  = train_catboost(X_train, y_train, X_test, y_test)

        print(f"  XGBoost:  AUROC = {auroc_xgb:.4f}")
        print(f"  CatBoost: AUROC = {auroc_cb:.4f}")

        out_dir = os.path.join(OUTPUT_BASE, ep_key)
        os.makedirs(out_dir, exist_ok=True)
        metrics_df = pd.DataFrame([{
            "endpoint": ep_key,
            "n_features": len(feat_cols),
            "n_train": len(X_train),
            "n_test": len(X_test),
            "xgboost": auroc_xgb,
            "catboost": auroc_cb,
        }])
        metrics_df.to_csv(os.path.join(out_dir, "metrics.csv"), index=False)

        all_metrics.append({
            "endpoint": ep_key,
            "label": label,
            "n_features": len(feat_cols),
            "xgboost": auroc_xgb,
            "catboost": auroc_cb,
        })

    summary = pd.DataFrame(all_metrics)
    summary_path = os.path.join(OUTPUT_BASE, "summary.csv")
    summary.to_csv(summary_path, index=False)

    print(f"\n{'='*60}")
    print("Summary")
    print('='*60)
    print(summary[["label", "n_features", "xgboost", "catboost"]].to_string(index=False))
    print(f"\nSaved: {summary_path}")


if __name__ == "__main__":
    main()
