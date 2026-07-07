"""
Ablation study: compare AUROC when each feature group (and combinations) is
included vs. excluded from XGBoost and CatBoost models.

Feature groups are defined by the '::' prefix in column names.

For each endpoint:
  - Train on train.csv.gz, evaluate on test.csv.gz
  - Run 'full' model (all features)
  - For each group G: run 'drop_G' model (all features EXCEPT group G)
  - For each group G: run 'only_G' model (ONLY features in group G)

Results saved to:
  results/ablation/{endpoint}/ablation_results.csv
  results/ablation/ablation_summary.csv   (all endpoints combined)
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier
from catboost import CatBoostClassifier

# ── Config ────────────────────────────────────────────────────────────────────

DATA_ROOT = Path("data/4-datasets")
FEATURE_SET = "2D+RdkitFP+dyesSelected+QM9pred+easyTargetPreds+xtb3_20260107"
OUTPUT_DIR = Path("results/ablation")
RANDOM_STATE = 42

ENDPOINTS = [
    "euos25_challenge_train_fluorescence340_450",
    "euos25_challenge_train_fluorescence480plus",
    "euos25_challenge_train_transmittance340",
    "euos25_challenge_train_transmittance450plus",
]

# All known feature groups (by :: prefix)
GROUPS = [
    "pred_fluo340",
    "pred_trans340",
    "RDkitFP-RDKit",
    "RDKit2D",
    "xtb3",
    "QM9",
    "dyesChromophores",
    "dyesSimMurcko",
]

# Combined groups for extra ablation conditions
COMBINED_GROUPS = {
    "pred_all": ["pred_fluo340", "pred_trans340"],
}

XGB_PARAMS = dict(
    n_estimators=500,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    eval_metric="logloss",
    tree_method="hist",
    random_state=RANDOM_STATE,
    n_jobs=-1,
)

CATBOOST_PARAMS = dict(
    iterations=500,
    depth=6,
    learning_rate=0.05,
    random_seed=RANDOM_STATE,
    verbose=0,
    thread_count=-1,
)


# ── Helpers ───────────────────────────────────────────────────────────────────

def load_endpoint(endpoint: str) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    base = DATA_ROOT / endpoint / FEATURE_SET
    train = pd.read_csv(base / "train.csv.gz")
    test = pd.read_csv(base / "test.csv.gz")

    y_train = train["activity"]
    X_train = train.drop(columns=["activity"])
    y_test = test["activity"]
    X_test = test.drop(columns=["activity"])

    return X_train, y_train, X_test, y_test


def preprocess(X_train: pd.DataFrame, X_test: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Drop all-NaN columns (based on train), fill remaining NaNs with train median."""
    nan_cols = X_train.columns[X_train.isna().all()].tolist()
    if nan_cols:
        X_train = X_train.drop(columns=nan_cols)
        X_test = X_test.drop(columns=nan_cols)

    medians = X_train.median(numeric_only=True)
    X_train = X_train.fillna(medians)
    X_test = X_test.fillna(medians)
    return X_train, X_test


def select_features(X_train: pd.DataFrame, X_test: pd.DataFrame,
                    drop_groups: list[str] | None = None,
                    only_groups: list[str] | None = None
                    ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    drop_groups: exclude columns whose :: prefix is in this list.
    only_groups: keep only columns whose :: prefix is in this list.
    If neither is set, returns all columns.
    """
    def group_of(col):
        return col.split("::")[0] if "::" in col else None

    if only_groups is not None:
        cols = [c for c in X_train.columns if group_of(c) in only_groups]
    elif drop_groups is not None:
        cols = [c for c in X_train.columns if group_of(c) not in drop_groups]
    else:
        cols = list(X_train.columns)

    return X_train[cols], X_test[cols]


def train_xgb(X_train, y_train, X_test, y_test) -> float:
    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
    model = XGBClassifier(**XGB_PARAMS, scale_pos_weight=scale_pos_weight)
    model.fit(X_train, y_train)
    proba = model.predict_proba(X_test)[:, 1]
    return roc_auc_score(y_test, proba)


def train_catboost(X_train, y_train, X_test, y_test) -> float:
    scale = (y_train == 0).sum() / (y_train == 1).sum()
    model = CatBoostClassifier(**CATBOOST_PARAMS, class_weights=[1.0, float(scale)])
    model.fit(X_train, y_train)
    proba = model.predict_proba(X_test)[:, 1]
    return roc_auc_score(y_test, proba)


def run_condition(X_train, y_train, X_test, y_test,
                  drop_groups=None, only_groups=None,
                  label: str = "") -> dict:
    Xtr, Xte = select_features(X_train, X_test,
                                drop_groups=drop_groups,
                                only_groups=only_groups)
    n_features = Xtr.shape[1]
    print(f"    [{label}] features={n_features}", flush=True)

    if n_features == 0:
        print(f"    [{label}] SKIP — no features remaining")
        return {"condition": label, "n_features": 0, "xgboost": float("nan"), "catboost": float("nan")}

    xgb_auc = train_xgb(Xtr, y_train, Xte, y_test)
    cb_auc = train_catboost(Xtr, y_train, Xte, y_test)
    print(f"    [{label}] XGB={xgb_auc:.4f}  CB={cb_auc:.4f}", flush=True)
    return {"condition": label, "n_features": n_features, "xgboost": xgb_auc, "catboost": cb_auc}


# ── Per-endpoint runner ───────────────────────────────────────────────────────

def run_endpoint(endpoint: str) -> pd.DataFrame:
    short = endpoint.replace("euos25_challenge_train_", "")
    print(f"\n{'='*60}")
    print(f"Endpoint: {short}")
    print(f"{'='*60}", flush=True)

    X_train, y_train, X_test, y_test = load_endpoint(endpoint)
    print(f"  Train: {len(y_train):,}  |  Test: {len(y_test):,}  |  Features: {X_train.shape[1]:,}")
    print(f"  Train pos: {(y_train==1).sum():,}  |  Test pos: {(y_test==1).sum():,}", flush=True)

    X_train, X_test = preprocess(X_train, X_test)

    records = []

    # Full model
    print("\n  [Conditions]")
    records.append(run_condition(X_train, y_train, X_test, y_test, label="full"))

    # Drop each individual group
    for g in GROUPS:
        # check if group exists in data
        group_cols = [c for c in X_train.columns if c.split("::")[0] == g if "::" in c]
        if not group_cols:
            continue
        records.append(run_condition(X_train, y_train, X_test, y_test,
                                     drop_groups=[g], label=f"drop_{g}"))

    # Drop combined groups
    for combo_name, combo_groups in COMBINED_GROUPS.items():
        present = [g for g in combo_groups
                   if any(c.split("::")[0] == g for c in X_train.columns if "::" in c)]
        if present:
            records.append(run_condition(X_train, y_train, X_test, y_test,
                                         drop_groups=present, label=f"drop_{combo_name}"))

    # Only each individual group
    for g in GROUPS:
        group_cols = [c for c in X_train.columns if "::" in c and c.split("::")[0] == g]
        if not group_cols:
            continue
        records.append(run_condition(X_train, y_train, X_test, y_test,
                                     only_groups=[g], label=f"only_{g}"))

    # Only combined groups
    for combo_name, combo_groups in COMBINED_GROUPS.items():
        present = [g for g in combo_groups
                   if any(c.split("::")[0] == g for c in X_train.columns if "::" in c)]
        if present:
            records.append(run_condition(X_train, y_train, X_test, y_test,
                                         only_groups=present, label=f"only_{combo_name}"))

    df = pd.DataFrame(records)
    df.insert(0, "endpoint", short)

    out_dir = OUTPUT_DIR / short
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "ablation_results.csv"
    df.to_csv(out_path, index=False)
    print(f"\n  Saved: {out_path}")
    return df


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    all_frames = []
    for endpoint in ENDPOINTS:
        df = run_endpoint(endpoint)
        all_frames.append(df)

    summary = pd.concat(all_frames, ignore_index=True)

    # Compute delta vs full for each endpoint
    full_rows = summary[summary["condition"] == "full"].set_index("endpoint")
    for model in ["xgboost", "catboost"]:
        summary[f"delta_{model}"] = summary.apply(
            lambda row: row[model] - full_rows.loc[row["endpoint"], model]
            if row["endpoint"] in full_rows.index else float("nan"),
            axis=1,
        )

    out_path = OUTPUT_DIR / "ablation_summary.csv"
    summary.to_csv(out_path, index=False)
    print(f"\nAll done. Summary saved: {out_path}")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
