"""
Feature Ablation Study for Data Leakage Verification

Trains CatBoost on original data with progressive removal of high-cardinality
identifier features to demonstrate that near-perfect performance is driven by
account/institution behavioral patterns, not data leakage.

Conditions:
  1. Full (all 9 features)
  2. -Account IDs (remove withdrawal/deposit account numbers)
  3. -Account IDs -FI IDs (also remove withdrawal/deposit bank codes)

Self-contained: loads data directly from numpy files without lib dependency.
Replicates the same split logic (read_changed_val with random_state=777)
and CatBoost config used in the main experiments.
"""

import os
import json
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score, matthews_corrcoef,
)
from catboost import CatBoostClassifier

# Feature layout after concat: [0]=거래금액(num), [1..8]=cat features
# X_cat columns: [0]=거래일자, [1]=거래시간대, [2]=출금금융회사, [3]=출금계좌,
#                [4]=입금금융회사, [5]=입금계좌, [6]=매체구분, [7]=자금구분

ABLATION_CONDITIONS = {
    "Full (all features)": {
        "drop_cat_cols": [],
        "description": "All 9 features",
    },
    "- Account IDs": {
        "drop_cat_cols": [3, 5],  # 출금계좌일련번호, 입금계좌일련번호
        "description": "Remove withdrawal/deposit account numbers",
    },
    "- Account IDs - FI IDs": {
        "drop_cat_cols": [2, 3, 4, 5],  # 출금금융회사, 출금계좌, 입금금융회사, 입금계좌
        "description": "Also remove withdrawal/deposit bank codes",
    },
}


def load_npy(path, name):
    return np.load(os.path.join(path, name), allow_pickle=True)


def read_changed_val(data_dir, val_size=0.2):
    """Replicate lib.data.read_changed_val: merge train+val, re-split with seed 777."""
    X_num_train = load_npy(data_dir, "X_num_train.npy")
    X_cat_train = load_npy(data_dir, "X_cat_train.npy")
    y_train = load_npy(data_dir, "y_train.npy")
    X_num_val = load_npy(data_dir, "X_num_val.npy")
    X_cat_val = load_npy(data_dir, "X_cat_val.npy")
    y_val = load_npy(data_dir, "y_val.npy")

    X_num = np.concatenate([X_num_train, X_num_val], axis=0)
    X_cat = np.concatenate([X_cat_train, X_cat_val], axis=0)
    y = np.concatenate([y_train, y_val], axis=0)

    ixs = np.arange(len(y))
    tr_ix, va_ix = train_test_split(ixs, test_size=val_size, random_state=777, stratify=y)

    return (X_num[tr_ix], X_cat[tr_ix], y[tr_ix],
            X_num[va_ix], X_cat[va_ix], y[va_ix])


def run_ablation(data_dir, config_path, seed=0):
    data_dir = str(data_dir)

    X_num_train, X_cat_train, y_train, X_num_val, X_cat_val, y_val = \
        read_changed_val(data_dir, val_size=0.2)

    X_num_test = load_npy(data_dir, "X_num_test.npy")
    X_cat_test = load_npy(data_dir, "X_cat_test.npy")
    y_test = load_npy(data_dir, "y_test.npy")

    with open(config_path) as f:
        catboost_base_config = json.load(f)

    base_params = {k: v for k, v in catboost_base_config.items() if k != 'cat_features'}

    results = []

    for cond_name, cond in ABLATION_CONDITIONS.items():
        drop_cols = cond["drop_cat_cols"]
        keep_cols = [i for i in range(X_cat_train.shape[1]) if i not in drop_cols]

        X_cat_tr = X_cat_train[:, keep_cols]
        X_cat_va = X_cat_val[:, keep_cols]
        X_cat_te = X_cat_test[:, keep_cols]

        n_num = X_num_train.shape[1]
        n_cat = len(keep_cols)
        n_feat = n_num + n_cat

        def build_df(X_num, X_cat):
            df = pd.DataFrame(X_num, columns=range(n_num))
            df_cat = pd.DataFrame(X_cat, columns=range(n_num, n_feat))
            return pd.concat([df, df_cat], axis=1)

        df_train = build_df(X_num_train, X_cat_tr)
        df_val = build_df(X_num_val, X_cat_va)
        df_test = build_df(X_num_test, X_cat_te)

        cat_features = list(range(n_num, n_feat))

        for col in range(n_feat):
            for df in [df_train, df_val, df_test]:
                if col in cat_features:
                    df[col] = df[col].astype(str)
                else:
                    df[col] = df[col].astype(float)

        train_neg = int((y_train == 0).sum())
        train_pos = int((y_train == 1).sum())
        spw = train_neg / train_pos if train_pos > 0 else 1.0

        model = CatBoostClassifier(
            loss_function="Logloss",
            eval_metric="TotalF1",
            random_seed=seed,
            class_names=["0", "1"],
            cat_features=cat_features,
            scale_pos_weight=spw,
            **base_params,
        )

        print(f"\n{'='*70}")
        print(f"Condition: {cond_name} ({cond['description']})")
        print(f"  Features: {n_feat} (num={n_num}, cat={n_cat})")
        print(f"  scale_pos_weight: {spw:.4f}")
        print(f"{'='*70}")

        model.fit(
            df_train, y_train,
            eval_set=(df_val, y_val),
            verbose=200,
        )

        y_pred_proba = model.predict_proba(df_test)[:, 1]
        y_pred = (y_pred_proba >= 0.5).astype(int)

        acc = accuracy_score(y_test, y_pred)
        f1_neg = f1_score(y_test, y_pred, pos_label=0)
        f1_pos = f1_score(y_test, y_pred, pos_label=1)
        auc = roc_auc_score(y_test, y_pred_proba)
        mcc = matthews_corrcoef(y_test, y_pred)

        row = {
            "Condition": cond_name,
            "Features": n_feat,
            "Accuracy": round(acc, 4),
            "F1-score (Neg.)": round(f1_neg, 4),
            "F1-score (Pos.)": round(f1_pos, 4),
            "ROC-AUC": round(auc, 4),
            "MCC": round(mcc, 4),
        }
        results.append(row)

        print(f"\n  Accuracy:       {acc:.4f}")
        print(f"  F1-score (Neg): {f1_neg:.4f}")
        print(f"  F1-score (Pos): {f1_pos:.4f}")
        print(f"  ROC-AUC:        {auc:.4f}")
        print(f"  MCC:            {mcc:.4f}")

    return results


def main():
    parser = argparse.ArgumentParser(description="Feature Ablation Study")
    parser.add_argument("--data_dir", type=str,
                        default="data/orig-micro-retry")
    parser.add_argument("--exp_dir", type=str,
                        default="exp/orig-micro-retry")
    parser.add_argument("--config", type=str,
                        default="tuned_models/catboost/orig-micro-retry_cv.json")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    results = run_ablation(Path(args.data_dir), args.config, seed=args.seed)

    out_dir = Path(args.exp_dir) / "feature_ablation"
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame(results)
    csv_path = out_dir / "feature_ablation_results.csv"
    df.to_csv(csv_path, index=False)

    json_path = out_dir / "feature_ablation_results.json"
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\n{'='*70}")
    print("Feature Ablation Results:")
    print(df.to_string(index=False))
    print(f"\nSaved to: {csv_path}")
    print(f"          {json_path}")


if __name__ == "__main__":
    main()
