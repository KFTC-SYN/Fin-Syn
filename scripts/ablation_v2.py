"""
피처 그룹 ablation (리뷰어 1: 기관 간/공동망 정보의 가치). 고정 LightGBM 설정, seed 5개, 시간 기준 test.

Usage:
    OMP_NUM_THREADS=8 python scripts/ablation_v2.py
"""
import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve

ROOT = Path(__file__).resolve().parents[1]
D = ROOT / "data/finsyn-v2"
info = json.loads((D / "info.json").read_text())
G = info["feature_groups"]


def load(s):
    X = pd.concat([pd.DataFrame(np.load(D / f"X_num_{s}.npy"), columns=info["num_cols"]),
                   pd.DataFrame(np.load(D / f"X_cat_{s}.npy", allow_pickle=True), columns=info["cat_cols"])], axis=1)
    return X, np.load(D / f"y_{s}.npy")


(Xtr, ytr), (Xva, yva), (Xte, yte) = load("train"), load("val"), load("test")
SETS = {
    "T": G["transaction"],
    "T+B": G["transaction"] + G["bank"],
    "T+R": G["transaction"] + G["receiver_crossbank_history"],
    "T+B+R": G["transaction"] + G["bank"] + G["receiver_crossbank_history"],
    "T+S": G["transaction"] + G["sender_history"],
    "T+B+S": G["transaction"] + G["bank"] + G["sender_history"],
    "T+B+S+R": G["transaction"] + G["bank"] + G["sender_history"] + G["receiver_crossbank_history"],
}


def run(cols, seed):
    A, B, C = (X[cols].copy() for X in (Xtr, Xva, Xte))
    for c in cols:
        if c in info["cat_cols"]:
            A[c] = A[c].astype("category")
            B[c] = pd.Categorical(B[c], categories=A[c].cat.categories)
            C[c] = pd.Categorical(C[c], categories=A[c].cat.categories)
    m = lgb.LGBMClassifier(n_estimators=2000, learning_rate=0.03, num_leaves=31, min_child_samples=20, subsample=0.8,
                           subsample_freq=1, colsample_bytree=0.8, random_state=seed, n_jobs=8, verbose=-1)
    m.fit(A, ytr, eval_set=[(B, yva)], eval_metric="average_precision", callbacks=[lgb.early_stopping(100, verbose=False)])
    p = m.predict_proba(C)[:, 1]
    fpr, tpr, _ = roc_curve(yte, p)
    return {"pr_auc": average_precision_score(yte, p), "roc_auc": roc_auc_score(yte, p),
            "recall@0.1%fpr": float(np.interp(0.001, fpr, tpr)), "recall@1%fpr": float(np.interp(0.01, fpr, tpr))}


res = {}
for name, cols in SETS.items():
    runs = [run(cols, s) for s in range(5)]
    res[name] = {"n_features": len(cols), **{k: [float(np.mean([r[k] for r in runs])), float(np.std([r[k] for r in runs]))] for k in runs[0]}}
    print(name, {k: round(v[0], 3) if isinstance(v, list) else v for k, v in res[name].items()}, flush=True)
(ROOT / "exp/finsyn-v2/ablation_feature_groups.json").write_text(json.dumps(res, indent=1))
