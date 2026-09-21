"""
가시성 기준 ablation: 송금 은행이 스스로 볼 수 있는 피처만 쓴 탐지기 vs 공동망 전용 피처를 더한 탐지기.

왜: 기존 그룹 ablation(ablation_v2.py)은 공동망 피처의 한계 기여가 +0.003(0.913→0.916)이라
    "interbank 동기"를 약하게 만든다. 그런데 그 평균은 전형적 사기 패턴과 그렇지 않은 거래를 섞은 값이다.
    테스트 기간 의심거래의 73%가 '처음 보내는 수취인'에게 가는 이체인데, 이때 수취 계좌의 과거는
    송금 은행이 볼 수 없고 공동망만 안다. 그 부분집합에서 따로 재 본다.

가시성은 부록 A 피처표의 "Visible to" 열과 같다(N = 공동망 전용 5개).
bankpair_n_30d(송금은행→수취은행 30일 건수)와 s_n_dbanks_prev(송금계좌의 과거 수취은행 수)는
모두 송금은행이 발신한 거래로만 이루어지므로 송금은행도 계산할 수 있다(S). 처음 버전은 이 둘을 N으로 두었고,
그 결과는 visibility_ablation_7N.json에 보관했다.
설정은 ablation_v2.py와 동일(고정 LightGBM, 시드 5개, val로 early stopping).

Usage:
    OMP_NUM_THREADS=8 python scripts/visibility_ablation_v2.py
"""
import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_curve

ROOT = Path(__file__).resolve().parents[1]
D = ROOT / "data/finsyn-v2"
info = json.loads((D / "info.json").read_text())
NETWORK_ONLY = ["r_n_prev_in", "r_n_in_30d", "r_n_payers_prev", "r_n_sendbanks_prev", "r_days_since_first"]
ALL = info["num_cols"] + info["cat_cols"]
SENDER_VISIBLE = [c for c in ALL if c not in NETWORK_ONLY]
assert len(NETWORK_ONLY) == 5 and len(SENDER_VISIBLE) == 19


def load(s):
    X = pd.concat([pd.DataFrame(np.load(D / f"X_num_{s}.npy"), columns=info["num_cols"]),
                   pd.DataFrame(np.load(D / f"X_cat_{s}.npy", allow_pickle=True), columns=info["cat_cols"])], axis=1)
    return X, np.load(D / f"y_{s}.npy").astype(int)


(Xtr, ytr), (Xva, yva), (Xte, yte) = load("train"), load("val"), load("test")


def predict(cols, seed):
    A, B, C = (X[cols].copy() for X in (Xtr, Xva, Xte))
    for c in cols:
        if c in info["cat_cols"]:
            A[c] = A[c].astype("category")
            B[c] = pd.Categorical(B[c], categories=A[c].cat.categories)
            C[c] = pd.Categorical(C[c], categories=A[c].cat.categories)
    m = lgb.LGBMClassifier(n_estimators=2000, learning_rate=0.03, num_leaves=31, min_child_samples=20, subsample=0.8,
                           subsample_freq=1, colsample_bytree=0.8, random_state=seed, n_jobs=8, verbose=-1)
    m.fit(A, ytr, eval_set=[(B, yva)], eval_metric="average_precision", callbacks=[lgb.early_stopping(100, verbose=False)])
    return m.predict_proba(C)[:, 1]


def recall_at(y, p, fpr_max):
    fpr, tpr, _ = roc_curve(y, p)
    return float(tpr[fpr <= fpr_max].max()) if (fpr <= fpr_max).any() else 0.0


def main():
    subsets = {"all": np.ones(len(yte), bool),
               "first_time_payee": (Xte["pair_n_prev"] == 0).values,
               "known_payee": (Xte["pair_n_prev"] > 0).values}
    res = {k: {"n": int(m.sum()), "positives": int(yte[m].sum())} for k, m in subsets.items()}
    for name, cols in [("sender_visible", SENDER_VISIBLE), ("with_network", ALL)]:
        preds = [predict(cols, s) for s in range(5)]
        for k, m in subsets.items():
            pr = [average_precision_score(yte[m], p[m]) for p in preds]
            r1 = [recall_at(yte[m], p[m], 0.01) for p in preds]
            res[k][name] = {"pr_auc": [float(np.mean(pr)), float(np.std(pr))],
                            "recall@1%fpr": [float(np.mean(r1)), float(np.std(r1))]}
    out = ROOT / "exp/finsyn-v2/visibility_ablation.json"
    out.write_text(json.dumps(res, indent=1))
    print(f"{'subset':18s} {'n':>6s} {'pos':>4s}  {'sender-only':>18s}  {'+network':>18s}  {'gain':>6s}")
    for k, v in res.items():
        a, b = v["sender_visible"]["pr_auc"], v["with_network"]["pr_auc"]
        print(f"{k:18s} {v['n']:6,d} {v['positives']:4d}  {a[0]:.3f} ± {a[1]:.3f}     {b[0]:.3f} ± {b[1]:.3f}     "
              f"{b[0]-a[0]:+.3f}")
    print("\nrecall@1%FPR")
    for k, v in res.items():
        a, b = v["sender_visible"]["recall@1%fpr"], v["with_network"]["recall@1%fpr"]
        print(f"{k:18s} {a[0]:.3f} -> {b[0]:.3f} ({b[0]-a[0]:+.3f})")
    print(f"saved -> {out}")


if __name__ == "__main__":
    main()
