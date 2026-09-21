"""
소수 클래스 augmentation 실험 (리뷰어 2의 두 번째 실무 용도).

시나리오: 실제 레이블 데이터가 부족한 기관이 공개 합성데이터로 학습 데이터를 보강한다.
  - 실제 train을 층화 표본으로 f in {5,10,25,50,100}% 만 사용
  - (a) 실제 부분집합만 학습  vs  (b) 실제 부분집합 + 합성 train 전체
  - 튜닝은 하지 않고 고정 설정(공정 비교), 평가는 비공개 test(시간 기준), 지표는 PR-AUC
생성기별로 augmentation이 이득인지/해가 되는지, leaderboard fidelity와 어떤 관계인지 본다.

--syn_ratio r (9/19 PAT 피드백): 합성 train 전체(약 6.4만 행)를 붙이면 5%(3,185행)에서 합성:실제가 약 20:1이 된다.
  그 비율 자체가 결과를 좌우하는지 보려고, 합성 행을 실제 부분집합의 r배로 층화 서브샘플해 붙인다.
  결과는 augmentation_ratio.json에 따로 쌓는다(키 끝에 |r<r>).

Usage:
    OMP_NUM_THREADS=8 python scripts/augmentation_v2.py --models smote,tabddpm,great,tabpfgen,tabpfgen-prior,ctgan,tvae,ctabgan,ctabgan-plus
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
E = ROOT / "exp/finsyn-v2"
REAL = ROOT / "data/finsyn-v2"
INFO = json.loads((REAL / "info.json").read_text())
NUM, CAT = INFO["num_cols"], INFO["cat_cols"]


def load(d, s):
    X = pd.concat([pd.DataFrame(np.load(Path(d) / f"X_num_{s}.npy", allow_pickle=True).astype(float), columns=NUM),
                   pd.DataFrame(np.load(Path(d) / f"X_cat_{s}.npy", allow_pickle=True), columns=CAT).astype(str)], axis=1)
    return X, np.load(Path(d) / f"y_{s}.npy", allow_pickle=True).astype(int)


def fit_eval(name, Xtr, ytr, Xva, yva, Xte, yte, seed):
    cats = {c: sorted(set(Xtr[c]) | set(Xva[c]) | set(Xte[c])) for c in CAT}
    A, B, C = (X.copy() for X in (Xtr, Xva, Xte))
    for c in CAT:
        for D in (A, B, C):
            D[c] = pd.Categorical(D[c], categories=cats[c])
    if name == "lgbm":
        import lightgbm as lgb
        m = lgb.LGBMClassifier(n_estimators=2000, learning_rate=0.05, num_leaves=31, subsample=0.8, subsample_freq=1,
                               colsample_bytree=0.8, random_state=seed, n_jobs=8, verbose=-1)
        m.fit(A, ytr, eval_set=[(B, yva)], eval_metric="average_precision",
              callbacks=[lgb.early_stopping(100, verbose=False)])
    elif name == "xgb":
        import xgboost as xgb
        m = xgb.XGBClassifier(n_estimators=2000, learning_rate=0.05, max_depth=6, subsample=0.8, colsample_bytree=0.8,
                              tree_method="hist", enable_categorical=True, max_cat_to_onehot=1, eval_metric="aucpr",
                              early_stopping_rounds=100, random_state=seed, n_jobs=8)
        m.fit(A, ytr, eval_set=[(B, yva)], verbose=False)
    else:
        from catboost import CatBoostClassifier
        A, B, C = (D.astype({c: str for c in CAT}) for D in (A, B, C))
        m = CatBoostClassifier(iterations=2000, learning_rate=0.05, depth=6, eval_metric="PRAUC", od_type="Iter",
                               od_wait=100, random_seed=seed, thread_count=8, verbose=False,
                               cat_features=[A.columns.get_loc(c) for c in CAT])
        m.fit(A, ytr, eval_set=(B, yva))
    return float(average_precision_score(yte, m.predict_proba(C)[:, 1]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", required=True)
    ap.add_argument("--fractions", default="0.05,0.10,0.25,0.50,1.00")
    ap.add_argument("--detectors", default="lgbm,xgb,catboost")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--out", default=None, help="기본: augmentation.json (--syn_ratio면 augmentation_ratio.json)")
    ap.add_argument("--syn_ratio", type=float, default=None, help="합성 행 수 = r x 실제 부분집합 행 수")
    args = ap.parse_args()

    Xtr, ytr = load(REAL, "train")
    Xva, yva = load(REAL, "val")
    Xte, yte = load(REAL, "test")
    out = Path(args.out or E / ("augmentation_ratio.json" if args.syn_ratio else "augmentation.json"))
    suffix = f"|r{args.syn_ratio:g}" if args.syn_ratio else ""
    res = json.loads(out.read_text()) if out.exists() else {}

    for frac in [float(f) for f in args.fractions.split(",")]:
        for seed in range(args.seeds):
            if frac < 1.0:
                idx, _ = train_test_split(np.arange(len(ytr)), train_size=frac, stratify=ytr, random_state=seed)
            else:
                idx = np.arange(len(ytr))
            Xs, ys = Xtr.iloc[idx].reset_index(drop=True), ytr[idx]
            for det in args.detectors.split(","):
                key = f"real|{frac}|{det}|{seed}"
                if key not in res:
                    res[key] = fit_eval(det, Xs, ys, Xva, yva, Xte, yte, seed)
                    out.write_text(json.dumps(res, indent=1))
                    print(f"{key}: {res[key]:.3f}  (n={len(ys):,}, pos={ys.sum()})", flush=True)
                for model in args.models.split(","):
                    key = f"{model}|{frac}|{det}|{seed}{suffix}"
                    if key in res:
                        continue
                    Xg, yg = load(E / f"synth/{model}/seed0", "train")
                    if args.syn_ratio:
                        n = min(len(yg), int(round(args.syn_ratio * len(ys))))
                        if n < len(yg):
                            strat = yg if np.bincount(yg, minlength=2).min() >= 2 else None
                            gi, _ = train_test_split(np.arange(len(yg)), train_size=n, stratify=strat, random_state=seed)
                            Xg, yg = Xg.iloc[gi].reset_index(drop=True), yg[gi]
                    Xa = pd.concat([Xs, Xg], ignore_index=True)
                    ya = np.concatenate([ys, yg])
                    res[key] = fit_eval(det, Xa, ya, Xva, yva, Xte, yte, seed)
                    out.write_text(json.dumps(res, indent=1))
                    print(f"{key}: {res[key]:.3f}  (+{len(yg):,} synthetic)", flush=True)

    # 요약: 생성기 × fraction 평균 (detector, seed 평균) 과 실제 전용 대비 증감
    rows = []
    for k, v in res.items():
        src, frac, det, seed = k.split("|")[:4]
        rows.append({"src": src, "frac": float(frac), "det": det, "seed": int(seed), "pr_auc": v})
    df = pd.DataFrame(rows).groupby(["src", "frac"]).pr_auc.mean().unstack()
    base = df.loc["real"]
    print("\nPR-AUC (mean over detectors/seeds)\n", df.round(3).to_string())
    print("\nDelta vs real-only\n", (df - base).drop(index="real").round(3).to_string())


if __name__ == "__main__":
    main()
