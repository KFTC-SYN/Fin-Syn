"""
생성기별 '통상' 합성데이터 평가 지표 (발견 (a): 통상 지표로 뽑은 최고 생성기 vs leaderboard fidelity로 뽑은 생성기).

각 생성기의 synthetic train을 real train과 비교한다 (기간별 생성기 중 가장 큰 train 기간 기준).
  fidelity : 수치형 컬럼 평균 KS 통계량, 범주형 컬럼 평균 TVD, 수치형 Spearman 상관행렬 차이(Frobenius/쌍 수)
  detection: real vs synthetic 판별 LightGBM ROC-AUC (5-fold, 각 2만 행) — 0.5에 가까울수록 구분 불가
  label    : 양성 비율 절대 오차(%p)
  utility  : TSTR (synthetic train → real test) CatBoost PR-AUC(ICDM 원고 프로토콜) 및 11개 탐지기 최고 PR-AUC
  privacy  : real train 행과 완전히 같은 synthetic 행 비율(복사율), 정규화 거리 기준 DCR 중앙값
결과: exp/finsyn-v2/standard_metrics.json, 그리고 fidelity 결과와 합친 표를 출력.

Usage:
    OMP_NUM_THREADS=4 python scripts/standard_metrics_v2.py --models smote,tvae,ctgan
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp, spearmanr
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.neighbors import NearestNeighbors

ROOT = Path(__file__).resolve().parents[1]
E = ROOT / "exp/finsyn-v2"


def load(d, s):
    info = json.loads((Path(d) / "info.json").read_text())
    Xn = pd.DataFrame(np.load(Path(d) / f"X_num_{s}.npy", allow_pickle=True).astype(float), columns=info["num_cols"])
    Xc = pd.DataFrame(np.load(Path(d) / f"X_cat_{s}.npy", allow_pickle=True), columns=info["cat_cols"]).astype(str)
    y = np.load(Path(d) / f"y_{s}.npy", allow_pickle=True).astype(int)
    return Xn, Xc, y


def fidelity(rn, rc, sn, sc):
    ks = np.mean([ks_2samp(rn[c], sn[c]).statistic for c in rn.columns])
    tvd = np.mean([0.5 * (rc[c].value_counts(normalize=True).sub(sc[c].value_counts(normalize=True), fill_value=0).abs().sum())
                   for c in rc.columns])
    cr = spearmanr(rn.values).statistic
    cs = spearmanr(sn.values).statistic
    cr, cs = np.nan_to_num(cr), np.nan_to_num(cs)
    k = rn.shape[1]
    corr = float(np.sqrt(((cr - cs) ** 2)[np.triu_indices(k, 1)].mean()))
    return float(ks), float(tvd), corr


def detection_auc(rn, rc, sn, sc, n=20000, seed=0):
    import lightgbm as lgb
    rng = np.random.default_rng(seed)
    ri = rng.choice(len(rn), min(n, len(rn)), replace=False)
    si = rng.choice(len(sn), min(n, len(sn)), replace=False)
    X = pd.concat([pd.concat([rn.iloc[ri], rc.iloc[ri]], axis=1), pd.concat([sn.iloc[si], sc.iloc[si]], axis=1)], ignore_index=True)
    for c in rc.columns:
        X[c] = X[c].astype("category")
    y = np.r_[np.zeros(len(ri)), np.ones(len(si))]
    p = np.zeros(len(y))
    for tr, te in StratifiedKFold(5, shuffle=True, random_state=seed).split(X, y):
        m = lgb.LGBMClassifier(n_estimators=300, learning_rate=0.05, num_leaves=31, n_jobs=4, verbose=-1).fit(X.iloc[tr], y[tr])
        p[te] = m.predict_proba(X.iloc[te])[:, 1]
    return float(roc_auc_score(y, p))


def privacy(rn, rc, sn, sc, n=20000, seed=0):
    key = lambda a, b: pd.concat([a.round(6).astype(str), b], axis=1).agg("|".join, axis=1)
    copy = float(key(sn, sc).isin(set(key(rn, rc))).mean())
    rng = np.random.default_rng(seed)
    lo, hi = rn.quantile(0.01), rn.quantile(0.99)
    scale = (hi - lo).replace(0, 1)
    f = lambda Xn, Xc: np.hstack([((Xn - lo) / scale).clip(-1, 2).values, pd.get_dummies(Xc).reindex(columns=pd.get_dummies(rc).columns, fill_value=0).values])
    R = f(rn, rc)
    si = rng.choice(len(sn), min(n, len(sn)), replace=False)
    S = f(sn.iloc[si], sc.iloc[si])
    d, _ = NearestNeighbors(n_neighbors=1).fit(R).kneighbors(S)
    return copy, float(np.median(d))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", required=True)
    ap.add_argument("--real", default="data/finsyn-v2")
    ap.add_argument("--exp", default="exp/finsyn-v2")
    ap.add_argument("--synth_seeds", default="0", help="릴리스 시드들(쉼표). 0 외가 있으면 standard_metrics_seeds.json에 '<model>|<seed>' 키로")
    args = ap.parse_args()
    global E
    E = ROOT / args.exp
    seeds = [int(s) for s in args.synth_seeds.split(",")]
    per_seed = seeds != [0]
    out_path = E / ("standard_metrics_seeds.json" if per_seed else "standard_metrics.json")
    res = json.loads(out_path.read_text()) if out_path.exists() else {}
    rn, rc, ry = load(ROOT / args.real, "train")
    jobs = [(m, k) for m in args.models.split(",") for k in seeds if (E / f"synth/{m}/seed{k}/info.json").exists()]
    for m, k in jobs:
        rk, suf = (f"{m}|{k}" if per_seed else m), ("" if k == 0 else f"_seed{k}")
        if per_seed and rk in res:
            continue
        syn = E / f"synth/{m}/seed{k}"
        sn, sc, sy = load(syn, "train")
        ks, tvd, corr = fidelity(rn, rc, sn, sc)
        det = detection_auc(rn, rc, sn, sc)
        copy, dcr = privacy(rn, rc, sn, sc)
        lb = json.loads((E / f"leaderboard_{m}_s2r{suf}/results.json").read_text())
        r = {"ks_mean": ks, "tvd_mean": tvd, "corr_rmse": corr, "detection_auc": det,
             "pos_rate_abs_err_pp": float(abs(sy.mean() - ry.mean()) * 100), "pos_rate": float(sy.mean()),
             "tstr_catboost_pr_auc": lb["catboost"]["summary"]["pr_auc"][0],
             "tstr_best_pr_auc": max(v["summary"]["pr_auc"][0] for v in lb.values()),
             "copy_rate": copy, "dcr_median": dcr}
        for tag in ["s2r", "s2s"]:
            fp = E / f"leaderboard_{m}_{tag}{suf}/fidelity_vs_leaderboard_real.json"
            if fp.exists():
                f = json.loads(fp.read_text())
                r[f"lf_tau_{tag}"], r[f"lf_pairs_{tag}"], r[f"lf_regret_{tag}"] = f["kendall_tau"], f["sig_pair_order_kept"], f["selection_regret_pr_auc"]
        res[rk] = r
        out_path.write_text(json.dumps(res, indent=1))
        print(rk, {kk: round(v, 4) for kk, v in r.items()}, flush=True)

    df = pd.DataFrame(res).T
    print("\n", df.round(3).to_string())
    # 지표별 '최고 생성기'
    best = {"ks_mean": df["ks_mean"].idxmin(), "tvd_mean": df["tvd_mean"].idxmin(), "corr_rmse": df["corr_rmse"].idxmin(),
            "detection_auc(closest 0.5)": (df["detection_auc"] - 0.5).abs().idxmin(), "tstr_catboost": df["tstr_catboost_pr_auc"].idxmax(),
            "tstr_best": df["tstr_best_pr_auc"].idxmax()}
    if "lf_tau_s2s" in df:
        best["leaderboard_fidelity_s2s(tau)"] = df["lf_tau_s2s"].astype(float).idxmax()
    print("\nbest generator by metric:", best)


if __name__ == "__main__":
    main()
