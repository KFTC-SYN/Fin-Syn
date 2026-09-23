"""
원고의 leaderboard-fidelity 수치를 한 곳에서 계산한다(본문·표·그림이 같은 정의를 쓰도록).

정의
  릴리스의 tau / 유의쌍 보존율 / regret = 생성기 시드 0,1,2 결과의 평균(GReaT는 시드 0 하나).
  표준지표는 생성기당 다섯 공개본의 평균을 쓴다(상관 분석). 프라이버시는 시드 0 값이다.

상관 분석 (릴리스 9개, n=9)
  Spearman rho, 각 지표는 "클수록 좋은 릴리스"가 되도록 부호를 맞춘다(오차·구분가능성은 음수화).
  p값: 9! = 362,880개 순열 전체에 대한 정확 순열검정(양측).
  95% CI: 릴리스를 복원추출하는 부트스트랩 10,000회(상수 표본은 제외), 백분위 구간.
  동률이 많은 지표(C2ST는 9개 중 6개가 1.000)는 동률 수를 함께 기록한다.

Usage:
    python scripts/lf_analysis_v2.py
"""
import itertools
import json
from pathlib import Path

import numpy as np
from scipy.stats import rankdata

import os
# 논문 메인 표/그림은 생성기마다 같은 수의 공개본을 쓴다(사용자 지시 9/20). 기본 5시드.
MAX_SEEDS = int(os.environ.get("FINSYN_MAX_SEEDS", "5"))

ROOT = Path(__file__).resolve().parents[1]
E = ROOT / "exp/finsyn-v2"
MODELS = ["smote", "tabpfgen", "tabpfgen-prior", "tabddpm", "great", "tvae", "ctabgan", "ctgan", "ctabgan-plus",
          "tabsyn", "tabdiff", "findiff"]  # 9/19: TabSyn, TabDiff, FinDiff 추가(12개)
COLLAPSED = {"tvae", "ctgan", "ctabgan", "ctabgan-plus"}
RNG = np.random.default_rng(0)


def seed_runs(model, tag):
    runs = []
    for s in range(MAX_SEEDS):  # 생성기마다 같은 수의 공개본만 집계한다
        d = E / (f"leaderboard_{model}_{tag}" if s == 0 else f"leaderboard_{model}_{tag}_seed{s}")
        f = d / "fidelity_vs_leaderboard_real.json"
        if f.exists():
            runs.append(json.loads(f.read_text()))
    return runs


def summarize(model, tag):
    runs = seed_runs(model, tag)
    tau = [r["kendall_tau"] for r in runs]
    return {
        "n_seeds": len(runs),
        "tau": tau, "tau_mean": float(np.mean(tau)), "tau_min": float(min(tau)), "tau_max": float(max(tau)),
        "pairs_mean": float(np.mean([r["sig_pair_order_kept"] for r in runs])),
        "regret": [r["selection_regret_pr_auc"] for r in runs],
        "regret_mean": float(np.mean([r["selection_regret_pr_auc"] for r in runs])),
        "top1": [r["cand_top1"] for r in runs],
    }


def spearman(x, y):
    rx, ry = rankdata(x), rankdata(y)
    if rx.std() == 0 or ry.std() == 0:
        return np.nan
    return float(np.corrcoef(rx, ry)[0, 1])


_PERMS = {}


def perms(n):
    """n <= 9면 모든 순열(정확 검정), 그보다 크면 무작위 순열 200,000개(몬테카를로 검정)."""
    if n not in _PERMS:
        if n <= 9:
            _PERMS[n] = np.array(list(itertools.permutations(range(n))), dtype=np.int8)
        else:
            rng = np.random.default_rng(12345)
            _PERMS[n] = np.array([rng.permutation(n) for _ in range(200_000)], dtype=np.int8)
    return _PERMS[n]


def perm_p(x, y):
    """순열검정: y의 순위를 섞어 |rho| 이상이 나올 비율(n <= 9 정확, 그 이상 몬테카를로)."""
    rx, ry = rankdata(x), rankdata(y)
    rx = (rx - rx.mean()) / rx.std()
    ryp = ry[perms(len(ry))]
    ryp = (ryp - ryp.mean(axis=1, keepdims=True)) / ryp.std(axis=1, keepdims=True)
    rho = (ryp * rx).mean(axis=1)
    obs = float(np.mean(rx * (ry - ry.mean()) / ry.std()))
    return float(np.mean(np.abs(rho) >= abs(obs) - 1e-12))


def boot_ci(x, y, n=10000):
    x, y = np.asarray(x), np.asarray(y)
    vals = []
    for _ in range(n):
        i = RNG.integers(0, len(x), len(x))
        r = spearman(x[i], y[i])
        if not np.isnan(r):
            vals.append(r)
    return [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))]


def assoc(x, y):
    return {"rho": spearman(x, y), "p": perm_p(x, y), "ci95": boot_ci(x, y),
            "n_ties_x": int(len(x) - len(set(np.round(x, 9))))}


def main():
    sm = json.loads((E / "standard_metrics.json").read_text())
    # 생성기 수준 상관은 생성기당 다섯 공개본의 평균으로 잰다. 시드 0 한 개로 재면
    # y축(다섯 공개본 평균 tau)과 x축의 기준이 어긋난다(9/22 점검).
    per = {}
    for k, v in json.loads((E / "standard_metrics_seeds.json").read_text()).items():
        m, sd = k.split("|")
        if int(sd) < MAX_SEEDS:
            per.setdefault(m, []).append(v)
    smm = {m: {k: float(np.mean([r[k] for r in rs])) for k in rs[0] if isinstance(rs[0][k], (int, float))}
           for m, rs in per.items()}
    sm = {m: {**sm.get(m, {}), **smm.get(m, {})} for m in set(sm) | set(smm)}
    pv = json.loads((E / "privacy.json").read_text())
    rel = {m: {"s2s": summarize(m, "s2s"), "s2r": summarize(m, "s2r")} for m in MODELS}
    tau = np.array([rel[m]["s2s"]["tau_mean"] for m in MODELS])

    # 표준지표: 클수록 좋게 부호 정렬
    metrics = {
        "ks": [-sm[m]["ks_mean"] for m in MODELS],
        "tvd": [-sm[m]["tvd_mean"] for m in MODELS],
        "c2st": [-sm[m]["detection_auc"] for m in MODELS],
        "label_rate_error": [-sm[m]["pos_rate_abs_err_pp"] for m in MODELS],
        "dependence_error": [-sm[m]["corr_rmse"] for m in MODELS],
        "tstr_catboost": [sm[m]["tstr_catboost_pr_auc"] for m in MODELS],
        "dcr_ratio": [pv[m]["dcr_ratio"] for m in MODELS],   # 클수록 비공개 데이터에서 멀다
    }
    corr = {k: assoc(v, tau) for k, v in metrics.items()}

    # tau가 사용자에게 무엇을 뜻하는지: 공개본 60개에서 tau와 나머지 두 기준의 관계.
    _ps = json.loads((E / "standard_metrics_seeds.json").read_text())
    _t, _r, _p = [], [], []
    for _k, _v in _ps.items():
        if int(_k.split("|")[1]) >= MAX_SEEDS:
            continue
        if "lf_tau_s2s" in _v and "lf_regret_s2s" in _v and "lf_pairs_s2s" in _v:
            _t.append(_v["lf_tau_s2s"]); _r.append(_v["lf_regret_s2s"]); _p.append(_v["lf_pairs_s2s"])
    _t, _r = np.array(_t), np.array(_r)
    _band = NFR["tau_vs_full_q05"] if "NFR" in dir() else json.loads(
        (E / "leaderboard_real/noise_floor.json").read_text())["tau_vs_full_q05"]
    tau_buys = {
        "n_releases": len(_t),
        "rho_tau_pairs": assoc(_t.tolist(), _p)["rho"],
        "rho_tau_regret": assoc(_t.tolist(), _r.tolist())["rho"],
        "band_regret_max": float(_r[_t >= _band].max()),
        "band_n": int((_t >= _band).sum()),
        "low_regret_mean": float(_r[_t < 0.5].mean()),
        "low_regret_max": float(_r[_t < 0.5].max()),
        "low_n": int((_t < 0.5).sum()),
    }

    # 증강 이득(레이블 5%)과의 연관
    aug = json.loads((E / "augmentation.json").read_text())
    def gain(model, frac=0.05):
        g = []
        for det in ("lgbm", "xgb"):
            for s in range(3):
                g.append(aug[f"{model}|{frac}|{det}|{s}"] - aug[f"real|{frac}|{det}|{s}"])
        return float(np.mean(g))
    g5 = [gain(m) for m in MODELS]
    aug_corr = {"tau_s2s": assoc(tau, g5), "tstr_catboost": assoc(metrics["tstr_catboost"], g5),
                "ks": assoc(metrics["ks"], g5), "c2st": assoc(metrics["c2st"], g5)}

    real = json.loads((E / "leaderboard_real/results.json").read_text())
    pr = np.array([real[k]["summary"]["pr_auc"][0] for k in real])
    nf = json.loads((E / "leaderboard_real/noise_floor.json").read_text())

    # 공개본 단위 변동: tau_S->S를 생성기 간 분산과 같은 생성기 안의 시드 간 분산으로 나눈다
    def decompose(models):
        runs = {m: rel[m]["s2s"]["tau"] for m in models if rel[m]["s2s"]["n_seeds"] >= 2}
        allv = np.concatenate([np.asarray(v) for v in runs.values()])
        within = float(np.mean([np.var(v, ddof=1) for v in runs.values()]))
        between = float(np.var([np.mean(v) for v in runs.values()], ddof=1))
        # 비중은 일원 분산분석의 제곱합으로 낸다: 생성기 안 제곱합 / 전체 제곱합(9/23 외부 검토).
        # 예전 정의(생성기 안 표본분산의 평균 / (그것 + 생성기 평균의 분산))는 within_share_var_ratio로 남긴다.
        grand = float(np.mean(allv))
        ss_w = float(sum(((np.asarray(v) - np.mean(v)) ** 2).sum() for v in runs.values()))
        ss_b = float(sum(len(v) * (np.mean(v) - grand) ** 2 for v in runs.values()))
        return {"generators": list(runs), "n_runs": int(len(allv)), "total_var": float(np.var(allv, ddof=1)),
                "ss_within": ss_w, "ss_between": ss_b, "within_share": ss_w / (ss_w + ss_b),
                "within_var": within, "between_var_of_means": between,
                "within_share_var_ratio": within / (within + between),
                "within_sd": float(np.sqrt(within)),
                "range_per_generator": {m: [float(min(v)), float(max(v))] for m, v in runs.items()}}
    variance = {"all": decompose(MODELS), "non_collapsed": decompose([m for m in MODELS if m not in COLLAPSED])}

    out = {"definition": f"tau/pairs/regret = mean over generator seeds 0-{MAX_SEEDS - 1}",
           "variance_decomposition": variance,
           "releases": rel, "corr_with_tau_s2s": corr, "what_tau_buys": tau_buys,
           "aug_gain_5pct": dict(zip(MODELS, g5)),
           "aug_corr": aug_corr, "random_choice_regret": float(pr.max() - pr.mean()),
           "noise_floor": {"q05": nf["tau_vs_full_q05"], "mean": nf["tau_vs_full_mean"]}}
    (E / "lf_analysis.json").write_text(json.dumps(out, indent=1))

    print("release          tau_s2s mean [min,max]    tau_s2r   pairs  regret(mean)")
    for m in sorted(MODELS, key=lambda m: -rel[m]["s2s"]["tau_mean"]):
        a, b = rel[m]["s2s"], rel[m]["s2r"]
        print(f"  {m:15s} {a['tau_mean']:+.3f} [{a['tau_min']:+.3f},{a['tau_max']:+.3f}]  {b['tau_mean']:+.3f}"
              f"   {a['pairs_mean']:.2f}   {a['regret_mean']:.3f}")
    print("\ncorrelation with mean tau_s2s (rho, exact p, 95% CI, ties in metric)")
    for k, v in corr.items():
        print(f"  {k:17s} {v['rho']:+.2f}  p={v['p']:.3f}  CI[{v['ci95'][0]:+.2f},{v['ci95'][1]:+.2f}]  ties={v['n_ties_x']}")
    print("\naugmentation gain at 5% labels vs ...")
    for k, v in aug_corr.items():
        print(f"  {k:17s} {v['rho']:+.2f}  p={v['p']:.3f}  CI[{v['ci95'][0]:+.2f},{v['ci95'][1]:+.2f}]")
    print(f"\nrandom-choice regret on the private leaderboard: {out['random_choice_regret']:.3f}")


if __name__ == "__main__":
    main()
