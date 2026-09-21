"""
TSTR과 leaderboard fidelity(LF)가 같은 것을 재는가?

리뷰어 예상 질문: "릴리스 간 rho(TSTR, tau_S->S)=0.92인데, 그냥 TSTR을 쓰면 되지 않나?"
네 가지로 나눠 본다.

 (1) 붕괴 릴리스 제외: GAN/VAE 4종(TSTR 0.01~0.13)은 두 지표 모두에서 바닥이라 상관을 부풀린다.
     나머지 5개만으로도 상관이 유지되는가.
 (2) 탐지기 의존성: TSTR은 "어떤 한 모델"의 점수다. 11개 탐지기 각각으로 TSTR을 정의하면
     릴리스 순위(및 tau와의 상관)가 얼마나 달라지는가. 달라진다면 TSTR은 단일한 기준이 아니다.
 (3) 시드 내 선택: 공개 결정은 "어느 생성기"뿐 아니라 "그 생성기의 어느 실행본"을 고르는 일이다.
     같은 생성기의 시드 3개 사이에서 TSTR이 높은 쪽이 tau도 높은가(부호 일치율).
 (4) 순위 불일치: 두 지표가 릴리스 순서를 다르게 매기는 쌍.

TSTR 값은 S->R 리더보드(합성 train/tune, 비공개 test)의 탐지기 PR-AUC에서 시드별로 읽는다.

Usage:
    python scripts/tstr_vs_lf_v2.py
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
          "tabsyn", "tabdiff", "findiff"]  # 9/19: 12개
COLLAPSED = {"tvae", "ctgan", "ctabgan", "ctabgan-plus"}
DETS = ["nb", "dt", "lr", "knn", "mlp", "rf", "et", "hgb", "lgbm", "xgb", "catboost"]


def run_dir(model, tag, seed):
    return E / (f"leaderboard_{model}_{tag}" if seed == 0 else f"leaderboard_{model}_{tag}_seed{seed}")


def seeds_of(model):
    return [s for s in range(MAX_SEEDS) if (run_dir(model, "s2s", s) / "fidelity_vs_leaderboard_real.json").exists()]


def tau(model, seed):
    return json.loads((run_dir(model, "s2s", seed) / "fidelity_vs_leaderboard_real.json").read_text())["kendall_tau"]


def tstr(model, seed, det):
    r = json.loads((run_dir(model, "s2r", seed) / "results.json").read_text())
    v = r.get(det, {}).get("summary", {}).get("pr_auc", [np.nan])[0]
    return float(v) if v is not None else np.nan


def spearman(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    ok = ~(np.isnan(x) | np.isnan(y))
    x, y = x[ok], y[ok]
    if len(x) < 3 or np.std(rankdata(x)) == 0 or np.std(rankdata(y)) == 0:
        return np.nan
    return float(np.corrcoef(rankdata(x), rankdata(y))[0, 1])


def exact_p(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    rx, ry = rankdata(x), rankdata(y)
    obs = abs(spearman(x, y))
    cnt = tot = 0
    for perm in itertools.permutations(range(len(y))):
        tot += 1
        cnt += abs(np.corrcoef(rx, ry[list(perm)])[0, 1]) >= obs - 1e-12
    return cnt / tot


def main():
    seeds = {m: seeds_of(m) for m in MODELS}
    tau_mean = {m: np.mean([tau(m, s) for s in seeds[m]]) for m in MODELS}
    tstr_mean = {d: {m: np.nanmean([tstr(m, s, d) for s in seeds[m]]) for m in MODELS} for d in DETS}
    tstr_best = {m: np.mean([max(tstr(m, s, d) for d in DETS) for s in seeds[m]]) for m in MODELS}
    out = {}

    # (0) 기준: 전체 9개, TSTR(CatBoost, 시드 평균) / TSTR(best, 시드 평균)
    t9 = [tau_mean[m] for m in MODELS]  # 이름은 예전 그대로(all9), 내용은 MODELS 전체
    out["all9"] = out["all"] = {"catboost": spearman([tstr_mean["catboost"][m] for m in MODELS], t9),
                   "best": spearman([tstr_best[m] for m in MODELS], t9)}

    # (1) 붕괴 릴리스 제외
    keep = [m for m in MODELS if m not in COLLAPSED]
    x5, y5 = [tstr_mean["catboost"][m] for m in keep], [tau_mean[m] for m in keep]
    out["non_collapsed"] = {"models": keep, "rho_catboost": spearman(x5, y5), "p_catboost": exact_p(x5, y5),
                            "rho_best": spearman([tstr_best[m] for m in keep], y5),
                            "tstr_catboost": dict(zip(keep, x5)), "tau": dict(zip(keep, y5))}

    # (2) 탐지기 의존성: 11개 탐지기 각각의 TSTR로 본 상관과 1위 릴리스
    per_det = {}
    for d in DETS:
        xs = [tstr_mean[d][m] for m in MODELS]
        order = [m for _, m in sorted(zip(xs, MODELS), key=lambda z: -np.nan_to_num(z[0], nan=-1))]
        per_det[d] = {"rho_all9": spearman(xs, t9),
                      "rho_non_collapsed": spearman([tstr_mean[d][m] for m in keep], y5),
                      "top_release": order[0], "second_release": order[1]}
    out["per_detector"] = per_det
    rhos = [v["rho_non_collapsed"] for v in per_det.values() if not np.isnan(v["rho_non_collapsed"])]
    out["per_detector_summary"] = {"rho_non_collapsed_min": float(min(rhos)), "rho_non_collapsed_max": float(max(rhos)),
                                   "distinct_top_releases": sorted({v["top_release"] for v in per_det.values()})}
    # TSTR 탐지기를 바꿨을 때 릴리스 순위끼리의 일치도(Kendall tau 대신 Spearman 평균)
    mats = []
    for a, b in itertools.combinations(DETS, 2):
        mats.append(spearman([tstr_mean[a][m] for m in keep], [tstr_mean[b][m] for m in keep]))
    out["per_detector_summary"]["mean_pairwise_rank_agreement_non_collapsed"] = float(np.nanmean(mats))

    # (3) 시드 내 선택: 같은 생성기의 시드 쌍에서 TSTR 차이와 tau 차이의 부호 일치율
    pairs = []
    for m in MODELS:
        ss = seeds[m]
        for a, b in itertools.combinations(ss, 2):
            dt = tstr(m, a, "catboost") - tstr(m, b, "catboost")
            dtau = tau(m, a) - tau(m, b)
            pairs.append({"model": m, "seeds": [a, b], "d_tstr": dt, "d_tau": dtau,
                          "agree": bool(np.sign(dt) == np.sign(dtau)) if dt != 0 and dtau != 0 else None})
    decided = [p for p in pairs if p["agree"] is not None]
    nc = [p for p in decided if p["model"] not in COLLAPSED]
    out["within_generator"] = {
        "n_pairs": len(decided), "agree_rate": float(np.mean([p["agree"] for p in decided])),
        "n_pairs_non_collapsed": len(nc), "agree_rate_non_collapsed": float(np.mean([p["agree"] for p in nc])),
        "pairs": pairs,
        # 시드 선택에 쓰였을 때의 결과: TSTR 최고 시드를 골랐을 때 tau 최고 시드를 맞히는가
        "pick": {m: {"by_tstr": int(max(seeds[m], key=lambda s: tstr(m, s, "catboost"))),
                     "by_tau": int(max(seeds[m], key=lambda s: tau(m, s))),
                     "tau_by_tstr_pick": tau(m, max(seeds[m], key=lambda s: tstr(m, s, "catboost"))),
                     "tau_best": max(tau(m, s) for s in seeds[m]),
                     "tau_worst": min(tau(m, s) for s in seeds[m])}
                 for m in MODELS if len(seeds[m]) > 1},
    }

    # (4) 릴리스 쌍 중 두 지표가 순서를 다르게 매기는 경우
    disc = []
    for a, b in itertools.combinations(MODELS, 2):
        s1 = np.sign(tstr_mean["catboost"][a] - tstr_mean["catboost"][b])
        s2 = np.sign(tau_mean[a] - tau_mean[b])
        if s1 and s2 and s1 != s2:
            disc.append({"pair": [a, b], "tstr": [tstr_mean["catboost"][a], tstr_mean["catboost"][b]],
                         "tau": [tau_mean[a], tau_mean[b]]})
    out["discordant_release_pairs"] = disc

    (E / "tstr_vs_lf.json").write_text(json.dumps(out, indent=1, default=float))

    print(f"(0) all 9 releases: rho(TSTR-CatBoost, tau)={out['all9']['catboost']:+.2f}, "
          f"rho(TSTR-best, tau)={out['all9']['best']:+.2f}")
    nc_ = out["non_collapsed"]
    print(f"(1) without the 4 collapsed releases (n=5): rho={nc_['rho_catboost']:+.2f} (exact p={nc_['p_catboost']:.3f}), "
          f"best-detector TSTR rho={nc_['rho_best']:+.2f}")
    for m in keep:
        print(f"      {m:15s} TSTR={nc_['tstr_catboost'][m]:.3f}  tau={nc_['tau'][m]:+.3f}")
    s = out["per_detector_summary"]
    print(f"(2) TSTR defined with each of 11 detectors, non-collapsed: rho from {s['rho_non_collapsed_min']:+.2f} "
          f"to {s['rho_non_collapsed_max']:+.2f}; top release by detector: {s['distinct_top_releases']}; "
          f"mean pairwise agreement of the TSTR release rankings = {s['mean_pairwise_rank_agreement_non_collapsed']:+.2f}")
    for d, v in per_det.items():
        print(f"      {d:9s} rho9={v['rho_all9']:+.2f} rho5={v['rho_non_collapsed']:+.2f} top={v['top_release']}")
    w = out["within_generator"]
    print(f"(3) within a generator, sign(dTSTR)==sign(dtau): {w['agree_rate']:.0%} of {w['n_pairs']} seed pairs "
          f"({w['agree_rate_non_collapsed']:.0%} of {w['n_pairs_non_collapsed']} for non-collapsed)")
    for m, v in w["pick"].items():
        print(f"      {m:15s} TSTR picks seed {v['by_tstr']} (tau {v['tau_by_tstr_pick']:+.3f}); "
              f"best seed {v['by_tau']} (tau {v['tau_best']:+.3f}), worst {v['tau_worst']:+.3f}")
    print(f"(4) discordant release pairs (TSTR vs tau): {len(disc)} of 36")
    for d in disc:
        print(f"      {d['pair']}: TSTR {d['tstr'][0]:.3f} vs {d['tstr'][1]:.3f} | tau {d['tau'][0]:+.3f} vs {d['tau'][1]:+.3f}")


if __name__ == "__main__":
    main()
