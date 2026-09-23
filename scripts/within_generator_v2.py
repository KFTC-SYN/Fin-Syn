"""
같은 생성기의 여러 실행(시드) 중에서 기존 지표가 '좋은 공개본'을 골라낼 수 있는가.

왜: 공개본 12개로 보면 생성기 수준에서는 분포 지표도 tau와 상관이 있다(KS rho 0.64, TSTR 0.92).
    그러나 데이터 보유자의 실제 결정은 "같은 설정으로 만든 여러 실행 중 무엇을 공개할까"이다.
    그 결정에서 각 지표가 쓸모 있는지 본다.

지표별로 두 가지를 잰다.
  agree : 같은 생성기의 시드 쌍에서 지표가 더 좋은 쪽이 tau도 더 높은 비율(0.5 = 동전 던지기)
  regret: 지표로 최고 실행을 골랐을 때 tau가 사후 최선보다 얼마나 낮은가(시드 평균)
비교 기준으로 tau 자체(=비공개 데이터로 직접 측정)도 함께 낸다.

유의성: 한 공개본이 쌍 4개에 들어가므로 쌍들은 독립이 아니고 이항검정은 쓸 수 없다(9/23 외부 검토).
생성기 안에서 지표값을 시드끼리 뒤섞는 순열검정으로 p를 낸다. 귀무가설(생성기 안에서 지표와 tau가 무관)을
그대로 구현하고, 쌍의 종속 구조는 보존된다.

Usage:
    python scripts/within_generator_v2.py
"""
import argparse
import itertools
import json
import os
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
E = ROOT / "exp/finsyn-v2"
OUT = "within_generator.json"
# 생성기마다 같은 수의 공개본을 쓴다. SMOTE는 시드 10개가 남아 있어 상한이 없으면 이 분석에만
# 쌍의 32%를 차지했다(9/22 발견). 논문이 말하는 "생성기당 다섯 공개본"과 어긋난다.
MAX_SEEDS = int(os.environ.get("FINSYN_MAX_SEEDS", "5"))
COLLAPSED = {"tvae", "ctgan", "ctabgan", "ctabgan-plus"}
# 부호: 클수록 좋은 공개본이 되도록 맞춘다
METRICS = {"ks": -1, "tvd": -1, "corr_rmse": -1, "detection_auc": -1, "pos_rate_abs_err_pp": -1,
           "tstr_catboost_pr_auc": +1, "tstr_best_pr_auc": +1}
KEY = {"ks": "ks_mean", "tvd": "tvd_mean", "corr_rmse": "corr_rmse", "detection_auc": "detection_auc",
       "pos_rate_abs_err_pp": "pos_rate_abs_err_pp", "tstr_catboost_pr_auc": "tstr_catboost_pr_auc",
       "tstr_best_pr_auc": "tstr_best_pr_auc"}


N_PERM = 10000


def agree_rate(runs, name, sign, grp, rng):
    """생성기 안에서 지표값을 시드끼리 뒤섞은 뒤의 일치율(순열 귀무분포 한 표본)."""
    hits = []
    for m, rs in runs.items():
        if len(rs) < 2 or (grp == "non_collapsed" and m in COLLAPSED):
            continue
        ss = sorted(rs)
        val = dict(zip(ss, rng.permutation([rs[s][KEY[name]] for s in ss])))
        for a, b in itertools.combinations(ss, 2):
            dm = sign * (val[a] - val[b])
            dt = rs[a]["lf_tau_s2s"] - rs[b]["lf_tau_s2s"]
            if dm != 0 and dt != 0:
                hits.append(np.sign(dm) == np.sign(dt))
    return float(np.mean(hits))


def main():
    global E, OUT, COLLAPSED
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", default="exp/finsyn-v2")
    ap.add_argument("--out", default="within_generator.json")
    a = ap.parse_args()
    E, OUT = ROOT / a.exp, a.out
    if E.name != "finsyn-v2":  # '붕괴' 구분은 본 벤치마크의 것이다. 다른 데이터에서는 전부 한 묶음으로 본다
        COLLAPSED = set()
    sm = json.loads((E / "standard_metrics_seeds.json").read_text())
    runs = {}
    for k, v in sm.items():
        m, s = k.split("|")
        if "lf_tau_s2s" in v and v["lf_tau_s2s"] == v["lf_tau_s2s"]:  # NaN: 합성 test에 양성이 없어 순위 미정의
            runs.setdefault(m, {})[int(s)] = v
    # 상한은 불러오는 곳에서 한 번만 건다. 분석마다 따로 걸다 무작위 선택 기준선에서 빠뜨린 적이 있다(9/23).
    runs = {m: {s_: rs[s_] for s_ in sorted(rs)[:MAX_SEEDS]} for m, rs in runs.items()}
    rng = np.random.default_rng(0)
    res = {}
    for name, sign in METRICS.items():
        # 지표값이 없는 공개본(TabReD 시드 1~4의 TSTR)은 그 지표의 비교에서만 뺀다
        runs_m = {m: {s_: v for s_, v in rs.items() if v.get(KEY[name]) is not None} for m, rs in runs.items()}
        if sum(len(rs) * (len(rs) - 1) // 2 for rs in runs_m.values()) < 2:
            continue
        agree = {"all": [], "non_collapsed": []}
        regret = {"all": [], "non_collapsed": []}
        per_gen_hits = {}  # 생성기별 일치 여부. 구간은 생성기를 단위로 재표집한다(쌍은 독립이 아니다)
        for m, rs in runs_m.items():
            if len(rs) < 2:
                continue
            grp = "non_collapsed" if m not in COLLAPSED else "collapsed"
            for a, b in itertools.combinations(sorted(rs), 2):
                dm = sign * (rs[a][KEY[name]] - rs[b][KEY[name]])
                dt = rs[a]["lf_tau_s2s"] - rs[b]["lf_tau_s2s"]
                if dm != 0 and dt != 0:
                    agree["all"].append(np.sign(dm) == np.sign(dt))
                    per_gen_hits.setdefault(m, []).append(np.sign(dm) == np.sign(dt))
                    if grp == "non_collapsed":
                        agree["non_collapsed"].append(np.sign(dm) == np.sign(dt))
            pick = max(rs, key=lambda s: sign * rs[s][KEY[name]])
            r = max(rs[s]["lf_tau_s2s"] for s in rs) - rs[pick]["lf_tau_s2s"]
            regret["all"].append(r)
            if grp == "non_collapsed":
                regret["non_collapsed"].append(r)
        res[name] = {g: {"n_pairs": len(agree[g]), "agree_rate": float(np.mean(agree[g])),
                         "n_generators": len(regret[g]), "tau_regret_mean": float(np.mean(regret[g]))}
                     for g in ("all", "non_collapsed")}
        hs = list(per_gen_hits.values())
        boot = []
        for _ in range(N_PERM):
            pick = rng.integers(0, len(hs), len(hs))
            boot.append(np.mean(np.concatenate([hs[i] for i in pick])))
        res[name]["all"]["ci95_generator_bootstrap"] = [float(np.quantile(boot, 0.025)), float(np.quantile(boot, 0.975))]
        for g in ("all", "non_collapsed"):
            obs = res[name][g]["agree_rate"]
            null = np.array([agree_rate(runs_m, name, sign, g, rng) for _ in range(N_PERM)])
            res[name][g]["p_perm_two_sided"] = float((np.abs(null - 0.5) >= abs(obs - 0.5) - 1e-12).mean())
            res[name][g]["null_q025_q975"] = [float(np.quantile(null, 0.025)), float(np.quantile(null, 0.975))]
    # 비교 기준: 비공개 데이터로 직접 tau를 재서 고르면 regret 0
    res["tau_oracle"] = {g: {"agree_rate": 1.0, "tau_regret_mean": 0.0} for g in ("all", "non_collapsed")}
    # 무작위 선택의 기대 regret
    rnd = {"all": [], "non_collapsed": []}
    for m, rs in runs.items():
        if len(rs) < 2:
            continue
        t = [rs[s]["lf_tau_s2s"] for s in rs]
        rnd["all"].append(max(t) - np.mean(t))
        if m not in COLLAPSED:
            rnd["non_collapsed"].append(max(t) - np.mean(t))
    res["random_pick"] = {g: {"agree_rate": 0.5, "tau_regret_mean": float(np.mean(rnd[g]))} for g in rnd}
    (E / OUT).write_text(json.dumps(res, indent=1))
    print(f"{'metric':22s} {'agree(all)':>11s} {'agree(non-col)':>15s} {'tau regret(all)':>16s} {'(non-col)':>10s}")
    for k, v in res.items():
        a, n = v["all"], v["non_collapsed"]
        print(f"{k:22s} {a['agree_rate']:11.2f} {n['agree_rate']:15.2f} {a['tau_regret_mean']:16.3f} {n['tau_regret_mean']:10.3f}")
    print("\npermutation p (two-sided, within-generator shuffles):",
          {k: round(v["all"]["p_perm_two_sided"], 3) for k, v in res.items() if "p_perm_two_sided" in v["all"]})
    print(f"runs per generator (capped at {MAX_SEEDS}): { {m: len(r) for m, r in runs.items()} }")


if __name__ == "__main__":
    main()
