"""
같은 생성기의 여러 실행(시드) 중에서 기존 지표가 '좋은 공개본'을 골라낼 수 있는가.

왜: 공개본 12개로 보면 생성기 수준에서는 분포 지표도 tau와 상관이 있다(KS rho 0.64, TSTR 0.92).
    그러나 데이터 보유자의 실제 결정은 "같은 설정으로 만든 여러 실행 중 무엇을 공개할까"이다.
    그 결정에서 각 지표가 쓸모 있는지 본다.

지표별로 두 가지를 잰다.
  agree : 같은 생성기의 시드 쌍에서 지표가 더 좋은 쪽이 tau도 더 높은 비율(0.5 = 동전 던지기)
  regret: 지표로 최고 실행을 골랐을 때 tau가 사후 최선보다 얼마나 낮은가(시드 평균)
비교 기준으로 tau 자체(=비공개 데이터로 직접 측정)도 함께 낸다.

Usage:
    python scripts/within_generator_v2.py
"""
import itertools
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
E = ROOT / "exp/finsyn-v2"
COLLAPSED = {"tvae", "ctgan", "ctabgan", "ctabgan-plus"}
# 부호: 클수록 좋은 공개본이 되도록 맞춘다
METRICS = {"ks": -1, "tvd": -1, "corr_rmse": -1, "detection_auc": -1, "pos_rate_abs_err_pp": -1,
           "tstr_catboost_pr_auc": +1, "tstr_best_pr_auc": +1}
KEY = {"ks": "ks_mean", "tvd": "tvd_mean", "corr_rmse": "corr_rmse", "detection_auc": "detection_auc",
       "pos_rate_abs_err_pp": "pos_rate_abs_err_pp", "tstr_catboost_pr_auc": "tstr_catboost_pr_auc",
       "tstr_best_pr_auc": "tstr_best_pr_auc"}


def main():
    sm = json.loads((E / "standard_metrics_seeds.json").read_text())
    runs = {}
    for k, v in sm.items():
        m, s = k.split("|")
        if "lf_tau_s2s" in v:
            runs.setdefault(m, {})[int(s)] = v
    res = {}
    for name, sign in METRICS.items():
        agree = {"all": [], "non_collapsed": []}
        regret = {"all": [], "non_collapsed": []}
        for m, rs in runs.items():
            if len(rs) < 2:
                continue
            grp = "non_collapsed" if m not in COLLAPSED else "collapsed"
            for a, b in itertools.combinations(sorted(rs), 2):
                dm = sign * (rs[a][KEY[name]] - rs[b][KEY[name]])
                dt = rs[a]["lf_tau_s2s"] - rs[b]["lf_tau_s2s"]
                if dm != 0 and dt != 0:
                    agree["all"].append(np.sign(dm) == np.sign(dt))
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
    (E / "within_generator.json").write_text(json.dumps(res, indent=1))
    print(f"{'metric':22s} {'agree(all)':>11s} {'agree(non-col)':>15s} {'tau regret(all)':>16s} {'(non-col)':>10s}")
    for k, v in res.items():
        a, n = v["all"], v["non_collapsed"]
        print(f"{k:22s} {a['agree_rate']:11.2f} {n['agree_rate']:15.2f} {a['tau_regret_mean']:16.3f} {n['tau_regret_mean']:10.3f}")
    print(f"\nruns per generator: { {m: len(r) for m, r in runs.items()} }")


if __name__ == "__main__":
    main()
