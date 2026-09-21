"""
"여러 번 생성하고, 비공개 데이터로 재서, 가장 좋은 공개본을 내라"는 권고가 통하는지 확인한다.

문제: 데이터 보유자가 같은 테스트 기간으로 여러 공개본의 tau를 재고 최고를 고르면, 그 선택은 테스트 기간의
      우연한 흔들림에 맞춰질 수 있다(winner's curse). 그러면 고른 공개본이 실제로는 다른 것보다 낫지 않을 수 있다.
방법: 비공개 test를 층화 반분(A/B)을 200번 반복한다.
      - 기준 리더보드: 각 탐지기의 시드 평균 예측(leaderboard_real/pred_test_*.npy)으로 A, B에서 각각 PR-AUC.
      - S->S 후보: 합성 test에서 잰 점수라 A/B와 무관(results.json의 시드 평균 PR-AUC).
      - S->R 후보: 후보 리더보드의 pred_test를 같은 A/B로 잘라 잰다.
      - A에서 tau가 가장 큰 공개본을 고르고, B에서 그 공개본의 tau를 무작위 선택(평균)·사후 최선(oracle)과 비교한다.
      생성기별(같은 생성기의 시드 중 선택)과 전체(모든 공개본 중 선택) 두 가지로 본다.

Usage:
    python scripts/release_selection_v2.py
"""
import json
from pathlib import Path

import numpy as np
from scipy.stats import kendalltau
from sklearn.metrics import average_precision_score
from sklearn.model_selection import StratifiedShuffleSplit

import os
# 논문 메인 표/그림은 생성기마다 같은 수의 공개본을 쓴다(사용자 지시 9/20). 기본 5시드.
MAX_SEEDS = int(os.environ.get("FINSYN_MAX_SEEDS", "5"))

ROOT = Path(__file__).resolve().parents[1]
E = ROOT / "exp/finsyn-v2"
DETS = ["nb", "dt", "lr", "knn", "mlp", "rf", "et", "hgb", "lgbm", "xgb", "catboost"]
MODELS = ["smote", "tabpfgen", "tabpfgen-prior", "tabddpm", "great", "tvae", "ctabgan", "ctgan", "ctabgan-plus",
          "tabsyn", "tabdiff", "findiff"]
N_SPLITS = 200


def run_dir(m, tag, s):
    return E / (f"leaderboard_{m}_{tag}" if s == 0 else f"leaderboard_{m}_{tag}_seed{s}")


def runs():
    out = []
    for m in MODELS:
        for s in range(MAX_SEEDS):
            if (run_dir(m, "s2s", s) / "fidelity_vs_leaderboard_real.json").exists() and \
               (run_dir(m, "s2r", s) / "fidelity_vs_leaderboard_real.json").exists():
                out.append((m, s))
    return out


def main():
    y = np.load(ROOT / "data/finsyn-v2/y_test.npy").astype(int)
    real = {d: np.load(E / f"leaderboard_real/pred_test_{d}.npy") for d in DETS}
    R = runs()
    s2s = {r: [json.loads((run_dir(*r[:1], "s2s", r[1]) / "results.json").read_text())[d]["summary"]["pr_auc"][0]
               for d in DETS] for r in R}
    s2r_pred = {r: {d: np.load(run_dir(r[0], "s2r", r[1]) / f"pred_test_{d}.npy") for d in DETS} for r in R}
    sss = StratifiedShuffleSplit(n_splits=N_SPLITS, test_size=0.5, random_state=0)
    rec = {"s2s": [], "s2r": []}
    for a, b in sss.split(np.zeros(len(y)), y):
        ref = {h: [average_precision_score(y[idx], real[d][idx]) for d in DETS] for h, idx in (("A", a), ("B", b))}
        for tag in ("s2s", "s2r"):
            tau = {}
            for r in R:
                for h, idx in (("A", a), ("B", b)):
                    cand = s2s[r] if tag == "s2s" else [average_precision_score(y[idx], s2r_pred[r][d][idx]) for d in DETS]
                    tau[(r, h)] = kendalltau(ref[h], cand).statistic
            rec[tag].append(tau)
    res = {"n_splits": N_SPLITS, "runs": [f"{m}|{s}" for m, s in R]}
    for tag in ("s2s", "s2r"):
        per_gen = {}
        for m in MODELS:
            rs = [r for r in R if r[0] == m]
            if len(rs) < 2:
                continue
            sel, rnd, orc, hit = [], [], [], []
            for tau in rec[tag]:
                pick = max(rs, key=lambda r: tau[(r, "A")])
                tb = {r: tau[(r, "B")] for r in rs}
                sel.append(tb[pick]); rnd.append(np.mean(list(tb.values()))); orc.append(max(tb.values()))
                hit.append(pick == max(rs, key=lambda r: tb[r]))
            q = lambda v: [float(np.percentile(v, 5)), float(np.percentile(v, 95))]
            per_gen[m] = {"n_seeds": len(rs), "tau_B_selected": float(np.mean(sel)), "tau_B_random": float(np.mean(rnd)),
                          "tau_B_oracle": float(np.mean(orc)), "gain_vs_random": float(np.mean(sel) - np.mean(rnd)),
                          "pick_is_B_best": float(np.mean(hit)),
                          "ci_selected": q(sel), "ci_random": q(rnd), "ci_oracle": q(orc)}
        sel, rnd, orc = [], [], []
        for tau in rec[tag]:
            pick = max(R, key=lambda r: tau[(r, "A")])
            tb = {r: tau[(r, "B")] for r in R}
            sel.append(tb[pick]); rnd.append(np.mean(list(tb.values()))); orc.append(max(tb.values()))
        res[tag] = {"per_generator": per_gen,
                    "all_releases": {"tau_B_selected": float(np.mean(sel)), "tau_B_random": float(np.mean(rnd)),
                                     "tau_B_oracle": float(np.mean(orc))}}
    out = E / "release_selection.json"
    out.write_text(json.dumps(res, indent=1))
    for tag in ("s2s", "s2r"):
        print(f"== {tag}: select on half A, evaluate on half B ({N_SPLITS} splits)")
        for m, v in res[tag]["per_generator"].items():
            print(f"  {m:15s} n={v['n_seeds']:2d}  selected {v['tau_B_selected']:.3f}  random {v['tau_B_random']:.3f}  "
                  f"oracle {v['tau_B_oracle']:.3f}  gain {v['gain_vs_random']:+.3f}  pick=B-best {v['pick_is_B_best']:.2f}")
        v = res[tag]["all_releases"]
        print(f"  ALL releases     selected {v['tau_B_selected']:.3f}  random {v['tau_B_random']:.3f}  oracle {v['tau_B_oracle']:.3f}")
    print(f"saved -> {out}")


if __name__ == "__main__":
    main()
