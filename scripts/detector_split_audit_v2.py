"""
감사가 감사에 쓰지 않은 탐지기에도 통하는가(9/23 외부 리뷰 W2·Q4 대응).

탐지기 11개를 선택용 6개와 나머지 5개로 나누는 모든 경우(462가지)에 대해,
  - 선택: 선택용 6개끼리의 15쌍에서 계산한 Kendall tau(공개본 리더보드 대 기준 리더보드)로 공개본을 고르고,
  - 평가: 나머지 탐지기가 하나라도 들어간 40쌍에서 같은 방식의 순위 일치도를 잰다.
선택에 쓴 쌍과 평가에 쓴 쌍이 겹치지 않으므로, 고른 공개본이 감사에 없던 탐지기와의 비교에서도 나은지 본다.
일치도는 tau_a 형태((일치 - 불일치) / 쌍 수)이며, 동점 쌍은 0으로 센다.

결과: exp/finsyn-v2/detector_split_audit.json

Usage:
    python scripts/detector_split_audit_v2.py
"""
import itertools
import json
import os
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
E = ROOT / "exp/finsyn-v2"
MAX_SEEDS = int(os.environ.get("FINSYN_MAX_SEEDS", "5"))
MODELS = ["smote", "tabpfgen", "tabpfgen-prior", "tabddpm", "great", "tvae", "ctabgan", "ctgan", "ctabgan-plus",
          "tabsyn", "tabdiff", "findiff"]
N_SELECT = 6


def agreement(ref, cand, pairs):
    s = [np.sign(ref[i] - ref[j]) * np.sign(cand[i] - cand[j]) for i, j in pairs]
    return float(np.mean(s))


def main():
    real = json.loads((E / "leaderboard_real/results.json").read_text())
    dets = sorted(real)
    ref = np.array([real[d]["summary"]["pr_auc"][0] for d in dets])
    runs = {}
    for m in MODELS:
        for s in range(MAX_SEEDS):
            f = E / (f"leaderboard_{m}_s2s" if s == 0 else f"leaderboard_{m}_s2s_seed{s}") / "results.json"
            r = json.loads(f.read_text())
            runs[(m, s)] = np.array([r[d]["summary"]["pr_auc"][0] for d in dets])
    all_pairs = list(itertools.combinations(range(len(dets)), 2))
    rec = {"all": [], "gen": []}
    for S in itertools.combinations(range(len(dets)), N_SELECT):
        sel_pairs = [p for p in all_pairs if p[0] in S and p[1] in S]
        ev_pairs = [p for p in all_pairs if not (p[0] in S and p[1] in S)]
        a = {k: agreement(ref, v, sel_pairs) for k, v in runs.items()}
        b = {k: agreement(ref, v, ev_pairs) for k, v in runs.items()}
        pick = max(runs, key=a.get)
        rec["all"].append((b[pick], np.mean(list(b.values())), max(b.values())))
        g = []
        for m in MODELS:
            ks = [k for k in runs if k[0] == m]
            g.append((b[max(ks, key=a.get)], np.mean([b[k] for k in ks]), max(b[k] for k in ks)))
        rec["gen"].append(np.mean(g, axis=0))
    out = {"n_splits": len(rec["all"]), "n_select_detectors": N_SELECT,
           "n_select_pairs": len(sel_pairs), "n_eval_pairs": len(ev_pairs)}
    for k in ("all", "gen"):
        v = np.array(rec[k])
        sel, rnd, orc = v.mean(axis=0)
        out["all_releases" if k == "all" else "within_generator"] = {
            "selected": float(sel), "random": float(rnd), "oracle": float(orc),
            "share_of_gap_recovered": float((sel - rnd) / (orc - rnd)),
            "selected_beats_random_share_of_splits": float(np.mean(v[:, 0] > v[:, 1]))}
    (E / "detector_split_audit.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
