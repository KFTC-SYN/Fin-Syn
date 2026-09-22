"""
가장 가까운 선행 연구(van Breugel et al., ICML'23)의 처방과 우리 처방을 같은 자로 비교한다.

그들의 처방: 한 생성기의 여러 실행을 모아 평균 낸다(비공개 데이터가 필요 없다).
우리 처방:   비공개 데이터로 tau를 재서 가장 좋은 실행을 고른다.

프로토콜은 release_selection_v2.py와 같다. 비공개 test를 층화 반분해 한쪽(A)에서 고르고
다른 쪽(B)에서 평가한다. 앙상블은 고를 것이 없으므로 B에서 바로 잰다.

Usage:
    python scripts/ensemble_vs_audit_v2.py
"""
import json
import os
from pathlib import Path

import numpy as np
from scipy.stats import kendalltau
from sklearn.metrics import average_precision_score
from sklearn.model_selection import StratifiedShuffleSplit

ROOT = Path(__file__).resolve().parents[1]
E = ROOT / "exp/finsyn-v2"
MAX_SEEDS = int(os.environ.get("FINSYN_MAX_SEEDS", "5"))
N_SPLITS = 200
GEN = ["smote", "tabsyn", "tabdiff", "findiff", "tabddpm", "tabpfgen", "tabpfgen-prior",
       "great", "tvae", "ctgan", "ctabgan", "ctabgan-plus"]


def main():
    dets = sorted(json.loads((E / "leaderboard_real/results.json").read_text()))
    y = np.load(ROOT / "data/finsyn-v2/y_test.npy").astype(int)
    real = {d: np.load(E / f"leaderboard_real/pred_test_{d}.npy") for d in dets}
    runs = {}
    for m in GEN:
        v = []
        for s in range(MAX_SEEDS):
            f = E / (f"leaderboard_{m}_s2s" if s == 0 else f"leaderboard_{m}_s2s_seed{s}") / "results.json"
            if f.exists():
                r = json.loads(f.read_text())
                v.append([r[d]["summary"]["pr_auc"][0] for d in dets])
        runs[m] = v

    acc = {k: {m: [] for m in GEN} for k in ("random", "ensemble", "selected", "oracle")}
    for a, b in StratifiedShuffleSplit(n_splits=N_SPLITS, test_size=0.5, random_state=0).split(np.zeros(len(y)), y):
        refA = [average_precision_score(y[a], real[d][a]) for d in dets]
        refB = [average_precision_score(y[b], real[d][b]) for d in dets]
        for m, vs in runs.items():
            tA = [kendalltau(refA, v).statistic for v in vs]
            tB = [kendalltau(refB, v).statistic for v in vs]
            acc["random"][m].append(float(np.mean(tB)))
            acc["ensemble"][m].append(float(kendalltau(refB, np.mean(vs, axis=0)).statistic))
            acc["selected"][m].append(float(tB[int(np.argmax(tA))]))
            acc["oracle"][m].append(float(max(tB)))

    out = {"n_splits": N_SPLITS, "n_generators": len(GEN), "seeds_per_generator": MAX_SEEDS}
    for k in ("random", "ensemble", "selected", "oracle"):
        out[k] = {"overall": float(np.mean([np.mean(acc[k][m]) for m in GEN])),
                  "per_generator": {m: float(np.mean(acc[k][m])) for m in GEN}}
    gap = out["oracle"]["overall"] - out["random"]["overall"]
    out["share_of_gap_recovered"] = {
        "ensemble": (out["ensemble"]["overall"] - out["random"]["overall"]) / gap,
        "selected": (out["selected"]["overall"] - out["random"]["overall"]) / gap}
    (E / "ensemble_vs_audit.json").write_text(json.dumps(out, indent=1))
    for k in ("random", "ensemble", "selected", "oracle"):
        print(f"  {k:10s} {out[k]['overall']:+.3f}")
    print(f"  회수 비율: 앙상블 {out['share_of_gap_recovered']['ensemble']:.0%}, "
          f"감사 {out['share_of_gap_recovered']['selected']:.0%}")


if __name__ == "__main__":
    main()
