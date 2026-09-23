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
import sys
from pathlib import Path

import numpy as np
from scipy.stats import kendalltau
from sklearn.model_selection import StratifiedShuffleSplit

ROOT = Path(__file__).resolve().parents[1]
E = ROOT / "exp/finsyn-v2"
MAX_SEEDS = int(os.environ.get("FINSYN_MAX_SEEDS", "5"))
N_SPLITS = 200
sys.path.insert(0, str(ROOT / "scripts"))
from leaderboard import board_score, seed_preds  # noqa: E402
GEN = ["smote", "tabsyn", "tabdiff", "findiff", "tabddpm", "tabpfgen", "tabpfgen-prior",
       "great", "tvae", "ctgan", "ctabgan", "ctabgan-plus"]


def main():
    dets = sorted(json.loads((E / "leaderboard_real/results.json").read_text()))
    y = np.load(ROOT / "data/finsyn-v2/y_test.npy").astype(int)
    real = {d: seed_preds(E / "leaderboard_real", d) for d in dets}
    assert all(v.ndim == 2 for v in real.values()), "시드별 예측이 없다: scripts/refit_seed_preds.py 먼저"
    runs = {}
    for m in GEN:
        v = []
        for s in range(MAX_SEEDS):
            f = E / (f"leaderboard_{m}_s2s" if s == 0 else f"leaderboard_{m}_s2s_seed{s}") / "results.json"
            if f.exists():
                r = json.loads(f.read_text())
                v.append([r[d]["summary"]["pr_auc"][0] for d in dets])
        runs[m] = v

    # DGE(van Breugel et al. 권고 (3)): 공개본 k로 학습해 같은 생성기의 다른 공개본 j의 test로 잰 점수를
    # k != j 20개 조합에 대해 평균한 리더보드. scripts/dge_cross_eval_v2.py 결과가 있어야 한다.
    dge = {}
    for m in GEN:
        fs = [E / f"dge/{m}_seed{k}.json" for k in range(MAX_SEEDS)]
        if all(f.exists() for f in fs):
            sc = [json.loads(f.read_text())["scores"] for f in fs]
            dge[m] = [float(np.nanmean([sc[k][str(j)][d] for k in range(MAX_SEEDS) for j in range(MAX_SEEDS) if j != k]))
                      for d in dets]
    # 같은 교차 평가를, 다른 네 공개본의 test를 이어 붙인 하나의 평가 집합에서 잰 변형(dge_pooled/, 9/24 리뷰 W5)
    dge_pooled = {}
    for m in GEN:
        fs = [E / f"dge_pooled/{m}_seed{k}.json" for k in range(MAX_SEEDS)]
        if all(f.exists() for f in fs):
            sc = [json.loads(f.read_text())["pooled"] for f in fs]
            dge_pooled[m] = [float(np.nanmean([sc[k][d] for k in range(MAX_SEEDS)])) for d in dets]
    kinds = ("random", "ensemble", "dge", "selected", "oracle") if len(dge) == len(GEN) else ("random", "ensemble", "selected", "oracle")
    if len(dge_pooled) == len(GEN):
        kinds = kinds[:3] + ("dge_pooled",) + kinds[3:]
    acc = {k: {m: [] for m in GEN} for k in kinds}
    for a, b in StratifiedShuffleSplit(n_splits=N_SPLITS, test_size=0.5, random_state=0).split(np.zeros(len(y)), y):
        refA = [board_score(y[a], real[d][:, a]) for d in dets]
        refB = [board_score(y[b], real[d][:, b]) for d in dets]
        for m, vs in runs.items():
            tA = [kendalltau(refA, v).statistic for v in vs]
            tB = [kendalltau(refB, v).statistic for v in vs]
            acc["random"][m].append(float(np.mean(tB)))
            acc["ensemble"][m].append(float(kendalltau(refB, np.mean(vs, axis=0)).statistic))
            if "dge" in acc:
                acc["dge"][m].append(float(kendalltau(refB, dge[m]).statistic))
            if "dge_pooled" in acc:
                acc["dge_pooled"][m].append(float(kendalltau(refB, dge_pooled[m]).statistic))
            acc["selected"][m].append(float(tB[int(np.argmax(tA))]))
            acc["oracle"][m].append(float(max(tB)))

    out = {"n_splits": N_SPLITS, "n_generators": len(GEN), "seeds_per_generator": MAX_SEEDS}
    for k in kinds:
        out[k] = {"overall": float(np.mean([np.mean(acc[k][m]) for m in GEN])),
                  "per_generator": {m: float(np.mean(acc[k][m])) for m in GEN}}
    gap = out["oracle"]["overall"] - out["random"]["overall"]
    out["share_of_gap_recovered"] = {k: (out[k]["overall"] - out["random"]["overall"]) / gap
                                     for k in kinds if k not in ("random", "oracle")}
    (E / "ensemble_vs_audit.json").write_text(json.dumps(out, indent=1))
    for k in kinds:
        print(f"  {k:10s} {out[k]['overall']:+.3f}")
    print("  회수 비율:", {k: f"{v:.0%}" for k, v in out["share_of_gap_recovered"].items()})


if __name__ == "__main__":
    main()
