"""
노이즈 기준선과 감사가 거래 간 의존성에 얼마나 민감한가(9/23 외부 리뷰 W3 대응).

기존 부트스트랩과 감사 반분은 거래 행 단위다. 같은 송금 계좌의 거래와 같은 주의 거래는 서로 독립이 아니므로,
행 대신 계좌 또는 주(ISO 주)를 단위로 재표집하면 기준선과 감사 이득이 얼마나 달라지는지 본다.

  1. 노이즈 기준선: 계좌 / 주를 복원추출한 1,000개 표본에서 리더보드(시드별 PR-AUC의 평균)를 다시 재고
     전체 표본 리더보드와의 Kendall tau 분포(평균, 5% 분위)와 분리 가능한 쌍 수를 낸다.
  2. 감사: 테스트 기간을 계좌 / 주 단위로 반분(200회)해 한쪽에서 공개본을 고르고 다른 쪽에서 평가한다(S->S).

계좌와 날짜는 기관 내부의 ID 포함 파일(data/finsyn-v2/full_with_ids_DO_NOT_RELEASE.parquet, 공개하지 않음)에서
평가 배열과 같은 순서로 읽는다. 결과 파일에는 집계 수치만 남긴다.

결과: exp/finsyn-v2/cluster_bootstrap.json

Usage:
    python scripts/cluster_bootstrap_v2.py
"""
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import kendalltau

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from leaderboard import board_score, load_split, seed_preds  # noqa: E402

E = ROOT / "exp/finsyn-v2"
MAX_SEEDS = int(os.environ.get("FINSYN_MAX_SEEDS", "5"))
MODELS = ["smote", "tabpfgen", "tabpfgen-prior", "tabddpm", "great", "tvae", "ctabgan", "ctgan", "ctabgan-plus",
          "tabsyn", "tabdiff", "findiff"]
N_BOOT, N_SPLITS = 1000, 200


def units():
    df = pd.read_parquet(ROOT / "data/finsyn-v2/full_with_ids_DO_NOT_RELEASE.parquet",
                         columns=["거래일자", "출금계좌일련번호", "y", "split"])
    te = df[df["split"] == "test"].reset_index(drop=True)
    _, y, _ = load_split(str(ROOT / "data/finsyn-v2"), "test")
    assert np.array_equal(te["y"].to_numpy(int), y), "ID 파일과 평가 배열의 행 순서가 다르다"
    week = pd.to_datetime(te["거래일자"], format="%Y%m%d").dt.isocalendar()
    return y, {"account": pd.factorize(te["출금계좌일련번호"])[0],
               "week": pd.factorize(week["year"].astype(str) + "-" + week["week"].astype(str))[0]}


def resample_rows(by, rng):
    """단위를 복원추출하고, 뽑힌 단위의 행을 모두 모은다. by[i]는 단위 i의 행 번호."""
    pick = rng.integers(0, len(by), len(by))
    return np.concatenate([by[i] for i in pick])


def noise(y, P, dets, groups, rng):
    full = np.array([board_score(y, P[d]) for d in dets])
    boots = np.empty((N_BOOT, len(dets)))
    by = None if groups is None else [np.where(groups == g)[0] for g in np.unique(groups)]
    for b in range(N_BOOT):
        idx = resample_rows(by, rng) if groups is not None else np.concatenate(
            [rng.choice(np.where(y == 1)[0], (y == 1).sum()), rng.choice(np.where(y == 0)[0], (y == 0).sum())])
        boots[b] = [board_score(y[idx], P[d][:, idx]) for d in dets]
    taus = np.array([kendalltau(full, bb).statistic for bb in boots])
    sep = 0
    for i in range(len(dets)):
        for j in range(i + 1, len(dets)):
            d = boots[:, i] - boots[:, j]
            w = (d > 0).mean() + 0.5 * (d == 0).mean()
            sep += int(w >= 0.975 or w <= 0.025)
    return {"tau_mean": float(np.nanmean(taus)), "tau_q05": float(np.nanquantile(taus, 0.05)), "separable_pairs": sep}


def audit(y, P, dets, groups, rng):
    runs = {}
    for m in MODELS:
        for s in range(MAX_SEEDS):
            f = E / (f"leaderboard_{m}_s2s" if s == 0 else f"leaderboard_{m}_s2s_seed{s}") / "results.json"
            if f.exists():
                r = json.loads(f.read_text())
                runs[(m, s)] = [r[d]["summary"]["pr_auc"][0] for d in dets]
    ids = np.unique(groups)
    sel, rnd, orc, per_gen = [], [], [], {m: [] for m in MODELS}
    for _ in range(N_SPLITS):
        half = rng.permutation(ids)[: len(ids) // 2]
        a = np.isin(groups, half)
        refA = [board_score(y[a], P[d][:, a]) for d in dets]
        refB = [board_score(y[~a], P[d][:, ~a]) for d in dets]
        tA = {k: kendalltau(refA, v).statistic for k, v in runs.items()}
        tB = {k: kendalltau(refB, v).statistic for k, v in runs.items()}
        pick = max(runs, key=tA.get)
        sel.append(tB[pick]); rnd.append(np.mean(list(tB.values()))); orc.append(max(tB.values()))
        for m in MODELS:
            ks = [k for k in runs if k[0] == m]
            per_gen[m].append((tB[max(ks, key=tA.get)], np.mean([tB[k] for k in ks]), max(tB[k] for k in ks)))
    pg = np.array([np.mean(v, axis=0) for v in per_gen.values()])  # (생성기, [선택, 무작위, 사후최선])
    gap = pg[:, 2].mean() - pg[:, 1].mean()
    return {"all_releases": {"selected": float(np.mean(sel)), "random": float(np.mean(rnd)), "oracle": float(np.mean(orc))},
            "within_generator": {"selected": float(pg[:, 0].mean()), "random": float(pg[:, 1].mean()),
                                 "oracle": float(pg[:, 2].mean()),
                                 "share_of_gap_recovered": float((pg[:, 0].mean() - pg[:, 1].mean()) / gap)}}


def main():
    y, U = units()
    dets = sorted(json.loads((E / "leaderboard_real/results.json").read_text()))
    P = {d: seed_preds(E / "leaderboard_real", d) for d in dets}
    assert all(p.ndim == 2 for p in P.values())
    out = {"n_units": {k: int(len(np.unique(v))) for k, v in U.items()}}
    for name, groups in [("row", None), ("account", U["account"]), ("week", U["week"])]:
        rng = np.random.default_rng(0)
        out[name] = {"noise": noise(y, P, dets, groups, rng)}
        if groups is not None:
            out[name]["audit"] = audit(y, P, dets, groups, np.random.default_rng(0))
        print(name, json.dumps(out[name]), flush=True)
    (E / "cluster_bootstrap.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
