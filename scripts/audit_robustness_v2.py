"""
감사의 견고성 추가 점검(9/24 외부 리뷰 W3·W4·Q2·Q4 대응). 결과: exp/finsyn-v2/audit_robustness.json

  time_split : 테스트 기간을 날짜로 나눠 앞 절반(2024년 7~9월)에서 공개본을 고르고 뒤 절반(10~12월)에서 평가한다.
               반대 방향도 낸다. 무작위 반분은 같은 기간을 섞지만, 이것은 선택이 뒤 기간으로 옮겨 가는지를 본다.
               날짜는 기관 내부의 ID 포함 파일에서 평가 배열과 같은 순서로 읽는다(공개하지 않음).
  sample_size: 비공개 테스트 기간의 25/50/75/100%만 있을 때(층화 부분표본) 노이즈 상한·밴드 하한·분리 쌍과
               감사 이득(부분표본의 층화 반분 100회)이 어떻게 변하는지 본다. 감사에 필요한 비공개 표본 규모의 감각.
  power      : 생성기 안 순열검정(within_generator_v2.py)의 검정력. 실제 공개본 tau에 상관 rho로 연결된 가상 지표를
               만들어, 생성기 12종 x 시드 5개 구조에서 5% 수준으로 유의해지는 비율을 잰다.

Usage:
    python scripts/audit_robustness_v2.py
"""
import itertools
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import kendalltau
from sklearn.model_selection import StratifiedShuffleSplit

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from leaderboard import bootstrap_noise, board_score, load_split, seed_preds  # noqa: E402

E = ROOT / "exp/finsyn-v2"
MAX_SEEDS = int(os.environ.get("FINSYN_MAX_SEEDS", "5"))
MODELS = ["smote", "tabpfgen", "tabpfgen-prior", "tabddpm", "great", "tvae", "ctabgan", "ctgan", "ctabgan-plus",
          "tabsyn", "tabdiff", "findiff"]


def runs(dets):
    out = {}
    for m in MODELS:
        for s in range(MAX_SEEDS):
            f = E / (f"leaderboard_{m}_s2s" if s == 0 else f"leaderboard_{m}_s2s_seed{s}") / "results.json"
            r = json.loads(f.read_text())
            out[(m, s)] = [r[d]["summary"]["pr_auc"][0] for d in dets]
    return out


def select(refA, refB, R):
    """A에서 고르고 B에서 평가: 생성기 안(평균)과 60개 전체."""
    tA = {k: kendalltau(refA, v).statistic for k, v in R.items()}
    tB = {k: kendalltau(refB, v).statistic for k, v in R.items()}
    g = []
    for m in MODELS:
        ks = [k for k in R if k[0] == m]
        g.append((tB[max(ks, key=tA.get)], np.mean([tB[k] for k in ks]), max(tB[k] for k in ks)))
    g = np.mean(g, axis=0)
    pick = max(R, key=tA.get)
    return {"within_generator": {"selected": float(g[0]), "random": float(g[1]), "oracle": float(g[2]),
                                 "share_of_gap_recovered": float((g[0] - g[1]) / (g[2] - g[1]))},
            "all_releases": {"selected": float(tB[pick]), "random": float(np.mean(list(tB.values()))),
                             "oracle": float(max(tB.values()))}}


def time_split(y, P, dets, R):
    df = pd.read_parquet(ROOT / "data/finsyn-v2/full_with_ids_DO_NOT_RELEASE.parquet", columns=["거래일자", "y", "split"])
    te = df[df["split"] == "test"].reset_index(drop=True)
    assert np.array_equal(te["y"].to_numpy(int), y)
    early = (te["거래일자"].astype(str) <= "20240930").to_numpy()
    ref = {h: [board_score(y[idx], P[d][:, idx]) for d in dets] for h, idx in (("early", early), ("late", ~early))}
    out = {"n_early": int(early.sum()), "pos_early": int(y[early].sum()),
           "n_late": int((~early).sum()), "pos_late": int(y[~early].sum()),
           "early_to_late": select(ref["early"], ref["late"], R),
           "late_to_early": select(ref["late"], ref["early"], R)}
    return out


def sample_size(y, P, dets, R, rng):
    out = {}
    for frac in (0.25, 0.5, 0.75, 1.0):
        if frac < 1.0:
            sub, _ = next(StratifiedShuffleSplit(n_splits=1, train_size=frac, random_state=0).split(np.zeros(len(y)), y))
        else:
            sub = np.arange(len(y))
        ys = y[sub]
        Ps = {d: P[d][:, sub] for d in dets}
        nz = bootstrap_noise(ys, Ps, n_boot=500)
        sep = sum(1 for v in nz["pairwise_win_prob"].values() if v >= 0.975 or v <= 0.025)
        acc = []
        for a, b in StratifiedShuffleSplit(n_splits=100, test_size=0.5, random_state=0).split(np.zeros(len(ys)), ys):
            refA = [board_score(ys[a], Ps[d][:, a]) for d in dets]
            refB = [board_score(ys[b], Ps[d][:, b]) for d in dets]
            acc.append(select(refA, refB, R)["within_generator"])
        m = {k: float(np.mean([x[k] for x in acc])) for k in ("selected", "random", "oracle")}
        m["share_of_gap_recovered"] = (m["selected"] - m["random"]) / (m["oracle"] - m["random"])
        out[str(frac)] = {"n": int(len(sub)), "positives": int(ys.sum()), "noise_ceiling": nz["tau_vs_full_mean"],
                          "band_q05": nz["tau_vs_full_q05"], "separable_pairs": sep, "audit_within_generator": m}
        print("sample", frac, json.dumps(out[str(frac)]), flush=True)
    return out


def power(rng, n_sim=300, n_perm=1000):
    """상관 rho의 가상 지표로, 실제 tau 구조에서 순열검정의 검정력과 기대 일치율을 잰다."""
    lf = json.loads((E / "lf_analysis.json").read_text())["releases"]
    T = np.array([lf[m]["s2s"]["tau"][:MAX_SEEDS] for m in MODELS])  # (생성기, 시드)
    pairs = list(itertools.combinations(range(T.shape[1]), 2))
    ia, ib = np.array([p[0] for p in pairs]), np.array([p[1] for p in pairs])
    dt = np.sign(T[:, ia] - T[:, ib])                                  # (생성기, 쌍)
    valid = dt != 0
    Z = (T - T.mean(1, keepdims=True)) / (T.std(1, keepdims=True) + 1e-12)

    def agree(C):  # C: (..., 생성기, 시드)
        dm = np.sign(C[..., ia] - C[..., ib])
        hit = (dm == dt) & valid & (dm != 0)
        return hit.sum(axis=(-1, -2)) / ((dm != 0) & valid).sum(axis=(-1, -2))

    out = {}
    for rho in (0.0, 0.2, 0.4, 0.6, 0.8):
        sig, rates = 0, []
        for _ in range(n_sim):
            C = rho * Z + np.sqrt(1 - rho ** 2) * rng.standard_normal(Z.shape)
            obs = agree(C)
            perm = np.argsort(rng.random((n_perm,) + C.shape), axis=-1)
            null = agree(np.take_along_axis(np.broadcast_to(C, perm.shape), perm, axis=-1))
            p = float((np.abs(null - 0.5) >= abs(obs - 0.5) - 1e-12).mean())
            sig += p < 0.05
            rates.append(obs)
        out[str(rho)] = {"power_at_0.05": sig / n_sim, "mean_agreement": float(np.mean(rates))}
        print("power rho", rho, out[str(rho)], flush=True)
    return out


def main():
    rng = np.random.default_rng(0)
    _, y, _ = load_split(str(ROOT / "data/finsyn-v2"), "test")
    dets = sorted(json.loads((E / "leaderboard_real/results.json").read_text()))
    P = {d: seed_preds(E / "leaderboard_real", d) for d in dets}
    R = runs(dets)
    res = {"time_split": time_split(y, P, dets, R)}
    print("time", json.dumps(res["time_split"]), flush=True)
    res["power"] = power(rng)
    res["sample_size"] = sample_size(y, P, dets, R, rng)
    (E / "audit_robustness.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
