"""
감사 반분 B를 합성 과정에서도 제외했을 때 공개본 선택이 유지되는가(9/23 외부 리뷰 W2 대응, 분석 단계).

생성 단계(scripts/halfA_driver.py, SMOTE는 scripts/synth_v2.py)는 비공개 테스트 기간의 A 절반만으로 각 공개본의
합성 테스트 split을 다시 만들었다. 여기서는
  assemble    : 생성 결과를 모으고(TabPFGen-prior는 A의 양성 비율로 서브샘플), 공개 정책과 같이 비공개 행과
                똑같은 합성 행을 지운다 -> exp/finsyn-v2-halfA/synth_test/<model>/seed<k>/{X_num,X_cat,y}_test.npy
  leaderboards: 원래 공개본의 합성 train/val과 S->S 튜닝 결과(best_params)로 탐지기를 다시 학습해(시드 0..4)
                새 합성 테스트에서 PR-AUC를 잰다 -> exp/finsyn-v2-halfA/boards/<model>_seed<k>.json
  analyse     : 기준 리더보드를 A와 B에서 각각 재고(시드별 PR-AUC의 평균), A에서 가장 높은 tau의 공개본을 골라
                B에서 평가한다. 같은 분할에서 원래 합성 테스트(A와 B를 모두 보고 만든 것)로 한 결과와 나란히 낸다.

A/B 분할은 release_selection_v2.py의 첫 분할(StratifiedShuffleSplit, random_state=0)이며 인덱스는
exp/finsyn-v2-halfA/split_idx_{A,B}.npy에 있다.

Usage:
    python scripts/halfA_analysis_v2.py assemble
    python scripts/halfA_analysis_v2.py leaderboards --jobs 3
    python scripts/halfA_analysis_v2.py analyse
"""
import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import kendalltau

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
E = ROOT / "exp/finsyn-v2"
H = ROOT / "exp/finsyn-v2-halfA"
REAL = ROOT / "data/finsyn-v2"
MODELS = ["smote", "tabpfgen", "tabpfgen-prior", "tabddpm", "great", "tvae", "ctabgan", "ctgan", "ctabgan-plus",
          "tabsyn", "tabdiff", "findiff"]
SEEDS = range(5)


def gen_dir(m, k):
    return H / "gen" / m / ("test" if k == 0 else f"test_seed{k}")


def out_dir(m, k):
    return H / "synth_test" / m / f"seed{k}"


def lb_dir(m, k):
    return E / (f"leaderboard_{m}_s2s" if k == 0 else f"leaderboard_{m}_s2s_seed{k}")


def assemble():
    from copy_filter_v2 import key, load
    from run_generators_v2 import match_prior
    private = set()
    for s in ("train", "val", "test"):
        n, c, _ = load(REAL, s)
        private |= set(key(n, c))
    yA = np.load(REAL / "y_test.npy", allow_pickle=True).astype(int)[np.load(H / "split_idx_A.npy")]
    info = json.loads((REAL / "info.json").read_text())
    report = {}
    for m in MODELS:
        for k in SEEDS:
            g = gen_dir(m, k)
            if not (g / "y_train.npy").exists():
                continue
            Xn = np.load(g / "X_num_train.npy", allow_pickle=True)
            Xc = np.load(g / "X_cat_train.npy", allow_pickle=True)
            y = np.load(g / "y_train.npy", allow_pickle=True).astype(int)
            if m == "tabpfgen-prior":
                Xn, Xc, y = match_prior(Xn, Xc, y, float(yA.mean()), np.random.default_rng(k))
            n = pd.DataFrame(Xn.astype(float), columns=info["num_cols"])
            c = pd.DataFrame(Xc, columns=info["cat_cols"]).astype(str)
            keep = ~key(n, c).isin(private).to_numpy()
            o = out_dir(m, k)
            o.mkdir(parents=True, exist_ok=True)
            np.save(o / "X_num_test.npy", Xn[keep])
            np.save(o / "X_cat_test.npy", Xc[keep], allow_pickle=True)
            np.save(o / "y_test.npy", y[keep])
            report[f"{m}|{k}"] = {"rows": int(keep.sum()), "removed": int((~keep).sum()), "positive_rate": float(y[keep].mean())}
    (H / "assemble_report.json").write_text(json.dumps(report, indent=1))
    print(f"조립 {len(report)}개; 복제 행 제거:", {k: v["removed"] for k, v in report.items() if v["removed"]})


def board(m, k):
    """원래 공개본의 train/val로 학습하고 A에서 만든 새 합성 테스트로 평가한 S->S 리더보드."""
    import optuna
    from sklearn.metrics import average_precision_score
    from leaderboard import Prep, fit_predict, load_split, space
    syn = E / f"synth/{m}/seed{k}"
    Xtr, ytr, info = load_split(str(syn), "train")
    Xva, yva, _ = load_split(str(syn), "val")
    o = out_dir(m, k)
    num = pd.DataFrame(np.load(o / "X_num_test.npy", allow_pickle=True).astype(float), columns=info["num_cols"])
    cat = pd.DataFrame(np.load(o / "X_cat_test.npy", allow_pickle=True), columns=info["cat_cols"]).astype(str)
    Xte, yte = pd.concat([num, cat], axis=1), np.load(o / "y_test.npy", allow_pickle=True).astype(int)
    prep = Prep(info, Xtr)
    res = json.loads((lb_dir(m, k) / "results.json").read_text())
    scores = {}
    for d, r in res.items():
        if not yte.any():
            scores[d] = float("nan")
            continue
        if r.get("best_params") is None or r.get("degenerate"):
            scores[d] = float(average_precision_score(yte, np.full(len(yte), ytr.mean())))
            continue
        best = space(d, optuna.trial.FixedTrial(r["best_params"]))
        s = []
        for run in r["runs"]:
            _, pt = fit_predict(d, best, prep, Xtr, ytr, Xva, yva, Xte, seed=run["seed"])
            s.append(float(average_precision_score(yte, pt)))
        scores[d] = float(np.mean(s))
    (H / "boards").mkdir(parents=True, exist_ok=True)
    (H / "boards" / f"{m}_seed{k}.json").write_text(json.dumps({"scores": scores, "n_test": int(len(yte)),
                                                                 "positives": int(yte.sum())}, indent=1))
    print(f"{m} seed{k}: done", flush=True)


def leaderboards(jobs):
    todo = [(m, k) for m in MODELS for k in SEEDS
            if (out_dir(m, k) / "y_test.npy").exists() and not (H / "boards" / f"{m}_seed{k}.json").exists()]
    (H / "logs").mkdir(parents=True, exist_ok=True)
    running, t0 = [], time.time()
    print(f"리더보드 {len(todo)}개", flush=True)
    while todo or running:
        while todo and len(running) < jobs:
            m, k = todo.pop(0)
            fh = open(H / "logs" / f"board_{m}_seed{k}.txt", "w")
            running.append(((m, k), fh, subprocess.Popen([sys.executable, "-u", __file__, "board", m, str(k)],
                                                         cwd=ROOT, stdout=fh, stderr=subprocess.STDOUT)))
        time.sleep(15)
        for it in running[:]:
            (m, k), fh, p = it
            if p.poll() is not None:
                fh.close(); running.remove(it)
                print(f"[{(time.time() - t0) / 60:5.1f}분] {m} seed{k} rc={p.returncode} (남은 {len(todo)})", flush=True)


def analyse():
    from leaderboard import board_score, seed_preds
    y = np.load(REAL / "y_test.npy", allow_pickle=True).astype(int)
    a, b = np.load(H / "split_idx_A.npy"), np.load(H / "split_idx_B.npy")
    dets = sorted(json.loads((E / "leaderboard_real/results.json").read_text()))
    P = {d: seed_preds(E / "leaderboard_real", d) for d in dets}
    refA = [board_score(y[a], P[d][:, a]) for d in dets]
    refB = [board_score(y[b], P[d][:, b]) for d in dets]
    out = {"split": "StratifiedShuffleSplit(n_splits=200, test_size=0.5, random_state=0), first split",
           "n_A": int(len(a)), "n_B": int(len(b))}
    for tag in ("B_excluded", "original"):
        tA, tB = {}, {}
        for m in MODELS:
            for k in SEEDS:
                if tag == "B_excluded":
                    f = H / "boards" / f"{m}_seed{k}.json"
                    if not f.exists():
                        continue
                    sc = json.loads(f.read_text())["scores"]
                else:
                    sc = {d: v["summary"]["pr_auc"][0] for d, v in json.loads((lb_dir(m, k) / "results.json").read_text()).items()}
                v = [sc[d] for d in dets]
                if any(x != x for x in v):
                    continue
                tA[(m, k)] = kendalltau(refA, v).statistic
                tB[(m, k)] = kendalltau(refB, v).statistic
        gens = [m for m in MODELS if sum(1 for r in tA if r[0] == m) >= 2]
        pg = {}
        for m in gens:
            ks = [r for r in tA if r[0] == m]
            pick = max(ks, key=tA.get)
            pg[m] = {"selected": float(tB[pick]), "random": float(np.mean([tB[r] for r in ks])),
                     "oracle": float(max(tB[r] for r in ks)), "n": len(ks)}
        sel, rnd, orc = (float(np.mean([v[x] for v in pg.values()])) for x in ("selected", "random", "oracle"))
        pick = max(tA, key=tA.get)
        out[tag] = {"n_releases": len(tA), "n_generators": len(gens),
                    "within_generator": {"selected": sel, "random": rnd, "oracle": orc,
                                         "share_of_gap_recovered": (sel - rnd) / (orc - rnd) if orc > rnd else float("nan")},
                    "all_releases": {"selected": float(tB[pick]), "random": float(np.mean(list(tB.values()))),
                                     "oracle": float(max(tB.values())), "pick": f"{pick[0]}|{pick[1]}"},
                    "per_generator": pg}
        print(tag, json.dumps({k: out[tag][k] for k in ("n_releases", "within_generator", "all_releases")}), flush=True)
    (H / "halfA_analysis.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["assemble", "leaderboards", "analyse", "board"])
    ap.add_argument("m", nargs="?")
    ap.add_argument("k", nargs="?", type=int)
    ap.add_argument("--jobs", type=int, default=3)
    a = ap.parse_args()
    if a.stage == "board":
        board(a.m, a.k)
    else:
        {"assemble": assemble, "leaderboards": lambda: leaderboards(a.jobs), "analyse": analyse}[a.stage]()
