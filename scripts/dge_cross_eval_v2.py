"""
van Breugel et al.(ICML'23)의 DGE 교차 평가를 우리 공개본에 그대로 적용한다(9/23 외부 리뷰 W1 대응).

그들의 모델 평가·선택 권고: 합성 데이터셋 하나로 학습하고 평가하지 말고, 어떤 합성 데이터셋으로 학습해
다른 합성 데이터셋으로 평가하라(§4.2, 권고 (3)). 우리 감사 비교의 '리더보드 평균'은 이 교차 평가가 빠진
단순화였다. 여기서는 같은 생성기의 공개본 k로 학습한 탐지기를 공개본 j(j != k)의 합성 테스트로 평가한다.

  학습: 공개본 k의 합성 train, 조기 종료는 합성 val, 설정은 그 공개본 S->S 리더보드의 튜닝 결과(best_params),
        시드 0..4 (leaderboard.py와 같은 절차, 튜닝은 다시 하지 않는다)
  평가: 같은 생성기 공개본 다섯 개의 합성 test 전부. k == j는 기존 S->S 점수를 재현해야 한다(재현 점검).

결과: exp/finsyn-v2/dge/<generator>_seed<k>.json  {"scores": {j: {detector: 시드 평균 PR-AUC}}, "self_check": ...}
      --pooled 를 주면 exp/finsyn-v2/dge_pooled/ 에 "pooled": {detector: ...}도 쓴다. k가 아닌 네 공개본의 test를
      이어 붙인 하나의 평가 집합에서 시드별 PR-AUC를 잰 평균이다. van Breugel et al.이 설명하는 "다른 합성 데이터셋들에서의
      평가"는 공개본별 점수의 평균과 일반적으로 같지 않으므로 두 방식을 모두 낸다(9/24 외부 리뷰 W5).
그 뒤 ensemble_vs_audit_v2.py가 DGE 리더보드(k != j 점수의 평균)를 감사·평균과 같은 반분 프로토콜로 비교한다.

Usage:
    python scripts/dge_cross_eval_v2.py --jobs 3
"""
import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
E = ROOT / "exp/finsyn-v2"
OUT = E / "dge"
POOLED = "--pooled" in sys.argv
if POOLED:
    OUT = E / "dge_pooled"
MODELS = ["smote", "tabpfgen", "tabpfgen-prior", "tabddpm", "great", "tvae", "ctabgan", "ctgan", "ctabgan-plus",
          "tabsyn", "tabdiff", "findiff"]
SEEDS = range(5)


def lb_dir(m, s):
    return E / (f"leaderboard_{m}_s2s" if s == 0 else f"leaderboard_{m}_s2s_seed{s}")


def one(m, k):
    import optuna
    from sklearn.metrics import average_precision_score
    from leaderboard import Prep, fit_predict, load_split, space
    syn = E / f"synth/{m}/seed{k}"
    Xtr, ytr, info = load_split(str(syn), "train")
    Xva, yva, _ = load_split(str(syn), "val")
    tests = {j: load_split(str(E / f"synth/{m}/seed{j}"), "test")[:2] for j in SEEDS}
    import pandas as pd
    Xte = pd.concat([tests[j][0] for j in SEEDS], ignore_index=True)
    cuts = np.cumsum([0] + [len(tests[j][1]) for j in SEEDS])
    prep = Prep(info, Xtr)
    res = json.loads((lb_dir(m, k) / "results.json").read_text())
    scores = {j: {} for j in SEEDS}
    pooled = {}
    others = [j for j in SEEDS if j != k]
    y_pool = np.concatenate([tests[j][1] for j in others])
    check = {}
    for d, r in res.items():
        runs = r["runs"]
        if r.get("best_params") is None or r.get("degenerate"):
            # 튜닝이 전부 실패한 탐지기는 leaderboard.py와 같이 학습 양성 비율 상수 예측으로 둔다
            per = {j: [average_precision_score(tests[j][1], np.full(len(tests[j][1]), ytr.mean()))
                       if tests[j][1].any() else float("nan") for _ in runs] for j in SEEDS}
            pool = [float(average_precision_score(y_pool, np.full(len(y_pool), ytr.mean())))]
        else:
            best = space(d, optuna.trial.FixedTrial(r["best_params"]))
            per = {j: [] for j in SEEDS}
            pool = []
            for run in runs:
                _, pt = fit_predict(d, best, prep, Xtr, ytr, Xva, yva, Xte, seed=run["seed"])
                pool.append(float(average_precision_score(y_pool, np.concatenate([pt[cuts[j]:cuts[j + 1]] for j in others]))))
                for j in SEEDS:
                    yj = tests[j][1]
                    per[j].append(float(average_precision_score(yj, pt[cuts[j]:cuts[j + 1]])) if yj.any() else float("nan"))
        for j in SEEDS:
            scores[j][d] = float(np.mean(per[j]))
        pooled[d] = float(np.mean(pool))
        check[d] = abs(scores[k][d] - r["summary"]["pr_auc"][0])
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"{m}_seed{k}.json").write_text(json.dumps(
        {"generator": m, "train_release": k, "scores": {str(j): v for j, v in scores.items()}, "pooled": pooled,
         "self_check_max_abs_diff": max(check.values()), "self_check": check}, indent=1))
    print(f"{m} seed{k}: self-check max |diff| {max(check.values()):.2e}", flush=True)


def driver(jobs):
    todo = [(m, k) for m in MODELS for k in SEEDS if not (OUT / f"{m}_seed{k}.json").exists()]
    print(f"남은 작업 {len(todo)}개", flush=True)
    running, t0 = [], time.time()
    (OUT / "logs").mkdir(parents=True, exist_ok=True)
    while todo or running:
        while todo and len(running) < jobs:
            m, k = todo.pop(0)
            fh = open(OUT / "logs" / f"{m}_seed{k}.txt", "w")
            running.append(((m, k), fh, subprocess.Popen([sys.executable, "-u", __file__, "--one", m, str(k)]
                                                         + (["--pooled"] if POOLED else []),
                                                         cwd=ROOT, stdout=fh, stderr=subprocess.STDOUT)))
        time.sleep(15)
        for it in running[:]:
            (m, k), fh, p = it
            if p.poll() is not None:
                fh.close(); running.remove(it)
                print(f"[{(time.time() - t0) / 60:5.1f}분] {m} seed{k} rc={p.returncode} (남은 {len(todo)})", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", type=int, default=3)
    ap.add_argument("--one", nargs=2)
    ap.add_argument("--pooled", action="store_true")
    a = ap.parse_args()
    if a.one:
        one(a.one[0], int(a.one[1]))
    else:
        driver(a.jobs)
