"""
TabReD(외부 재현) 파이프라인: 생성 종료 대기 → assemble → 리더보드(s2s/s2r) → fidelity → 표준지표 → 상관분석.

목적은 두 번째 본격 연구가 아니라, 원고 5.3절의 상관 주장을 **공개 실제 데이터에서 독립적으로 확인**하는 것이다.
축소 예산: 생성기 5종(시드 1개), 탐지기 11종, Optuna 10 trial, 시드 3개.

Usage:
    python scripts/tabred_pipeline.py --wait-pid <생성 PID> --jobs 4
"""
import argparse
import itertools
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
from scipy.stats import rankdata

ROOT = Path(__file__).resolve().parents[1]
E = ROOT / "exp/tabred-hc"
DATA = "data/tabred-hc"
PY = sys.executable
GPU_MODELS = ["ctgan", "tvae", "ctabgan-plus", "tabddpm"]
MODELS = ["smote"] + GPU_MODELS
ENV = {"OMP_NUM_THREADS": "4", "OPENBLAS_NUM_THREADS": "4", "MKL_NUM_THREADS": "4", "LB_N_JOBS": "4"}


def sh(cmd, log, env_extra=None):
    with open(log, "a") as fh:
        fh.write(f"\n$ {' '.join(map(str, cmd))}\n")
        fh.flush()
        return subprocess.run(cmd, cwd=ROOT, env={**os.environ, **(env_extra or {})},
                              stdout=fh, stderr=subprocess.STDOUT).returncode


def spearman(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    rx, ry = rankdata(x), rankdata(y)
    if rx.std() == 0 or ry.std() == 0:
        return float("nan")
    return float(np.corrcoef(rx, ry)[0, 1])


def exact_p(x, y):
    rx, ry = rankdata(x), rankdata(y)
    obs, cnt, tot = abs(spearman(x, y)), 0, 0
    for perm in itertools.permutations(range(len(y))):
        tot += 1
        cnt += abs(np.corrcoef(rx, ry[list(perm)])[0, 1]) >= obs - 1e-12
    return cnt / tot


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--wait-pid", type=int, default=0)
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--models", default=None, help="이 실행에서 다룰 생성기(기본: 전체)")
    ap.add_argument("--all-models", default=None, help="상관분석에 포함할 전체 목록(기본: --models)")
    args = ap.parse_args()
    global MODELS, GPU_MODELS
    if args.models:
        MODELS = args.models.split(',')
        GPU_MODELS = [m for m in MODELS if m != 'smote']
    log = E / "logs/pipeline.txt"
    (E / "logs").mkdir(parents=True, exist_ok=True)

    while args.wait_pid and Path(f"/proc/{args.wait_pid}").exists():
        time.sleep(60)
    print("generation finished; assembling", flush=True)

    ok = []
    for m in GPU_MODELS:
        sh([PY, "scripts/run_generators_v2.py", "assemble", "--models", m,
            "--real", DATA, "--exp", "exp/tabred-hc", "--seed", "0"], log)
    for m in MODELS:
        if (E / f"synth/{m}/seed0/y_train.npy").exists():
            ok.append(m)
        else:
            print(f"  {m}: 생성 실패, 제외", flush=True)
    print("releases:", ok, flush=True)

    # 리더보드 (동시 jobs개)
    tasks = []
    for m in ok:
        syn = f"exp/tabred-hc/synth/{m}/seed0"
        for tag, test in [("s2r", f"{DATA}:test"), ("s2s", f"{syn}:test")]:
            tasks.append([PY, "-u", "scripts/leaderboard.py", "--train", syn, "--tune", f"{syn}:val",
                          "--test", test, "--out", f"exp/tabred-hc/leaderboard_{m}_{tag}",
                          "--models", "all", "--trials", "10", "--seeds", "3"])
    running, fh = [], open(log, "a")
    t0 = time.time()
    while tasks or running:
        while tasks and len(running) < args.jobs:
            cmd = tasks.pop(0)
            fh.write(f"\n$ {' '.join(cmd)}\n"); fh.flush()
            running.append((cmd[cmd.index("--out") + 1],
                            subprocess.Popen(cmd, cwd=ROOT, env={**os.environ, **ENV},
                                             stdout=fh, stderr=subprocess.STDOUT)))
        time.sleep(20)
        for it in list(running):
            if it[1].poll() is not None:
                running.remove(it)
                print(f"[{time.strftime('%H:%M')}] done {Path(it[0]).name} rc={it[1].returncode} "
                      f"(남은 {len(tasks)}) [{(time.time()-t0)/60:.0f}분]", flush=True)
    fh.close()

    for m in ok:
        sh([PY, "scripts/fidelity.py", "--real", "exp/tabred-hc/leaderboard_real",
            "--cand", f"exp/tabred-hc/leaderboard_{m}_s2r", f"exp/tabred-hc/leaderboard_{m}_s2s"], log)
    sh([PY, "-u", "scripts/standard_metrics_v2.py", "--models", ",".join(ok),
        "--real", DATA, "--exp", "exp/tabred-hc"], log, ENV)

    # 상관 분석 (n=len(ok))
    sm = json.loads((E / "standard_metrics.json").read_text())
    pool = (args.all_models.split(",") if args.all_models else sorted(sm))
    tau = {}
    for m in pool:
        f = E / f"leaderboard_{m}_s2s/fidelity_vs_leaderboard_real.json"
        if f.exists():
            v = json.loads(f.read_text())["kendall_tau"]
            if v == v:  # NaN 제외 (합성 test에 양성이 없어 tau가 정의되지 않는 릴리스)
                tau[m] = v
    ok = [m for m in tau if m in sm]
    nf = json.loads((E / "leaderboard_real/noise_floor.json").read_text())
    metrics = {"ks": [-sm[m]["ks_mean"] for m in ok], "tvd": [-sm[m]["tvd_mean"] for m in ok],
               "c2st": [-sm[m]["detection_auc"] for m in ok],
               "dependence_error": [-sm[m]["corr_rmse"] for m in ok],
               "tstr_catboost": [sm[m]["tstr_catboost_pr_auc"] for m in ok]}
    t = [tau[m] for m in ok]
    corr = {k: {"rho": spearman(v, t), "p": exact_p(v, t)} for k, v in metrics.items()}
    out = {"dataset": "TabReD homecredit-default (public)", "releases": ok, "tau_s2s": tau,
           "noise_floor": {"q05": nf["tau_vs_full_q05"], "mean": nf["tau_vs_full_mean"]},
           "corr_with_tau_s2s": corr}
    (E / "lf_analysis.json").write_text(json.dumps(out, indent=1))
    print("\n== TabReD 결과 ==")
    print(f"noise floor: q05={nf['tau_vs_full_q05']:+.3f}, mean={nf['tau_vs_full_mean']:+.3f}")
    for m in sorted(ok, key=lambda m: -tau[m]):
        print(f"  {m:14s} tau_s2s={tau[m]:+.3f}")
    for k, v in corr.items():
        print(f"  rho({k:16s}, tau) = {v['rho']:+.2f}  p={v['p']:.3f}")


if __name__ == "__main__":
    main()
