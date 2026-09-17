"""
seed0 재생성의 남은 단계(리더보드 이후)를 병렬로 끝낸다.

교체 이유: regen_seed0_gans.py 는 리더보드 8건을 순차 실행해 20코어 중 4개만 쓰고 있었다
(부하 4, 건당 약 70분 -> 총 9시간 예상). 생성 단계는 이미 끝났으므로 리더보드부터 다시 잡는다.
leaderboard.py 는 results.json 에 있는 모델을 건너뛰므로 중단된 작업도 이어서 진행된다.

Usage:
    python scripts/regen_seed0_finish.py --jobs 4
"""
import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
E = ROOT / "exp/finsyn-v2"
PY = sys.executable
MODELS = ["ctgan", "tvae", "ctabgan", "ctabgan-plus"]
BACKUP = E / "_backup_preseed"
ENV = {"OMP_NUM_THREADS": "4", "OPENBLAS_NUM_THREADS": "4", "MKL_NUM_THREADS": "4", "LB_N_JOBS": "4"}


def sh(cmd, log, env_extra=None):
    env = {**os.environ, **(env_extra or {})}
    with open(log, "a") as fh:
        fh.write(f"\n$ {' '.join(map(str, cmd))}\n")
        fh.flush()
        return subprocess.run(cmd, cwd=ROOT, env=env, stdout=fh, stderr=subprocess.STDOUT).returncode


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", type=int, default=4, help="동시 실행 리더보드 수")
    args = ap.parse_args()
    log = E / "logs/regen_seed0.txt"
    t0 = time.time()

    # 1) 리더보드 8건을 jobs개씩 동시 실행 (완료분은 leaderboard.py가 건너뛴다)
    tasks = []
    for m in MODELS:
        syn = f"exp/finsyn-v2/synth/{m}/seed0"
        for tag, test in [("s2r", "data/finsyn-v2:test"), ("s2s", f"{syn}:test")]:
            tasks.append([PY, "-u", "scripts/leaderboard.py", "--train", syn, "--tune", f"{syn}:val",
                          "--test", test, "--out", f"exp/finsyn-v2/leaderboard_{m}_{tag}",
                          "--models", "all", "--trials", "20", "--seeds", "5"])
    running = []
    fh = open(log, "a")
    while tasks or running:
        while tasks and len(running) < args.jobs:
            cmd = tasks.pop(0)
            fh.write(f"\n$ {' '.join(cmd)}\n")
            fh.flush()
            running.append((cmd[cmd.index("--out") + 1],
                            subprocess.Popen(cmd, cwd=ROOT, env={**os.environ, **ENV},
                                             stdout=fh, stderr=subprocess.STDOUT)))
        time.sleep(20)
        for item in list(running):
            name, p = item
            if p.poll() is not None:
                running.remove(item)
                print(f"[{time.strftime('%H:%M')}] done {Path(name).name} rc={p.returncode} "
                      f"(남은 {len(tasks)}건, 실행 {len(running)}) [{(time.time()-t0)/60:.0f}분]", flush=True)
    fh.close()

    # 2) fidelity
    for m in MODELS:
        sh([PY, "scripts/fidelity.py", "--real", "exp/finsyn-v2/leaderboard_real",
            "--cand", f"exp/finsyn-v2/leaderboard_{m}_s2r", f"exp/finsyn-v2/leaderboard_{m}_s2s"], log)

    # 3) 지표 / 프라이버시 / augmentation
    ms = ",".join(MODELS)
    sh([PY, "-u", "scripts/standard_metrics_v2.py", "--models", ms], log, ENV)
    sh([PY, "-u", "scripts/privacy_v2.py", "--models", ms, "--n", "20000"], log, ENV)
    aug = E / "augmentation.json"
    if aug.exists():
        r = json.loads(aug.read_text())
        drop = [k for k in r if k.split("|")[0] in MODELS]
        for k in drop:
            del r[k]
        aug.write_text(json.dumps(r, indent=1))
        print(f"augmentation: dropped {len(drop)} stale keys", flush=True)
    sh([PY, "-u", "scripts/augmentation_v2.py", "--models", ms, "--detectors", "lgbm,xgb"], log, ENV)

    # 4) 표 / 그림
    sh([PY, "scripts/make_tables_v2.py"], log)
    sh([PY, "scripts/fig_leaderboard_fidelity.py", "--out",
        str(ROOT.parent / "Fin-Syn-paper/202609_iclr/figures/leaderboard_fidelity.pdf")], log)

    # 5) 변경 요약
    print(f"\ndone in {(time.time()-t0)/3600:.1f} h\n\ntau_s2s before -> after")
    for m in MODELS:
        old = BACKUP / f"leaderboard_{m}_s2s/fidelity_vs_leaderboard_real.json"
        o = json.loads(old.read_text())["kendall_tau"] if old.exists() else float("nan")
        n = json.loads((E / f"leaderboard_{m}_s2s/fidelity_vs_leaderboard_real.json").read_text())["kendall_tau"]
        print(f"  {m:14s} {o:+.3f} -> {n:+.3f}")


if __name__ == "__main__":
    main()
