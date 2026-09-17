"""
생성기 결과가 기간 3개 모두 준비되면 assemble → synth/real, synth/synth 리더보드 → fidelity 를 순서대로 실행하는 감시자.

Usage (Fin-Syn/):
    OMP_NUM_THREADS=8 python scripts/pipeline_after_gen_v2.py --models tvae,ctgan,tabddpm,ctabgan-plus,ctabgan,tabpfgen
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


def parts_done(model):
    runs = E / "gen_runs.jsonl"
    if not runs.exists():
        return False
    recs = [json.loads(l) for l in runs.read_text().splitlines() if l.strip()]
    ok = {r["part"] for r in recs if r["model"] == model and r["rc"] == 0}
    bad = [r for r in recs if r["model"] == model and r["rc"] != 0]
    if bad:
        return "failed"
    return ok >= {"train", "val", "test"} and all((E / f"gen/{model}/{p}/y_train.npy").exists() for p in ["train", "val", "test"])


def sh(cmd, log):
    with open(log, "a") as fh:
        fh.write(f"\n$ {' '.join(cmd)}\n")
        fh.flush()
        return subprocess.run(cmd, cwd=ROOT, stdout=fh, stderr=subprocess.STDOUT).returncode


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", required=True)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    todo = args.models.split(",")
    (E / "logs").mkdir(exist_ok=True)
    while todo:
        for model in list(todo):
            st = parts_done(model)
            if st == "failed":
                print(f"{model}: generation failed, skipped", flush=True)
                todo.remove(model)
                continue
            if not st:
                continue
            log = E / f"logs/after_gen_{model}.txt"
            syn = f"exp/finsyn-v2/synth/{model}/seed{args.seed}"
            sh([PY, "scripts/run_generators_v2.py", "assemble", "--models", model, "--seed", str(args.seed)], log)
            for tag, test in [("s2r", "data/finsyn-v2:test"), ("s2s", f"{syn}:test")]:
                out = f"exp/finsyn-v2/leaderboard_{model}_s2r" if tag == "s2r" else f"exp/finsyn-v2/leaderboard_{model}_s2s"
                rc = sh([PY, "-u", "scripts/leaderboard.py", "--train", syn, "--tune", f"{syn}:val", "--test", test,
                         "--out", out, "--models", "all", "--trials", "20", "--seeds", "5"], log)
                print(f"{model} {tag} leaderboard rc={rc}", flush=True)
            sh([PY, "scripts/fidelity.py", "--real", "exp/finsyn-v2/leaderboard_real",
                "--cand", f"exp/finsyn-v2/leaderboard_{model}_s2r", f"exp/finsyn-v2/leaderboard_{model}_s2s"], log)
            print(f"{model}: done", flush=True)
            todo.remove(model)
        time.sleep(60)


if __name__ == "__main__":
    main()
