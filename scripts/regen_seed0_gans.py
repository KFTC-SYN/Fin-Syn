"""
시드 고정 패치 이후, GAN 계열 4종의 seed0 릴리스를 재생성하고 하위 결과를 전부 다시 계산한다.

배경: CTGAN/TVAE/CTAB-GAN/CTAB-GAN+ 는 학습에 시드가 없어서 기존 seed0 산출물이 재현 불가능했다.
      trainer에 _seed_all()을 추가한 뒤이므로, seed0을 다시 만들면 "seed 0 -> 공개 데이터"가 참이 된다.

주의: 기존 산출물은 exp/finsyn-v2/_backup_preseed/ 로 옮겨 보관한다(삭제하지 않는다).
      진행 중인 작업이 있으면 --wait-pid 로 그 프로세스가 끝난 뒤 시작한다.

Usage:
    python scripts/regen_seed0_gans.py --wait-pid 3363267
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
E = ROOT / "exp/finsyn-v2"
PY = sys.executable
MODELS = ["ctgan", "tvae", "ctabgan", "ctabgan-plus"]
BACKUP = E / "_backup_preseed"


def sh(cmd, log, env_extra=None):
    env = {**os.environ, **(env_extra or {})}
    with open(log, "a") as fh:
        fh.write(f"\n$ {' '.join(map(str, cmd))}\n")
        fh.flush()
        return subprocess.run(cmd, cwd=ROOT, env=env, stdout=fh, stderr=subprocess.STDOUT).returncode


def wait_for(pid, log):
    if not pid:
        return
    print(f"waiting for pid {pid} ...", flush=True)
    while Path(f"/proc/{pid}").exists():
        time.sleep(120)
    print(f"pid {pid} finished; starting regeneration", flush=True)


def backup(log):
    BACKUP.mkdir(parents=True, exist_ok=True)
    for m in MODELS:
        # gen/<m> 전체가 아니라 seed0 part 디렉터리만 옮긴다(시드 1,2 산출물은 제자리에 둔다)
        for src in [E / f"synth/{m}/seed0", E / f"leaderboard_{m}_s2s", E / f"leaderboard_{m}_s2r",
                    E / f"gen/{m}/train", E / f"gen/{m}/val", E / f"gen/{m}/test"]:
            if src.exists():
                dst = BACKUP / src.relative_to(E)
                dst.parent.mkdir(parents=True, exist_ok=True)
                if dst.exists():
                    shutil.rmtree(dst)
                shutil.move(str(src), str(dst))
    print(f"backed up pre-seeding artefacts to {BACKUP}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--wait-pid", type=int, default=0)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()
    log = E / "logs/regen_seed0.txt"
    (E / "logs").mkdir(parents=True, exist_ok=True)
    wait_for(args.wait_pid, log)
    t0 = time.time()

    backup(log)

    # 1) 재생성 (시드 고정된 학습)
    ms = ",".join(MODELS)
    sh([PY, "scripts/run_generators_v2.py", "prepare", "--models", ms, "--seed", "0"], log)
    for m in MODELS:
        rc = sh([PY, "-u", "scripts/run_generators_v2.py", "run", "--models", m,
                 "--device", args.device, "--seed", "0"], log)
        sh([PY, "scripts/run_generators_v2.py", "assemble", "--models", m, "--seed", "0"], log)
        print(f"{m}: regenerated (rc={rc}) [{time.time()-t0:.0f}s]", flush=True)

    # 2) 리더보드 + fidelity
    cpu = {"OMP_NUM_THREADS": "8", "OPENBLAS_NUM_THREADS": "8", "MKL_NUM_THREADS": "8", "LB_N_JOBS": "8"}  # 단독 실행이므로 8스레드
    for m in MODELS:
        syn = f"exp/finsyn-v2/synth/{m}/seed0"
        for tag, test in [("s2r", "data/finsyn-v2:test"), ("s2s", f"{syn}:test")]:
            sh([PY, "-u", "scripts/leaderboard.py", "--train", syn, "--tune", f"{syn}:val", "--test", test,
                "--out", f"exp/finsyn-v2/leaderboard_{m}_{tag}", "--models", "all",
                "--trials", "20", "--seeds", "5"], log, cpu)
        sh([PY, "scripts/fidelity.py", "--real", "exp/finsyn-v2/leaderboard_real",
            "--cand", f"exp/finsyn-v2/leaderboard_{m}_s2r", f"exp/finsyn-v2/leaderboard_{m}_s2s"], log)
        print(f"{m}: leaderboards done [{time.time()-t0:.0f}s]", flush=True)

    # 3) 표준 지표 / 프라이버시
    sh([PY, "-u", "scripts/standard_metrics_v2.py", "--models", ms], log, cpu)
    sh([PY, "-u", "scripts/privacy_v2.py", "--models", ms, "--n", "20000"], log, cpu)

    # 4) augmentation: 해당 4종의 기존 키를 지우고 다시 계산
    aug = E / "augmentation.json"
    if aug.exists():
        r = json.loads(aug.read_text())
        drop = [k for k in r if k.split("|")[0] in MODELS]
        for k in drop:
            del r[k]
        aug.write_text(json.dumps(r, indent=1))
        print(f"augmentation: dropped {len(drop)} stale keys", flush=True)
    sh([PY, "-u", "scripts/augmentation_v2.py", "--models", ms, "--detectors", "lgbm,xgb"], log, cpu)

    # 5) 표/그림 재생성
    sh([PY, "scripts/make_tables_v2.py"], log)
    sh([PY, "scripts/fig_leaderboard_fidelity.py", "--out",
        str(ROOT.parent / "Fin-Syn-paper/202609_iclr/figures/leaderboard_fidelity.pdf")], log)

    # 6) 이전 값과 비교해서 무엇이 바뀌었는지 출력
    old = {}
    for m in MODELS:
        f = BACKUP / f"leaderboard_{m}_s2s/fidelity_vs_leaderboard_real.json"
        if f.exists():
            old[m] = json.loads(f.read_text())["kendall_tau"]
    print(f"\ndone in {(time.time()-t0)/3600:.1f} h\n\ntau_s2s before -> after")
    for m in MODELS:
        new = json.loads((E / f"leaderboard_{m}_s2s/fidelity_vs_leaderboard_real.json").read_text())["kendall_tau"]
        print(f"  {m:14s} {old.get(m, float('nan')):+.3f} -> {new:+.3f}")


if __name__ == "__main__":
    main()
