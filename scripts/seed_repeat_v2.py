"""
생성기 시드 반복: 같은 설정으로 생성기를 다시 학습해 leaderboard fidelity(tau)의 변동폭을 얻는다.

왜: 지금까지의 릴리스는 모두 seed 0 한 번뿐이라 tau에 구간이 없다(ICDM 리뷰어 3의 유의성 지적).
    탐지기 시드 5개와 test 부트스트랩은 이미 반영돼 있으므로, 여기서 재는 것은 '생성기 재학습' 변동이다.

GReaT는 제외한다: 시드 1개당 9.5 GPU-시간으로 전체 13.2시간의 72%를 차지하는데,
tau 0.273으로 논지의 중심이 아니다. 부록에 제외 사실을 명시한다.

흐름(시드마다): 생성(GPU, 순차) -> assemble -> 리더보드 s2s/s2r(CPU) -> fidelity
리더보드는 백그라운드로 돌려 다음 시드의 GPU 생성과 겹치게 한다.

Usage:
    python scripts/seed_repeat_v2.py --seeds 1,2
"""
import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
E = ROOT / "exp/finsyn-v2"
PY = sys.executable
GPU_MODELS = ["tabddpm", "tabpfgen", "tabpfgen-prior", "ctgan", "tvae", "ctabgan", "ctabgan-plus"]
ALL_MODELS = ["smote"] + GPU_MODELS


def sh(cmd, log, env_extra=None, background=False):
    import os
    env = {**os.environ, **(env_extra or {})}
    fh = open(log, "a")
    fh.write(f"\n$ {' '.join(map(str, cmd))}\n")
    fh.flush()
    if background:
        return subprocess.Popen(cmd, cwd=ROOT, env=env, stdout=fh, stderr=subprocess.STDOUT)
    p = subprocess.run(cmd, cwd=ROOT, env=env, stdout=fh, stderr=subprocess.STDOUT)
    fh.close()
    return p.returncode


def leaderboards(model, seed, log):
    """한 릴리스의 s2s/s2r 리더보드 + fidelity. CPU 작업이라 백그라운드로 돌린다."""
    syn = f"exp/finsyn-v2/synth/{model}/seed{seed}"
    script = E / f"logs/lb_{model}_seed{seed}.sh"
    lines = ["set -e"]
    for tag, test in [("s2r", "data/finsyn-v2:test"), ("s2s", f"{syn}:test")]:
        out = f"exp/finsyn-v2/leaderboard_{model}_{tag}_seed{seed}"
        lines.append(f'"{PY}" -u scripts/leaderboard.py --train {syn} --tune {syn}:val --test {test} '
                     f'--out {out} --models all --trials 20 --seeds 5')
    cands = " ".join(f"exp/finsyn-v2/leaderboard_{model}_{t}_seed{seed}" for t in ("s2r", "s2s"))
    lines.append(f'"{PY}" scripts/fidelity.py --real exp/finsyn-v2/leaderboard_real --cand {cands}')
    script.write_text("\n".join(lines) + "\n")
    return sh(["bash", str(script)], log, env_extra={"OMP_NUM_THREADS": "4", "OPENBLAS_NUM_THREADS": "4",
                                                     "MKL_NUM_THREADS": "4"}, background=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="1,2")
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()
    (E / "logs").mkdir(parents=True, exist_ok=True)
    running = []

    for seed in [int(s) for s in args.seeds.split(",")]:
        log = E / f"logs/seed_repeat_{seed}.txt"
        t0 = time.time()
        print(f"=== seed {seed}: SMOTE ===", flush=True)
        sh([PY, "scripts/synth_v2.py", "--model", "smote", "--real", "data/finsyn-v2",
            "--out", f"exp/finsyn-v2/synth/smote/seed{seed}", "--seed", str(seed)], log)
        running.append(("smote", seed, leaderboards("smote", seed, log)))

        print(f"=== seed {seed}: GPU generators ===", flush=True)
        sh([PY, "scripts/run_generators_v2.py", "prepare", "--models", ",".join(GPU_MODELS), "--seed", str(seed)], log)
        for model in GPU_MODELS:
            rc = sh([PY, "-u", "scripts/run_generators_v2.py", "run", "--models", model,
                     "--device", args.device, "--seed", str(seed)], log)
            sh([PY, "scripts/run_generators_v2.py", "assemble", "--models", model, "--seed", str(seed)], log)
            syn = E / f"synth/{model}/seed{seed}"
            if not (syn / "y_train.npy").exists():
                print(f"{model} seed{seed}: assemble produced nothing, skipped", flush=True)
                continue
            running.append((model, seed, leaderboards(model, seed, log)))
            print(f"{model} seed{seed}: generated (rc={rc}), leaderboards launched "
                  f"[{time.time()-t0:.0f}s elapsed]", flush=True)
        print(f"=== seed {seed}: generation done in {(time.time()-t0)/3600:.1f} h ===", flush=True)

    print("waiting for leaderboards ...", flush=True)
    for model, seed, p in running:
        p.wait()
        print(f"{model} seed{seed}: leaderboards rc={p.returncode}", flush=True)
    summarize()


def summarize():
    """시드별 tau를 모아 릴리스마다 범위를 출력한다."""
    out = {}
    for d in sorted(E.glob("leaderboard_*_s2*")):
        f = d / "fidelity_vs_leaderboard_real.json"
        if not f.exists():
            continue
        name = d.name.replace("leaderboard_", "")
        seed = 0
        if "_seed" in name:
            name, seed = name.rsplit("_seed", 1)
            seed = int(seed)
        model, tag = name.rsplit("_", 1)
        out.setdefault((model, tag), {})[seed] = json.loads(f.read_text())["kendall_tau"]
    (E / "seed_repeat_tau.json").write_text(json.dumps({f"{m}_{t}": v for (m, t), v in out.items()}, indent=1))
    print("\ntau by seed")
    for (model, tag), v in sorted(out.items()):
        vals = [v[s] for s in sorted(v)]
        if len(vals) > 1:
            print(f"  {model:16s} {tag}: " + " ".join(f"{x:+.3f}" for x in vals) +
                  f"   range {max(vals)-min(vals):.3f}")


if __name__ == "__main__":
    main()
