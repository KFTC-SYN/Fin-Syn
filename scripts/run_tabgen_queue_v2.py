"""
TabSyn·TabDiff 릴리스 3시드를 끝까지 만든다: 생성(GPU) -> assemble -> 복제 행 필터 -> 리더보드 s2r/s2s(CPU) -> fidelity.

왜: ICLR 리뷰어가 가장 먼저 물을 최신 생성기 두 개(TabSyn ICLR'24, TabDiff ICLR'25)를 기존 9개 릴리스와
    같은 프로토콜(공식 기본값, 기간당 3시간 상한 = 에폭 예산, 시드 0-2, 복제 행 제거)로 추가한다.

상태는 매번 디스크와 프로세스 목록에서 다시 읽는다(재시작해도 이어서 돈다).
  - 생성 완료: exp/finsyn-v2/gen_runs.jsonl 에 rc 0 + rows 기록
  - 실행 중  : 'run_generators_v2.py run --models M --parts P --seed S' 프로세스가 있음(이어받기 프로세스 포함)
  - 실패      : rc != 0 기록. 2번 실패하면 더 시도하지 않는다.
  - assemble : 세 기간이 모두 완료되고 synth/<m>/seed<s>/info.json 이 없을 때 -> 필터(flock)
  - 리더보드 : noise_floor.json 이 있으면 완료, '--out <dir>' 프로세스가 있으면 실행 중. 동시 2개(각 4스레드).
  - 시드 0 리더보드는 접미사 없이(leaderboard_<m>_s2s), 시드 1-2는 _seed<k> (기존 규칙).

Usage:
    nohup python -u scripts/run_tabgen_queue_v2.py > exp/finsyn-v2/logs/tabgen_queue.txt 2>&1 &
"""
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
E = ROOT / "exp/finsyn-v2"
PY = sys.executable
# 환경변수로 덮어쓸 수 있다(자동 파이프라인 auto_finalize_v2.py가 추가 시드를 돌릴 때 사용).
MODELS = os.environ.get("FINSYN_QUEUE_MODELS", "tabdiff,tabsyn,findiff").split(",")
SEEDS = [int(s) for s in os.environ.get("FINSYN_QUEUE_SEEDS", "0,1,2,3,4").split(",")]
PARTS = ["train", "val", "test"]
# GReaT처럼 무거운 생성기는 동시 실행이 늘수록 총처리량이 떨어진다(9/20 실측: 1개 0.30 s/it, 3개 2.12 s/it,
# 4개 1.78 s/it). 모델별로 슬롯을 환경변수로 조절한다.
GPU_SLOTS = int(os.environ.get("FINSYN_GPU_SLOTS", "4"))
LB_SLOTS, MAX_FAIL = 2, 2
ENV_LB = {"OMP_NUM_THREADS": "4", "OPENBLAS_NUM_THREADS": "4", "MKL_NUM_THREADS": "4", "LB_N_JOBS": "4"}
LOG = E / "logs/tabgen"
LOG.mkdir(parents=True, exist_ok=True)
JOBS = [(m, p, s) for s in SEEDS for p in PARTS for m in MODELS]  # 시드 0, 긴 train 기간 먼저


def stamp():
    return time.strftime("%m-%d %H:%M")


def proc_args():
    out = subprocess.run(["ps", "-eo", "pid=,args="], capture_output=True, text=True).stdout.splitlines()
    return [l.strip().split(None, 1)[1] for l in out if len(l.strip().split(None, 1)) == 2]


def records():
    f = E / "gen_runs.jsonl"
    rs = [json.loads(l) for l in f.read_text().splitlines() if l.strip()] if f.exists() else []
    return [{**r, "seed": r.get("seed", 0)} for r in rs if r.get("model") in MODELS]  # 예전 기록엔 seed가 없다


def lb_name(model, tag, seed):
    return f"leaderboard_{model}_{tag}" + ("" if seed == 0 else f"_seed{seed}")


def popen(cmd, log, env_extra=None):
    fh = open(log, "a")
    fh.write(f"\n[{stamp()}] $ {' '.join(map(str, cmd))}\n"); fh.flush()
    return subprocess.Popen(cmd, cwd=ROOT, env={**os.environ, **(env_extra or {})}, stdout=fh, stderr=subprocess.STDOUT,
                            start_new_session=True)


def main():
    t0, seen, prepared, launched = time.time(), set(), set(), {}
    while True:
        recs, ps = records(), proc_args()
        # rc 0이면 완료로 본다. 'rows'는 래퍼 생성기(TabSyn/TabDiff/FinDiff)만 남기므로 조건에 넣으면
        # 설정 기반 생성기(CTGAN 등)가 영원히 미완료로 보여 같은 작업을 반복한다(9/20 실제 발생).
        ok = {(r["model"], r["part"], r["seed"]) for r in recs if r["rc"] == 0}
        fails = {}
        for r in recs:
            if r["rc"] != 0:
                k = (r["model"], r["part"], r["seed"])
                fails[k] = fails.get(k, 0) + 1
        # 기록조차 남기지 못하고 죽는 실패(설정 파일 없음 등)도 실패로 센다. 그렇지 않으면 무한 반복한다(9/20 실제 발생).
        for k, n in launched.items():
            if k not in ok:
                fails[k] = max(fails.get(k, 0), n)
        running = {j for j in JOBS if any(f"run_generators_v2.py run --models {j[0]} --parts {j[1]} --seed {j[2]}" in a
                                          for a in ps)}
        for j in JOBS:
            if j in ok and ("ok",) + j not in seen:
                seen.add(("ok",) + j)
                r = [x for x in recs if (x["model"], x["part"], x["seed"]) == j and x["rc"] == 0][-1]
                print(f"[{stamp()}] gen done {j}: {r['minutes']} min, pos {r.get('positive_rate')}", flush=True)
        # 설정 기반 생성기는 시드마다 config.toml이 필요하다. 없으면 prepare를 먼저 돌린다(9/20 시드 3에서 누락).
        for m, part, s in [j for j in JOBS if j not in ok and j not in running]:
            if m in ("tabsyn", "tabdiff", "findiff") or (m, s) in prepared:
                continue
            if not (E / f"gen/{m}/{part if s == 0 else f'{part}_seed{s}'}/config.toml").exists():
                prepared.add((m, s))
                subprocess.run([PY, "scripts/run_generators_v2.py", "prepare", "--models", m, "--seed", str(s)],
                               cwd=ROOT, stdout=open(LOG / f"prepare_{m}_seed{s}.txt", "w"), stderr=subprocess.STDOUT)
                print(f"[{stamp()}] prepared configs for {m} seed{s}", flush=True)
        pending = [j for j in JOBS if j not in ok and j not in running and fails.get(j, 0) < MAX_FAIL]
        dead = [j for j in JOBS if j not in ok and j not in running and fails.get(j, 0) >= MAX_FAIL]
        # 1) 생성
        n_run = len(running)
        for j in pending:
            if n_run >= GPU_SLOTS:
                break
            m, p, s = j
            popen([PY, "-u", "scripts/run_generators_v2.py", "run", "--models", m, "--parts", p, "--seed", str(s)],
                  LOG / f"gen_{m}_{p}_seed{s}.txt")
            launched[j] = launched.get(j, 0) + 1
            print(f"[{stamp()}] start gen {j} (attempt {launched[j]})", flush=True)
            n_run += 1
        # 2) assemble + 필터
        for m in MODELS:
            for s in SEEDS:
                if all((m, p, s) in ok for p in PARTS) and not (E / f"synth/{m}/seed{s}/info.json").exists():
                    lg = open(LOG / f"post_{m}_seed{s}.txt", "a")
                    rc1 = subprocess.run([PY, "scripts/run_generators_v2.py", "assemble", "--models", m, "--seed", str(s)],
                                         cwd=ROOT, stdout=lg, stderr=subprocess.STDOUT).returncode
                    rc2 = subprocess.run(["flock", str(E / ".copy_filter.lock"), PY, "scripts/copy_filter_v2.py", "--filter"],
                                         cwd=ROOT, stdout=lg, stderr=subprocess.STDOUT).returncode
                    print(f"[{stamp()}] assembled+filtered {m} seed{s} (rc {rc1},{rc2})", flush=True)
        # 3) 리더보드 + fidelity
        lb_running = sum(1 for a in ps if "scripts/leaderboard.py" in a and "--models all" in a
                         and any(f"exp/finsyn-v2/leaderboard_{m}_" in a for m in MODELS))
        lb_left = 0
        for s in SEEDS:
            for m in MODELS:
                if not (E / f"synth/{m}/seed{s}/info.json").exists():
                    lb_left += 2
                    continue
                syn = f"exp/finsyn-v2/synth/{m}/seed{s}"
                for tag in ("s2r", "s2s"):
                    out = f"exp/finsyn-v2/{lb_name(m, tag, s)}"
                    if (ROOT / out / "noise_floor.json").exists():
                        continue
                    lb_left += 1
                    if any(f"--out {out} " in a + " " for a in ps) or lb_running >= LB_SLOTS:
                        continue
                    test = "data/finsyn-v2:test" if tag == "s2r" else f"{syn}:test"
                    popen([PY, "-u", "scripts/leaderboard.py", "--train", syn, "--tune", f"{syn}:val", "--test", test,
                           "--out", out, "--models", "all", "--trials", "20", "--seeds", "5"],
                          LOG / f"lb_{m}_{tag}_seed{s}.txt", ENV_LB)
                    print(f"[{stamp()}] start leaderboard {Path(out).name}", flush=True)
                    lb_running += 1
                cands = [E / lb_name(m, t, s) for t in ("s2r", "s2s")]
                if all((c / "noise_floor.json").exists() for c in cands) and \
                        not all((c / "fidelity_vs_leaderboard_real.json").exists() for c in cands):
                    subprocess.run([PY, "scripts/fidelity.py", "--real", "exp/finsyn-v2/leaderboard_real", "--cand",
                                    *map(str, cands)], cwd=ROOT, stdout=open(LOG / f"post_{m}_seed{s}.txt", "a"),
                                   stderr=subprocess.STDOUT)
                    print(f"[{stamp()}] fidelity {m} seed{s}", flush=True)
        gen_left = [j for j in JOBS if j not in ok and j not in dead]
        if not gen_left and not lb_left:
            break
        if not gen_left and dead and all(not (E / f"synth/{m}/seed{s}/info.json").exists()
                                         for m, _, s in dead) and lb_left == 2 * len({(m, s) for m, _, s in dead}):
            break
        time.sleep(60)
    print(f"[{stamp()}] finished [{(time.time()-t0)/3600:.1f}h]; gave up on {dead}", flush=True)


if __name__ == "__main__":
    main()
