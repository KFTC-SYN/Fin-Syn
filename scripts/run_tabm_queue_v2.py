"""
TabM 리더보드 대기열(부록 견고성 점검 B안): 실제 리더보드 + 모든 릴리스의 시드 0 리더보드(s2r, s2s).

본문 11개 탐지기 결과와 섞이지 않도록 exp/finsyn-v2/tabm/<리더보드 이름>/ 에 TabM만 저장한다.
설정은 다른 탐지기와 같다(Optuna TPE 20 trials, 튜닝 split PR-AUC, 최적 설정으로 시드 5개).
TabSyn/TabDiff 릴리스는 생성·필터·기존 리더보드가 끝난 뒤에 들어간다(없으면 기다렸다가 마지막에 시도).
동시 실행 수: TabSyn/TabDiff 생성('run_generators_v2.py run')이 돌고 있으면 --jobs(기본 1),
끝나면 --jobs-free(기본 2)로 늘린다. 실행 중인 TabM 리더보드는 프로세스 목록에서 찾아 이어받는다(재시작해도 중복 실행 없음).

Usage:
    nohup python -u scripts/run_tabm_queue_v2.py --jobs 1 > exp/finsyn-v2/logs/tabm_queue.txt 2>&1 &
"""
import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
E = ROOT / "exp/finsyn-v2"
PY = sys.executable
FIRST = ["smote", "tabpfgen", "tabpfgen-prior", "tabddpm", "great", "tvae", "ctgan", "ctabgan", "ctabgan-plus"]
LATER = ["tabsyn", "tabdiff", "findiff"]


def job(name, train, tune, test):
    return name, [PY, "-u", "scripts/leaderboard.py", "--train", train, "--tune", tune, "--test", test,
                  "--out", f"exp/finsyn-v2/tabm/{name}", "--models", "tabm", "--trials", "20", "--seeds", "5"]


def release_jobs(m):
    syn = f"exp/finsyn-v2/synth/{m}/seed0"
    return [job(f"leaderboard_{m}_s2r", syn, f"{syn}:val", "data/finsyn-v2:test"),
            job(f"leaderboard_{m}_s2s", syn, f"{syn}:val", f"{syn}:test")]


def ready(m):
    """기존 11개 리더보드가 끝났으면(=필터된 릴리스 확정) TabM도 같은 릴리스로 돌릴 수 있다."""
    return all((E / f"leaderboard_{m}_{t}/fidelity_vs_leaderboard_real.json").exists() for t in ("s2r", "s2s"))


def ps_args():
    out = subprocess.run(["ps", "-eo", "args="], capture_output=True, text=True).stdout
    return out.splitlines()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", type=int, default=1)
    ap.add_argument("--jobs-free", type=int, default=2)
    a = ap.parse_args()
    q = [job("leaderboard_real", "data/finsyn-v2", "data/finsyn-v2:val", "data/finsyn-v2:test")]
    for m in FIRST:
        q += release_jobs(m)
    waiting = list(LATER)
    (E / "logs/tabm").mkdir(parents=True, exist_ok=True)
    running, t0 = [], time.time()
    external = set()
    while True:
        ps = ps_args()
        # 이 드라이버가 띄우지 않았지만 돌고 있는 TabM 리더보드(재시작 전의 작업)
        external = {name for name, _ in q if any(f"--out exp/finsyn-v2/tabm/{name} " in s + " " for s in ps)}
        external -= {n for n, _ in running}
        if not (q or running or waiting or external):
            break
        gen_busy = any("run_generators_v2.py run" in s for s in ps)
        slots = a.jobs if gen_busy else a.jobs_free
        for m in list(waiting):
            if ready(m):
                waiting.remove(m)
                q += release_jobs(m)
        busy = len(running) + len(external)
        for item in list(q):
            if busy >= slots:
                break
            name, cmd = item
            if name in external:
                continue
            q.remove(item)
            if (E / f"tabm/{name}/results.json").exists():
                print(f"[{time.strftime('%m-%d %H:%M')}] skip {name} (done)", flush=True)
                continue
            busy += 1
            fh = open(E / f"logs/tabm/{name}.txt", "a")
            running.append((name, subprocess.Popen(cmd, cwd=ROOT, env={**os.environ, "OMP_NUM_THREADS": "4"},
                                                   stdout=fh, stderr=subprocess.STDOUT)))
            print(f"[{time.strftime('%m-%d %H:%M')}] start {name} (slots {slots})", flush=True)
        if not q and not running and waiting:
            if time.time() - t0 > 5 * 24 * 3600:
                break
        time.sleep(60)
        for it in list(running):
            if it[1].poll() is not None:
                running.remove(it)
                print(f"[{time.strftime('%m-%d %H:%M')}] done {it[0]} rc={it[1].returncode} "
                      f"(queue {len(q)}, waiting {waiting}) [{(time.time()-t0)/3600:.1f}h]", flush=True)
    print("all done", flush=True)


if __name__ == "__main__":
    main()
