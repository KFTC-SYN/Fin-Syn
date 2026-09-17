"""
동시 실행 중인 leaderboard.py 개수를 목표치로 유지한다(SIGSTOP/SIGCONT).

배경: 드라이버가 생성기 완료마다 리더보드를 추가 투입해 동시 실행이 11개까지 늘었고,
      20코어에 부하 59(3배 과점유)가 되면서 시간당 처리량이 3.6건 → 2.9건으로 떨어졌다.
      총 작업량은 그대로이므로, 동시 실행을 줄이면 문맥 전환 낭비만큼 회복된다.

정책: 진행이 많이 된 것부터 target개를 실행 상태로 두고 나머지는 정지시킨다.
      (거의 끝난 작업을 먼저 끝내야 슬롯이 빨리 비고 전체가 빨리 끝난다)
      정지된 프로세스는 CPU를 쓰지 않고 메모리만 유지하므로 진행분 손실이 없다.

안전장치: 어떤 경로로 끝나든(정상 종료/kill/예외) 종료 직전에 모든 프로세스를 SIGCONT 한다.
          감독 대상은 '--out ..._seed<k>' 인 시드 반복 리더보드로 한정하고,
          단독 실행되는 seed0 재생성 리더보드는 건드리지 않는다.

Usage:
    python scripts/throttle_leaderboards.py --target 5 --driver-pid 3363267
"""
import argparse
import json
import os
import re
import signal
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def leaderboard_procs(only_seed_repeat=True):
    """(pid, out_dir, state) 목록. state 'T'면 정지 상태."""
    out = []
    for d in Path("/proc").iterdir():
        if not d.name.isdigit():
            continue
        try:
            cmd = (d / "cmdline").read_bytes().decode("utf-8", "replace").replace("\x00", " ")
            if "scripts/leaderboard.py" not in cmd:
                continue
            m = re.search(r"--out (\S+)", cmd)
            outdir = m.group(1) if m else ""
            if only_seed_repeat and not re.search(r"_seed\d+$", outdir):
                continue  # seed0 재생성 리더보드는 감독하지 않는다
            state = (d / "stat").read_text().split(") ")[1].split()[0]
            out.append((int(d.name), outdir, state))
        except (FileNotFoundError, ProcessLookupError, IndexError, PermissionError):
            continue
    return out


def progress(outdir):
    try:
        return len(json.loads((ROOT / outdir / "results.json").read_text()))
    except Exception:
        return 0


def resume_all(procs):
    for pid, _, _ in procs:
        try:
            os.kill(pid, signal.SIGCONT)
        except ProcessLookupError:
            pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", type=int, default=5, help="동시에 실행할 리더보드 수")
    ap.add_argument("--driver-pid", type=int, default=0, help="이 프로세스가 끝나고 대상이 없으면 종료")
    ap.add_argument("--interval", type=int, default=60)
    args = ap.parse_args()
    print(f"throttling leaderboards to {args.target} concurrent "
          f"(driver pid {args.driver_pid or 'n/a'})", flush=True)
    last = None
    try:
        while True:
            procs = leaderboard_procs()
            if not procs and (not args.driver_pid or not Path(f"/proc/{args.driver_pid}").exists()):
                print("no leaderboards left and driver finished; exiting", flush=True)
                break
            # 진행이 많이 된 것 우선으로 실행 슬롯 배정
            ranked = sorted(procs, key=lambda p: -progress(p[1]))
            run, stop = ranked[:args.target], ranked[args.target:]
            for pid, outdir, state in run:
                if state == "T":
                    os.kill(pid, signal.SIGCONT)
            for pid, outdir, state in stop:
                if state != "T":
                    os.kill(pid, signal.SIGSTOP)
            cur = (len(run), len(stop))
            if cur != last:
                names = ", ".join(f"{Path(o).name.replace('leaderboard_','')}({progress(o)}/11)" for _, o, _ in run)
                print(f"[{time.strftime('%H:%M')}] running {cur[0]}, stopped {cur[1]} | {names}", flush=True)
                last = cur
            time.sleep(args.interval)
    finally:
        procs = leaderboard_procs()
        resume_all(procs)
        print(f"resumed {len(procs)} process(es) on exit", flush=True)


if __name__ == "__main__":
    main()
