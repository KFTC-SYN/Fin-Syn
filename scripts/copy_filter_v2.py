"""
공개 정책: 비공개 데이터(모든 분할)의 어떤 행과도 완전히 일치하는 합성 행은 공개본에서 제거한다.

배경: SMOTE는 실제 거래 두 건 사이를 보간하므로 보간 결과가 실제 거래와 똑같아지는 경우가 있다.
      모든 시드·모든 분할에서 약 1%의 행이 비공개 거래를 그대로 복제하고 있었다(양성은 0건).
      "합성 데이터 공개"라는 주장과 파일 내용이 어긋나지 않도록 제거하고,
      평가한 산출물 = 공개하는 산출물이 되도록 영향받은 리더보드를 다시 계산한다.
      (BAF도 같은 정책을 쓴다: 생성 후 원본과 일치하는 행을 제거.)

단계
  1) 복제 행 제거: synth/<model>/seed<k>/ 를 제자리에서 걸러내고, 원본은 seed<k>_unfiltered/ 로 보관.
  2) 영향받은 (model, seed)의 리더보드 s2r/s2s를 새로 계산(기존 디렉터리는 *_unfiltered 로 보관).
     leaderboard.py는 results.json이 있으면 이어서 계산하므로, 옛 결과를 반드시 치워야 한다.
  3) fidelity 재계산. 이후 표준지표·프라이버시·augmentation·분석은 --post 로 일괄 재실행.

Usage:
    python scripts/copy_filter_v2.py --filter            # 1단계
    python scripts/copy_filter_v2.py --rerun --jobs 4    # 2~3단계
    python scripts/copy_filter_v2.py --post              # 하위 지표 재계산
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
E = ROOT / "exp/finsyn-v2"
REAL = ROOT / "data/finsyn-v2"
INFO = json.loads((REAL / "info.json").read_text())
NUM, CAT = INFO["num_cols"], INFO["cat_cols"]
PY = sys.executable
ENV = {"OMP_NUM_THREADS": "4", "OPENBLAS_NUM_THREADS": "4", "MKL_NUM_THREADS": "4", "LB_N_JOBS": "4"}
REPORT = E / "copy_filter.json"


def load(d, s):
    Xn = pd.DataFrame(np.load(Path(d) / f"X_num_{s}.npy", allow_pickle=True).astype(float), columns=NUM)
    Xc = pd.DataFrame(np.load(Path(d) / f"X_cat_{s}.npy", allow_pickle=True), columns=CAT).astype(str)
    return Xn, Xc, np.load(Path(d) / f"y_{s}.npy", allow_pickle=True).astype(int)


def key(n, c):
    return pd.concat([n.round(6).astype(str), c], axis=1).agg("|".join, axis=1)


def do_filter():
    private = set()
    for s in ("train", "val", "test"):
        n, c, _ = load(REAL, s)
        private |= set(key(n, c))
    report = {}
    for mdir in sorted((E / "synth").iterdir()):
        for sd in sorted(mdir.glob("seed[0-9]")):
            backup = sd.parent / f"{sd.name}_unfiltered"
            src = backup if backup.exists() else sd  # 항상 원본에서 판정해 여러 번 실행해도 결과가 같다
            counts, masks = {}, {}
            for s in ("train", "val", "test"):
                n, c, y = load(src, s)
                hit = key(n, c).isin(private).values
                counts[s] = {"rows": int(len(y)), "copies": int(hit.sum()), "copies_positive": int((hit & (y == 1)).sum())}
                masks[s] = ~hit
            if not any(v["copies"] for v in counts.values()):
                continue
            if not backup.exists():
                shutil.copytree(sd, backup)
            src = backup
            for s in ("train", "val", "test"):
                n = np.load(src / f"X_num_{s}.npy", allow_pickle=True)
                c = np.load(src / f"X_cat_{s}.npy", allow_pickle=True)
                y = np.load(src / f"y_{s}.npy", allow_pickle=True)
                keep = masks[s]
                np.save(sd / f"X_num_{s}.npy", n[keep])
                np.save(sd / f"X_cat_{s}.npy", c[keep], allow_pickle=True)
                np.save(sd / f"y_{s}.npy", y[keep])
            info = json.loads((sd / "info.json").read_text())
            for s in ("train", "val", "test"):
                info[f"{s}_size"] = counts[s]["rows"] - counts[s]["copies"]
            info["copy_filter"] = "rows identical to any private record removed"
            (sd / "info.json").write_text(json.dumps(info, ensure_ascii=False, indent=2))
            report[f"{mdir.name}/{sd.name}"] = counts
            tot = sum(v["copies"] for v in counts.values())
            print(f"  {mdir.name:15s} {sd.name}: removed {tot} rows "
                  f"(train {counts['train']['copies']}, val {counts['val']['copies']}, test {counts['test']['copies']}; "
                  f"positives {sum(v['copies_positive'] for v in counts.values())})", flush=True)
    REPORT.write_text(json.dumps(report, indent=1))
    print(f"report -> {REPORT}")


def affected():
    rep = json.loads(REPORT.read_text())
    return [(k.split("/")[0], int(k.split("/")[1].replace("seed", ""))) for k in rep]


def do_rerun(jobs):
    tasks = []
    for model, seed in affected():
        syn = f"exp/finsyn-v2/synth/{model}/seed{seed}"
        suffix = "" if seed == 0 else f"_seed{seed}"
        for tag, test in [("s2r", "data/finsyn-v2:test"), ("s2s", f"{syn}:test")]:
            out = E / f"leaderboard_{model}_{tag}{suffix}"
            if out.exists() and not (E / f"_unfiltered/{out.name}").exists():
                (E / "_unfiltered").mkdir(exist_ok=True)
                shutil.move(str(out), str(E / f"_unfiltered/{out.name}"))
            tasks.append([PY, "-u", "scripts/leaderboard.py", "--train", syn, "--tune", f"{syn}:val", "--test", test,
                          "--out", f"exp/finsyn-v2/{out.name}", "--models", "all", "--trials", "20", "--seeds", "5"])
    log = open(E / "logs/copy_filter_rerun.txt", "a")
    running, t0 = [], time.time()
    while tasks or running:
        while tasks and len(running) < jobs:
            cmd = tasks.pop(0)
            log.write(f"\n$ {' '.join(cmd)}\n"); log.flush()
            running.append((cmd[cmd.index("--out") + 1],
                            subprocess.Popen(cmd, cwd=ROOT, env={**os.environ, **ENV}, stdout=log, stderr=subprocess.STDOUT)))
        time.sleep(20)
        for it in list(running):
            if it[1].poll() is not None:
                running.remove(it)
                print(f"[{time.strftime('%H:%M')}] {Path(it[0]).name} rc={it[1].returncode} "
                      f"(남은 {len(tasks)}, 실행 {len(running)}) [{(time.time()-t0)/60:.0f}분]", flush=True)
    for model, seed in affected():
        suffix = "" if seed == 0 else f"_seed{seed}"
        subprocess.run([PY, "scripts/fidelity.py", "--real", "exp/finsyn-v2/leaderboard_real", "--cand",
                        f"exp/finsyn-v2/leaderboard_{model}_s2r{suffix}", f"exp/finsyn-v2/leaderboard_{model}_s2s{suffix}"],
                       cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
    print("leaderboards and fidelity done", flush=True)


def do_post():
    """seed0 릴리스에 기대는 하위 지표를 다시 계산한다."""
    seed0 = sorted({m for m, s in affected() if s == 0})
    log = open(E / "logs/copy_filter_post.txt", "a")
    run = lambda cmd: subprocess.run(cmd, cwd=ROOT, env={**os.environ, **ENV}, stdout=log, stderr=subprocess.STDOUT)
    if seed0:
        run([PY, "-u", "scripts/standard_metrics_v2.py", "--models", ",".join(seed0)])
        run([PY, "-u", "scripts/privacy_v2.py", "--models", ",".join(seed0), "--n", "20000"])
        aug = E / "augmentation.json"
        r = json.loads(aug.read_text())
        for k in [k for k in r if k.split("|")[0] in seed0]:
            del r[k]
        aug.write_text(json.dumps(r, indent=1))
        run([PY, "-u", "scripts/augmentation_v2.py", "--models", ",".join(seed0), "--detectors", "lgbm,xgb"])
    run([PY, "scripts/lf_analysis_v2.py"])
    run([PY, "scripts/tstr_vs_lf_v2.py"])
    run([PY, "scripts/make_tables_v2.py"])
    run([PY, "scripts/fig_leaderboard_fidelity.py", "--out",
         str(ROOT.parent / "Fin-Syn-paper/202609_iclr/figures/leaderboard_fidelity.pdf")])
    print("post-processing done", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--filter", action="store_true")
    ap.add_argument("--rerun", action="store_true")
    ap.add_argument("--post", action="store_true")
    ap.add_argument("--jobs", type=int, default=4)
    a = ap.parse_args()
    if a.filter:
        do_filter()
    if a.rerun:
        do_rerun(a.jobs)
    if a.post:
        do_post()
