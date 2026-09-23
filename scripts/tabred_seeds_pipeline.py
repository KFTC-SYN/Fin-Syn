"""
TabReD 복제 확장: 시드 1~4의 리더보드(s2s/s2r)를 만들고, 생성기 내 변동을 본 벤치마크와 같은 방식으로 잰다.

왜: 원고 부록 F의 복제는 생성기당 시드 1개여서 이 논문의 중심 주장(생성기가 공개본을 결정하지 않는다)을
    공개 데이터에서 확인하지 못한다. 시드를 늘려 within-generator 분산과 기준 비교를 재현한다.

Usage:
    python scripts/tabred_seeds_pipeline.py --stage leaderboards --jobs 3
    python scripts/tabred_seeds_pipeline.py --stage analyse
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
from scipy.stats import kendalltau

ROOT = Path(__file__).resolve().parents[1]
E = ROOT / "exp/tabred-hc"
MODELS = ["smote", "ctgan", "tvae", "ctabgan", "ctabgan-plus", "tabddpm", "tabpfgen"]
SEEDS = [0, 1, 2, 3, 4]
ENV = {"OMP_NUM_THREADS": "4", "OPENBLAS_NUM_THREADS": "4", "MKL_NUM_THREADS": "4", "LB_N_JOBS": "4"}


def lb_dir(m, tag, s):
    return E / (f"leaderboard_{m}_{tag}" if s == 0 else f"leaderboard_{m}_{tag}_seed{s}")


def leaderboards(jobs):
    tasks = []
    for m, s in itertools.product(MODELS, SEEDS):
        syn = E / f"synth/{m}/seed{s}"
        if not (syn / "info.json").exists():
            continue
        # 시드 1~4는 s2s만 만든다. 이 확장의 목적은 생성기 내 tau_{S->S} 변동이고,
        # s2r까지 만들면 리더보드가 두 배가 되어 며칠이 걸린다(9/23).
        tags = [("s2s", f"{syn.relative_to(ROOT)}:test")]
        if s == 0:
            tags.append(("s2r", "data/tabred-hc:test"))
        for tag, test in tags:
            out = lb_dir(m, tag, s)
            if (out / "results.json").exists():
                continue
            tasks.append([sys.executable, "-u", "scripts/leaderboard.py",
                          "--train", str(syn.relative_to(ROOT)), "--tune", f"{syn.relative_to(ROOT)}:val",
                          "--test", test, "--out", str(out.relative_to(ROOT)),
                          "--models", "all", "--trials", "10", "--seeds", "3"])
    print(f"남은 리더보드 {len(tasks)}개")
    (E / "logs").mkdir(parents=True, exist_ok=True)
    running = []
    while tasks or running:
        while tasks and len(running) < jobs:
            cmd = tasks.pop(0)
            name = Path(cmd[cmd.index("--out") + 1]).name
            fh = open(E / "logs" / f"lb_{name}.txt", "a")
            running.append((name, subprocess.Popen(cmd, cwd=ROOT, env={**os.environ, **ENV},
                                                   stdout=fh, stderr=subprocess.STDOUT), fh))
            print(f"  시작 {name}", flush=True)
        time.sleep(20)
        for item in running[:]:
            name, p, fh = item
            if p.poll() is not None:
                fh.close(); running.remove(item)
                print(f"  끝   {name} rc={p.returncode} (남은 {len(tasks)}개)", flush=True)


def fidelity():
    """시드 0과 같은 방식으로 tau_b와 분리쌍을 계산한다."""
    for m, s in itertools.product(MODELS, SEEDS):
        cands = [str(lb_dir(m, t, s).relative_to(ROOT)) for t in ("s2r", "s2s")
                 if (lb_dir(m, t, s) / "results.json").exists()
                 and not (lb_dir(m, t, s) / "fidelity_vs_leaderboard_real.json").exists()]
        if cands:
            subprocess.run([sys.executable, "scripts/fidelity.py", "--real",
                            "exp/tabred-hc/leaderboard_real", "--cand", *cands],
                           cwd=ROOT, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
    print("fidelity 계산 완료")


def analyse():
    tau, vecs = {}, {}
    for m in MODELS:
        tau[m], vecs[m] = {}, {}
        for s in SEEDS:
            f = lb_dir(m, "s2s", s) / "fidelity_vs_leaderboard_real.json"
            if not f.exists():
                continue
            t = json.loads(f.read_text())["kendall_tau"]
            if t == t:  # NaN 제외: 합성 test에 양성이 없어 순위가 정의되지 않는 공개본
                tau[m][s] = float(t)
    usable = {m: v for m, v in tau.items() if len(v) >= 2}
    n_undefined = sum(len(SEEDS) - len(v) for v in tau.values())
    all_t = [t for v in usable.values() for t in v.values()]
    # 본 벤치마크(lf_analysis_v2.py)와 같은 정의: 생성기 안 제곱합 / 전체 제곱합
    grand = float(np.mean(all_t))
    ss_w = float(sum(sum((t - np.mean(list(v.values()))) ** 2 for t in v.values()) for v in usable.values()))
    ss_b = float(sum(len(v) * (np.mean(list(v.values())) - grand) ** 2 for v in usable.values()))
    within, between = ss_w, ss_b
    out = {"n_generators_usable": len(usable), "n_releases": len(all_t), "n_undefined": n_undefined,
           "tau_per_release": {m: v for m, v in tau.items()},
           "ss_within": ss_w, "ss_between": ss_b,
           "within_share": ss_w / (ss_w + ss_b),
           "range_per_generator": {m: [min(v.values()), max(v.values())] for m, v in usable.items()}}
    # 생성기 사이: 다섯 공개본 평균의 표준 지표와 평균 tau의 Spearman rho(정확 순열 p). 본 벤치마크 그림 3(a)와 같은 축.
    smf = E / "standard_metrics_seeds.json"
    if smf.exists():
        from scipy.stats import spearmanr
        sm = json.loads(smf.read_text())
        ms = sorted(usable)
        t_mean = [float(np.mean(list(usable[m].values()))) for m in ms]
        across = {}
        for key in ("ks_mean", "tvd_mean", "corr_rmse", "detection_auc", "pos_rate_abs_err_pp"):
            x = [-float(np.mean([sm[f"{m}|{s}"][key] for s in usable[m]])) for m in ms]
            rho = float(spearmanr(x, t_mean).statistic)
            null = [spearmanr(x, [t_mean[i] for i in pp]).statistic for pp in itertools.permutations(range(len(ms)))]
            across[key] = {"rho": rho, "p_exact": float(np.mean([abs(r) >= abs(rho) - 1e-12 for r in null]))}
        out["across_generator_5seed"] = {"generators": ms, "tau_mean": dict(zip(ms, t_mean)), "corr": across}
        print("  생성기 사이(5시드 평균):", {k: (round(v["rho"], 2), round(v["p_exact"], 3)) for k, v in across.items()})
    (E / "within_generator.json").write_text(json.dumps(out, indent=1))
    print(f"쓸 수 있는 생성기 {len(usable)}종, 공개본 {len(all_t)}개")
    print(f"  within 제곱합 {within:.4f}, between {between:.4f} -> within 비중 {out['within_share']:.2f}")
    for m, (lo, hi) in sorted(out["range_per_generator"].items(), key=lambda z: -(z[1][1] - z[1][0])):
        print(f"  {m:14s} {lo:+.2f} ~ {hi:+.2f}  (폭 {hi-lo:.2f})")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["leaderboards", "fidelity", "analyse"], required=True)
    ap.add_argument("--jobs", type=int, default=3)
    a = ap.parse_args()
    {"leaderboards": lambda: leaderboards(a.jobs), "fidelity": fidelity, "analyse": analyse}[a.stage]()
