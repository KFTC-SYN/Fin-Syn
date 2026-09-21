"""
마지막 실험(GReaT 시드 2)이 끝나면 전체 분석을 돌리고, 필요하면 추가 시드까지 자동으로 이어 간다.

흐름
  1) GReaT 시드 2의 리더보드(fidelity 파일)가 생길 때까지 기다린다. great_lb_watch.sh가 만든다.
  2) 분석 일괄 실행: 표준지표(시드 0 / 시드별), 프라이버시(무작위·맞춘 비교표본, 시드별), augmentation,
     lf_analysis, tstr_vs_lf, within_generator, release_selection, tabm_extension, 표, 그림.
  3) 추가 실험이 필요한지 규칙으로 판단한다.
       - 시드가 5개 미만이고, 기존 시드의 tau 범위가 0.15보다 넓어 변동을 더 재야 하는 생성기
       - 시드당 GPU 시간이 2시간을 넘지 않는 생성기만(gen_runs.jsonl 실측). GReaT는 제외(시드당 약 13시간)
       - 새로 도는 part 작업 수는 24개로 제한
     해당하면 생성 대기열을 그 생성기·시드로 다시 돌리고(리더보드까지 자동), 끝나면 2)를 한 번 더 한다.
  4) exp/finsyn-v2/auto_finalize_report.json 에 무엇을 돌렸고 무엇이 실패했는지 남긴다.

Usage:
    nohup python -u scripts/auto_finalize_v2.py > exp/finsyn-v2/logs/auto_finalize.txt 2>&1 &
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
ALL = ["smote", "tabpfgen", "tabpfgen-prior", "tabddpm", "great", "tvae", "ctgan", "ctabgan", "ctabgan-plus",
       "tabsyn", "tabdiff", "findiff"]
# 메인 표/그림은 생성기마다 같은 수의 공개본을 쓴다(사용자 지시 9/20). 기본 5시드.
MAX_SEEDS = int(os.environ.get("FINSYN_MAX_SEEDS", "5"))
SEEDS_ARG = ",".join(str(s) for s in range(MAX_SEEDS))
WAIT_FOR = [E / f"leaderboard_great_{t}_seed2/fidelity_vs_leaderboard_real.json" for t in ("s2r", "s2s")]
MAX_NEW_PART_JOBS = 24
TAU_RANGE_TRIGGER = 0.15
GPU_MIN_PER_SEED_LIMIT = 120
report = {"started": time.strftime("%F %T"), "steps": [], "extra_experiments": None}


def log(msg):
    print(f"[{time.strftime('%m-%d %H:%M')}] {msg}", flush=True)


def run(name, cmd, env_extra=None):
    t0 = time.time()
    fh = open(E / "logs/auto_finalize_steps.txt", "a")
    fh.write(f"\n===== {name} :: {' '.join(map(str, cmd))}\n"); fh.flush()
    rc = subprocess.run(cmd, cwd=ROOT, env={**os.environ, "OMP_NUM_THREADS": "4", **(env_extra or {})},
                        stdout=fh, stderr=subprocess.STDOUT).returncode
    report["steps"].append({"step": name, "rc": rc, "minutes": round((time.time() - t0) / 60, 1)})
    (E / "auto_finalize_report.json").write_text(json.dumps(report, indent=1))
    log(f"{name}: rc={rc} ({(time.time()-t0)/60:.1f} min)")
    return rc


def analysis_chain(tag):
    models = ",".join(ALL)
    steps = [
        (f"{tag}/standard_metrics", [PY, "-u", "scripts/standard_metrics_v2.py", "--models", models]),
        (f"{tag}/standard_metrics_seeds", [PY, "-u", "scripts/standard_metrics_v2.py", "--models", models,
                                           "--synth_seeds", SEEDS_ARG]),
        (f"{tag}/privacy", [PY, "-u", "scripts/privacy_v2.py", "--models", models]),
        (f"{tag}/privacy_matched", [PY, "-u", "scripts/privacy_v2.py", "--holdout", "matched", "--models", models]),
        (f"{tag}/privacy_seeds", [PY, "-u", "scripts/privacy_v2.py", "--models", models, "--synth_seeds", SEEDS_ARG]),
        (f"{tag}/augmentation", [PY, "-u", "scripts/augmentation_v2.py", "--models", models, "--detectors", "lgbm,xgb"]),
        (f"{tag}/augmentation_r1", [PY, "-u", "scripts/augmentation_v2.py", "--models", models, "--fractions",
                                    "0.05,0.10", "--detectors", "lgbm,xgb", "--syn_ratio", "1"]),
        (f"{tag}/augmentation_r4", [PY, "-u", "scripts/augmentation_v2.py", "--models", models, "--fractions",
                                    "0.05,0.10", "--detectors", "lgbm,xgb", "--syn_ratio", "4"]),
        (f"{tag}/lf_analysis", [PY, "-u", "scripts/lf_analysis_v2.py"]),
        (f"{tag}/tstr_vs_lf", [PY, "-u", "scripts/tstr_vs_lf_v2.py"]),
        (f"{tag}/within_generator", [PY, "-u", "scripts/within_generator_v2.py"]),
        (f"{tag}/release_selection", [PY, "-u", "scripts/release_selection_v2.py"]),
        (f"{tag}/tabm_extension", [PY, "-u", "scripts/tabm_extension_v2.py"]),
        (f"{tag}/tables", [PY, "-u", "scripts/make_tables_v2.py"]),
        (f"{tag}/figure", [PY, "-u", "scripts/fig_leaderboard_fidelity.py", "--out",
                           str(ROOT.parent / "Fin-Syn-paper/figures/leaderboard_fidelity.pdf")]),
        (f"{tag}/figure_metric", [PY, "-u", "scripts/fig_metric_vs_tau.py", "--out",
                                  str(ROOT.parent / "Fin-Syn-paper/figures/metric_vs_tau.pdf")]),
        (f"{tag}/figure_select", [PY, "-u", "scripts/fig_release_selection.py", "--out",
                                  str(ROOT.parent / "Fin-Syn-paper/figures/release_selection.pdf")]),
    ]
    for name, cmd in steps:
        run(name, cmd)


def gpu_minutes_per_seed():
    """생성기별 시드 1개(3기간 합계)의 GPU 시간 중앙값. 기록이 없으면 None."""
    recs = [json.loads(l) for l in (E / "gen_runs.jsonl").read_text().splitlines() if l.strip()]
    per = {}
    for r in recs:
        if r.get("rc") == 0 and "minutes" in r:
            per.setdefault(r["model"], {}).setdefault(r.get("seed", 0), 0)
            per[r["model"]][r.get("seed", 0)] += r["minutes"]
    import statistics
    return {m: statistics.median(v.values()) for m, v in per.items() if v}


def decide_extra():
    lf = json.loads((E / "lf_analysis.json").read_text())["releases"]
    cost = gpu_minutes_per_seed()
    picks = []
    for m, v in lf.items():
        s = v["s2s"]
        if m == "great" or s["n_seeds"] >= 5:
            continue
        spread = s["tau_max"] - s["tau_min"]
        c = cost.get(m)
        if spread > TAU_RANGE_TRIGGER and c is not None and c <= GPU_MIN_PER_SEED_LIMIT:
            picks.append({"model": m, "n_seeds": s["n_seeds"], "tau_range": round(spread, 3),
                          "gpu_min_per_seed": round(c, 1), "add_seeds": list(range(s["n_seeds"], 5))})
    # part 작업 수 제한
    total = sum(len(p["add_seeds"]) * 3 for p in picks)
    while total > MAX_NEW_PART_JOBS and picks:
        picks.sort(key=lambda p: p["gpu_min_per_seed"])
        dropped = picks.pop()
        log(f"skip {dropped['model']} (budget)")
        total = sum(len(p["add_seeds"]) * 3 for p in picks)
    return picks


def main():
    log("waiting for GReaT seed 2 leaderboards")
    deadline = time.time() + 6 * 3600
    while not all(p.exists() for p in WAIT_FOR):
        if time.time() > deadline:
            log("timeout waiting for GReaT seed 2; continuing without it")
            break
        time.sleep(120)
    log("running analysis chain (pass 1)")
    analysis_chain("pass1")
    picks = decide_extra()
    report["extra_experiments"] = picks
    (E / "auto_finalize_report.json").write_text(json.dumps(report, indent=1))
    if picks:
        models = ",".join(p["model"] for p in picks)
        log(f"extra seeds needed: {picks}")
        rc = run("extra_seeds", [PY, "-u", "scripts/run_tabgen_queue_v2.py"],
                 {"FINSYN_QUEUE_MODELS": models, "FINSYN_QUEUE_SEEDS": "0,1,2,3,4"})
        log(f"extra seed queue finished rc={rc}; re-running analysis")
        analysis_chain("pass2")
    else:
        log("no extra experiments needed")
    report["finished"] = time.strftime("%F %T")
    (E / "auto_finalize_report.json").write_text(json.dumps(report, indent=1))
    bad = [s for s in report["steps"] if s["rc"] != 0]
    log(f"done. failed steps: {[s['step'] for s in bad] or 'none'}")


if __name__ == "__main__":
    main()
