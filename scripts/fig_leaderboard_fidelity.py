"""
그림 1: 공개본별 leaderboard fidelity. 같은 생성기의 실행(시드)들을 점으로 찍어 공개본 단위 변동을 보인다.

메시지: 순위 보존은 생성기가 아니라 공개본 하나하나의 속성이다. 생성기 평균(막대)보다
       같은 생성기 안의 실행 간 폭이 큰 경우가 많다.
스타일은 탑티어 벤치마크 논문(TabArena, TabReD)의 관습을 따른다: 막대 + 오차막대, 위쪽 한 줄 범례,
기준선, 색 3개 이하, 본문과 같은 Times 계열 8pt(이전 그림은 인쇄 시 약 4.4pt였다).

Usage:
    python scripts/fig_leaderboard_fidelity.py --out ../Fin-Syn-paper/figures/leaderboard_fidelity.pdf
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import os
# 논문 메인 표/그림은 생성기마다 같은 수의 공개본을 쓴다(사용자 지시 9/20). 기본 5시드.
MAX_SEEDS = int(os.environ.get("FINSYN_MAX_SEEDS", "5"))

ROOT = Path(__file__).resolve().parents[1]
E = ROOT / "exp/finsyn-v2"
LABELS = {"smote": "SMOTE", "tabpfgen": "TabPFGen", "tabpfgen-prior": "TabPFGen-prior", "tabddpm": "TabDDPM",
          "great": "GReaT", "tvae": "TVAE", "ctabgan": "CTAB-GAN", "ctgan": "CTGAN", "ctabgan-plus": "CTAB-GAN+",
          "tabsyn": "TabSyn", "tabdiff": "TabDiff", "findiff": "FinDiff"}
BLUE, GREY = "#1f4e79", "#8c8c8c"


def setup():
    plt.rcParams.update({
        "font.family": "serif", "font.serif": ["Liberation Serif", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8, "xtick.labelsize": 7.5,
        "ytick.labelsize": 7.5, "legend.fontsize": 7.5, "pdf.fonttype": 42, "ps.fonttype": 42,
        "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 0.6,
        "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    })


def runs(model, tag):
    out = []
    for s in range(MAX_SEEDS):
        f = E / (f"leaderboard_{model}_{tag}" if s == 0 else f"leaderboard_{model}_{tag}_seed{s}")
        f = f / "fidelity_vs_leaderboard_real.json"
        if f.exists():
            out.append(json.loads(f.read_text())["kendall_tau"])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    setup()
    nf = json.loads((E / "leaderboard_real/noise_floor.json").read_text())
    lo, mid = nf["tau_vs_full_q05"], nf["tau_vs_full_mean"]
    data = {m: runs(m, "s2s") for m in LABELS if runs(m, "s2s")}
    order = sorted(data, key=lambda m: np.mean(data[m]))          # 아래에서 위로 좋아지게
    fig, ax = plt.subplots(figsize=(5.5, 2.7))
    ax.axvspan(lo, 1.0, color=BLUE, alpha=0.07, lw=0)
    ax.axvline(mid, color=BLUE, ls="--", lw=0.8, zorder=1)
    ax.axvline(0, color="black", lw=0.6, zorder=1)
    means = [float(np.mean(data[m])) for m in order]
    lo_err = [means[i] - min(data[m]) for i, m in enumerate(order)]
    hi_err = [max(data[m]) - means[i] for i, m in enumerate(order)]
    ax.barh(range(len(order)), means, height=0.62, color=BLUE, alpha=0.85, zorder=2,
            xerr=[lo_err, hi_err], error_kw=dict(ecolor=GREY, elinewidth=0.8, capsize=2, capthick=0.8, zorder=3))
    for i, m in enumerate(order):
        ax.scatter(data[m], np.full(len(data[m]), i), s=7, facecolor="white", edgecolor=GREY, lw=0.6, zorder=4)
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels([LABELS[m] for m in order])
    ax.set_xlabel(r"Kendall $\tau$ vs. private leaderboard ($\uparrow$)")
    ax.set_xlim(-0.55, 1.02)
    ax.set_ylim(-0.75, len(order) - 0.25)
    ax.grid(axis="x", color=GREY, alpha=0.25, lw=0.5)
    ax.set_axisbelow(True)
    ax.set_xticks(np.arange(-0.4, 1.01, 0.2))
    ax.text(1.0, len(order) - 0.6, "resampling noise ", color=BLUE, ha="right", va="center", fontsize=7)
    ax.bar(0, 0, color=BLUE, alpha=0.85, label="Generator mean")
    ax.scatter([], [], s=7, facecolor="white", edgecolor=GREY, lw=0.6, label="Single release")
    ax.legend(loc="lower right", frameon=False, handletextpad=0.4, borderaxespad=0.4)
    fig.tight_layout(pad=0.4)
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight")
    print(f"saved -> {out}  ({len(order)} releases, runs: { {m: len(v) for m, v in data.items()} })")


if __name__ == "__main__":
    main()
