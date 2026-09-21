"""
그림 1: 공개본 60개를 leaderboard fidelity로 정렬하고 생성기 계열로 칠한다.

메시지: 순위 보존은 생성기가 아니라 공개본 하나하나의 속성이다. 생성기가 결정한다면 같은 색이
       다섯 개씩 뭉쳐야 하는데, 상위 10개가 3개 계열에서 나오고 같은 생성기의 공개본이
       1등과 47등에 함께 놓인다.

스타일 근거(선례): 순위대로 정렬한 막대를 범주 색으로 칠하고 범례를 붙이는 방식은
  TableShift(NeurIPS'23 D&B) Fig.4(a)가 모델 19개를 "Model Type" 색으로 칠한 것과 같다.
  파선 기준선에 그림 안 라벨을 붙이는 것은 TabArena(NeurIPS'25 D&B) Fig.1을 따른다.

Usage:
    python scripts/fig_release_ranking.py --out ../Fin-Syn-paper/figures/release_ranking.pdf
"""
import argparse
import json
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

MAX_SEEDS = int(os.environ.get("FINSYN_MAX_SEEDS", "5"))
ROOT = Path(__file__).resolve().parents[1]
E = ROOT / "exp/finsyn-v2"

GEN = ["smote", "tabsyn", "tabdiff", "findiff", "tabddpm", "tabpfgen", "tabpfgen-prior",
       "great", "tvae", "ctgan", "ctabgan", "ctabgan-plus"]
FAMILY = {"smote": "Interpolation",
          "tabsyn": "Diffusion", "tabdiff": "Diffusion", "findiff": "Diffusion", "tabddpm": "Diffusion",
          "tabpfgen": "Foundation model", "tabpfgen-prior": "Foundation model",
          "great": "Language model",
          "tvae": "GAN or VAE", "ctgan": "GAN or VAE", "ctabgan": "GAN or VAE", "ctabgan-plus": "GAN or VAE"}
# viridis 계열 다섯 단계. 흰 배경에서 약해지는 밝은 끝은 피한다.
COLOR = {"Interpolation": "#440154", "Diffusion": "#3b528b", "Foundation model": "#21918c",
         "Language model": "#5ec962", "GAN or VAE": "#b8b8b8"}
ORDER = ["Interpolation", "Diffusion", "Foundation model", "Language model", "GAN or VAE"]
GREY = "#8c8c8c"


def setup(tick):
    plt.rcParams.update({
        "font.family": "serif", "font.serif": ["Liberation Serif", "DejaVu Serif"],
        "mathtext.fontset": "stix", "font.size": 8, "axes.labelsize": 8,
        "xtick.labelsize": tick, "ytick.labelsize": tick, "legend.fontsize": tick - 0.5,
        "pdf.fonttype": 42, "ps.fonttype": 42, "axes.spines.top": False, "axes.spines.right": False,
        "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    })


def releases():
    out = []
    for m in GEN:
        for s in range(MAX_SEEDS):
            f = E / (f"leaderboard_{m}_s2s" if s == 0 else f"leaderboard_{m}_s2s_seed{s}")
            f = f / "fidelity_vs_leaderboard_real.json"
            if f.exists():
                out.append((json.loads(f.read_text())["kendall_tau"], m))
    out.sort(key=lambda z: -z[0])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--width", type=float, default=2.75)
    ap.add_argument("--height", type=float, default=2.0)
    ap.add_argument("--tick", type=float, default=6.5)
    a = ap.parse_args()
    setup(a.tick)

    rel = releases()
    lo = json.loads((E / "leaderboard_real/noise_floor.json").read_text())["tau_vs_full_q05"]
    n_band = sum(1 for t, _ in rel if t >= lo)
    n_gen = len({m for t, m in rel if t >= lo})

    fig, ax = plt.subplots(figsize=(a.width, a.height))
    x = np.arange(len(rel))
    ax.bar(x, [t for t, _ in rel], width=0.85,
           color=[COLOR[FAMILY[m]] for _, m in rel], edgecolor="none", zorder=2)
    ax.axhline(0, color=GREY, lw=0.6, zorder=1)
    ax.axhline(lo, color="black", ls="--", lw=0.8, zorder=3)

    ax.set_xlim(-1, len(rel))
    ax.set_ylim(-0.4, 1.0)
    ax.set_xticks([0, 19, 39, 59])
    ax.set_xticklabels(["1", "20", "40", "60"])
    ax.set_xlabel("Releases, ranked by leaderboard fidelity")
    ax.set_ylabel(r"Kendall $\tau$ ($\uparrow$)")
    ax.grid(axis="y", color=GREY, alpha=0.25, lw=0.5)
    ax.set_axisbelow(True)
    ax.text(len(rel) - 1, lo + 0.02, "Resampling noise", ha="right", va="bottom", fontsize=a.tick)

    handles = [plt.Rectangle((0, 0), 1, 1, color=COLOR[f]) for f in ORDER]
    leg = ax.legend(handles, ORDER, loc="lower left", ncol=2, frameon=True,
                    handlelength=1.1, handletextpad=0.4, columnspacing=1.0,
                    borderpad=0.35, labelspacing=0.3)
    leg.get_frame().set(edgecolor="#b0b0b0", facecolor="white", linewidth=0.6)

    fig.tight_layout(pad=0.3)
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight")
    print(f"saved -> {out}  ({len(rel)} releases, {n_band} in the band from {n_gen} generators)")


if __name__ == "__main__":
    main()
