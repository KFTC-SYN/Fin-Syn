"""
그림 1: 공개본별 leaderboard fidelity. 생성기마다 5개 실행(시드)의 분포를 상자그림으로 보인다.

메시지: 순위 보존은 생성기가 아니라 공개본 하나하나의 속성이다. 같은 생성기 안의 실행 간 폭이
       생성기 사이의 차이보다 큰 경우가 많다.

스타일 근거(선례): 범주별 분포를 세로 상자그림으로 그리는 방식은
  - SynReal(ICML'23) Fig.6: 세로 상자그림, 회전한 x축 라벨, 파선 기준선
  - Why-tree-based(NeurIPS'22) Fig.3: 방법별 세로 상자그림
파선 기준선에 그림 안 라벨을 붙이는 방식은 TabArena(NeurIPS'25 D&B) Fig.1("AutoGluon 1.3 (4h)")의 관례를 따른다.
점 5개를 직접 찍는 strip/dot 방식과 기준 구간을 띠로 칠하는 방식은 참고 문헌에서 선례를 찾지 못해 쓰지 않는다.

Usage:
    python scripts/fig_leaderboard_fidelity.py --out ../Fin-Syn-paper/figures/leaderboard_fidelity.pdf
"""
import argparse
import json
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# 논문 메인 표/그림은 생성기마다 같은 수의 공개본을 쓴다(사용자 지시 9/20). 기본 5시드.
MAX_SEEDS = int(os.environ.get("FINSYN_MAX_SEEDS", "5"))

ROOT = Path(__file__).resolve().parents[1]
E = ROOT / "exp/finsyn-v2"
LABELS = {"smote": "SMOTE", "tabpfgen": "TabPFGen", "tabpfgen-prior": "TabPFGen-prior", "tabddpm": "TabDDPM",
          "great": "GReaT", "tvae": "TVAE", "ctabgan": "CTAB-GAN", "ctgan": "CTGAN", "ctabgan-plus": "CTAB-GAN+",
          "tabsyn": "TabSyn", "tabdiff": "TabDiff", "findiff": "FinDiff"}
# 색은 viridis 계열로 통일한다: 상자 채움 viridis(0.25) 연하게, 테두리·중앙값 viridis(0.25).
DARK, FILL, GREY = "#3b528b", "#b9c3db", "#8c8c8c"


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
    ap.add_argument("--width", type=float, default=5.5)
    ap.add_argument("--height", type=float, default=1.95)
    ap.add_argument("--rotation", type=float, default=35)
    ap.add_argument("--tick", type=float, default=7.5)
    a = ap.parse_args()
    setup()
    plt.rcParams.update({"xtick.labelsize": a.tick, "ytick.labelsize": a.tick})
    nf = json.loads((E / "leaderboard_real/noise_floor.json").read_text())
    lo = nf["tau_vs_full_q05"]
    data = {m: runs(m, "s2s") for m in LABELS if runs(m, "s2s")}
    order = sorted(data, key=lambda m: -np.mean(data[m]))      # 좋은 쪽이 왼쪽

    fig, ax = plt.subplots(figsize=(a.width, a.height))
    ax.axhline(0, color=GREY, lw=0.6, zorder=1)
    ax.axhline(lo, color="black", ls="--", lw=0.8, zorder=1)

    bp = ax.boxplot([data[m] for m in order], positions=range(len(order)), widths=0.58,
                    patch_artist=True, whis=(0, 100), zorder=3)
    for b in bp["boxes"]:
        b.set(facecolor=FILL, edgecolor=DARK, lw=0.8)
    for k in ("whiskers", "caps"):
        for e in bp[k]:
            e.set(color=DARK, lw=0.8)
    for e in bp["medians"]:
        e.set(color=DARK, lw=1.3)

    ax.set_xticks(range(len(order)))
    ax.set_xticklabels([LABELS[m] for m in order], rotation=a.rotation, ha="right", rotation_mode="anchor")
    ax.set_ylabel(r"Kendall $\tau$ ($\uparrow$)")
    ax.set_xlim(-0.65, len(order) - 0.35)
    ax.set_ylim(-0.4, 1.0)
    ax.grid(axis="y", color=GREY, alpha=0.25, lw=0.5)
    ax.set_axisbelow(True)
    ax.text(len(order) - 0.4, lo + 0.02, "Resampling noise", ha="right", va="bottom", fontsize=7)

    fig.tight_layout(pad=0.4)
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight")
    print(f"saved -> {out}  ({len(order)} generators, runs: { {m: len(v) for m, v in data.items()} })")


if __name__ == "__main__":
    main()
