"""
그림: 비공개 데이터로 재서 공개본을 고르면 실제로 더 좋은 공개본이 나온다.

비공개 test를 층화 반분해 한쪽(A)에서 tau가 가장 큰 실행을 고르고, 다른 쪽(B)에서 그 실행의 tau를 본다.
무작위 선택(그 생성기 실행들의 평균)과 사후 최선(oracle)을 함께 그린다. 막대는 200회 반분의 평균,
수염은 5-95% 구간이다. 스타일은 탑티어 벤치마크 논문 관습(가로 막대, 모델 이름은 y축, 방향 표시, 한 줄 범례)을 따른다.

Usage:
    python scripts/fig_release_selection.py --out ../Fin-Syn-paper/figures/release_selection.pdf
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
E = ROOT / "exp/finsyn-v2"
LABELS = {"smote": "SMOTE", "tabpfgen": "TabPFGen", "tabpfgen-prior": "TabPFGen-prior", "tabddpm": "TabDDPM",
          "great": "GReaT", "tvae": "TVAE", "ctabgan": "CTAB-GAN", "ctgan": "CTGAN", "ctabgan-plus": "CTAB-GAN+",
          "tabsyn": "TabSyn", "tabdiff": "TabDiff", "findiff": "FinDiff"}
BLUE, LIGHT, GREY = "#1f4e79", "#9db8d2", "#8c8c8c"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    plt.rcParams.update({
        "font.family": "serif", "font.serif": ["Liberation Serif", "DejaVu Serif"], "mathtext.fontset": "stix",
        "font.size": 8, "axes.labelsize": 8, "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "legend.fontsize": 7.5,
        "pdf.fonttype": 42, "ps.fonttype": 42, "axes.spines.top": False, "axes.spines.right": False,
        "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6})
    res = json.loads((E / "release_selection.json").read_text())["s2s"]["per_generator"]
    nf = json.loads((E / "leaderboard_real/noise_floor.json").read_text())
    ms = sorted(res, key=lambda m: res[m]["tau_B_selected"])
    y = np.arange(len(ms))
    get = lambda k: np.array([res[m][k] for m in ms])
    sel, rnd, orc = get("tau_B_selected"), get("tau_B_random"), get("tau_B_oracle")
    ci = lambda k, v: np.abs(np.array([res[m][k] for m in ms]).T - v)

    fig, ax = plt.subplots(figsize=(5.5, 2.7))
    ax.axvspan(nf["tau_vs_full_q05"], 1.0, color=BLUE, alpha=0.07, lw=0)
    ax.axvline(nf["tau_vs_full_mean"], color=BLUE, ls="--", lw=0.8, zorder=1)
    ax.barh(y - 0.2, rnd, height=0.4, color=LIGHT, label="Random run",
            xerr=ci("ci_random", rnd) if "ci_random" in res[ms[0]] else None,
            error_kw=dict(ecolor=GREY, elinewidth=0.7, capsize=1.5, capthick=0.7))
    ax.barh(y + 0.2, sel, height=0.4, color=BLUE, label="Audit-selected run",
            xerr=ci("ci_selected", sel) if "ci_selected" in res[ms[0]] else None,
            error_kw=dict(ecolor=GREY, elinewidth=0.7, capsize=1.5, capthick=0.7))
    ax.scatter(orc, y + 0.2, s=16, marker="D", facecolor="white", edgecolor="black", lw=0.7, zorder=4,
               label="Best run (oracle)")
    ax.set_yticks(y)
    ax.set_yticklabels([LABELS[m] for m in ms])
    ax.set_xlabel(r"Kendall $\tau$ on the held-out half ($\uparrow$)")
    ax.set_xlim(0, 1.0)
    ax.set_ylim(-0.7, len(ms) + 0.05)
    ax.grid(axis="x", color=GREY, alpha=0.25, lw=0.5)
    ax.set_axisbelow(True)
    ax.text(1.0, len(ms) - 0.25, "resampling noise ", color=BLUE, ha="right", va="center", fontsize=7)
    ax.legend(loc="lower right", frameon=False, handletextpad=0.5, borderaxespad=0.3)
    fig.tight_layout(pad=0.4)
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight")
    print(f"saved -> {out} ({len(ms)} generators)")


if __name__ == "__main__":
    main()
