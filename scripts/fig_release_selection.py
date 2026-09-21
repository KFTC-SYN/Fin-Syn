"""
그림 3: 비공개 데이터로 재서 공개본을 고르면 실제로 더 좋은 공개본이 나온다.

비공개 test를 층화 반분해 한쪽(A)에서 tau가 가장 큰 실행을 고르고, 다른 쪽(B)에서 그 실행의 tau를 본다.
무작위 선택(그 생성기 실행들의 평균)과 사후 최선(oracle)을 함께 그린다. 값은 200회 반분의 평균,
수염은 5-95% 구간이다.

스타일 근거(선례): 범주(모델)별로 정책 세 가지를 묶어 세로 막대와 오차막대로 그리는 방식은
  - TabArena(NeurIPS'25 D&B) Fig.1: 모델별 Default/Tuned/Tuned+Ensembled 세 값, 오차막대, 축 위 한 줄 범례,
    파선 기준선에 그림 안 라벨
  - TableShift(NeurIPS'23 D&B) Fig.4(a): 막대 + 오차막대, 한 줄 범례
무작위와 선택을 잇는 띠, 기준 구간 음영처럼 참고 문헌에서 선례를 찾지 못한 표현은 쓰지 않는다.

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
# viridis 계열 세 단계: 연한 파랑, viridis(0.25), viridis(0.55).
PALE, DARK, TEAL, GREY = "#b9c3db", "#3b528b", "#21918c", "#8c8c8c"


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
    ms = sorted(res, key=lambda m: -res[m]["tau_B_selected"])   # 좋은 쪽이 왼쪽
    x = np.arange(len(ms))
    get = lambda k: np.array([res[m][k] for m in ms])
    err = lambda k, v: np.abs(np.array([res[m][k] for m in ms]).T - v)

    fig, ax = plt.subplots(figsize=(5.5, 1.95))
    w = 0.27
    for off, key, ckey, color, lab in (
            (-w, "tau_B_random", "ci_random", PALE, "Random run"),
            (0.0, "tau_B_selected", "ci_selected", DARK, "Audit-selected run"),
            (w, "tau_B_oracle", "ci_oracle", TEAL, "Best run (oracle)")):
        v = get(key)
        ax.bar(x + off, v, width=w, color=color, edgecolor="none", zorder=2, label=lab)
        ax.errorbar(x + off, v, yerr=err(ckey, v), fmt="none", ecolor=GREY,
                    elinewidth=0.6, capsize=0, zorder=3)

    ax.axhline(nf["tau_vs_full_q05"], color="black", ls="--", lw=0.8, zorder=4)
    ax.set_xticks(x)
    ax.set_xticklabels([LABELS[m] for m in ms], rotation=35, ha="right", rotation_mode="anchor")
    ax.set_ylabel(r"Kendall $\tau$ ($\uparrow$)")
    ax.set_xlim(-0.6, len(ms) - 0.4)
    lo_all = min(min(res[m][k]) if isinstance(res[m][k], list) else res[m][k]
                 for m in ms for k in ("ci_random", "ci_selected", "ci_oracle"))
    ax.set_ylim(min(0.0, lo_all) - 0.05, 1.0)
    if lo_all < 0:
        ax.axhline(0, color=GREY, lw=0.6, zorder=1)
    ax.grid(axis="y", color=GREY, alpha=0.25, lw=0.5)
    ax.set_axisbelow(True)
    ax.text(len(ms) - 0.4, nf["tau_vs_full_q05"] + 0.015, "Resampling noise",
            ha="right", va="bottom", fontsize=7)
    # 범례: TabArena Fig.1처럼 축 위 가운데 한 줄, 얇은 테두리 상자.
    leg = ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.01), ncol=3, frameon=True,
                    handletextpad=0.5, borderaxespad=0, columnspacing=1.6, handlelength=1.5,
                    borderpad=0.4)
    leg.get_frame().set(edgecolor="#b0b0b0", facecolor="white", linewidth=0.6)
    fig.tight_layout(pad=0.4)
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight")
    print(f"saved -> {out} ({len(ms)} generators)")


if __name__ == "__main__":
    main()
