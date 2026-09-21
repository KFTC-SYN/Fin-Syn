"""
부록 그림: 은행 간 이체가 거치는 단계.

9/21 재작성 이유: 기존 PDF가 Type 3 DejaVu 글꼴을 써서 제출본 글꼴 점검에 걸렸고,
본문 그림 3개(Liberation Serif 8pt, pdf.fonttype 42)와 서체가 달랐다.

Usage:
    python scripts/fig_interbank_flow.py --out ../Fin-Syn-paper/figures/interbank_flow.pdf
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

# viridis 계열: 테두리 viridis(0.25), 채움은 그 옅은 톤.
BLUE, FILL, GREY = "#3b528b", "#eceef6", "#8c8c8c"
BOXES = ["Customer", "Access\nchannel", "Sending\nbank", "Clearing\nnetwork", "Receiving\nbank"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    plt.rcParams.update({
        "font.family": "serif", "font.serif": ["Liberation Serif", "DejaVu Serif"],
        "mathtext.fontset": "stix", "font.size": 8, "pdf.fonttype": 42, "ps.fonttype": 42})

    fig, ax = plt.subplots(figsize=(5.5, 1.35))
    ax.set_xlim(0, 103); ax.set_ylim(0, 34); ax.axis("off")

    w, h, gap = 15.0, 14.0, 6.0
    xs = [2 + i * (w + gap) for i in range(len(BOXES))]
    for x, label in zip(xs, BOXES):
        ax.add_patch(FancyBboxPatch((x, 10), w, h, boxstyle="round,pad=0,rounding_size=1.2",
                                    linewidth=0.8, edgecolor=BLUE, facecolor=FILL))
        ax.text(x + w / 2, 17, label, ha="center", va="center", fontsize=8)

    for i in range(len(BOXES) - 1):
        x0, x1 = xs[i] + w, xs[i + 1]
        ax.add_patch(FancyArrowPatch((x0 + 0.6, 20), (x1 - 0.6, 20), arrowstyle="-|>",
                                     mutation_scale=7, linewidth=0.8, color="black"))
        ax.add_patch(FancyArrowPatch((x1 - 0.6, 14), (x0 + 0.6, 14), arrowstyle="-|>",
                                     mutation_scale=7, linewidth=0.8, color="black"))
        ax.text((x0 + x1) / 2, 25.5, f"({i + 1})", ha="center", va="center", fontsize=7.5)
        ax.text((x0 + x1) / 2, 7.5, f"({8 - i})", ha="center", va="center", fontsize=7.5)

    ax.text(51.5, 1.5, "(1) to (4) transfer request and clearing message    |    "
                     "(5) to (8) credit decision and status response",
            ha="center", va="center", fontsize=7.5, color=GREY)

    fig.tight_layout(pad=0.2)
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight")
    print(f"saved -> {out}")


if __name__ == "__main__":
    main()
