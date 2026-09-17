"""
Figure 1 (teaser): 릴리스별 leaderboard fidelity + 벤치마크 구축 조건.

(a) 9개 합성 릴리스의 Kendall tau (s2s = 공개 데이터만 쓰는 사용자, s2r = TSTR),
    비공개 test를 부트스트랩 재표집했을 때 나오는 tau 분포(노이즈 밴드)를 배경에 표시.
(b) 벤치마크 구축 방식(자연 분포 / 랜덤 분할 / 중복+랜덤 분할)의 tau.

Usage:
    python scripts/fig_leaderboard_fidelity.py \
        --out ../Fin-Syn-paper/202609_iclr/figures/leaderboard_fidelity.pdf
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

LABELS = {
    "smote": "SMOTE", "tabpfgen": "TabPFGen", "tabpfgen-prior": "TabPFGen-prior",
    "tabddpm": "TabDDPM", "great": "GReaT", "tvae": "TVAE",
    "ctabgan": "CTAB-GAN", "ctgan": "CTGAN", "ctabgan-plus": "CTAB-GAN+",
}
ORDER = ["smote", "tabpfgen", "tabpfgen-prior", "tabddpm", "great", "tvae", "ctabgan", "ctgan", "ctabgan-plus"]


def tau(name):
    f = E / f"leaderboard_{name}/fidelity_vs_leaderboard_real.json"
    return json.loads(f.read_text())["kendall_tau"]


def tau_seeds(model, tag):
    """생성기 시드 0,1,2의 tau. 없는 시드는 건너뛴다."""
    out = []
    for s in (0, 1, 2):
        d = E / (f"leaderboard_{model}_{tag}" if s == 0 else f"leaderboard_{model}_{tag}_seed{s}")
        f = d / "fidelity_vs_leaderboard_real.json"
        if f.exists():
            out.append(json.loads(f.read_text())["kendall_tau"])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    nf = json.loads((E / "leaderboard_real/noise_floor.json").read_text())
    lo, mid = nf["tau_vs_full_q05"], nf["tau_vs_full_mean"]

    seeds_s2s = [tau_seeds(m, "s2s") for m in ORDER]
    seeds_s2r = [tau_seeds(m, "s2r") for m in ORDER]
    s2s = [sum(v) / len(v) for v in seeds_s2s]
    s2r = [sum(v) / len(v) for v in seeds_s2r]
    cond = [("Natural prevalence", tau("real_natural")),
            ("Random split", tau("cond_random")),
            ("Duplicates + random split", tau("cond_dup"))]

    plt.rcParams.update({"font.family": "serif", "font.size": 8.5, "axes.linewidth": 0.7})
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(11.0, 3.0), gridspec_kw={"width_ratios": [2.15, 1.0]})

    # ---- (a) 릴리스 ----
    y = np.arange(len(ORDER))[::-1]
    h = 0.36
    ax.axvspan(lo, 1.0, color="#cfe0f3", alpha=0.55, lw=0, zorder=0)
    ax.axvline(mid, color="#2f4b7c", lw=0.9, ls="--", zorder=1)
    ax.axvline(0, color="#666666", lw=0.8, zorder=1)
    ax.barh(y + h / 2, s2s, height=h, color="#2f4b7c", zorder=2, label="public-user ($S\\!\\to\\!S$), mean of 3 seeds")
    ax.barh(y - h / 2, s2r, height=h, color="#9cb4d4", zorder=2, label="TSTR ($S\\!\\to\\!R$), mean of 3 seeds")
    for yy, vs, off in [(y + h / 2, seeds_s2s, 0), (y - h / 2, seeds_s2r, 0)]:
        for yi, v in zip(yy, vs):
            if len(v) > 1:
                ax.plot([min(v), max(v)], [yi, yi], color="#222222", lw=0.9, zorder=3,
                        solid_capstyle="butt")
                ax.plot(v, [yi] * len(v), "|", color="#222222", ms=4, mew=0.9, zorder=4)
    for yy, v, vs in zip(y + h / 2, s2s, seeds_s2s):
        e = max(vs) if v >= 0 else min(vs)
        ax.text(e + (0.02 if v >= 0 else -0.02), yy, f"{v:+.2f}", va="center",
                ha="left" if v >= 0 else "right", fontsize=7.5)
    ax.set_yticks(y, [LABELS[m] for m in ORDER])
    ax.set_xlim(-0.62, 1.04)
    ax.set_xlabel("Kendall $\\tau$ against the private leaderboard")
    ax.set_title("(a) Synthetic releases", fontsize=9, loc="left")
    ax.text((lo + 1.0) / 2, len(ORDER) - 1.35, "bootstrap noise band", fontsize=7.5, color="#2f4b7c",
            rotation=90, va="top", ha="center")
    ax.legend(loc="upper left", fontsize=7.5, frameon=False)
    ax.spines[["top", "right"]].set_visible(False)

    # ---- (b) 구축 조건 ----
    yb = np.arange(len(cond))[::-1]
    bx.axvspan(lo, 1.0, color="#cfe0f3", alpha=0.55, lw=0, zorder=0)
    bx.axvline(mid, color="#2f4b7c", lw=0.9, ls="--", zorder=1)
    bx.barh(yb, [c[1] for c in cond], height=0.42, color="#7a5195", zorder=2)
    for yy, (_, v) in zip(yb, cond):
        bx.text(v + 0.02, yy, f"{v:+.2f}", va="center", fontsize=7.5)
    bx.set_yticks(yb, [c[0] for c in cond])
    bx.set_xlim(0, 1.04)
    bx.set_xlabel("Kendall $\\tau$")
    bx.set_title("(b) Benchmark construction", fontsize=9, loc="left")
    bx.spines[["top", "right"]].set_visible(False)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight", pad_inches=0.02)
    print(f"wrote {out}  (noise band: q05={lo:.3f}, mean={mid:.3f})")


if __name__ == "__main__":
    main()
