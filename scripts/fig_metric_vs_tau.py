"""
그림: 분포 지표는 생성기는 고르지만 공개본은 고르지 못한다.

(a) 생성기 사이: 생성기 평균 KS(작을수록 좋음)와 평균 tau. 우리 데이터와 공개 데이터(TabReD)를 함께.
(b) 같은 생성기 안: 각 실행의 KS와 tau를 생성기 평균에서 뺀 편차로 그린다. 관계가 없으면 구름처럼 퍼진다.

Usage:
    python scripts/fig_metric_vs_tau.py --out ../Fin-Syn-paper/figures/metric_vs_tau.pdf
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import rankdata

ROOT = Path(__file__).resolve().parents[1]
E = ROOT / "exp/finsyn-v2"
LABELS = {"smote": "SMOTE", "tabpfgen": "TabPFGen", "tabpfgen-prior": "TabPFGen-prior", "tabddpm": "TabDDPM",
          "great": "GReaT", "tvae": "TVAE", "ctabgan": "CTAB-GAN", "ctgan": "CTGAN", "ctabgan-plus": "CTAB-GAN+",
          "tabsyn": "TabSyn", "tabdiff": "TabDiff", "findiff": "FinDiff"}
# viridis 계열: 주색 viridis(0.25), 대비색 viridis(0.45) teal.
BLUE, ORANGE, GREY = "#3b528b", "#25848e", "#8c8c8c"


def spearman(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    ok = ~(np.isnan(x) | np.isnan(y))
    if ok.sum() < 3:
        return np.nan
    return float(np.corrcoef(rankdata(x[ok]), rankdata(y[ok]))[0, 1])


def setup():
    plt.rcParams.update({
        "font.family": "serif", "font.serif": ["Liberation Serif", "DejaVu Serif"], "mathtext.fontset": "stix",
        "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8, "xtick.labelsize": 7.5, "ytick.labelsize": 7.5,
        "legend.fontsize": 7, "pdf.fonttype": 42, "ps.fonttype": 42, "axes.spines.top": False,
        "axes.spines.right": False, "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    setup()
    lf = json.loads((E / "lf_analysis.json").read_text())["releases"]
    sm = json.loads((E / "standard_metrics.json").read_text())
    per_seed = json.loads((E / "standard_metrics_seeds.json").read_text())
    # 패널을 각각 독립 파일로 낸다. 본문에서 subfigure로 조립하므로 그림 안 제목은 두지 않는다(9/21).
    fig1, ax1 = plt.subplots(figsize=(2.72, 2.35))
    fig2, ax2 = plt.subplots(figsize=(2.72, 2.35))
    fig = fig1   # 아래 canvas.draw()가 쓰는 이름

    # (a) 생성기 사이
    ms = [m for m in LABELS if m in lf and m in sm]
    placed = []
    x = [sm[m]["ks_mean"] for m in ms]
    y = [lf[m]["s2s"]["tau_mean"] for m in ms]
    rho_pub = float("nan")
    rho_ours = spearman([-v for v in x], y)
    ax1.scatter(x, y, s=18, color=BLUE, zorder=3, label="Interbank")
    others = list(zip(x, y))
    for m, xi, yi in zip(ms, x, y):
        placed.append((m, xi, yi))
    tr = E.parent / "tabred-hc/lf_analysis.json"
    if tr.exists():
        t = json.loads(tr.read_text())
        tsm = json.loads((E.parent / "tabred-hc/standard_metrics.json").read_text())
        taus = t["tau_s2s"]
        mt = [m for m in taus if m in tsm and not np.isnan(taus[m])]
        if mt:
            xt = [tsm[m]["ks_mean"] for m in mt]
            yt = [taus[m] for m in mt]
            rho_pub = spearman([-v for v in xt], yt)
            ax1.scatter(xt, yt, s=18, facecolor="white", edgecolor=ORANGE, lw=0.9, zorder=3,
                        label="TabReD")
            others += list(zip(xt, yt))
    # 이름표 배치: 실제 글자 상자를 재서 후보 위치를 시도하고, 전부 겹치면 겹침이 가장 적은 곳을 쓴다.
    # (9/21) 겹침이 남는 두 가지 원인을 고쳤다: 범례가 나중에 그려져 충돌 검사에서 빠졌고,
    #        모든 후보가 겹칠 때 검사 없이 기본 위치에 두었다.
    ax1.margins(x=0.20, y=0.18)
    leg = ax1.legend(frameon=True, loc="lower left", handletextpad=0.4, borderaxespad=0.2,
                     borderpad=0.35)
    leg.get_frame().set(edgecolor="#b0b0b0", facecolor="white", linewidth=0.6)
    fig.canvas.draw()
    rend = fig.canvas.get_renderer()
    pt_boxes = [ax1.transData.transform(pt) for pt in others]
    taken = [leg.get_window_extent(renderer=rend).expanded(1.02, 1.05)]
    axbb = ax1.get_window_extent()
    cands = [(5, 2), (5, -8), (-5, 2), (-5, -8), (5, 8), (-5, 8), (0, 8), (0, -12),
             (13, 2), (-13, 2), (13, -8), (-13, -8), (0, 17), (0, -21)]

    def cost(bb):
        """다른 상자·점과 겹치는 넓이. 축 밖으로 나가면 큰 벌점."""
        c = 0.0
        for b in taken:
            dx = min(bb.x1, b.x1) - max(bb.x0, b.x0)
            dy = min(bb.y1, b.y1) - max(bb.y0, b.y0)
            if dx > 0 and dy > 0:
                c += dx * dy
        c += 400.0 * sum(bb.x0 - 2 < px < bb.x1 + 2 and bb.y0 - 2 < py < bb.y1 + 2 for px, py in pt_boxes)
        if not (axbb.x0 <= bb.x0 and bb.x1 <= axbb.x1 and axbb.y0 <= bb.y0 and bb.y1 <= axbb.y1):
            c += 5000.0
        return c

    def annotate(m, xi, yi, dx, dy):
        arrow = dict(arrowstyle="-", lw=0.4, color=BLUE, shrinkA=0.5, shrinkB=2.5) if abs(dy) > 10 or abs(dx) > 10 else None
        return ax1.annotate(LABELS[m], (xi, yi), textcoords="offset points", xytext=(dx, dy), fontsize=6.2,
                            color=BLUE, ha="left" if dx > 0 else ("right" if dx < 0 else "center"),
                            arrowprops=arrow)

    for m, xi, yi in placed:
        # 후보마다 임시로 그려 비용만 재고 바로 지운다. 확정된 위치에 한 번만 다시 그린다.
        scored = []
        for dx, dy in cands:
            t = annotate(m, xi, yi, dx, dy)
            bb = t.get_window_extent(renderer=rend).expanded(1.12, 1.45)
            scored.append((cost(bb), dx, dy))
            t.remove()
            if scored[-1][0] == 0:
                break
        _, dx, dy = min(scored)
        t = annotate(m, xi, yi, dx, dy)
        taken.append(t.get_window_extent(renderer=rend).expanded(1.12, 1.45))

    ax1.set_xlabel(r"Marginal fidelity error, KS ($\downarrow$)")
    ax1.set_ylabel(r"Leaderboard fidelity, Kendall $\tau$ ($\uparrow$)")
    print(f"  (a) Spearman rho = {rho_ours:.2f} (interbank), {rho_pub:.2f} (public)")

    # (b) 같은 생성기 안: 각 기준이 두 실행 중 더 좋은 쪽을 맞히는 비율
    wg = json.loads((E / "within_generator.json").read_text())
    names = {"ks": "Marginal fidelity (KS)", "tvd": "TVD", "corr_rmse": "Dependence", "detection_auc": "C2ST",
             "pos_rate_abs_err_pp": "Label rate", "tstr_best_pr_auc": "TSTR (best detector)",
             "tstr_catboost_pr_auc": "TSTR (CatBoost)"}
    items = [(names[k], wg[k]["all"]["agree_rate"], wg[k]["all"]["n_pairs"]) for k in names if k in wg]
    items.sort(key=lambda z: z[1])
    # 수치는 y축에 둔다(그림 1, 그림 3과 같은 방향). 막대 + 오차막대와 파선 기준선은
    # TabArena(NeurIPS'25 D&B) Fig.1의 구성이다.
    items.sort(key=lambda z: -z[1])
    ax2.axhline(0.5, color="black", ls="--", lw=0.8, zorder=1)
    err = [1.96 * np.sqrt(max(v * (1 - v), 1e-9) / max(n, 1)) for _, v, n in items]
    ax2.bar(range(len(items)), [v for _, v, _ in items], width=0.6, color=BLUE, zorder=2,
            yerr=err, error_kw=dict(ecolor=GREY, elinewidth=0.8, capsize=2, capthick=0.8))
    ax2.set_xticks(range(len(items)))
    ax2.set_xticklabels([n for n, _, _ in items], rotation=35, ha="right", rotation_mode="anchor")
    ax2.set_ylim(0, 0.8)
    ax2.set_ylabel("Agreement on paired runs")
    ax2.set_xlim(-0.7, len(items) - 0.3)
    ax2.text(len(items) - 0.35, 0.51, "Chance", fontsize=6.8, va="bottom", ha="right")
    ax2.grid(axis="y", color=GREY, alpha=0.25, lw=0.5)
    ax2.set_axisbelow(True)
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    for f, suffix in ((fig1, "_a"), (fig2, "_b")):
        f.tight_layout(pad=0.3)
        dst = out.with_name(out.stem + suffix + out.suffix)
        f.savefig(dst, bbox_inches="tight")
        print(f"saved -> {dst}")
    print(f"  (a: {len(ms)} generators, b: {len(items)} criteria)")


if __name__ == "__main__":
    main()
