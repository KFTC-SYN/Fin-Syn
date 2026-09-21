"""
결과 파일에서 원고용 LaTeX 표를 생성한다(수치 전사 오류 방지).

Usage:
    python scripts/make_tables_v2.py --out ../Fin-Syn-paper/202609_iclr/tables
"""
import argparse
import json
from pathlib import Path

import numpy as np

import os
# 논문 메인 표/그림은 생성기마다 같은 수의 공개본을 쓴다(사용자 지시 9/20). 기본 5시드.
MAX_SEEDS = int(os.environ.get("FINSYN_MAX_SEEDS", "5"))

ROOT = Path(__file__).resolve().parents[1]
E = ROOT / "exp/finsyn-v2"
DET = {"nb": "Naive Bayes", "dt": "Decision tree", "lr": "Logistic regression", "knn": "$k$-NN", "mlp": "MLP",
       "rf": "Random forest", "et": "Extra trees", "hgb": "HistGB", "lgbm": "LightGBM", "xgb": "XGBoost", "catboost": "CatBoost"}
GEN = {"smote": "SMOTE", "tvae": "TVAE", "ctgan": "CTGAN", "ctabgan": "CTAB-GAN", "ctabgan-plus": "CTAB-GAN+",
       "tabsyn": "TabSyn", "tabdiff": "TabDiff", "findiff": "FinDiff",
       "tabddpm": "TabDDPM", "great": "GReaT", "tabpfgen": "TabPFGen", "tabpfgen-prior": "TabPFGen-prior", "tabpfgen-nobal": "TabPFGen (no class balancing)"}


def wrap(body, caption, label, note="", wrapwidth=None):
    """wrapwidth를 주면 본문이 표 옆으로 흐르는 wraptable로 출력한다(지면 절약)."""
    if wrapwidth:
        head = ("\\begin{wraptable}{r}{" + wrapwidth + "}\n\\vspace{-\\intextsep}\n\\centering\\small\n"
                "\\caption{" + caption + "}\n\\label{" + label + "}\n")
        tail = "\n\\end{wraptable}\n"  # 각주는 캡션에 포함시킨다
    else:
        cap = caption + ((" " + note) if note else "")  # 표 아래 각주를 쓰지 않고 캡션에 합친다(9/20 요청)
        head = "\\begin{table}[t]\n\\centering\n\\caption{" + cap + "}\n\\label{" + label + "}\n"
        tail = "\n\\end{table}\n"
    return head + body + tail


def real_leaderboard(out):
    r = json.loads((E / "leaderboard_real/results.json").read_text())
    n = json.loads((E / "leaderboard_real/noise_floor.json").read_text())
    order = sorted(r, key=lambda m: -r[m]["summary"]["pr_auc"][0])
    rows = []
    for m in order:
        s = r[m]["summary"]
        ci = n["pr_auc_ci95"][m]
        rows.append(f"{DET[m]} & {s['pr_auc'][0]:.3f} & [{ci[0]:.3f}, {ci[1]:.3f}] \\\\")
    # 11행 한 덩어리. 6+5로 나누면 홀수라 마지막 칸이 비어 표가 어색해진다(9/21 되돌림).
    body = ("\\begin{tabular}{@{}lcc@{}}\n\\toprule\nDetector & PR-AUC & 95\\% CI \\\\\n\\midrule\n"
            + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}")
    # 본문이 표 옆으로 흐르도록 wraptable로 낸다(9/21 요청). 폭이 좁아 캡션도 세 문장으로 줄였고,
    # 재표집 일치도 0.910은 5.1절 본문이 그대로 싣는다.
    # wraptable이라 캡션이 길면 표가 쪽 아래 끝에 닿는다. 자세한 조건은 4.3절과 부록이 싣는다.
    cap = ("Reference leaderboard on the private test period. Test PR-AUC, seed mean, with a 95\\% bootstrap CI. "
           "Recall and ROC-AUC are in Appendix~\\ref{app:extra}.")
    (out / "tab_real_leaderboard.tex").write_text(wrap(body, cap, "tab:real", wrapwidth="0.46\\textwidth"))



def tau_seeds(model, tag):
    """생성기 공개본들의 tau 목록(없는 시드는 건너뜀)."""
    out = []
    for s in range(MAX_SEEDS):
        d = E / (f"leaderboard_{model}_{tag}" if s == 0 else f"leaderboard_{model}_{tag}_seed{s}")
        f = d / "fidelity_vs_leaderboard_real.json"
        if f.exists():
            out.append(json.loads(f.read_text())["kendall_tau"])
    return out


def generators(out):
    """본문 표 2(핵심 열만)와 부록 표(나머지 표준지표). 9/19 스타일 점검: 15열을 축소해 넣던 표를 나눈다."""
    sm = json.loads((E / "standard_metrics.json").read_text())
    pv_path = E / "privacy.json"
    pv = json.loads(pv_path.read_text()) if pv_path.exists() else {}
    lf = json.loads((E / "lf_analysis.json").read_text())["releases"]  # 시드 평균 정의 (scripts/lf_analysis_v2.py)
    cf = json.loads((E / "copy_filter.json").read_text()) if (E / "copy_filter.json").exists() else {}
    removed = {m: (cf[f"{m}/seed0"]["train"]["copies"] / cf[f"{m}/seed0"]["train"]["rows"] * 100
                   if f"{m}/seed0" in cf else 0.0) for m in sm}  # 공개 전 제거한 완전 복제 행(%)
    order = sorted(sm, key=lambda m: -lf[m]["s2s"]["tau_mean"])
    best_tau = max(lf[m]["s2s"]["tau_mean"] for m in order)
    counts = {lf[m]["s2s"]["n_seeds"] for m in order}
    uniform = len(counts) == 1          # 메인 표는 생성기마다 같은 수의 공개본을 써야 한다(사용자 지시 9/20)
    n_rel = next(iter(counts)) if uniform else None
    if not uniform:
        print("  [경고] 생성기별 공개본 수가 다릅니다:",
              {GEN.get(m, m): lf[m]["s2s"]["n_seeds"] for m in order})

    def num(x, fmt="{:.2f}", signed=False):
        s = ("{:+.2f}" if signed else fmt).format(x)
        return s.replace("-", "$-$")

    main, appx = [], []
    for m in order:
        v, p = sm[m], pv.get(m, {})
        a, b = lf[m]["s2s"], lf[m]["s2r"]
        rng = (f"[{num(a['tau_min'], signed=True)}, {num(a['tau_max'], signed=True)}]" if a["n_seeds"] > 1 else "one run")
        tau_s2s = num(a["tau_mean"], signed=True)
        if abs(a["tau_mean"] - best_tau) < 1e-9:
            tau_s2s = "\\textbf{" + tau_s2s + "}"
        priv = (f"{num(p['dcr_ratio'])} & {p['mia_auc']:.2f}" if p else "n/a & n/a")
        name = GEN.get(m, m) + ("" if uniform else f" ({a['n_seeds']})")
        main.append(f"{name} & {v['ks_mean']:.2f} & {v['tstr_catboost_pr_auc']:.2f} & "
                    f"{tau_s2s} & {rng} & {num(b['tau_mean'], signed=True)} & {priv} \\\\")
        appx.append(f"{GEN.get(m, m)} & {a['pairs_mean']*100:.0f} & {a['regret_mean']:.3f} & {v['tvd_mean']:.3f} & "
                    f"{v['corr_rmse']:.3f} & {100*v['pos_rate']:.2f} & {v['tstr_best_pr_auc']:.3f} & "
                    f"{removed[m]:.2f} & {p.get('dcr_share', float('nan')):.3f} \\\\")

    body = ("\\small\n\\setlength{\\tabcolsep}{4.5pt}\n\\begin{tabular}{lccccccc}\n\\toprule\n"
            "\\multirow{2}{*}{Generator} & \\multicolumn{2}{c}{Standard metrics} & \\multicolumn{3}{c}{Leaderboard fidelity} & "
            "\\multicolumn{2}{c}{Privacy} \\\\\n\\cmidrule(lr){2-3}\\cmidrule(lr){4-6}\\cmidrule(lr){7-8}\n"
            " & KS $\\downarrow$ & TSTR $\\uparrow$ & $\\tau_{S\\to S}$ $\\uparrow$ & [min, max] & "
            "$\\tau_{S\\to R}$ $\\uparrow$ & DCR $\\to 1$ & MIA $\\to .5$ \\\\\n\\midrule\n"
            + "\n".join(main) + "\n\\bottomrule\n\\end{tabular}")
    n_word = {3: "three", 4: "four", 5: "five", 10: "ten"}.get(n_rel, str(n_rel))
    lead = (f"Every generator produced {n_word} releases, one per seed. " if uniform
            else "The number of releases per generator is given in parentheses. ")
    note = (lead + "$\\tau_{S\\to S}$ is the mean over them and "
            "[min, max] their range; the other columns use the "
            "released seed-0 run. Arrows give the better direction; DCR near 1.0 means as far from the private data as a fresh real "
            "sample, and bold marks the highest $\\tau_{S\\to S}$. TVAE keeps one positive row per split, so its $\\tau$ is "
            "not interpretable. Definitions are in Appendix~\\ref{app:metrics}; separable pairs, regret and the remaining "
            "metrics are in Table~\\ref{tab:genfull}.")
    (out / "tab_generators.tex").write_text(wrap(
        body, "Leaderboard fidelity varies more between runs of one generator than between generators.", "tab:gen", note))

    abody = ("\\small\n\\setlength{\\tabcolsep}{4pt}\n\\begin{tabular}{lcccccccc}\n\\toprule\n"
             "Generator & Pairs \\% $\\uparrow$ & Regret $\\downarrow$ & TVD $\\downarrow$ & Dep. $\\downarrow$ & "
             "Pos. \\% & TSTR $\\uparrow$ & Copies \\% & DCR sh. $\\to .5$ \\\\\n\\midrule\n"
             + "\n".join(appx) + "\n\\bottomrule\n\\end{tabular}")
    anote = ("Separable-pair preservation and selection regret are means over that generator's runs; the remaining columns use "
             "the released run. The Copies removed column gives the share of rows identical to a private record, deleted before "
             "publication. Across the sixty releases this affected all fifteen SMOTE splits, three TabDDPM runs and "
             "two GReaT runs; at seed 0, which this table reports, only SMOTE produced them. The private prevalence is 1.28\\%.")
    (out / "tab_generators_full.tex").write_text(wrap(
        abody, "Remaining standard metrics for the seed-0 release of each generator.", "tab:genfull", anote))


WORD = {3: "three", 4: "four", 5: "five", 6: "six", 7: "seven", 8: "eight"}


def conditions(out):
    """구성 비교표. SynReal(ICML'23) Table 4처럼 조건을 행, 탐지기를 열로 두어
    독자가 순위 뒤집힘을 직접 읽게 한다. 요약 통계(최고 점수, 포화 개수, 승자)는
    표 안에서 바로 보이므로 별도 열을 두지 않는다(9/22 조사)."""
    ref = json.loads((E / "leaderboard_real/results.json").read_text())
    order = sorted(ref, key=lambda m: -ref[m]["summary"]["pr_auc"][0])
    short = {"nb": "NB", "dt": "DT", "lr": "LR", "knn": "kNN", "mlp": "MLP", "rf": "RF", "et": "ET",
             "hgb": "HGB", "lgbm": "LGBM", "xgb": "XGB", "catboost": "CB"}
    PRIV = {k: v["summary"]["pr_auc"][0] for k, v in ref.items()}
    rows, sep_counts, regrets, n_tied, reg_all = [], [], [], [], []
    for tag, name in [("real", "Temporal split"), ("cond_random", "Random split"),
                      ("cond_dup", "Duplicated + random split")]:
        d = E / ("leaderboard_real" if tag == "real" else f"leaderboard_{tag}")
        r = json.loads((d / "results.json").read_text())
        n = json.loads((d / "noise_floor.json").read_text())
        sep_counts.append((sum(1 for v in n["pairwise_win_prob"].values() if v >= 0.975 or v <= 0.025),
                           len(n["pairwise_win_prob"])))
        f = json.loads((d / "fidelity_vs_leaderboard_real.json").read_text()) if tag != "real" else None
        tau = "1.000" if f is None else f"{f['kendall_tau']:.3f}"
        reg = "0.000" if f is None else f"{f['selection_regret_pr_auc']:.3f}"
        hi = max(r[d_]["summary"]["pr_auc"][0] for d_ in order)
        cells = " & ".join((f"\\textbf{{{r[d_]['summary']['pr_auc'][0]:.2f}}}"
                            if abs(r[d_]["summary"]["pr_auc"][0] - hi) < 1e-9
                            else f"{r[d_]['summary']['pr_auc'][0]:.2f}") for d_ in order)
        n_tied.append(sum(1 for d_ in order if abs(r[d_]["summary"]["pr_auc"][0] - hi) < 1e-9))
        reg_all.append([round(max(PRIV.values()) - PRIV[d_], 3)
                        for d_ in order if abs(r[d_]["summary"]["pr_auc"][0] - hi) < 1e-9])
        regrets.append(reg)
        rows.append(f"{name} & {cells} & {tau} \\\\")
    body = ("\\footnotesize\n\\setlength{\\tabcolsep}{3pt}\n"
            "\\begin{tabular}{@{}l" + "c" * len(order) + "c@{}}\n\\toprule\n"
            "Construction & " + " & ".join(short[d_] for d_ in order)
            + " & $\\tau$ \\\\\n\\midrule\n" + "\n".join(rows)
            + "\n\\bottomrule\n\\end{tabular}")
    note = ("Test PR-AUC of every detector, with the columns ordered by the reference leaderboard of the first row, so a "
            "construction preserves the ranking when its row orders the columns as the first row does. All three use the "
            "same transfers and features and differ only in the split; the last row also replicates 2{,}157 distinct rows "
            f"to 91{{,}}005. Bold marks the top score in each row. The first row is the reference leaderboard, "
            f"and $\\tau$ is the rank agreement with it. "
            f"Separable pairs fall from {sep_counts[0][0]} of {sep_counts[0][1]} to {sep_counts[1][0]} and then "
            f"{sep_counts[2][0]}. In the last row {WORD.get(n_tied[2], n_tied[2])} detectors are exactly tied at a "
            f"perfect score, so which one "
            f"a user would deploy is arbitrary and the private PR-AUC it costs ranges from "
            f"{min(reg_all[2]):.3f} to {max(reg_all[2]):.3f}.")
    (out / "tab_conditions.tex").write_text(wrap(body, "Effect of dataset construction on the private-data leaderboard.", "tab:cond", note))

def ablation(out):
    a = json.loads((E / "ablation_feature_groups.json").read_text())
    names = {"T": "Transaction", "T+B": "+ Bank codes, bank-pair volume", "T+R": "+ Receiver cross-bank history",
             "T+B+R": "+ Bank, receiver cross-bank", "T+S": "+ Sender history", "T+B+S": "+ Bank, sender history",
             "T+B+S+R": "All groups"}
    rows = [f"{names[k]} & {v['n_features']} & {v['pr_auc'][0]:.3f} $\\pm$ {v['pr_auc'][1]:.3f} & {v['recall@0.1%fpr'][0]:.3f} & {v['roc_auc'][0]:.3f} \\\\"
            for k, v in a.items()]
    body = ("\\begin{tabular}{lcccc}\n\\toprule\nFeature groups & \\# feat. & PR-AUC & R@0.1\\%FPR & ROC-AUC \\\\\n\\midrule\n"
            + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}")
    note = ("LightGBM with a fixed configuration, five seeds, private test period. Only the receiver's cross-bank history "
            "needs records the sending bank does not hold; bank codes and the sender's own history do not.")
    (out / "tab_ablation.tex").write_text(wrap(body, "Feature-group ablation: value of cross-institution information.", "tab:ablation", note))


def augmentation(out):
    """실제 레이블이 부족할 때 공개 합성데이터로 보강하면 이득인가."""
    import pandas as pd
    from scipy.stats import spearmanr

    r = json.loads((E / "augmentation.json").read_text())
    rows = [dict(zip(["src", "frac", "det", "seed"], k.split("|")), pr=v) for k, v in r.items()]
    df = pd.DataFrame(rows)
    df["frac"] = df.frac.astype(float)
    df = df[df.det.isin(["lgbm", "xgb"])]  # CatBoost는 교착으로 중단, 전 구간 완주한 두 탐지기만 사용
    piv = df.groupby(["src", "frac"]).pr.mean().unstack()
    base, fracs = piv.loc["real"], sorted(piv.columns)
    delta = (piv - base).drop(index="real")
    order = sorted(delta.index, key=lambda m: -delta[fracs[0]][m])

    lfa = json.loads((E / "lf_analysis.json").read_text())
    tau = {m: lfa["releases"][m]["s2s"]["tau_mean"] for m in delta.index}
    head = " & ".join(f"{int(f * 100)}\\%" for f in fracs)
    rows = [f"Real labels only & {' & '.join(f'{base[f]:.3f}' for f in fracs)} & none \\\\\n\\midrule"]
    rows += [f"+ {GEN.get(m, m)} & " + " & ".join(f"{delta[f][m]:+.3f}".replace("-", "$-$") for f in fracs)
             + f" & ${tau[m]:+.3f}$ \\\\".replace("$-", "$-") for m in order]
    body = ("\\begin{tabular}{l" + "c" * len(fracs) + "c}\n\\toprule\n\\multirow{2}{*}{Training data} & \\multicolumn{" + str(len(fracs))
            + "}{c}{Fraction of real training labels} & \\multirow{2}{*}{$\\tau_{S\\to S}$} \\\\\n\\cmidrule(lr){2-" + str(len(fracs) + 1) + "}\n"
            + " & " + head + " & \\\\\n\\midrule\n" + "\n".join(rows)
            + "\n\\bottomrule\n\\end{tabular}")
    ac = lfa["aug_corr"]["tau_s2s"]
    note = ("Test PR-AUC on the private test period; first row absolute, the rest change against it. "
            "LightGBM and XGBoost at fixed configurations, three stratified subsamples each; "
            "$\\tau_{S\\to S}$ is the seed-mean value of Table~\\ref{tab:gen}. "
            f"At 5\\% of labels, Spearman $\\rho$ between gain and $\\tau_{{S\\to S}}$ is {ac['rho']:.2f} "
            f"(exact permutation $p={ac['p']:.3f}$, $n={len(delta)}$).")
    (out / "tab_augmentation.tex").write_text(
        wrap(body, "Augmenting scarce real labels with a synthetic release.", "tab:aug", note))


def augmentation_ratio(out):
    """부록: 합성/실제 비율을 1:1, 4:1, 기본(약 20:1)로 바꿨을 때 보강 이득의 부호와 순서가 유지되는가."""
    import pandas as pd

    rr = json.loads((E / "augmentation_ratio.json").read_text())
    base_keys = [v for k, v in rr.items() if k.startswith("real|0.05|lgbm")]
    base = sum(base_keys) / len(base_keys)
    per = {}
    for k, v in rr.items():
        parts = k.split("|")
        if len(parts) == 5 and parts[1] == "0.05" and parts[2] == "lgbm":
            per.setdefault(parts[0], {}).setdefault(parts[4], []).append(v)
    default = json.loads((E / "lf_analysis.json").read_text())["aug_gain_5pct"]
    order = sorted(per, key=lambda m: -default.get(m, float("-inf")))
    rows = []
    for m in order:
        r1 = sum(per[m].get("r1", [float("nan")])) / max(len(per[m].get("r1", [1])), 1) - base
        r4 = sum(per[m].get("r4", [float("nan")])) / max(len(per[m].get("r4", [1])), 1) - base
        d = default.get(m, float("nan"))
        cells = " & ".join(f"{x:+.3f}".replace("-", "$-$") for x in (r1, r4, d))
        rows.append(f"{GEN.get(m, m)} & {cells} \\\\")
    body = ("\\small\n\\begin{tabular}{lccc}\n\\toprule\n"
            "Release & 1:1 & 4:1 & Unrestricted \\\\\n\\midrule\n" + "\n".join(rows)
            + "\n\\bottomrule\n\\end{tabular}")
    note = ("Change in test PR-AUC against training on the 5\\% real subset alone "
            f"(LightGBM, absolute baseline {base:.3f}), when the synthetic rows appended are capped at one and four "
            "times the number of real rows, and when the whole synthetic training split is appended "
            "(about twenty times). The magnitude depends on the ratio, the sign and the ordering do not.")
    (out / "tab_aug_ratio.tex").write_text(wrap(
        body, "Augmentation gain is ratio-dependent in magnitude but not in sign.", "tab:augratio", note))


def appendix_tables(out):
    """부록: 참조 리더보드 전체 지표 + 릴리스별 탐지기 상세."""
    r = json.loads((E / "leaderboard_real/results.json").read_text())
    n = json.loads((E / "leaderboard_real/noise_floor.json").read_text())
    order = sorted(r, key=lambda m: -r[m]["summary"]["pr_auc"][0])
    rows = []
    for m in order:
        s = r[m]["summary"]
        ci = n["pr_auc_ci95"][m]
        rows.append(f"{DET[m]} & {s['pr_auc'][0]:.3f} $\\pm$ {s['pr_auc'][1]:.3f} & [{ci[0]:.3f}, {ci[1]:.3f}] & "
                    f"{s['recall@0.1%fpr'][0]:.3f} & {s['recall@1%fpr'][0]:.3f} & {s['roc_auc'][0]:.3f} \\\\")
    body = ("\\small\n\\setlength{\\tabcolsep}{5pt}\n\\begin{tabular}{lccccc}\n\\toprule\nDetector & PR-AUC & 95\\% CI & R@0.1\\%FPR & R@1\\%FPR & ROC-AUC \\\\\n\\midrule\n"
            + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}")
    note = ("Seed mean $\\pm$ standard deviation over five runs; CI from 1{,}000 stratified bootstrap resamples of the test period. "
            "The columns omitted from Table~\\ref{tab:real} are included here.")
    (out / "tab_real_full.tex").write_text(wrap(body, "Reference leaderboard, all metrics.", "tab:realfull", note))

    # 릴리스별 탐지기 PR-AUC (s2s / s2r)
    models = [m for m in GEN if (E / f"leaderboard_{m}_s2s").exists()]
    sm = json.loads((E / "standard_metrics.json").read_text())
    lfa = json.loads((E / "lf_analysis.json").read_text())["releases"]
    models = sorted(models, key=lambda m: -lfa[m]["s2s"]["tau_mean"])
    for tag, cap in [("s2s", "trained, tuned, and tested on the release ($S\\to S$)"),
                     ("s2r", "trained and tuned on the release, tested on the private test period ($S\\to R$)")]:
        lb = {m: json.loads((E / f"leaderboard_{m}_{tag}/results.json").read_text()) for m in models}
        short = {"nb": "NB", "dt": "DT", "lr": "LR", "knn": "kNN", "mlp": "MLP", "rf": "RF", "et": "ET",
                 "hgb": "HGB", "lgbm": "LGBM", "xgb": "XGB", "catboost": "CB"}
        rows = ["Private data & " + " & ".join(f"{r[d]['summary']['pr_auc'][0]:.2f}" for d in order) + " \\\\",
                "\\midrule"]
        for m in models:
            cells = " & ".join(f"{lb[m][d]['summary']['pr_auc'][0]:.2f}" if d in lb[m] else "n/a" for d in order)
            rows.append(f"{GEN.get(m, m)} & {cells} \\\\")
        head = " & ".join(short[d] for d in order)
        body = ("\\footnotesize\n\\setlength{\\tabcolsep}{4pt}\n\\begin{tabular}{l" + "c" * len(order) + "}\n"
                "\\toprule\nRelease & " + head + " \\\\\n\\midrule\n" + "\n".join(rows)
                + "\n\\bottomrule\n\\end{tabular}")
        note = ("Test PR-AUC of the released run. Columns are ordered by the private leaderboard and rows by "
                "$\\tau_{S\\to S}$; a release preserves the leaderboard when its row orders the columns as the first row does.")
        (out / f"tab_detectors_{tag}.tex").write_text(
            wrap(body, f"Per-detector results, {cap}.", f"tab:det{tag}", note))


def tstr_vs_tau(out):
    """부록: 같은 생성기의 실행본(시드)별 TSTR과 tau의 폭. 시드 수가 생성기마다 달라 열 수는 고정(범위로 요약)."""
    import sys
    sys.path.insert(0, str(ROOT / "scripts"))
    from tstr_vs_lf_v2 import MODELS, seeds_of, tau, tstr
    res = json.loads((E / "tstr_vs_lf.json").read_text())
    rows = []
    for m in MODELS:
        ss = seeds_of(m)
        if len(ss) < 2:
            continue
        ts, ta = [tstr(m, s, "catboost") for s in ss], [tau(m, s) for s in ss]
        best_by_tstr = ta[int(max(range(len(ss)), key=lambda i: ts[i]))]
        rows.append((f"{GEN.get(m, m)} & {len(ss)} & {min(ts):.3f} to {max(ts):.3f} & "
                     f"{min(ta):+.2f} to {max(ta):+.2f} & {best_by_tstr:+.2f} & {max(ta):+.2f} \\\\")
                    .replace("-", "$-$"))
    body = ("\\small\n\\begin{tabular}{lccccc}\n\\toprule\n\\multirow{2}{*}{Generator} & \\multirow{2}{*}{Runs} & \\multicolumn{2}{c}{Range over runs} & "
            "\\multicolumn{2}{c}{$\\tau_{S\\to S}$ of the run picked by} \\\\\n"
            "\\cmidrule(lr){3-4}\\cmidrule(lr){5-6}\n"
            " & & TSTR & $\\tau_{S\\to S}$ & TSTR & Oracle \\\\\n\\midrule\n"
            + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}")
    w = res["within_generator"]
    note = (f"Runs of one generator differ in $\\tau_{{S\\to S}}$ far more than in TSTR. Across the {w['n_pairs']} pairs of runs "
            f"of the same generator, the run with the higher TSTR also has the higher $\\tau_{{S\\to S}}$ in "
            f"{w['agree_rate']*100:.0f}\\% of cases ({w['agree_rate_non_collapsed']*100:.0f}\\% of the "
            f"{w['n_pairs_non_collapsed']} pairs among non-collapsed generators). The last two columns compare the run TSTR would "
            "select with the best run.")
    (out / "tab_tstr_vs_tau.tex").write_text(
        wrap(body, "TSTR cannot tell a good run of a generator from a bad one.", "tab:tstrtau", note))


def tabred(out):
    """부록: 공개 데이터(TabReD homecredit-default)에서의 외부 재현."""
    T = ROOT / "exp/tabred-hc"
    if not (T / "standard_metrics.json").exists():
        print("  (tabred 결과 없음, 건너뜀)")
        return
    sm = json.loads((T / "standard_metrics.json").read_text())
    nf = json.loads((T / "leaderboard_real/noise_floor.json").read_text())

    def tau(m, tag):
        f = T / f"leaderboard_{m}_{tag}/fidelity_vs_leaderboard_real.json"
        if not f.exists():
            return None
        v = json.loads(f.read_text())["kendall_tau"]
        return v if v == v else None

    rows = []
    for m in sorted(sm, key=lambda m: -(tau(m, "s2s") if tau(m, "s2s") is not None else -9)):
        v, a, b = sm[m], tau(m, "s2s"), tau(m, "s2r")
        rows.append(f"{GEN.get(m, m)} & {v['ks_mean']:.3f} & {v['tvd_mean']:.3f} & {v['corr_rmse']:.3f} & "
                    f"{v['detection_auc']:.3f} & {100*v['pos_rate']:.2f} & {v['tstr_catboost_pr_auc']:.3f} & "
                    + (f"{b:+.3f}".replace("-", "$-$") if b is not None else "n/a") + " & "
                    + (f"{a:+.3f}".replace("-", "$-$") if a is not None else "undefined$^\\dagger$") + " \\\\")
    body = ("\\begin{tabular}{lccccccc c}\n\\toprule\n\\multirow{2}{*}{Generator} & \\multicolumn{4}{c}{Standard metrics} & "
            "\\multicolumn{2}{c}{Label / utility} & \\multicolumn{2}{c}{Leaderboard fidelity} \\\\\n"
            "\\cmidrule(lr){2-5}\\cmidrule(lr){6-7}\\cmidrule(lr){8-9}\n"
            " & KS & TVD & Corr & C2ST & Pos.\\ \\% & TSTR & $\\tau_{S\\to R}$ & $\\tau_{S\\to S}$ \\\\\n"
            "\\midrule\n" + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}")
    note = (f"Public replication on TabReD homecredit-default (32{{,}}076 / 7{{,}}924 / 10{{,}}000 rows by time, "
            f"5.0 / 3.3 / 2.3\\% positive), same protocol at a reduced budget (one generator seed, 10 tuning trials, "
            f"three detector seeds). The noise ceiling here is lower than on our benchmark: bootstrap $\\tau$ ceiling "
            f"{nf['tau_vs_full_mean']:.3f} (5th percentile {nf['tau_vs_full_q05']:.3f}) against 0.910 (0.818), and only "
            f"{sum(1 for v in nf['pairwise_win_prob'].values() if v >= 0.975 or v <= 0.025)} of "
            f"{len(nf['pairwise_win_prob'])} detector pairs are separable. "
            "$^\\dagger$The synthetic test split contains no positive row, so no metric, and hence no ranking, is defined.")
    (out / "tab_tabred.tex").write_text(
        wrap(body, "External replication of the protocol on public data.", "tab:tabred", note))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT.parent / "Fin-Syn-paper/202609_iclr/tables"))
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    real_leaderboard(out)
    generators(out)
    conditions(out)
    ablation(out)
    augmentation(out)
    augmentation_ratio(out)
    appendix_tables(out)
    tstr_vs_tau(out)
    tabred(out)
    print("wrote:", *(p.name for p in sorted(out.glob("*.tex"))))
