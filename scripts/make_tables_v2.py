"""
결과 파일에서 원고용 LaTeX 표를 생성한다(수치 전사 오류 방지). 표 내용은 파란색(\\color{blue})으로 출력.

Usage:
    python scripts/make_tables_v2.py --out ../Fin-Syn-paper/202609_iclr/tables
"""
import argparse
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
E = ROOT / "exp/finsyn-v2"
DET = {"nb": "Naive Bayes", "dt": "Decision tree", "lr": "Logistic regression", "knn": "$k$-NN", "mlp": "MLP",
       "rf": "Random forest", "et": "Extra trees", "hgb": "HistGB", "lgbm": "LightGBM", "xgb": "XGBoost", "catboost": "CatBoost"}
GEN = {"smote": "SMOTE", "tvae": "TVAE", "ctgan": "CTGAN", "ctabgan": "CTAB-GAN", "ctabgan-plus": "CTAB-GAN+",
       "tabddpm": "TabDDPM", "great": "GReaT", "tabpfgen": "TabPFGen", "tabpfgen-prior": "TabPFGen-prior", "tabpfgen-nobal": "TabPFGen (no class balancing)"}


def wrap(body, caption, label, note="", wrapwidth=None):
    """wrapwidth를 주면 본문이 표 옆으로 흐르는 wraptable로 출력한다(지면 절약)."""
    if wrapwidth:
        head = ("\\begin{wraptable}[15]{r}{" + wrapwidth + "}\n\\vspace{-\\intextsep}\n\\centering\\color{blue}\\small\n"
                "\\caption{\\BLUE{" + caption + "}}\n\\label{" + label + "}\n")
        tail = "\n\\end{wraptable}\n"  # 각주는 캡션에 포함시킨다
    else:
        head = "\\begin{table}[t]\n\\centering\n\\color{blue}\n\\caption{\\BLUE{" + caption + "}}\n\\label{" + label + "}\n"
        tail = ("\n\\vspace{2pt}\\footnotesize " + note if note else "") + "\n\\end{table}\n"
    return head + body + tail


def real_leaderboard(out):
    r = json.loads((E / "leaderboard_real/results.json").read_text())
    n = json.loads((E / "leaderboard_real/noise_floor.json").read_text())
    order = sorted(r, key=lambda m: -r[m]["summary"]["pr_auc"][0])
    rows = []
    for m in order:
        s = r[m]["summary"]
        ci = n["pr_auc_ci95"][m]
        rows.append(f"{DET[m]} & {s['pr_auc'][0]:.3f} & [{ci[0]:.3f}, {ci[1]:.3f}] & {s['recall@1%fpr'][0]:.3f} \\\\")
    body = ("\\resizebox{\\linewidth}{!}{%\n\\begin{tabular}{lccc}\n\\toprule\nDetector & PR-AUC & 95\\% CI & R@1\\%FPR \\\\\n\\midrule\n"
            + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}}")
    cap = ("Reference leaderboard on the private test period. Seed means over five runs; CI from 1{,}000 stratified "
           "bootstrap resamples. Recall at 0.1\\% FPR and ROC-AUC in Appendix~\\ref{app:extra}.")
    (out / "tab_real_leaderboard.tex").write_text(wrap(body, cap, "tab:real", wrapwidth="0.52\\textwidth"))



def tau_seeds(model, tag):
    """생성기 시드 0,1,2의 tau 목록(없는 시드는 건너뜀)."""
    out = []
    for s in (0, 1, 2):
        d = E / (f"leaderboard_{model}_{tag}" if s == 0 else f"leaderboard_{model}_{tag}_seed{s}")
        f = d / "fidelity_vs_leaderboard_real.json"
        if f.exists():
            out.append(json.loads(f.read_text())["kendall_tau"])
    return out


def generators(out):
    sm = json.loads((E / "standard_metrics.json").read_text())
    pv_path = E / "privacy.json"
    pv = json.loads(pv_path.read_text()) if pv_path.exists() else {}
    lf = json.loads((E / "lf_analysis.json").read_text())["releases"]  # 시드 평균 정의 (scripts/lf_analysis_v2.py)
    rows = []
    for m, v in sorted(sm.items(), key=lambda kv: -lf[kv[0]]["s2s"]["tau_mean"]):
        p = pv.get(m, {})
        priv = (f"{p['dcr_ratio']:.2f} & {p['mia_auc']:.3f}" if p else "--- & ---")
        a, b = lf[m]["s2s"], lf[m]["s2r"]
        dag = "$^\\dagger$" if m == "tvae" else ""
        rng = f"{a['tau_min']:+.2f}, {a['tau_max']:+.2f}" if a["n_seeds"] > 1 else "one seed"
        tau_cols = f"{b['tau_mean']:+.3f} & {a['tau_mean']:+.3f}{dag} & [{rng}]"
        rows.append(f"{GEN.get(m, m)} & {v['ks_mean']:.3f} & {v['tvd_mean']:.3f} & {v['corr_rmse']:.3f} & {v['detection_auc']:.3f} & "
                    f"{100*v['pos_rate']:.2f} & {v['tstr_best_pr_auc']:.3f} & {v['copy_rate']*100:.1f} & {priv} & "
                    f"{tau_cols} & {a['pairs_mean']*100:.0f} & {a['regret_mean']:.3f} \\\\")
    body = ("\\resizebox{\\textwidth}{!}{%\n\\begin{tabular}{lcccccccccccccc}\n\\toprule\n & \\multicolumn{4}{c}{Standard metrics} & "
            "\\multicolumn{2}{c}{Label / utility} & \\multicolumn{3}{c}{Privacy} & \\multicolumn{5}{c}{Leaderboard fidelity} \\\\\n"
            "\\cmidrule(lr){2-5}\\cmidrule(lr){6-7}\\cmidrule(lr){8-10}\\cmidrule(lr){11-15}\n"
            "Generator & KS & TVD & corr & C2ST & pos.\\ \\% & TSTR & copy \\% & DCR & MIA & $\\tau_{S\\to R}$ & $\\tau_{S\\to S}$ & seed min, max & pairs \\% & regret \\\\\n\\midrule\n"
            + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}}")
    base = pv.get("_baseline", {}).get("dcr_holdout_to_train")
    note = ("KS/TVD/corr: mean marginal and dependence error vs.\\ the private training period (lower is better); C2ST: real-vs-synthetic "
            "detection AUC (0.5 is indistinguishable); pos.\\ \\%: synthetic positive rate (private: 1.28\\%); TSTR: best detector "
            "PR-AUC when trained on synthetic and tested on private data; copy \\%: synthetic rows identical to a private training row; "
            "DCR: median distance to the closest private training record, as a ratio to the same distance for real transfers of the same "
            "period that are not in the benchmark (1.0 = as far as a fresh real sample"
            + (f", baseline {base:.3f}" if base else "") + "); MIA: AUC of a nearest-neighbour membership-inference attack that sees only "
            "the release (0.5 = no leakage). Leaderboard-fidelity columns are means over three generator seeds (GReaT: one): "
            "$\\tau$, Kendall rank agreement with Table~\\ref{tab:real} (noise band from 0.818); seed min, max of $\\tau_{S\\to S}$; "
            "pairs \\%, separable pairs whose order is kept; regret, private PR-AUC lost by deploying the public-leaderboard winner "
            "(0.102 for a detector picked uniformly at random). Standard and privacy columns are computed on the seed-0 release. "
            "$^\\dagger$TVAE releases contain one positive row per split, so seven of eleven detectors tie at the prevalence floor "
            "and its $\\tau$ is not interpretable.")
    (out / "tab_generators.tex").write_text(wrap(body, "Standard metrics, privacy risk, and leaderboard fidelity, sorted by $\\tau_{S\\to S}$.", "tab:gen", note))


def conditions(out):
    rows = []
    for tag, name in [("real", "Temporal split (ours)"), ("cond_random", "Random split"), ("cond_dup", "Duplicated rows + random split")]:
        d = E / ("leaderboard_real" if tag == "real" else f"leaderboard_{tag}")
        r = json.loads((d / "results.json").read_text())
        n = json.loads((d / "noise_floor.json").read_text())
        sep = sum(1 for v in n["pairwise_win_prob"].values() if v >= 0.975 or v <= 0.025)
        best = max(r, key=lambda m: r[m]["summary"]["pr_auc"][0])
        sat = sum(1 for m in r if r[m]["summary"]["pr_auc"][0] >= 0.999)
        f = json.loads((d / "fidelity_vs_leaderboard_real.json").read_text()) if tag != "real" else None
        rows.append(f"{name} & {r[best]['summary']['pr_auc'][0]:.3f} & {sat} & {sep}/{len(n['pairwise_win_prob'])} & "
                    + ("--- & --- & ---" if f is None else f"{f['kendall_tau']:.3f} & {DET[f['cand_top1']]} & {f['selection_regret_pr_auc']:.3f}") + " \\\\")
    body = ("\\resizebox{\\textwidth}{!}{%\n\\begin{tabular}{lcccccc}\n\\toprule\nConstruction & best PR-AUC & detectors $\\ge$0.999 & separable pairs & $\\tau$ vs.\\ ours & top-1 detector & regret \\\\\n\\midrule\n"
            + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}}")
    note = ("All three use the same transfers and features; only the split (and, in the last row, replication of 2{,}157 distinct rows to "
            "91{,}005 as in a naive construction) differ.")
    (out / "tab_conditions.tex").write_text(wrap(body, "Effect of dataset construction on the private-data leaderboard.", "tab:cond", note))


def ablation(out):
    a = json.loads((E / "ablation_feature_groups.json").read_text())
    names = {"T": "Transaction", "T+B": "+ bank codes, bank-pair volume", "T+R": "+ receiver cross-bank history",
             "T+B+R": "+ bank, receiver cross-bank", "T+S": "+ sender history", "T+B+S": "+ bank, sender history",
             "T+B+S+R": "All groups"}
    rows = [f"{names[k]} & {v['n_features']} & {v['pr_auc'][0]:.3f} $\\pm$ {v['pr_auc'][1]:.3f} & {v['recall@0.1%fpr'][0]:.3f} & {v['roc_auc'][0]:.3f} \\\\"
            for k, v in a.items()]
    body = ("\\begin{tabular}{lcccc}\n\\toprule\nFeature groups & \\# feat. & PR-AUC & R@0.1\\%FPR & ROC-AUC \\\\\n\\midrule\n"
            + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}")
    note = "LightGBM, fixed configuration, five seeds, private test period. Bank codes and receiver cross-bank history are visible only to the clearing network, not to the sending bank."
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
    rows = [f"Real labels only & {' & '.join(f'{base[f]:.3f}' for f in fracs)} & --- \\\\\n\\midrule"]
    rows += [f"+ {GEN.get(m, m)} & " + " & ".join(f"{delta[f][m]:+.3f}" for f in fracs) + f" & ${tau[m]:+.3f}$ \\\\"
             for m in order]
    body = ("\\begin{tabular}{l" + "c" * len(fracs) + "c}\n\\toprule\n & \\multicolumn{" + str(len(fracs))
            + "}{c}{Fraction of real training labels} & \\\\\n\\cmidrule(lr){2-" + str(len(fracs) + 1) + "}\n"
            + "Training data & " + head + " & $\\tau_{S\\to S}$ \\\\\n\\midrule\n" + "\n".join(rows)
            + "\n\\bottomrule\n\\end{tabular}")
    ac = lfa["aug_corr"]["tau_s2s"]
    note = ("Test PR-AUC on the private test period; first row absolute, the rest change against it. "
            "LightGBM and XGBoost at fixed configurations, three stratified subsamples each; "
            "$\\tau_{S\\to S}$ is the seed-mean value of Table~\\ref{tab:gen}. "
            f"At 5\\% of labels, Spearman $\\rho$ between gain and $\\tau_{{S\\to S}}$ is {ac['rho']:.2f} "
            f"(exact permutation $p={ac['p']:.3f}$, $n=9$).")
    (out / "tab_augmentation.tex").write_text(
        wrap(body, "Augmenting scarce real labels with a synthetic release.", "tab:aug", note))


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
    body = ("\\begin{tabular}{lccccc}\n\\toprule\nDetector & PR-AUC & 95\\% CI & R@0.1\\%FPR & R@1\\%FPR & ROC-AUC \\\\\n\\midrule\n"
            + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}")
    note = ("Seed mean $\\pm$ standard deviation over five runs; CI from 1{,}000 stratified bootstrap resamples of the test period. "
            "The columns omitted from Table~\\ref{tab:real} are included here.")
    (out / "tab_real_full.tex").write_text(wrap(body, "Reference leaderboard, all metrics.", "tab:realfull", note))

    # 릴리스별 탐지기 PR-AUC (s2s / s2r)
    models = [m for m in GEN if (E / f"leaderboard_{m}_s2s").exists()]
    sm = json.loads((E / "standard_metrics.json").read_text())
    models = sorted(models, key=lambda m: -sm[m]["lf_tau_s2s"])
    for tag, cap in [("s2s", "trained, tuned, and tested on the release ($S\\to S$)"),
                     ("s2r", "trained and tuned on the release, tested on the private test period ($S\\to R$)")]:
        lb = {m: json.loads((E / f"leaderboard_{m}_{tag}/results.json").read_text()) for m in models}
        rows = []
        for d in order:
            cells = " & ".join(f"{lb[m][d]['summary']['pr_auc'][0]:.3f}" if d in lb[m] else "---" for m in models)
            rows.append(f"{DET[d]} & {r[d]['summary']['pr_auc'][0]:.3f} & {cells} \\\\")
        head = " & ".join(GEN.get(m, m) for m in models)
        body = ("\\resizebox{\\textwidth}{!}{%\n\\begin{tabular}{l" + "c" * (len(models) + 1) + "}\n\\toprule\n"
                "Detector & private & " + head + " \\\\\n\\midrule\n" + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}}")
        note = ("Test PR-AUC, seed means. Rows are ordered by the private leaderboard; columns by $\\tau_{S\\to S}$. "
                "A release preserves the leaderboard when its column orders the rows as the \\emph{private} column does.")
        (out / f"tab_detectors_{tag}.tex").write_text(
            wrap(body, f"Per-detector results, {cap}.", f"tab:det{tag}", note))


def tstr_vs_tau(out):
    """부록: 같은 생성기의 실행본(시드)별 TSTR과 tau. TSTR은 거의 같은데 tau는 크게 다를 수 있음을 보인다."""
    import sys
    sys.path.insert(0, str(ROOT / "scripts"))
    from tstr_vs_lf_v2 import MODELS, seeds_of, tau, tstr
    res = json.loads((E / "tstr_vs_lf.json").read_text())
    rows = []
    for m in MODELS:
        ss = seeds_of(m)
        if len(ss) < 3:
            continue
        ts, ta = [tstr(m, s, "catboost") for s in ss], [tau(m, s) for s in ss]
        rows.append(f"{GEN.get(m, m)} & " + " & ".join(f"{x:.3f}" for x in ts) + f" & {max(ts)-min(ts):.3f} & "
                    + " & ".join(f"{x:+.3f}" for x in ta) + f" & {max(ta)-min(ta):.3f} \\\\")
    body = ("\\begin{tabular}{lcccccccc}\n\\toprule\n & \\multicolumn{4}{c}{TSTR (CatBoost PR-AUC)} & "
            "\\multicolumn{4}{c}{$\\tau_{S\\to S}$} \\\\\n\\cmidrule(lr){2-5}\\cmidrule(lr){6-9}\n"
            "Generator & seed 0 & seed 1 & seed 2 & range & seed 0 & seed 1 & seed 2 & range \\\\\n\\midrule\n"
            + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}")
    w = res["within_generator"]
    note = (f"Across the {w['n_pairs']} pairs of runs of the same generator, the run with the higher TSTR also has the higher "
            f"$\\tau_{{S\\to S}}$ in {w['agree_rate']*100:.0f}\\% of cases ({w['agree_rate_non_collapsed']*100:.0f}\\% of the "
            f"{w['n_pairs_non_collapsed']} pairs among non-collapsed generators). GReaT was run once and is omitted.")
    (out / "tab_tstr_vs_tau.tex").write_text(
        wrap(body, "TSTR and leaderboard fidelity of individual runs of the same generator.", "tab:tstrtau", note))


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
                    + (f"{b:+.3f}" if b is not None else "---") + " & "
                    + (f"{a:+.3f}" if a is not None else "undefined$^\\dagger$") + " \\\\")
    body = ("\\begin{tabular}{lccccccc c}\n\\toprule\n & \\multicolumn{4}{c}{Standard metrics} & "
            "\\multicolumn{2}{c}{Label / utility} & \\multicolumn{2}{c}{Leaderboard fidelity} \\\\\n"
            "\\cmidrule(lr){2-5}\\cmidrule(lr){6-7}\\cmidrule(lr){8-9}\n"
            "Generator & KS & TVD & corr & C2ST & pos.\\ \\% & TSTR & $\\tau_{S\\to R}$ & $\\tau_{S\\to S}$ \\\\\n"
            "\\midrule\n" + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}")
    note = (f"Public replication on TabReD homecredit-default (32{{,}}076 / 7{{,}}924 / 10{{,}}000 rows by time, "
            f"5.0 / 3.3 / 2.3\\% positive), same protocol at a reduced budget (one generator seed, 10 tuning trials, "
            f"three detector seeds). The noise floor here is lower than on our benchmark: bootstrap $\\tau$ ceiling "
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
    appendix_tables(out)
    tstr_vs_tau(out)
    tabred(out)
    print("wrote:", *(p.name for p in sorted(out.glob("*.tex"))))
