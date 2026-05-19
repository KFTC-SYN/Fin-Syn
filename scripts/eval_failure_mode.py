"""
Failure Mode Analysis for Synthetic Data Generation Models

Computes:
1. Suspicious label ratio per model (vs original training data)
2. Per-feature Jensen-Shannon divergence between original and synthetic suspicious samples
3. Failure mode classification: Faithful / Mild distortion / Label collapse

Uses existing synthetic CSVs (×1.0) from exp/orig-micro-retry/.
AUC values are taken from the paper's Table 4 (not recomputed) to ensure consistency.
"""

import os
import json
import argparse
import numpy as np
import pandas as pd
from scipy.spatial.distance import jensenshannon
from pathlib import Path


TABLE4_AUC = {
    "smote": 0.9997,
    "ctgan": 0.7592,
    "tvae": 0.6378,
    "ctabgan": 0.7407,
    "ctabgan-plus": 0.8458,
    "ddpm_cb_best": 0.9995,
    "great": 0.9987,
    "tabpfgen": 0.7368,
}

MODEL_DISPLAY = {
    "smote": "SMOTE",
    "ctgan": "CTGAN",
    "tvae": "TVAE",
    "ctabgan": "CTAB-GAN",
    "ctabgan-plus": "CTAB-GAN+",
    "ddpm_cb_best": "TabDDPM",
    "great": "GReaT",
    "tabpfgen": "TabPFGen",
}

FEATURE_DISPLAY = {
    "거래금액": "Amount",
    "거래일자": "Date",
    "거래시간대": "Time",
    "출금금융회사일련번호": "WD Bank",
    "출금계좌일련번호": "WD Acct",
    "입금금융회사일련번호": "DP Bank",
    "입금계좌일련번호": "DP Acct",
    "매체구분": "Media",
    "자금구분": "Fund",
}

FEATURES = [
    "거래금액", "거래일자", "거래시간대",
    "출금금융회사일련번호", "출금계좌일련번호",
    "입금금융회사일련번호", "입금계좌일련번호",
    "매체구분", "자금구분",
]

# 거래금액 is numeric; discretize for JS divergence
NUM_FEATURES = {"거래금액"}


def discretize_numeric(series, n_bins=50):
    """Discretize a numeric series into equal-width bins for JS divergence."""
    all_vals = series.dropna()
    if len(all_vals) == 0:
        return series
    lo, hi = all_vals.min(), all_vals.max()
    if lo == hi:
        return pd.Series(np.zeros(len(series), dtype=int), index=series.index)
    bins = np.linspace(lo, hi, n_bins + 1)
    return pd.cut(series, bins=bins, labels=False, include_lowest=True).fillna(0).astype(int)


def compute_js_divergence(orig_series, syn_series, is_numeric=False, n_bins=50):
    """Compute Jensen-Shannon divergence between two categorical/discretized distributions."""
    if is_numeric:
        combined = pd.concat([orig_series, syn_series])
        lo, hi = combined.min(), combined.max()
        if lo == hi:
            return 0.0
        bins = np.linspace(lo, hi, n_bins + 1)
        orig_binned = pd.cut(orig_series, bins=bins, labels=False, include_lowest=True).fillna(0)
        syn_binned = pd.cut(syn_series, bins=bins, labels=False, include_lowest=True).fillna(0)
        all_cats = range(n_bins)
    else:
        orig_series = orig_series.astype(str)
        syn_series = syn_series.astype(str)
        all_cats = sorted(set(orig_series.unique()) | set(syn_series.unique()))
        orig_binned = orig_series
        syn_binned = syn_series

    orig_counts = orig_binned.value_counts()
    syn_counts = syn_binned.value_counts()

    p = np.array([orig_counts.get(c, 0) for c in all_cats], dtype=float)
    q = np.array([syn_counts.get(c, 0) for c in all_cats], dtype=float)

    p_sum, q_sum = p.sum(), q.sum()
    if p_sum == 0 or q_sum == 0:
        return 1.0

    p = p / p_sum
    q = q / q_sum

    return float(jensenshannon(p, q) ** 2)  # JS divergence (squared JS distance)


def classify_failure_mode(delta_ratio):
    """Classify failure mode based on suspicious ratio deviation."""
    if delta_ratio <= 0.5:
        return "Faithful"
    elif delta_ratio <= 3.0:
        return "Mild distortion"
    else:
        return "Label collapse"


def load_original_train(data_dir):
    """Load original training data as DataFrame."""
    info = json.load(open(os.path.join(data_dir, "info.json")))
    col_names = info["column_names"]

    X_num = np.load(os.path.join(data_dir, "X_num_train.npy"), allow_pickle=True)
    X_cat = np.load(os.path.join(data_dir, "X_cat_train.npy"), allow_pickle=True)
    y = np.load(os.path.join(data_dir, "y_train.npy"), allow_pickle=True)

    num_cols = info["num_cols"]
    cat_cols = info["cat_cols"]

    df = pd.DataFrame(X_num, columns=num_cols)
    df_cat = pd.DataFrame(X_cat, columns=cat_cols)
    df = pd.concat([df, df_cat], axis=1)
    df["y"] = y
    return df


def main():
    parser = argparse.ArgumentParser(description="Failure Mode Analysis")
    parser.add_argument("--data_dir", type=str,
                        default="data/orig-micro-retry")
    parser.add_argument("--exp_dir", type=str,
                        default="exp/orig-micro-retry")
    parser.add_argument("--n_bins", type=int, default=50,
                        help="Number of bins for numeric feature discretization")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    exp_dir = Path(args.exp_dir)
    out_dir = exp_dir / "failure_mode_analysis"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Loading original training data...")
    df_orig = load_original_train(str(data_dir))
    orig_total = len(df_orig)
    orig_sus = (df_orig["y"] == 1).sum()
    orig_ratio = orig_sus / orig_total * 100
    print(f"  Original: {orig_total} total, {orig_sus} suspicious ({orig_ratio:.2f}%)")

    df_orig_sus = df_orig[df_orig["y"] == 1].copy()

    # --- Part 1: Suspicious ratio & failure mode ---
    ratio_rows = []
    js_rows = []

    for model_key in MODEL_DISPLAY:
        csv_path = exp_dir / f"{model_key}_syn_data_size_1.0x.csv"
        if not csv_path.exists():
            print(f"  [SKIP] {model_key}: {csv_path} not found")
            continue

        display = MODEL_DISPLAY[model_key]
        print(f"\nAnalyzing {display}...")

        df_syn = pd.read_csv(csv_path)
        syn_total = len(df_syn)
        syn_sus = (df_syn["y"] == 1).sum()
        syn_ratio = syn_sus / syn_total * 100
        delta = abs(syn_ratio - orig_ratio)
        auc = TABLE4_AUC.get(model_key, None)
        mode = classify_failure_mode(delta)

        ratio_rows.append({
            "Model": display,
            "Suspicious_Count": syn_sus,
            "Total": syn_total,
            "Suspicious_Ratio(%)": round(syn_ratio, 2),
            "Delta(%p)": round(delta, 2),
            "ROC-AUC": auc,
            "Failure_Mode": mode,
        })
        print(f"  Ratio: {syn_ratio:.2f}% (Δ={delta:.2f}%p) → {mode}, AUC={auc}")

        # --- Part 2: JS divergence for suspicious samples ---
        df_syn_sus = df_syn[df_syn["y"] == 1].copy()

        if len(df_syn_sus) == 0:
            js_row = {"Model": display}
            for feat in FEATURES:
                js_row[FEATURE_DISPLAY[feat]] = 1.0
            js_row["Mean_JS"] = 1.0
            js_rows.append(js_row)
            print(f"  No suspicious samples in synthetic data — JS=1.0 for all features")
            continue

        js_row = {"Model": display}
        js_vals = []

        for feat in FEATURES:
            is_num = feat in NUM_FEATURES
            orig_vals = df_orig_sus[feat].dropna()
            syn_vals = df_syn_sus[feat].dropna()

            if len(orig_vals) == 0 or len(syn_vals) == 0:
                js_val = 1.0
            else:
                js_val = compute_js_divergence(orig_vals, syn_vals,
                                               is_numeric=is_num,
                                               n_bins=args.n_bins)

            js_row[FEATURE_DISPLAY[feat]] = round(js_val, 3)
            js_vals.append(js_val)

        js_row["Mean_JS"] = round(np.mean(js_vals), 3)
        js_rows.append(js_row)
        print(f"  Mean JS divergence: {js_row['Mean_JS']:.3f}")

    # --- Save results ---
    df_ratio = pd.DataFrame(ratio_rows)
    df_js = pd.DataFrame(js_rows)

    ratio_path = out_dir / "failure_mode_ratio.csv"
    js_path = out_dir / "failure_mode_js_divergence.csv"

    df_ratio.to_csv(ratio_path, index=False)
    df_js.to_csv(js_path, index=False)

    print(f"\n{'='*80}")
    print("Failure Mode - Label Ratio Preservation:")
    print(df_ratio.to_string(index=False))
    print(f"\nSaved to: {ratio_path}")

    print(f"\n{'='*80}")
    print("Failure Mode - JS Divergence (Suspicious Samples):")
    print(df_js.to_string(index=False))
    print(f"\nSaved to: {js_path}")

    # --- Save combined summary JSON ---
    summary = {
        "original": {
            "total": int(orig_total),
            "suspicious": int(orig_sus),
            "ratio_pct": round(orig_ratio, 3),
        },
        "models": {},
    }
    for r, j in zip(ratio_rows, js_rows):
        model = r["Model"]
        summary["models"][model] = {
            "ratio": r,
            "js_divergence": j,
        }

    summary_path = out_dir / "failure_mode_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False, default=lambda x: int(x) if isinstance(x, (np.integer,)) else float(x) if isinstance(x, (np.floating,)) else x)
    print(f"\nSummary JSON saved to: {summary_path}")


if __name__ == "__main__":
    main()
