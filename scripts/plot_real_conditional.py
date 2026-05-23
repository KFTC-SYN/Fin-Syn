"""
원본 데이터의 label-conditional 분포 figure 생성 (real-only, 논문 Figure 2용)
"""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pandas.api.types import is_numeric_dtype


def main():
    real_data_path = Path("data/orig-micro-retry")
    output_dir = Path("exp/orig-micro-retry/ddpm_cb_best/fraud_fidelity")
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(real_data_path / "info.json") as f:
        info = json.load(f)
    num_cols = info.get("num_cols", [])
    cat_cols = info.get("cat_cols", [])

    X_num = np.load(real_data_path / "X_num_train.npy", allow_pickle=True) if (real_data_path / "X_num_train.npy").exists() else None
    X_cat = np.load(real_data_path / "X_cat_train.npy", allow_pickle=True) if (real_data_path / "X_cat_train.npy").exists() else None
    y = np.load(real_data_path / "y_train.npy", allow_pickle=True)

    dfs = []
    if X_num is not None:
        dfs.append(pd.DataFrame(X_num, columns=num_cols))
    if X_cat is not None:
        df_cat = pd.DataFrame(X_cat, columns=cat_cols)
        for c in df_cat.columns:
            df_cat[c] = df_cat[c].astype(str)
        dfs.append(df_cat)
    df = pd.concat(dfs, axis=1) if dfs else pd.DataFrame()
    df["y"] = y

    feature_alias = {
        "입금계좌일련번호": "deposit_account",
        "출금계좌일련번호": "withdraw_account",
        "출금금융회사일련번호": "withdraw_bank",
        "입금금융회사일련번호": "deposit_bank",
        "거래금액": "amount",
        "거래일자": "date",
        "거래시간대": "timeband",
        "매체구분": "mediatype",
        "자금구분": "fundtype",
    }
    force_categorical = {
        "입금계좌일련번호", "출금계좌일련번호",
        "출금금융회사일련번호", "입금금융회사일련번호",
    }
    color_map = {0: "#8EC6E8", 1: "#F6BD60"}
    label_map = {0: "Benign", 1: "Suspicious"}

    features = [
        "거래일자", "거래시간대", "출금금융회사일련번호",
        "출금계좌일련번호", "입금금융회사일련번호", "입금계좌일련번호",
        "거래금액", "매체구분", "자금구분",
    ]

    df_0 = df[df["y"] == 0]
    df_1 = df[df["y"] == 1]

    for feature in features:
        if feature not in df.columns:
            print(f"[SKIP] {feature}")
            continue

        alias = feature_alias.get(feature, feature)
        out_path = str(output_dir / f"label_conditional_{alias}_real.png")

        s0 = df_0[feature].dropna()
        s1 = df_1[feature].dropna()
        if len(s0) == 0 or len(s1) == 0:
            continue

        fig, ax = plt.subplots(figsize=(3, 3))

        if is_numeric_dtype(df[feature]) and feature not in force_categorical:
            from scipy.stats import gaussian_kde
            all_data = pd.concat([s0, s1])
            upper = np.percentile(all_data, 99.5)
            s0c = s0.clip(upper=upper)
            s1c = s1.clip(upper=upper)
            lo = min(s0c.min(), s1c.min())
            hi = max(s0c.max(), s1c.max())
            if lo == hi:
                plt.close()
                continue
            xs = np.linspace(lo, hi, 200)
            for s, label_idx in [(s0c, 0), (s1c, 1)]:
                kde = gaussian_kde(s)
                vals = kde(xs)
                ax.plot(xs, vals, label=label_map[label_idx],
                        color=color_map[label_idx], linewidth=1, alpha=0.8)
                ax.fill_between(xs, vals, alpha=0.2, color=color_map[label_idx])
        else:
            top_k = 20
            vc = s0.value_counts()
            top_cats = vc.head(top_k).index
            s0p = s0.where(s0.isin(top_cats), "OTHER").astype(str).replace("nan", "OTHER")
            s1p = s1.where(s1.isin(top_cats), "OTHER").astype(str).replace("nan", "OTHER")
            cats = sorted(set(s0p.unique()) | set(s1p.unique()))
            x = np.arange(len(cats))
            p0 = s0p.value_counts(normalize=True).reindex(cats, fill_value=0)
            p1 = s1p.value_counts(normalize=True).reindex(cats, fill_value=0)
            width = 0.35
            ax.bar(x - width / 2, p0, width=width, label=label_map[0],
                   color=color_map[0], alpha=0.8, edgecolor="black", linewidth=0.5)
            ax.bar(x + width / 2, p1, width=width, label=label_map[1],
                   color=color_map[1], alpha=0.8, edgecolor="black", linewidth=0.5)

        ax.set_xlabel(alias, fontsize=11)
        ax.set_ylabel("Probability", fontsize=12)
        ax.legend(fontsize=9, loc="best")
        ax.grid(True, alpha=0.3, linestyle=":")
        plt.tight_layout()
        plt.savefig(out_path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"Saved: {out_path}")

    print("Done.")


if __name__ == "__main__":
    main()
