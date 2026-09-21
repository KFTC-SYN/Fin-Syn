"""
공개용 패키지 생성: 내부 .npy 형식의 합성 공개본을 열 이름이 붙은 Parquet으로 바꾼다.

왜: exp/finsyn-v2/synth/<generator>/seed<k>/ 는 X_num/X_cat/y 를 .npy로 나눠 두는 내부 형식이라
    외부 사용자가 읽을 수 없다. 논문이 약속한 "공개본 60개"는 열 이름이 있는 한 장의 표여야 한다.

출력: release/data/<generator>/seed<k>/{train,val,test}.parquet
      release/manifest.csv  (공개본별 측정된 leaderboard fidelity 등)

Usage:
    python scripts/package_release.py --out release
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
E = ROOT / "exp/finsyn-v2"
SPLITS = ["train", "val", "test"]
# 논문에서 평가한 12개 생성기. tabpfgen-nobal은 비교 대상이 되지 못해 논문에서 제외했다.
GENERATORS = ["smote", "tabsyn", "tabdiff", "findiff", "tabpfgen", "tabpfgen-prior",
              "tabddpm", "tvae", "great", "ctgan", "ctabgan", "ctabgan-plus"]
SEEDS = range(5)


def load_split(d: Path, split: str, cols: dict) -> pd.DataFrame:
    num = np.load(d / f"X_num_{split}.npy", allow_pickle=True)
    cat = np.load(d / f"X_cat_{split}.npy", allow_pickle=True)
    y = np.load(d / f"y_{split}.npy", allow_pickle=True)
    df = pd.DataFrame(num, columns=cols["num_cols"])
    # 18개 수치 열 가운데 17개는 개수·금액이라 정수다. float64로 두면 파일이 두 배가 된다.
    for c in df.columns:
        v = df[c].to_numpy()
        if np.allclose(v, np.round(v)):
            lo, hi = v.min(), v.max()
            df[c] = v.astype("int32" if -2**31 < lo and hi < 2**31 else "int64")
        else:
            df[c] = v.astype("float32")
    for i, c in enumerate(cols["cat_cols"]):
        df[c] = pd.Series(cat[:, i]).astype("category")
    df["label"] = pd.Series(y).astype("int8")
    return df[cols["num_cols"] + cols["cat_cols"] + ["label"]]


def fidelity(model: str, seed: int, tag: str):
    """저장된 결과에서 이 공개본의 leaderboard fidelity를 읽는다."""
    name = f"leaderboard_{model}_{tag}" + ("" if seed == 0 else f"_seed{seed}")
    f = E / name / "fidelity_vs_leaderboard_real.json"
    return round(json.loads(f.read_text())["kendall_tau"], 3) if f.exists() else None



def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="release")
    a = ap.parse_args()
    out = ROOT / a.out
    cols = json.loads((ROOT / "data/finsyn-v2/info.json").read_text())
    rows, missing = [], []

    for m in GENERATORS:
        for s in SEEDS:
            d = E / f"synth/{m}/seed{s}"
            if not (d / "info.json").exists():
                missing.append(f"{m}/seed{s}")
                continue
            dst = out / "data" / m / f"seed{s}"
            dst.mkdir(parents=True, exist_ok=True)
            sizes = {}
            for sp in SPLITS:
                df = load_split(d, sp, cols)
                df.to_parquet(dst / f"{sp}.parquet", index=False, compression="zstd")
                sizes[sp] = len(df)
            rows.append({
                "generator": m, "seed": s,
                "train_rows": sizes["train"], "val_rows": sizes["val"], "test_rows": sizes["test"],
                "positive_rate_train": round(float(load_split(d, "train", cols)["label"].mean()), 5),
                "tau_s2s": fidelity(m, s, "s2s"), "tau_s2r": fidelity(m, s, "s2r"),
            })
            print(f"  {m}/seed{s}: {sizes}")

    out.mkdir(parents=True, exist_ok=True)
    man = pd.DataFrame(rows).sort_values(["generator", "seed"])
    man.to_csv(out / "manifest.csv", index=False)
    total = sum(f.stat().st_size for f in (out / "data").rglob("*.parquet"))
    print(f"\n공개본 {len(man)}개, 파일 {len(list((out / 'data').rglob('*.parquet')))}개, 합계 {total / 1048576:.1f} MB")
    if missing:
        print("빠진 공개본:", missing)


if __name__ == "__main__":
    main()
