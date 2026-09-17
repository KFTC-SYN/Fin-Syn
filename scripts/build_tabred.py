"""
외부 재현용 데이터셋: TabReD(homecredit-default)를 finsyn-v2와 같은 형식으로 변환한다.

왜: 우리 프로토콜이 비공개 데이터 하나에서만 검증됐다는 것이 가장 큰 약점이다.
    공개된 실제 데이터(시간 기준 분할, 불균형)에서 같은 절차를 반복해
    "분포 지표는 리더보드 충실도를 예측하지 못한다"는 주장을 독립적으로 확인한다.
출처: huggingface.co/datasets/danil-e/harbor-datasets-tabred (TabReD, Rubachev et al., ICLR 2025)의
      40k/10k 슬라이스. train 18286~18382일, test 18519~18540일로 시간 간격이 있다.

축소: 원본은 피처 694개(수치 612 + 범주 82)로 생성기 시간 상한에 맞지 않는다.
      **학습 분할만 사용해** LightGBM 중요도 상위 수치 18개 + 범주 6개(= 우리 벤치마크와 같은 24개)를 고른다.
      결측은 수치 -1(우리 규칙과 동일), 범주는 "NA" 문자열로 채운다.
분할: 제공된 train을 timestamp 기준 앞 80% = train, 뒤 20% = val. 제공된 test는 그대로 test(레이블은 answer.csv).

Usage:
    python scripts/build_tabred.py --out data/tabred-hc
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT.parent / "_datasets/tabred"
N_NUM, N_CAT = 18, 6


def select_features(tr, nums, cats, seed=0):
    """학습 분할만으로 LightGBM 중요도 상위 피처를 고른다(테스트 정보 미사용)."""
    import lightgbm as lgb
    X = tr[nums + cats].copy()
    for c in cats:
        X[c] = X[c].astype("category")
    m = lgb.LGBMClassifier(n_estimators=300, learning_rate=0.05, num_leaves=31,
                           random_state=seed, n_jobs=8, verbose=-1)
    m.fit(X, tr.target)
    imp = pd.Series(m.feature_importances_, index=X.columns)
    return (list(imp[nums].sort_values(ascending=False).head(N_NUM).index),
            list(imp[cats].sort_values(ascending=False).head(N_CAT).index))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "data/tabred-hc"))
    ap.add_argument("--val_frac", type=float, default=0.2)
    args = ap.parse_args()

    tr = pd.read_csv(SRC / "train_full.csv", low_memory=False)
    te = pd.read_csv(SRC / "test_full.csv", low_memory=False)
    te = te.merge(pd.read_csv(SRC / "answer.csv"), on="id", how="left")

    nums = [c for c in tr.columns if c.startswith("num_")]
    cats = [c for c in tr.columns if c.startswith("cat_")]
    nums = [c for c in nums if tr[c].notna().any()]          # 전부 결측인 열 제거
    cut = tr.timestamp.quantile(1 - args.val_frac)
    tr_part = tr[tr.timestamp <= cut]
    sel_num, sel_cat = select_features(tr_part, nums, cats)
    print(f"selected {len(sel_num)} numeric + {len(sel_cat)} categorical features on the training part only")

    def pack(df):
        Xn = df[sel_num].astype(float).fillna(-1.0).values
        Xc = df[sel_cat].astype("Int64").astype(object).where(df[sel_cat].notna(), "NA").astype(str).values
        return Xn, Xc, df.target.astype(int).values

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    splits = {"train": tr[tr.timestamp <= cut], "val": tr[tr.timestamp > cut], "test": te}
    info = {"name": out.name, "id": f"{out.name}--id", "task_type": "binclass", "n_classes": 2,
            "num_cols": sel_num, "cat_cols": sel_cat,
            "n_num_features": len(sel_num), "n_cat_features": len(sel_cat),
            "column_names": sel_num + sel_cat + ["y"],
            "column_mapping": {str(i): c for i, c in enumerate(sel_num + sel_cat + ["y"])},
            "split": {"train": "timestamp<=q80 of provided train", "val": "rest of provided train",
                      "test": "provided test (later period)"},
            "source": "TabReD homecredit-default 40k/10k slice (danil-e/harbor-datasets-tabred)",
            "note": "top-24 features by LightGBM importance on the training split; NaN -> -1 (numeric) / 'NA' (categorical)"}
    for s, df in splits.items():
        Xn, Xc, y = pack(df)
        np.save(out / f"X_num_{s}.npy", Xn)
        np.save(out / f"X_cat_{s}.npy", Xc, allow_pickle=True)
        np.save(out / f"y_{s}.npy", y)
        info[f"{s}_size"] = int(len(y))
        print(f"  {s}: {len(y):,} rows, {y.mean():.3%} positive, "
              f"timestamp {df.timestamp.min()}--{df.timestamp.max()}")
    (out / "info.json").write_text(json.dumps(info, indent=2))
    print("saved ->", out)


if __name__ == "__main__":
    main()
