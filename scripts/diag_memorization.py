"""
원천 데이터 중복 / train-test 중복 / 합성데이터 복제율 진단 스크립트 (ICDM R1-3, R2 대응).

1) 원천 split별 고유 행 비율, 클래스별 중복, 동일 피처-상이 레이블 충돌
2) test(val) 행이 train에 그대로 존재하는 비율 (전체 / suspicious)
3) CatBoost(Original)를 전체 test vs. train에 없는 test 행(novel)으로 나눠 평가
4) 합성데이터(x1.0)의 고유 행 비율과 원천 train 행 복제율

Usage:
    python scripts/diag_memorization.py --real data/orig-micro-retry --exp exp/orig-micro-retry
"""
import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.metrics import f1_score, roc_auc_score, average_precision_score

CAT_COLS = ["거래일자", "거래시간대", "출금금융회사일련번호", "출금계좌일련번호",
            "입금금융회사일련번호", "입금계좌일련번호", "매체구분", "자금구분"]
NUM_COL = "거래금액"
FEATS = CAT_COLS + [NUM_COL]
MODELS = ["smote", "ctgan", "tvae", "ctabgan", "ctabgan-plus", "ddpm_cb_best", "great", "tabpfgen"]


def load_split(real_dir: Path, split: str) -> pd.DataFrame:
    x_cat = np.load(real_dir / f"X_cat_{split}.npy", allow_pickle=True)
    x_num = np.load(real_dir / f"X_num_{split}.npy", allow_pickle=True)
    y = np.load(real_dir / f"y_{split}.npy", allow_pickle=True)
    df = pd.DataFrame(x_cat, columns=CAT_COLS).astype(str)
    df[NUM_COL] = x_num[:, 0].astype(float)
    df["y"] = y.astype(int)
    return df


def norm_key(df: pd.DataFrame, with_amount: bool = True) -> pd.Series:
    """행 비교용 키. 범주형은 문자열('1.0'->'1') 정규화, 금액은 정수 반올림."""
    cats = df[CAT_COLS].astype(str).apply(lambda s: s.str.replace(r"\.0$", "", regex=True))
    parts = [cats[c] for c in CAT_COLS]
    if with_amount:
        parts.append(df[NUM_COL].astype(float).round(0).astype("int64").astype(str))
    return pd.concat(parts, axis=1).agg("|".join, axis=1)


def dup_stats(df: pd.DataFrame) -> dict:
    k = norm_key(df)
    c = Counter(k)
    pos = df["y"] == 1
    kp = k[pos]
    # 동일 피처 벡터에 서로 다른 레이블이 붙은 경우
    lab = df.assign(_k=k).groupby("_k")["y"].nunique()
    return {
        "n": len(df),
        "unique_row_ratio": round(len(c) / len(df), 4),
        "max_freq": max(c.values()),
        "rows_in_dup_groups": round(float((k.map(c) > 1).mean()), 4),
        "pos_n": int(pos.sum()),
        "pos_unique_ratio": round(kp.nunique() / max(1, len(kp)), 4),
        "label_conflict_keys": int((lab > 1).sum()),
        "unique_ratio_wo_amount": round(norm_key(df, False).nunique() / len(df), 4),
    }


def overlap(a: pd.DataFrame, b: pd.DataFrame) -> dict:
    """b의 행 중 a(train)에 동일 피처 벡터가 존재하는 비율."""
    ka, kb = set(norm_key(a)), norm_key(b)
    seen = kb.isin(ka)
    pos = b["y"] == 1
    return {
        "frac_rows_seen_in_train": round(float(seen.mean()), 4),
        "frac_pos_rows_seen_in_train": round(float(seen[pos].mean()), 4) if pos.any() else None,
        "n_novel_rows": int((~seen).sum()),
        "n_novel_pos": int((~seen & pos).sum()),
    }


def eval_original(tr, va, te, params_path: Path) -> dict:
    params = json.loads(params_path.read_text()) if params_path.exists() else {}
    params = {k: v for k, v in params.items() if k not in ("task_type", "devices", "cat_features")}
    params.update(task_type="CPU", verbose=False, random_seed=0)
    clf = CatBoostClassifier(**params, cat_features=list(range(len(CAT_COLS))))
    clf.fit(tr[FEATS], tr["y"], eval_set=(va[FEATS], va["y"]))
    seen = norm_key(te).isin(set(norm_key(tr)))
    out = {}
    for name, mask in [("test_all", np.ones(len(te), bool)), ("test_seen", seen.values), ("test_novel", ~seen.values)]:
        sub = te[mask]
        if len(sub) == 0 or sub["y"].nunique() < 2:
            out[name] = {"n": int(len(sub)), "pos": int(sub["y"].sum()), "note": "single class"}
            continue
        p = clf.predict_proba(sub[FEATS])[:, 1]
        out[name] = {
            "n": int(len(sub)), "pos": int(sub["y"].sum()),
            "f1_pos": round(f1_score(sub["y"], p >= 0.5), 4),
            "roc_auc": round(roc_auc_score(sub["y"], p), 4),
            "pr_auc": round(average_precision_score(sub["y"], p), 4),
        }
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--real", default="data/orig-micro-retry")
    ap.add_argument("--exp", default="exp/orig-micro-retry")
    ap.add_argument("--params", default="tuned_models/catboost/orig-micro-retry_cv.json")
    ap.add_argument("--out", default="exp/orig-micro-retry/diag_memorization.json")
    args = ap.parse_args()

    real = Path(args.real)
    tr, va, te = (load_split(real, s) for s in ("train", "val", "test"))
    full = pd.concat([tr, va, te], ignore_index=True)

    res = {"real": {s: dup_stats(d) for s, d in [("train", tr), ("val", va), ("test", te), ("all", full)]}}
    res["split_overlap"] = {"val_vs_train": overlap(tr, va), "test_vs_train": overlap(tr, te)}
    res["original_catboost"] = eval_original(tr, va, te, Path(args.params))

    ktr = set(norm_key(tr))
    res["synthetic_x1.0"] = {}
    for m in MODELS:
        f = Path(args.exp) / f"{m}_syn_data_size_1.0x.csv"
        if not f.exists():
            continue
        s = pd.read_csv(f, dtype={c: str for c in CAT_COLS})
        ks = norm_key(s)
        res["synthetic_x1.0"][m] = {
            **dup_stats(s),
            "frac_rows_copied_from_train": round(float(ks.isin(ktr).mean()), 4),
            "n_unique_novel_rows": int(ks[~ks.isin(ktr)].nunique()),
        }

    Path(args.out).write_text(json.dumps(res, ensure_ascii=False, indent=2))
    print(json.dumps(res, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
