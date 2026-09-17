"""
자연 분포 검증셋: test 기간(2024H2)의 추출본 전체 이체로 평가셋을 만든다(case-control 표본이 아닌 실제 비율).

벤치마크(finsyn-v2)는 의심거래 계좌 주변에서 정상 거래를 뽑은 case-control 표본이라 의심거래 비율이 1.28%로 높다.
같은 피처·같은 규칙(과거 전용)으로 2024년 하반기 전체 이체에 대한 평가셋을 만들어, 리더보드 순위가 자연 비율에서도
유지되는지 확인한다. train/val은 finsyn-v2를 그대로 복사하므로 leaderboard.py에 바로 넣을 수 있다.

Usage:
    python scripts/build_natural_test_v2.py --out data/finsyn-v2-natural
"""
import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_dataset_v2 import build_features, load, RAW  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--orig", default=str(ROOT / "_datasets/orig.parquet"))
    ap.add_argument("--src", default=str(ROOT / "data/finsyn-v2"))
    ap.add_argument("--out", default=str(ROOT / "data/finsyn-v2-natural"))
    ap.add_argument("--start", default="20240701")
    ap.add_argument("--end", default="20241231")
    args = ap.parse_args()

    orig = load(args.orig)
    rows = orig[(orig["거래일자"] >= args.start) & (orig["거래일자"] <= args.end)].reset_index(drop=True)
    print(f"natural test rows: {len(rows):,} ({rows.y.mean():.3%} flagged)")
    feats = build_features(orig, rows)
    cat = pd.DataFrame({
        "hour_band": rows["거래시간대"],
        "dow": pd.to_datetime(rows["거래일자"], format="%Y%m%d").dt.dayofweek.astype(str),
        "withdraw_bank": rows["출금금융회사일련번호"],
        "deposit_bank": rows["입금금융회사일련번호"],
        "media_type": rows["매체구분"],
        "fund_type": rows["자금구분"],
    })
    num = pd.concat([rows[["거래금액"]].rename(columns={"거래금액": "amount"}), feats], axis=1).astype(float)

    src, out = Path(args.src), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for s in ["train", "val"]:  # 학습/튜닝은 벤치마크와 동일
        for f in ["X_num", "X_cat", "y"]:
            shutil.copyfile(src / f"{f}_{s}.npy", out / f"{f}_{s}.npy")
    np.save(out / "X_num_test.npy", num.values)
    np.save(out / "X_cat_test.npy", cat.values.astype(object), allow_pickle=True)
    np.save(out / "y_test.npy", rows["y"].values)
    info = json.loads((src / "info.json").read_text())
    info.update(name=out.name, id=f"{out.name}--id", test_size=int(len(rows)),
                note="test split = all extract transfers in 2024H2 (natural prevalence), same features/rules as finsyn-v2")
    (out / "info.json").write_text(json.dumps(info, ensure_ascii=False, indent=2))
    print("saved ->", out)


if __name__ == "__main__":
    main()
