"""
GReaT 생성기 (v2 전용 어댑터).

저장소의 be_great/pipeline_great.py는 컬럼 이름을 "0".."23" 숫자로 넘기는데, GReaT은 "컬럼명 is 값" 문장을 학습하므로
의미 없는 숫자 컬럼명에서는 이름과 값을 혼동해(예: "22 is 2, 8 is 1062") 행 구조를 학습하지 못하고 유효 행을 만들지 못했다.
여기서는 info.json의 실제 피처 이름을 그대로 넘긴다(원 논문이 전제하는 사용법). 그 외 설정은 be_great 기본값
(distilgpt2, temperature 0.7, random feature order, conditional column = 마지막 컬럼 y)이며, epochs만 공통 학습시간 상한에 맞춘다.
샘플링 max_length는 24개 피처 문장이 잘리지 않도록 500으로 둔다(학습 하이퍼파라미터가 아닌 디코딩 길이).

Usage:
    python scripts/great_v2.py --part test --epochs 3 --n 300 --out /tmp/great_test
    python scripts/great_v2.py --part train --epochs 14 --out exp/finsyn-v2/gen/great/train
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--real", default=str(ROOT / "data/finsyn-v2"))
    ap.add_argument("--part", required=True, choices=["train", "val", "test"])
    ap.add_argument("--epochs", type=float, required=True)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--llm", default="distilgpt2")
    ap.add_argument("--n", type=int, default=None, help="샘플 수 (기본: 해당 기간 실제 행 수)")
    ap.add_argument("--max_length", type=int, default=500)
    ap.add_argument("--sample_batch", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    import torch
    from be_great import GReaT

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    real, out = Path(args.real), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    info = json.loads((real / "info.json").read_text())
    num, cat = info["num_cols"], info["cat_cols"]
    Xn = np.load(real / f"X_num_{args.part}.npy")
    Xc = np.load(real / f"X_cat_{args.part}.npy", allow_pickle=True)
    y = np.load(real / f"y_{args.part}.npy")
    df = pd.concat([pd.DataFrame(Xn, columns=num), pd.DataFrame(Xc, columns=cat).astype(str),
                    pd.Series(y.astype(int), name="suspicious")], axis=1)
    n = args.n or len(df)

    t0 = time.time()
    # HF Trainer는 기본 시드 42로 셔플·dropout을 고정한다. 생성기 시드를 학습에도 반영하되,
    # 기존 seed 0 릴리스(Trainer 시드 42로 학습됨)가 그대로 재현되도록 42 + seed 로 둔다.
    model = GReaT(llm=args.llm, experiment_dir=str(out / "trainer_great"), epochs=args.epochs, batch_size=args.batch_size,
                  save_strategy="no", logging_steps=50, report_to=[], seed=42 + args.seed, data_seed=42 + args.seed)
    model.fit(df)
    t_train = time.time() - t0

    t1 = time.time()
    syn = model.sample(n_samples=n, k=args.sample_batch, max_length=args.max_length, device="cuda")
    t_sample = time.time() - t1

    syn = syn[syn["suspicious"].astype(str).isin(["0", "1", "0.0", "1.0"])].copy()
    for c in num:
        syn[c] = pd.to_numeric(syn[c], errors="coerce")
    syn = syn.dropna(subset=num).reset_index(drop=True)
    np.save(out / "X_num_train.npy", syn[num].to_numpy(float))
    np.save(out / "X_cat_train.npy", syn[cat].astype(str).to_numpy(object), allow_pickle=True)
    np.save(out / "y_train.npy", syn["suspicious"].astype(float).astype(int).to_numpy())
    rec = {"model": "great", "part": args.part, "epochs": args.epochs, "rows_requested": n, "rows_valid": int(len(syn)),
           "pos_rate": float(syn["suspicious"].astype(float).mean()) if len(syn) else None,
           "train_min": round(t_train / 60, 1), "sample_min": round(t_sample / 60, 1),
           "rows_per_sec": round(len(syn) / max(t_sample, 1e-9), 2)}
    (out / "great_run.json").write_text(json.dumps(rec, indent=1))
    print(json.dumps(rec), flush=True)


if __name__ == "__main__":
    main()
