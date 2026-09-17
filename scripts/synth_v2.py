"""
v2 합성데이터 생성 (split별 생성기).

공개 시나리오: 기간(split)마다 그 기간의 실제 데이터로 생성기를 학습해 같은 크기의 합성 데이터를 만든다.
  real train(2021-09~2023) -> synth train, real val(2024H1) -> synth val, real test(2024H2) -> synth test
출력은 finsyn-v2와 같은 npy 형식이라 leaderboard.py에 그대로 넣을 수 있다.

현재 구현: smote (TabDDPM 기준선과 같은 방식 — 클래스별로 원래 개수만큼 SMOTENC 보간 샘플을 새로 만들고 합성 행만 남김).
GPU 생성기(ctgan, tvae, ctabgan, ctabgan-plus, tabddpm, great, tabpfgen)는 같은 인터페이스(fit_sample)로 추가 예정.

Usage:
    python scripts/synth_v2.py --model smote --real data/finsyn-v2 --out exp/finsyn-v2/synth/smote/seed0 --seed 0
"""
import argparse
import json
import shutil
from pathlib import Path

import numpy as np
from imblearn.over_sampling import SMOTENC
from sklearn.preprocessing import MinMaxScaler


def fit_sample_smote(Xn, Xc, y, seed, k_neighbors=5):
    scaler = MinMaxScaler().fit(Xn)
    X = np.concatenate([scaler.transform(Xn).astype(object), Xc.astype(object)], axis=1)
    cat_idx = list(range(Xn.shape[1], X.shape[1]))
    strat = {int(c): int(2 * (y == c).sum()) for c in np.unique(y)}
    k = int(min(k_neighbors, min((y == c).sum() for c in np.unique(y)) - 1))
    sm = SMOTENC(categorical_features=cat_idx, sampling_strategy=strat, k_neighbors=k, random_state=seed)
    Xr, yr = sm.fit_resample(X, y)
    Xr, yr = Xr[len(y):], yr[len(y):]  # 합성 행만
    num = scaler.inverse_transform(Xr[:, : Xn.shape[1]].astype(float))
    return num, Xr[:, Xn.shape[1]:].astype(object), yr.astype(int)


GENERATORS = {"smote": fit_sample_smote}


def round_integer_columns(real_num, syn_num):
    for j in range(real_num.shape[1]):
        col = real_num[:, j]
        if np.all(np.isclose(col, np.round(col))):
            syn_num[:, j] = np.clip(np.round(syn_num[:, j]), col.min(), col.max())
    return syn_num


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=sorted(GENERATORS))
    ap.add_argument("--real", default="data/finsyn-v2")
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    real, out = Path(args.real), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    info = json.loads((real / "info.json").read_text())
    for s in ["train", "val", "test"]:
        Xn = np.load(real / f"X_num_{s}.npy")
        Xc = np.load(real / f"X_cat_{s}.npy", allow_pickle=True)
        y = np.load(real / f"y_{s}.npy")
        sn, sc, sy = GENERATORS[args.model](Xn, Xc, y, seed=args.seed)
        sn = round_integer_columns(Xn, sn)
        np.save(out / f"X_num_{s}.npy", sn.astype(float))
        np.save(out / f"X_cat_{s}.npy", sc, allow_pickle=True)
        np.save(out / f"y_{s}.npy", sy)
        info[f"{s}_size"] = int(len(sy))
        print(f"{args.model} {s}: real {len(y):,} ({y.mean():.2%}) -> synth {len(sy):,} ({sy.mean():.2%})", flush=True)
    info.update(name=f"synth-{args.model}-seed{args.seed}", generator=args.model, seed=args.seed, source=str(real))
    (out / "info.json").write_text(json.dumps(info, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
