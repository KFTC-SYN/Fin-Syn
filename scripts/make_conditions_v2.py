"""
발견 (b)용 통제 조건 데이터셋: v2와 같은 행·피처에서 분할 방식과 중복만 바꾼다.

  finsyn-v2         : 시간 기준 분할 (주 설정, build_dataset_v2.py)
  finsyn-v2-random  : 같은 행을 층화 무작위 70/15/15 분할
  finsyn-v2-dup     : v1(orig-micro-retry) 방식 재현 — v1과 같은 클래스 구성의 고유 행 풀(정상 1,530 / 의심 627)을
                      v2에서 뽑아 v1의 클래스별 복제 배수(정상 ≈58.3×, 의심 ≈2.78×)로 복제한 뒤 층화 무작위 70/15/15 분할

Usage:
    python scripts/make_conditions_v2.py --src data/finsyn-v2
"""
import argparse
import json
import shutil
from pathlib import Path

import numpy as np
from sklearn.model_selection import train_test_split

V1_POOL = {0: 1530, 1: 627}          # v1 고유 행의 클래스 구성
V1_ROWS = {0: 91005 - 1745, 1: 1745}  # v1 전체 행의 클래스 구성


def load_all(src: Path):
    parts = {s: (np.load(src / f"X_num_{s}.npy"), np.load(src / f"X_cat_{s}.npy", allow_pickle=True), np.load(src / f"y_{s}.npy"))
             for s in ["train", "val", "test"]}
    return tuple(np.concatenate([parts[s][i] for s in ["train", "val", "test"]]) for i in range(3))


def save(out: Path, src: Path, Xn, Xc, y, idx_by_split, note: str):
    out.mkdir(parents=True, exist_ok=True)
    info = json.loads((src / "info.json").read_text())
    for s, idx in idx_by_split.items():
        np.save(out / f"X_num_{s}.npy", Xn[idx])
        np.save(out / f"X_cat_{s}.npy", Xc[idx], allow_pickle=True)
        np.save(out / f"y_{s}.npy", y[idx])
        info[f"{s}_size"] = int(len(idx))
    info["name"], info["id"], info["split"] = out.name, f"{out.name}--id", note
    (out / "info.json").write_text(json.dumps(info, ensure_ascii=False, indent=2))
    msg = " | ".join(f"{s} {len(i):,} ({y[i].mean():.2%})" for s, i in idx_by_split.items())
    print(f"{out.name}: {msg}")


def random_split(y, seed):
    idx = np.arange(len(y))
    tr, rest = train_test_split(idx, test_size=0.30, stratify=y, random_state=seed)
    va, te = train_test_split(rest, test_size=0.50, stratify=y[rest], random_state=seed)
    return {"train": tr, "val": va, "test": te}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="data/finsyn-v2")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    src = Path(args.src)
    rng = np.random.default_rng(args.seed)
    Xn, Xc, y = load_all(src)

    save(Path(f"{src}-random"), src, Xn, Xc, y, random_split(y, args.seed), "stratified random 70/15/15 of v2 rows")

    pool = np.concatenate([rng.choice(np.where(y == c)[0], V1_POOL[c], replace=False) for c in (0, 1)])
    reps = []
    for c in (0, 1):
        members = pool[y[pool] == c]
        base, extra = divmod(V1_ROWS[c], len(members))
        counts = np.full(len(members), base)
        counts[rng.choice(len(members), extra, replace=False)] += 1
        reps.append(np.repeat(members, counts))
    rep = np.concatenate(reps)
    Xn2, Xc2, y2 = Xn[rep], Xc[rep], y[rep]
    save(Path(f"{src}-dup"), src, Xn2, Xc2, y2, random_split(y2, args.seed),
         f"v1-like: pool {V1_POOL} replicated to {V1_ROWS}, stratified random 70/15/15")


if __name__ == "__main__":
    main()
