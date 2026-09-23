"""
네 생성기(CTGAN, TVAE, CTAB-GAN, CTAB-GAN+)의 학습 시딩이 재현성을 주는지 짧은 실행으로 확인한다(9/23).

원 구현은 표집 단계만 시드를 받는다. 우리는 학습 시작에 파이썬·넘파이·토치 난수를 고정했다(train_sample_*.py의
_seed_all). 여기서는 TabReD 학습 기간(part-train)의 앞 2,000행으로 2에폭만 학습해, 같은 시드 두 번은 같은 출력을,
다른 시드는 다른 출력을 내는지 본다. 결과: exp/generator_seeding_check.json

Usage:
    python scripts/check_generator_seeding.py --device cuda:0
"""
import argparse
import importlib
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
GENS = {  # 모듈 경로, 학습 함수, 표집 함수, 추가 학습 인자
    "ctgan": ("CTGAN", "train_sample_ctgan", "train_ctgan", "sample_ctgan", {"batch_size": 500, "epochs": 2}),
    "tvae": ("CTGAN", "train_sample_tvae", "train_tvae", "sample_tvae", {"batch_size": 500, "epochs": 2}),
    "ctabgan": ("CTAB-GAN", "train_sample_ctabgan", "train_ctabgan", "sample_ctabgan", {"batch_size": 500, "epochs": 2}),
    "ctabgan-plus": ("CTAB-GAN-Plus", "train_sample_ctabganp", "train_ctabgan", "sample_ctabgan",
                     {"batch_size": 500, "epochs": 2}),
}


def small_copy(dst, n=2000):
    src = ROOT / "data/tabred-hc-part-train"
    dst.mkdir(parents=True)
    shutil.copy(src / "info.json", dst / "info.json")
    for split in ("train", "val", "test"):
        for k in ("X_num", "X_cat", "y"):
            a = np.load(src / f"{k}_{split}.npy", allow_pickle=True)
            np.save(dst / f"{k}_{split}.npy", a[:n])


def run_once(gen, data, seed, device, work):
    folder, mod, tr, sa, params = GENS[gen]
    sys.path[:0] = [str(ROOT), str(ROOT / folder)]
    m = importlib.import_module(mod)
    out = work / f"{gen}_s{seed}_{len(list(work.iterdir()))}"
    out.mkdir()
    syn = getattr(m, tr)(out, data, train_params=dict(params), device=device, seed=seed)
    getattr(m, sa)(syn, out, data, 500, train_params=dict(params), device=device, seed=seed)
    return np.load(out / "X_num_train.npy", allow_pickle=True).astype(float)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:0")
    a = ap.parse_args()
    os.chdir(ROOT)
    res = {}
    with tempfile.TemporaryDirectory() as t:
        data = Path(t) / "tabred-hc-part-train"  # CTAB-GAN은 columns.json을 데이터 폴더 이름으로 찾는다
        small_copy(data)
        work = Path(t) / "runs"
        work.mkdir()
        for gen in GENS:
            try:
                a0, a1, b = (run_once(gen, data, s, a.device, work) for s in (0, 0, 1))
                res[gen] = {"same_seed_identical": bool(np.array_equal(a0, a1)),
                            "same_seed_max_abs_diff": float(np.nanmax(np.abs(a0 - a1))),
                            "different_seed_identical": bool(np.array_equal(a0, b))}
            except Exception as e:  # noqa: BLE001
                res[gen] = {"error": repr(e)}
            print(gen, res[gen], flush=True)
    (ROOT / "exp/generator_seeding_check.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
