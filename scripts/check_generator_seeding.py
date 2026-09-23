"""
네 생성기(CTGAN, TVAE, CTAB-GAN, CTAB-GAN+)의 학습 시딩이 짧은 실행에서 재현성을 주는지 확인한다(9/23, 같은 날 재작성).

원 구현은 표집 단계만 시드를 받는다. 우리 래퍼(train_sample_*.py의 _seed_all)는 학습 시작에 파이썬·넘파이·토치
난수를 고정한다. 여기서는 TabReD 학습 기간(part-train)의 앞 2,000행으로 2에폭만 학습·표집해
  - 같은 시드 두 번의 출력이 같은지,
  - 다른 시드의 출력이 다른지
를 수치형 특성, 범주형 특성, 레이블 각각에서 본다.

재작성 이유: 첫 버전은 네 생성기를 한 프로세스에서 불러왔다. CTAB-GAN과 CTAB-GAN+는 둘 다 최상위 패키지
이름이 `model`이라, 뒤에 불러온 CTAB-GAN+가 CTAB-GAN의 클래스를 재사용했다(외부 검토에서 발견). 이제 실행마다
독립 프로세스를 쓰고, 실제로 불러온 생성기 클래스의 파일 경로를 결과에 남겨 두 구현이 구분됐는지 확인한다.

범위: 짧은 실행에서의 비트 단위 동일성만 확인한다. 전체 학습(수 시간)의 재현성이나 분포상의 재현성은 확인하지 않는다.

결과: exp/generator_seeding_check_{cpu,gpu,gpu_cudnn_deterministic}.json

Usage:
    python scripts/check_generator_seeding.py --device cpu
    python scripts/check_generator_seeding.py --device cuda:0
    python scripts/check_generator_seeding.py --device cuda:0 --models ctabgan,ctabgan-plus --cudnn-deterministic
"""
import argparse
import importlib
import inspect
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
GENS = {  # 폴더, 모듈, 학습 함수, 표집 함수
    "ctgan": ("CTGAN", "train_sample_ctgan", "train_ctgan", "sample_ctgan"),
    "tvae": ("CTGAN", "train_sample_tvae", "train_tvae", "sample_tvae"),
    "ctabgan": ("CTAB-GAN", "train_sample_ctabgan", "train_ctabgan", "sample_ctabgan"),
    "ctabgan-plus": ("CTAB-GAN-Plus", "train_sample_ctabganp", "train_ctabgan", "sample_ctabgan"),
}
PARAMS = {"batch_size": 500, "epochs": 2}
N_ROWS = 2000


def small_copy(dst):
    src = ROOT / "data/tabred-hc-part-train"  # CTAB-GAN은 columns.json을 데이터 폴더 이름으로 찾는다
    dst.mkdir(parents=True)
    shutil.copy(src / "info.json", dst / "info.json")
    for split in ("train", "val", "test"):
        for k in ("X_num", "X_cat", "y"):
            a = np.load(src / f"{k}_{split}.npy", allow_pickle=True)
            np.save(dst / f"{k}_{split}.npy", a[:N_ROWS])


def worker(gen, data, seed, device, out):
    """한 번 학습·표집하고 출력과 실제 클래스 경로를 남긴다. 반드시 새 프로세스에서 호출된다."""
    os.chdir(ROOT)
    if os.environ.get("CUDNN_DETERMINISTIC") == "1":  # 원인 확인용: cuDNN의 비결정적 합성곱 알고리즘을 끈다
        import torch
        torch.backends.cudnn.deterministic, torch.backends.cudnn.benchmark = True, False
    folder, mod, tr, sa = GENS[gen]
    sys.path[:0] = [str(ROOT), str(ROOT / folder)]
    m = importlib.import_module(mod)
    out = Path(out)
    out.mkdir(parents=True)
    syn = getattr(m, tr)(out, Path(data), train_params=dict(PARAMS), device=device, seed=seed)
    getattr(m, sa)(syn, out, Path(data), 500, train_params=dict(PARAMS), device=device, seed=seed)
    (out / "class_file.txt").write_text(inspect.getfile(type(syn)))


def load(out):
    out = Path(out)
    return {k: np.load(out / f"{k}_train.npy", allow_pickle=True) for k in ("X_num", "X_cat", "y")}


def same(a, b):
    if a.dtype.kind in "fc":
        return bool(np.array_equal(a, b, equal_nan=True))
    return bool(np.array_equal(a.astype(str), b.astype(str)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--worker", nargs=5, metavar=("GEN", "DATA", "SEED", "DEVICE", "OUT"))
    ap.add_argument("--models", default=",".join(GENS))
    ap.add_argument("--cudnn-deterministic", action="store_true")
    a = ap.parse_args()
    if a.worker:
        gen, data, seed, device, out = a.worker
        worker(gen, data, int(seed), device, out)
        return
    res = {"device": a.device, "cudnn_deterministic": a.cudnn_deterministic, "rows": N_ROWS, "epochs": PARAMS["epochs"],
           "generators": {}}
    with tempfile.TemporaryDirectory() as t:
        data = Path(t) / "tabred-hc-part-train"
        small_copy(data)
        for gen in a.models.split(","):
            runs = {}
            for tag, seed in (("a", 0), ("b", 0), ("c", 1)):
                out = Path(t) / f"{gen}_{tag}"
                env = {**os.environ, "CUDNN_DETERMINISTIC": "1" if a.cudnn_deterministic else "0"}
                p = subprocess.run([sys.executable, __file__, "--worker", gen, str(data), str(seed), a.device, str(out)],
                                   cwd=ROOT, capture_output=True, text=True, env=env)
                if p.returncode != 0:
                    runs = {"error": p.stderr[-800:]}
                    break
                runs[tag] = (load(out), (out / "class_file.txt").read_text())
            if "error" in runs:
                res["generators"][gen] = runs
                print(gen, "ERROR", runs["error"][-300:], flush=True)
                continue
            (o0, f0), (o1, f1), (o2, f2) = runs["a"], runs["b"], runs["c"]
            assert f0 == f1 == f2, (gen, f0, f1, f2)
            res["generators"][gen] = {
                "class_file": str(Path(f0).relative_to(ROOT)),
                "same_seed_identical": {k: same(o0[k], o1[k]) for k in o0},
                "different_seed_identical": {k: same(o0[k], o2[k]) for k in o0},
                # 같은 시드인데 수치형이 다를 때 그 크기(열 표준편차로 나눈 최대 차이, 달라진 값의 비율)
                "same_seed_X_num_share_differing": float(np.mean(o0["X_num"].astype(float) != o1["X_num"].astype(float))),
                "same_seed_X_num_max_diff_in_sd": float(np.nanmax(np.abs(o0["X_num"].astype(float) - o1["X_num"].astype(float))
                                                          / (np.nanstd(o0["X_num"].astype(float), axis=0) + 1e-12))),
            }
            print(gen, res["generators"][gen], flush=True)
    files = {v["class_file"] for v in res["generators"].values() if "class_file" in v}
    res["distinct_implementations"] = sorted(files)
    tag = ("cpu" if a.device == "cpu" else "gpu") + ("_cudnn_deterministic" if a.cudnn_deterministic else "")
    (ROOT / f"exp/generator_seeding_check_{tag}.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
