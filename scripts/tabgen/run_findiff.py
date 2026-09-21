"""공식 FinDiff(ICAIF'23, github.com/sattarov/FinDiff)를 기본 설정으로 학습·샘플링한다(저장소 코드 수정 없음).

usage: python run_findiff.py <part_dir> <out_dir> <seed> [--epochs N] [--device cuda:0] [--time_only K]
  * 설정: FinDiff 클래스 기본값(MLP backbone, cat_emb_dim 2, 1000 diffusion steps, linear schedule,
    lr 1e-4, batch 128, num_epochs 100)과 DataTransformer 기본값(standard scaler).
    학습 길이만 공통 상한(기간당 3시간)에 맞춰 --epochs로 준다.
  * 레이블: FinDiff의 레이블 조건부 생성을 쓰고, 조건 레이블은 해당 기간 실제 레이블을 섞은 것
    (= 실제 클래스 비율 그대로). TabDDPM(y 조건부, 경험적 레이블 분포)과 같은 방식이다.
  * 시드: python/numpy/torch 시드를 고정하고 DataLoader 셔플에도 같은 시드의 generator를 쓴다.
  * 출력: <out_dir>/{X_num,X_cat,y}_train.npy (다른 생성기의 gen_dir와 같은 형식).
    실제 데이터에서 정수인 수치 열은 반올림한다(adapter.py와 같은 규칙).
  * --time_only K: K 에폭만 학습하고 에폭당 시간을 출력한 뒤 끝낸다(상한 산정용).
"""
import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "FinDiff"))
from findiff.data import DataTransformer, FinDiffDataset  # noqa: E402
from findiff.model import FinDiff  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("part_dir"); ap.add_argument("out_dir"); ap.add_argument("seed", type=int)
ap.add_argument("--epochs", type=int, default=100)
ap.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
ap.add_argument("--time_only", type=int, default=0)
a = ap.parse_args()

random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed); torch.cuda.manual_seed_all(a.seed)
part, out = Path(a.part_dir), Path(a.out_dir)
info = json.loads((part / "info.json").read_text())
num, cat = info["num_cols"], info["cat_cols"]
Xn = np.load(part / "X_num_train.npy", allow_pickle=True).astype(np.float64)
Xc = np.load(part / "X_cat_train.npy", allow_pickle=True).astype(str)
y = np.load(part / "y_train.npy", allow_pickle=True).astype(np.int64)
X = pd.concat([pd.DataFrame(Xn, columns=num), pd.DataFrame(Xc, columns=cat)], axis=1)

tr = DataTransformer(categorical_cols=cat, numerical_cols=num)
enc = tr.fit_transform(X, pd.Series(y, name="y"))
ds = FinDiffDataset(cat_dataset=torch.tensor(enc["cat"]), num_dataset=torch.tensor(enc["num"], dtype=torch.float32),
                    labels=torch.tensor(enc["label"]))
epochs = a.time_only or a.epochs
model = FinDiff(data_transformer=tr, device=a.device, num_epochs=epochs)
dl = DataLoader(ds, batch_size=model.batch_size, shuffle=True, generator=torch.Generator().manual_seed(a.seed))
t0 = time.time()
model.fit(dl)
t_train = time.time() - t0
if a.time_only:
    print(f"TIME_PER_EPOCH {t_train / epochs:.2f} s ({len(y):,} rows, {len(dl)} batches/epoch)", flush=True)
    sys.exit(0)

labels = torch.tensor(np.random.default_rng(a.seed).permutation(enc["label"]))
t1 = time.time()
syn = model.sample(label=labels)
t_sample = time.time() - t1
y_syn = tr.label_encoder.inverse_transform(labels.numpy()).astype(np.int64)
Sn = syn[num].to_numpy(dtype=np.float64)
ints = [j for j in range(Xn.shape[1]) if np.all(np.mod(Xn[:, j], 1) == 0)]
Sn[:, ints] = np.round(Sn[:, ints])
out.mkdir(parents=True, exist_ok=True)
np.save(out / "X_num_train.npy", Sn)
np.save(out / "X_cat_train.npy", syn[cat].astype(str).to_numpy(dtype=object), allow_pickle=True)
np.save(out / "y_train.npy", y_syn)
(out / "findiff_run.json").write_text(json.dumps({
    "seed": a.seed, "epochs": epochs, "rows": int(len(y_syn)), "positive_rate": float(y_syn.mean()),
    "train_min": round(t_train / 60, 1), "sample_min": round(t_sample / 60, 1)}, indent=1))
print(f"Time: train {t_train/60:.1f} min, sample {t_sample/60:.1f} min, rows {len(y_syn):,}, pos {y_syn.mean():.4f}", flush=True)
