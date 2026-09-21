"""Fin-Syn npy 형식 <-> TabSyn / TabDiff 데이터 형식.

export : <part_dir>/{X_num,X_cat,y}_train.npy + info.json
         -> <repo>/data/<name>/{X_num,X_cat,y}_{train,test}.npy + info.json (두 저장소의 process_dataset.py 스키마)
         -> <repo>/synthetic/<name>/{real,test}.csv (TabDiff는 real.csv 행 수만큼 샘플링한다)
import : 샘플 CSV(열 이름 = 우리 열 이름 + 'y') -> <out>/{X_num,X_cat,y}_train.npy (다른 생성기의 gen_dir와 같은 형식)
         실제 데이터에서 정수인 수치 열은 반올림한다(두 구현 모두 기본적으로 반올림하지 않는다; CTAB-GAN, TabDDPM은 반올림).
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd


def _int_cols(Xn):
    return [j for j in range(Xn.shape[1]) if np.all(np.mod(Xn[:, j], 1) == 0)]


def export(part_dir, repo, name):
    part_dir, repo = Path(part_dir), Path(repo)
    info = json.loads((part_dir / "info.json").read_text())
    Xn = np.load(part_dir / "X_num_train.npy", allow_pickle=True).astype(np.float32)
    Xc = np.load(part_dir / "X_cat_train.npy", allow_pickle=True).astype(str).astype(object)
    y = np.load(part_dir / "y_train.npy", allow_pickle=True).astype(np.int64)
    num, cat = info["num_cols"], info["cat_cols"]
    cols = num + cat + ["y"]
    n_num, n_cat = len(num), len(cat)
    num_idx, cat_idx, tgt_idx = list(range(n_num)), list(range(n_num, n_num + n_cat)), [n_num + n_cat]
    ident = {i: i for i in range(len(cols))}  # 열 순서가 이미 num | cat | target
    int_wrt_num = _int_cols(Xn)  # TabDiff process_dataset.py와 같은 규칙

    d = repo / "data" / name
    d.mkdir(parents=True, exist_ok=True)
    # 90/10 분할 없음: 해당 기간 전체가 학습 데이터다(run_generators_v2.prepare와 동일).
    # 'test'는 사본이며 TabSyn VAE가 LR plateau·최적 체크포인트 선택에만 쓴다.
    for s in ("train", "test"):
        np.save(d / f"X_num_{s}.npy", Xn)
        np.save(d / f"X_cat_{s}.npy", Xc, allow_pickle=True)
        np.save(d / f"y_{s}.npy", y.reshape(-1, 1))
    meta = {"columns": {**{i: {"sdtype": "numerical", "computer_representation": "Float"} for i in num_idx},
                        **{i: {"sdtype": "categorical"} for i in cat_idx + tgt_idx}}}
    out = {"name": name, "task_type": "binclass", "header": "infer", "column_names": cols,
           "num_col_idx": num_idx, "cat_col_idx": cat_idx, "target_col_idx": tgt_idx,
           "file_type": "csv", "data_path": f"data/{name}/{name}.csv", "test_path": None, "val_path": None,
           "train_num": int(len(y)), "test_num": int(len(y)),
           "idx_mapping": ident, "inverse_idx_mapping": ident, "idx_name_mapping": dict(enumerate(cols)),
           "int_col_idx": int_wrt_num, "int_columns": [num[j] for j in int_wrt_num], "int_col_idx_wrt_num": int_wrt_num,
           "metadata": meta, "n_classes": 2}
    (d / "info.json").write_text(json.dumps(out, indent=2))
    df = pd.concat([pd.DataFrame(Xn, columns=num), pd.DataFrame(Xc, columns=cat), pd.DataFrame({"y": y})], axis=1)
    sd = repo / "synthetic" / name
    sd.mkdir(parents=True, exist_ok=True)
    df.to_csv(sd / "real.csv", index=False)
    df.to_csv(sd / "test.csv", index=False)
    return len(y)


def import_samples(csv_path, part_dir, out_dir):
    part_dir = Path(part_dir)
    info = json.loads((part_dir / "info.json").read_text())
    num, cat = info["num_cols"], info["cat_cols"]
    df = pd.read_csv(csv_path, dtype={c: str for c in cat})
    Xn = df[num].to_numpy(dtype=np.float64)
    ints = _int_cols(np.load(part_dir / "X_num_train.npy", allow_pickle=True).astype(np.float64))
    Xn[:, ints] = np.round(Xn[:, ints])
    y = np.round(pd.to_numeric(df["y"])).astype(np.int64).to_numpy()
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    np.save(out / "X_num_train.npy", Xn)
    np.save(out / "X_cat_train.npy", df[cat].astype(str).to_numpy(dtype=object), allow_pickle=True)
    np.save(out / "y_train.npy", y)
    return len(df), float(y.mean())
