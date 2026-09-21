"""공식 TabDiff(학습 가능한 노이즈 스케줄, 기본 설정)를 호환 패치와 학습 길이 상한만 걸어 실행한다.

usage: python run_tabdiff.py <repo> <dataname> <seed> <train|test> [--steps N]
  * torch 2.x: ReduceLROnPlateau(verbose=...) 제거
  * wandb -> no-op (공식 실행도 --no_wandb)
  * tabdiff.metrics.TabMetrics -> 스텁. 학습 중 평가는 로그용이고(sdmetrics/xgboost/kaleido 필요)
    체크포인트 선택은 학습 손실 기준이다. 샘플 수(real.csv 행 수)만 넘긴다.
  * 시드: 저장소의 --deterministic은 시드 0 고정이라 여기서 시드를 건다.
  * 학습 길이: 설정의 steps = 8000을 상한 값으로 교체하고, "epoch > 4000 이후에만 최적 체크포인트 저장" 규칙을
    같은 비율(학습의 후반부)로 steps // 2 로 맞춘다. 그대로 두면 상한에서 멈출 때 체크포인트가 하나도 없다.
    check_val_every(2000, 로그용 중간 샘플링)는 학습 길이보다 크게 둬 시간을 쓰지 않게 한다.
"""
import argparse
import os
import random
import sys
import types
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from patch import load_patched, torch_compat  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("repo"); ap.add_argument("name"); ap.add_argument("seed", type=int); ap.add_argument("mode")
ap.add_argument("--steps", type=int, default=8000)
a = ap.parse_args()

torch_compat()


class _Run:
    def define_metric(self, *x, **k): pass
    def log(self, *x, **k): pass


sys.modules["wandb"] = types.SimpleNamespace(init=lambda *x, **k: _Run())


class TabMetrics:
    def __init__(self, real_data_path, test_data_path, val_data_path, info, device, metric_list):
        self.info, self.real_data_size = info, len(pd.read_csv(real_data_path))

    def evaluate(self, syn):
        return {}, {}

    def plot_density(self, syn):
        from PIL import Image
        return Image.new("RGB", (1, 1))


sys.modules["tabdiff.metrics"] = types.SimpleNamespace(TabMetrics=TabMetrics)
random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed); torch.cuda.manual_seed_all(a.seed)
repo = Path(a.repo).resolve()
os.chdir(repo)
sys.path.insert(0, str(repo))
import src  # noqa: E402

half = a.steps // 2
load_patched("tabdiff.trainer", repo / "tabdiff/trainer.py",
             [("if total_loss < best_loss and self.curr_epoch > 4000:",
               f"if total_loss < best_loss and self.curr_epoch > {half}:"),
              ("if ema_total_loss < best_ema_loss and self.curr_epoch > 4000:",
               f"if ema_total_loss < best_ema_loss and self.curr_epoch > {half}:")])
_load = src.load_config


def load_config(path):
    cfg = _load(path)
    cfg["train"]["main"]["steps"] = a.steps
    cfg["train"]["main"]["check_val_every"] = a.steps + 1
    return cfg


src.load_config = load_config
from tabdiff.main import main  # noqa: E402

args = argparse.Namespace(dataname=a.name, mode=a.mode, method="tabdiff", gpu=0, debug=False, no_wandb=True,
                          exp_name=f"seed{a.seed}", deterministic=False, y_only=False, non_learnable_schedule=False,
                          num_samples_to_generate=None, ckpt_path=None, report=False, num_runs=20, impute=False,
                          trial_start=0, trial_size=50, resample_rounds=1, impute_condition="x_t",
                          y_only_model_path=None, w_num=0.6, w_cat=0.6)
args.device = "cuda:0" if torch.cuda.is_available() else "cpu"
main(args)
