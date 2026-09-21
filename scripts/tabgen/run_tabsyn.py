"""공식 TabSyn(VAE -> 잠재 확산 -> 샘플링)을 호환 패치와 학습 길이 상한만 걸어 실행한다.

usage: python run_tabsyn.py <repo> <dataname> <seed> <vae|diff|sample> [--vae_epochs N] [--diff_epochs N] [--save CSV]
  * torch 2.x: ReduceLROnPlateau(verbose=...) 제거
  * diffusion_utils.sample()의 device='cuda:0' 고정 -> args.device
  * TabSyn은 시드를 설정하지 않는다 -> 여기서 python/numpy/torch 시드 고정
  * 학습 길이: vae/main.py의 num_epochs = 4000, tabsyn/main.py의 num_epochs = 10000 + 1 을 상한 값으로 교체
    (공통 학습시간 상한 3시간/기간에 맞춤; 기본값이 상한 안에 들면 기본값 그대로)
"""
import argparse
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from patch import load_patched, torch_compat  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("repo"); ap.add_argument("name"); ap.add_argument("seed", type=int); ap.add_argument("stage")
ap.add_argument("--vae_epochs", type=int, default=4000)
ap.add_argument("--diff_epochs", type=int, default=10001)
ap.add_argument("--save", default=None)
a = ap.parse_args()

torch_compat()
random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed); torch.cuda.manual_seed_all(a.seed)
repo = Path(a.repo).resolve()
os.chdir(repo)
sys.path.insert(0, str(repo))
from utils import get_args  # noqa: E402

device = "cuda:0" if torch.cuda.is_available() else "cpu"
if a.stage == "vae":
    mod = load_patched("tabsyn.vae.main", repo / "tabsyn/vae/main.py",
                       [("num_epochs = 4000", f"num_epochs = {a.vae_epochs}")])
    method, mode = "vae", "train"
elif a.stage == "diff":
    mod = load_patched("tabsyn.main", repo / "tabsyn/main.py",
                       [("num_epochs = 10000 + 1", f"num_epochs = {a.diff_epochs}")])
    method, mode = "tabsyn", "train"
else:
    import tabsyn.diffusion_utils as du
    du.sample.__defaults__ = (50, device)
    import importlib
    mod = importlib.import_module("tabsyn.sample")
    method, mode = "tabsyn", "sample"
sys.argv = ["main.py", "--dataname", a.name, "--method", method, "--mode", mode]
args = get_args()
args.device = device
args.save_path = a.save or f"synthetic/{a.name}/tabsyn.csv"
mod.main(args)
