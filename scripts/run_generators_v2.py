"""
v2 합성데이터 생성 오케스트레이터 (GPU 생성기, 공식 기본값 + 동일 학습시간 상한).

기간(split)별로 생성기를 따로 학습한다: real train -> synth train, real val -> synth val, real test -> synth test.
기존 저장소 파이프라인은 데이터 폴더의 'train' split만 학습하므로, val/test 기간을 'train'으로 둔 part 폴더를 만든다.

단계
  prepare  : part 폴더(data/finsyn-v2-part-{train,val,test}), CTAB-GAN(+) columns.json 항목, 모델×part config 생성
  run      : 파이프라인 --train --sample 실행 (학습시간 상한 초과 시 중단하고 기록)
  assemble : part별 합성 결과를 exp/finsyn-v2/synth/<model>/seed<k>/ 로 모아 leaderboard.py 입력 형식으로 저장

Usage:
    python scripts/run_generators_v2.py prepare
    python scripts/run_generators_v2.py run --models tvae,ctgan --device cuda:0
    python scripts/run_generators_v2.py assemble --models tvae,ctgan
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import tomli
import tomli_w

ROOT = Path(__file__).resolve().parents[1]
REAL = ROOT / "data/finsyn-v2"      # --real 로 교체 가능
PARTS = ["train", "val", "test"]
TEMPLATE = ROOT / "exp/orig-micro-retry"  # 키 구조 참조용 (값은 아래 DEFAULTS로 덮어씀)
GEN_ROOT = ROOT / "exp/finsyn-v2/gen"
SYN_ROOT = ROOT / "exp/finsyn-v2/synth"
TIME_CAP_H = 3.0  # 모든 생성기 공통 학습시간 상한(시간, part당)

# 공식 구현/원 논문 기본값. 근거는 각 주석.
DEFAULTS = {
    # CTGAN/TVAE: ctgan 라이브러리 기본값 (epochs=300, batch_size=500)
    "ctgan": {"template": "ctgan", "pipeline": "CTGAN/pipeline_ctgan.py", "train_params": {
        "epochs": 300, "batch_size": 500, "embedding_dim": 128, "generator_dim": [256, 256], "discriminator_dim": [256, 256],
        "generator_lr": 2e-4, "discriminator_lr": 2e-4, "generator_decay": 1e-6, "discriminator_decay": 1e-6,
        "discriminator_steps": 1, "pac": 10, "log_frequency": True}},
    "tvae": {"template": "tvae", "pipeline": "CTGAN/pipeline_tvae.py", "train_params": {
        "epochs": 300, "batch_size": 500, "embedding_dim": 128, "compress_dims": [128, 128], "decompress_dims": [128, 128],
        "lr": 1e-3, "loss_factor": 2}},
    # CTAB-GAN: 코드 기본값(class_dim, random_dim, num_channels, lr, batch) + 원 논문 실험의 epochs=150 (코드 기본 10은 학습 부족)
    "ctabgan": {"template": "ctabgan", "pipeline": "CTAB-GAN/pipeline_ctabgan.py", "columns": "CTAB-GAN/columns.json", "train_params": {
        "epochs": 150, "batch_size": 512, "class_dim": [256, 256, 256, 256], "random_dim": 128, "num_channels": 64, "lr": 2e-4}},
    # CTAB-GAN+: 코드 기본값 그대로 (epochs=150)
    "ctabgan-plus": {"template": "ctabgan-plus", "pipeline": "CTAB-GAN-Plus/pipeline_ctabganp.py", "columns": "CTAB-GAN-Plus/columns.json", "train_params": {
        "epochs": 150, "batch_size": 500, "class_dim": [256, 256, 256, 256], "random_dim": 100, "num_channels": 64, "lr": 2e-4}},
    # TabDDPM: 라이브러리 기본값이 없어 원 논문 탐색공간의 중앙값을 고정 설정으로 사용, 원래 MLP 백본(ResNet 교체 없음)
    "tabddpm": {"template": "ddpm_cb_best", "pipeline": "scripts/pipeline.py", "tabddpm": {
        "model_type": "mlp", "rtdl_params": {"d_layers": [512, 1024, 1024, 512], "dropout": 0.0},
        "diffusion_params": {"num_timesteps": 1000, "gaussian_loss_type": "mse", "scheduler": "cosine"},
        "train_main": {"steps": 30000, "lr": 1e-3, "weight_decay": 1e-5, "batch_size": 4096}}},
    # GReaT: be_great 기본 LLM(distilgpt2); epochs는 공통 학습시간 상한에 맞춰 9/17 실측 후 확정(기본 100은 상한 초과)
    "great": {"template": "great", "pipeline": "be_great/pipeline_great.py", "train_params": {
        "llm": "distilgpt2", "epochs": 40, "batch_size": 32, "save_strategy": "no", "save_steps": 1000000,
        "save_total_limit": 0, "logging_steps": 50}},
    # TabPFGen: 코드 기본값 (n_sgld_steps=1000, step 0.01, noise 0.01); 구현이 학습 데이터를 10k로 서브샘플
    "tabpfgen": {"template": "tabpfgen", "pipeline": "TabPFGen/pipeline_tabpfgen.py", "train_params": {
        "n_sgld_steps": 1000, "sgld_step_size": 0.01, "sgld_noise_scale": 0.01}},
    # TabPFGen 변형(레이블 비율 보존): 생성기는 기본값 그대로(균형화 유지)이고, 출력만 실제 클래스 비율로 서브샘플한다.
    # balance_classes=False 경로는 SGLD 뒤 TabPFN argmax 재레이블링 때문에 소수 클래스가 전부 사라져(0%) 비교 대상이 되지 못한다.
    "tabpfgen-prior": {"template": "tabpfgen", "pipeline": "TabPFGen/pipeline_tabpfgen.py", "oversample": 2.0, "postprocess": "match_prior",
                       "train_params": {"n_sgld_steps": 1000, "sgld_step_size": 0.01, "sgld_noise_scale": 0.01}},
}


def part_dir(part: str) -> Path:
    return ROOT / f"data/{REAL.name}-part-{part}"


def gen_dir(model: str, part: str, seed: int = 0) -> Path:
    """생성기 산출 디렉터리. seed 0은 기존 경로를 그대로 써서 이미 끝난 결과를 보존한다."""
    return GEN_ROOT / model / (part if seed == 0 else f"{part}_seed{seed}")


def prepare(models, seed=0):
    info = json.loads((REAL / "info.json").read_text())
    for part in PARTS:
        d = part_dir(part)
        d.mkdir(parents=True, exist_ok=True)
        pinfo = dict(info)
        for s in PARTS:  # 파이프라인이 train만 쓰지만 TabDDPM make_dataset은 val/test 파일도 읽음
            for f in ["X_num", "X_cat", "y"]:
                shutil.copyfile(REAL / f"{f}_{part}.npy", d / f"{f}_{s}.npy")
            pinfo[f"{s}_size"] = info[f"{part}_size"]
        pinfo.update(name=d.name, id=f"{d.name}--id")
        (d / "info.json").write_text(json.dumps(pinfo, ensure_ascii=False, indent=2))

    n_num = info["n_num_features"]
    cat_idx = [str(i) for i in range(n_num, n_num + info["n_cat_features"])]
    int_idx = [str(j) for j in range(n_num)
               if np.allclose(np.load(REAL / "X_num_train.npy")[:, j], np.round(np.load(REAL / "X_num_train.npy")[:, j]))]
    for model in models:
        spec = DEFAULTS[model]
        if "columns" in spec:
            p = ROOT / spec["columns"]
            cols = json.loads(p.read_text())
            for part in PARTS:
                cols[part_dir(part).name] = {"categorical_columns": cat_idx + ["y"], "mixed_columns": {},
                                             "integer_columns": int_idx, "problem_type": {"Classification": "y"}}
            p.write_text(json.dumps(cols, indent=4))
        for part in PARTS:
            cfg = tomli.loads((TEMPLATE / spec["template"] / "config.toml").read_text())
            out = gen_dir(model, part, seed)
            out.mkdir(parents=True, exist_ok=True)
            cfg.update(parent_dir=str(out.relative_to(ROOT)), real_data_path=str(part_dir(part).relative_to(ROOT)) + "/", seed=seed)
            cfg["sample"] = {**cfg.get("sample", {}),
                             "num_samples": int(info[f"{part}_size"] * spec.get("oversample", 1.0)), "seed": seed}
            if model == "tabddpm":
                t = spec["tabddpm"]
                cfg["model_type"] = t["model_type"]
                cfg["num_numerical_features"] = n_num
                cfg["model_params"] = {"num_classes": 2, "is_y_cond": True, "rtdl_params": t["rtdl_params"]}
                cfg["diffusion_params"] = t["diffusion_params"]
                cfg["train"]["main"] = t["train_main"]
                cfg["sample"]["batch_size"] = 10000
            else:
                cfg["train_params"] = spec["train_params"]
            (out / "config.toml").write_text(tomli_w.dumps(cfg))
        print(f"prepared {model} (seed {seed}): {', '.join(PARTS)}")


def run(models, device, dry, seed=0):
    env = {**os.environ, "PYTHONPATH": f"{ROOT}:{ROOT}/scripts/_stubs", "PYTHONDONTWRITEBYTECODE": "1"}
    (ROOT / "scripts/_stubs").mkdir(exist_ok=True)
    stub = ROOT / "scripts/_stubs/eval_syntheval.py"  # scripts/pipeline.py가 import하지만 저장소에 없는 모듈
    if not stub.exists():
        stub.write_text("def train_syntheval(*a, **k):\n    raise NotImplementedError('eval_syntheval is not in the repository')\n")
    log = SYN_ROOT.parent / "gen_runs.jsonl"
    for model in models:
        spec = DEFAULTS[model]
        for part in PARTS:
            cfg_path = gen_dir(model, part, seed) / "config.toml"
            cfg = tomli.loads(cfg_path.read_text())
            cfg["device"] = device
            cfg_path.write_text(tomli_w.dumps(cfg))
            cmd = [sys.executable, spec["pipeline"], "--config", str(cfg_path.relative_to(ROOT)), "--train", "--sample"]
            if model == "tabpfgen":
                cmd = [sys.executable, spec["pipeline"], "--config", str(cfg_path.relative_to(ROOT)), "--train", "--sample"]
            print(" ".join(cmd), flush=True)
            if dry:
                continue
            t0 = time.time()
            with open(gen_dir(model, part, seed) / "run.log", "w") as fh:
                try:
                    p = subprocess.run(cmd, cwd=ROOT, env=env, stdout=fh, stderr=subprocess.STDOUT, timeout=TIME_CAP_H * 3600 * 1.5)
                    rc = p.returncode
                except subprocess.TimeoutExpired:
                    rc = "timeout"
            rec = {"model": model, "part": part, "seed": seed, "rc": rc, "minutes": round((time.time() - t0) / 60, 1), "device": device}
            with log.open("a") as fh:
                fh.write(json.dumps(rec) + "\n")
            print(rec, flush=True)


def match_prior(Xn, Xc, y, target_rate, rng):
    """생성 풀에서 실제 클래스 비율에 맞게 서브샘플(최대 크기). 생성기 자체는 건드리지 않는다."""
    pos, neg = np.where(y == 1)[0], np.where(y == 0)[0]
    if len(pos) == 0 or len(neg) == 0 or target_rate <= 0:
        return Xn, Xc, y
    n_by_neg = int(len(neg) / (1 - target_rate))
    n_by_pos = int(len(pos) / target_rate)
    n = min(n_by_neg, n_by_pos)
    n_pos = max(1, int(round(n * target_rate)))
    idx = np.concatenate([rng.choice(pos, n_pos, replace=False), rng.choice(neg, n - n_pos, replace=False)])
    rng.shuffle(idx)
    return Xn[idx], Xc[idx], y[idx]


def assemble(models, seed):
    info = json.loads((REAL / "info.json").read_text())
    for model in models:
        spec = DEFAULTS[model]
        out = SYN_ROOT / model / f"seed{seed}"
        out.mkdir(parents=True, exist_ok=True)
        sinfo = dict(info)
        rng = np.random.default_rng(seed)
        for part in PARTS:
            src = gen_dir(model, part, seed)
            if spec.get("postprocess") == "match_prior":
                Xn = np.load(src / "X_num_train.npy", allow_pickle=True)
                Xc = np.load(src / "X_cat_train.npy", allow_pickle=True)
                yy = np.load(src / "y_train.npy", allow_pickle=True).astype(int)
                rate = float(np.load(REAL / f"y_{part}.npy").mean())
                Xn, Xc, yy = match_prior(Xn, Xc, yy, rate, rng)
                np.save(out / f"X_num_{part}.npy", Xn)
                np.save(out / f"X_cat_{part}.npy", Xc, allow_pickle=True)
                np.save(out / f"y_{part}.npy", yy)
            else:
                for f in ["X_num", "X_cat", "y"]:
                    shutil.copyfile(src / f"{f}_train.npy", out / f"{f}_{part}.npy")
            y = np.load(out / f"y_{part}.npy", allow_pickle=True)
            sinfo[f"{part}_size"] = int(len(y))
            print(f"{model} {part}: {len(y):,} rows ({y.astype(int).mean():.2%})")
        sinfo.update(name=f"synth-{model}-seed{seed}", generator=model, seed=seed, defaults=DEFAULTS[model])
        (out / "info.json").write_text(json.dumps(sinfo, ensure_ascii=False, indent=2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["prepare", "run", "assemble"])
    ap.add_argument("--models", default=",".join(DEFAULTS))
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dry", action="store_true")
    ap.add_argument("--real", default=None, help="데이터셋 디렉터리(기본 data/finsyn-v2)")
    ap.add_argument("--exp", default=None, help="실험 출력 디렉터리(기본 exp/finsyn-v2)")
    args = ap.parse_args()
    global REAL, GEN_ROOT, SYN_ROOT
    if args.real:
        REAL = Path(args.real) if Path(args.real).is_absolute() else ROOT / args.real
    if args.exp:
        E = Path(args.exp) if Path(args.exp).is_absolute() else ROOT / args.exp
        GEN_ROOT, SYN_ROOT = E / "gen", E / "synth"
    models = args.models.split(",")
    {"prepare": lambda: prepare(models, args.seed), "run": lambda: run(models, args.device, args.dry, args.seed),
     "assemble": lambda: assemble(models, args.seed)}[args.stage]()


if __name__ == "__main__":
    main()
