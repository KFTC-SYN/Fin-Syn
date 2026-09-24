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
TEMPLATE = ROOT / "exp/finsyn-v2/gen"  # 시드 0 train 기간 설정을 키 구조 참조용으로 쓴다(값은 아래 DEFAULTS로 덮어씀)
GEN_ROOT = ROOT / "exp/finsyn-v2/gen"
SYN_ROOT = ROOT / "exp/finsyn-v2/synth"
TIME_CAP_H = 3.0  # 모든 생성기 공통 학습시간 상한(시간, part당)
# TabSyn/TabDiff는 상한을 에폭 수로 정했다(단독 실행 실측 기준). 공유 GPU에서는 같은 에폭이 더 오래 걸리므로
# 벽시계 제한은 안전장치로만 둔다. 9/18 처음 설정(4.5시간)은 경합 탓에 TabDiff val 시드 0을 에폭 3,786/4,400에서 끊었다.
EXT_TIMEOUT_H = 12.0
# GReaT(distilgpt2 파인튜닝)는 같은 epoch 예산이라도 벽시계가 훨씬 길고 GPU 경합에 민감하다.
# 시드 0-2도 6.6-25.6시간 걸렸으므로 공통 4.5시간 상한을 적용하면 무조건 죽는다(9/20 확인).
GEN_TIMEOUT_H = {"great": 48.0}

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
    # GReaT: be_great 기본 LLM(distilgpt2); epochs는 학습시간 예산에 맞춰 9/17 실측 후 14로 확정(기본 100은 예산 초과).
    # 주의: 시드 0-2 릴리스는 모두 14 epoch로 만들어졌다(great_run.json). 40으로 두면 시드 3-4만 2.9배 더 학습해
    # "같은 설정, 다른 시드"라는 전제가 깨진다(9/20 발견).
    "great": {"template": "great", "pipeline": "be_great/pipeline_great.py", "train_params": {
        "llm": "distilgpt2", "epochs": 14, "batch_size": 32, "save_strategy": "no", "save_steps": 1000000,
        "save_total_limit": 0, "logging_steps": 50}},
    # TabPFGen: 코드 기본값 (n_sgld_steps=1000, step 0.01, noise 0.01); 구현이 학습 데이터를 10k로 서브샘플
    "tabpfgen": {"template": "tabpfgen", "pipeline": "TabPFGen/pipeline_tabpfgen.py", "train_params": {
        "n_sgld_steps": 1000, "sgld_step_size": 0.01, "sgld_noise_scale": 0.01}},
    # TabPFGen 변형(레이블 비율 보존): 생성기는 기본값 그대로(균형화 유지)이고, 출력만 실제 클래스 비율로 서브샘플한다.
    # balance_classes=False 경로는 SGLD 뒤 TabPFN argmax 재레이블링 때문에 소수 클래스가 전부 사라져(0%) 비교 대상이 되지 못한다.
    "tabpfgen-prior": {"template": "tabpfgen", "pipeline": "TabPFGen/pipeline_tabpfgen.py", "oversample": 2.0, "postprocess": "match_prior",
                       "train_params": {"n_sgld_steps": 1000, "sgld_step_size": 0.01, "sgld_noise_scale": 0.01}},
    # TabSyn (ICLR'24), TabDiff (ICLR'25): 공식 저장소(저장소 루트에 클론, 수정 없음)를 scripts/tabgen/ 래퍼로 실행.
    # 설정은 공식 기본값. 학습 길이만 공통 상한(기간당 3시간)에 맞춘다: 9/18 실측(64k행 train 기간, GPU 공유 상태)
    #   TabSyn  VAE 4.0 s/epoch, 확산 2.1 s/epoch -> train 기간 VAE 1,300 + 확산 2,500 epoch(기본 4,000 + 최대 10,001, 조기종료 500)
    #   TabDiff 9.4 s/epoch -> train 기간 1,100 epoch(기본 8,000). val/test 기간(15k행)은 약 1/4 시간이라
    #   TabSyn은 기본값 그대로, TabDiff는 4,400 epoch.
    "tabsyn": {"runner": "tabsyn", "repo": "tabsyn", "commit": "cb5ac0f",
               "budget": {"train": {"vae_epochs": 1300, "diff_epochs": 2500},
                          "val": {"vae_epochs": 4000, "diff_epochs": 10001},
                          "test": {"vae_epochs": 4000, "diff_epochs": 10001}}},
    "tabdiff": {"runner": "tabdiff", "repo": "TabDiff", "commit": "5ecdb33",
                "budget": {"train": {"steps": 1100}, "val": {"steps": 4400}, "test": {"steps": 4400}}},
    # FinDiff (ICAIF'23, 금융 표 데이터 전용 확산 모델): 클래스 기본값 그대로(MLP, 100 epoch, batch 128, lr 1e-4).
    # 9/19 실측 20.5 s/epoch(train 기간, GPU 공유) -> 100 epoch = 34분으로 상한 안이라 모든 기간 기본값.
    # 레이블 조건부 생성, 조건 레이블은 실제 레이블을 섞은 것(TabDDPM과 같은 방식).
    "findiff": {"runner": "findiff", "repo": "FinDiff", "commit": "45e9563",
                "budget": {"train": {"epochs": 100}, "val": {"epochs": 100}, "test": {"epochs": 100}}},
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
        if "runner" in spec:  # 외부 래퍼가 part 폴더를 직접 읽는다(설정 파일 불필요)
            print(f"prepared {model} (seed {seed}): part folders only")
            continue
        if "columns" in spec:
            p = ROOT / spec["columns"]
            cols = json.loads(p.read_text())
            for part in PARTS:
                cols[part_dir(part).name] = {"categorical_columns": cat_idx + ["y"], "mixed_columns": {},
                                             "integer_columns": int_idx, "problem_type": {"Classification": "y"}}
            p.write_text(json.dumps(cols, indent=4))
        for part in PARTS:
            cfg = tomli.loads((TEMPLATE / model / "train" / "config.toml").read_text())
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


def run(models, device, dry, seed=0, parts=None, after_pid=None, start_stage=0):
    env = {**os.environ, "PYTHONPATH": f"{ROOT}:{ROOT}/scripts/_stubs", "PYTHONDONTWRITEBYTECODE": "1"}
    (ROOT / "scripts/_stubs").mkdir(exist_ok=True)
    stub = ROOT / "scripts/_stubs/eval_syntheval.py"  # scripts/pipeline.py가 import하지만 저장소에 없는 모듈
    if not stub.exists():
        stub.write_text("def train_syntheval(*a, **k):\n    raise NotImplementedError('eval_syntheval is not in the repository')\n")
    log = SYN_ROOT.parent / "gen_runs.jsonl"
    for model in models:
        spec = DEFAULTS[model]
        if "runner" in spec:
            for part in parts or PARTS:
                run_external(model, part, seed, device, dry, env, log, after_pid, start_stage)
            continue
        for part in parts or PARTS:
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
                    p = subprocess.run(cmd, cwd=ROOT, env=env, stdout=fh, stderr=subprocess.STDOUT, timeout=GEN_TIMEOUT_H.get(model, TIME_CAP_H * 1.5) * 3600)
                    rc = p.returncode
                except subprocess.TimeoutExpired:
                    rc = "timeout"
            rec = {"model": model, "part": part, "seed": seed, "rc": rc, "minutes": round((time.time() - t0) / 60, 1), "device": device}
            with log.open("a") as fh:
                fh.write(json.dumps(rec) + "\n")
            print(rec, flush=True)


def run_external(model, part, seed, device, dry, env, log, after_pid=None, start_stage=0):
    """TabSyn/TabDiff: part 폴더를 저장소 형식으로 내보내고, 단계별로 학습·샘플링한 뒤 gen_dir 형식으로 가져온다.

    after_pid/start_stage: 이미 돌고 있는 단계(부모가 종료된 학습 프로세스)가 끝나기를 기다렸다가 start_stage부터 잇는다.
    """
    sys.path.insert(0, str(ROOT / "scripts/tabgen"))
    from adapter import export, import_samples
    spec = DEFAULTS[model]
    repo = ROOT / spec["repo"]
    name = f"{REAL.name}-{part}-s{seed}"  # 기간·시드별 이름(체크포인트 폴더 충돌 방지)
    out = gen_dir(model, part, seed)
    out.mkdir(parents=True, exist_ok=True)
    b = spec["budget"][part]
    w = str(ROOT / f"scripts/tabgen/run_{model}.py")
    if model == "findiff":  # 래퍼가 학습·샘플링·gen_dir 저장을 한 번에 한다
        cmds = [[sys.executable, w, str(part_dir(part)), str(out), str(seed), "--epochs", str(b["epochs"])]]
    elif model == "tabsyn":
        csv = repo / f"synthetic/{name}/tabsyn_seed{seed}.csv"
        cmds = [[sys.executable, w, str(repo), name, str(seed), "vae", "--vae_epochs", str(b["vae_epochs"])],
                [sys.executable, w, str(repo), name, str(seed), "diff", "--diff_epochs", str(b["diff_epochs"])],
                [sys.executable, w, str(repo), name, str(seed), "sample", "--save", str(csv)]]
    else:
        cmds = [[sys.executable, w, str(repo), name, str(seed), "train", "--steps", str(b["steps"])],
                [sys.executable, w, str(repo), name, str(seed), "test", "--steps", str(b["steps"])]]
    print(f"{model} {part} seed{seed}: " + " && ".join(" ".join(c[1:]) for c in cmds), flush=True)
    if dry:
        return
    t0, rc = time.time(), 0
    if after_pid:
        while Path(f"/proc/{after_pid}").exists():
            time.sleep(60)
        done_marker = {"tabdiff": "Ending Trainnig Loop", "tabsyn": "Time:", "findiff": "Time:"}[model]  # 각 저장소가 단계 끝에 찍는 문구
        if done_marker not in (out / "run.log").read_text(errors="ignore"):
            rc = f"stage {start_stage - 1} did not finish (pid {after_pid})"
    elif model != "findiff":
        export(part_dir(part), repo, name)
    genv = {**env, "CUDA_VISIBLE_DEVICES": device.split(":")[-1] if device.startswith("cuda") else ""}
    with open(out / "run.log", "a" if start_stage else "w") as fh:
        for c in (cmds[start_stage:] if rc == 0 else []):
            try:
                rc = subprocess.run(c, cwd=ROOT, env=genv, stdout=fh, stderr=subprocess.STDOUT,
                                    timeout=EXT_TIMEOUT_H * 3600).returncode
            except subprocess.TimeoutExpired:
                rc = "timeout"
            if rc != 0:
                break
    rec = {"model": model, "part": part, "seed": seed, "rc": rc, "minutes": round((time.time() - t0) / 60, 1),
           "device": device, "budget": b}
    if rc == 0 and model == "findiff":
        run = json.loads((out / "findiff_run.json").read_text())
        rec.update(rows=run["rows"], positive_rate=round(run["positive_rate"], 5),
                   train_min=run["train_min"], sample_min=run["sample_min"])
    elif rc == 0:
        if model == "tabdiff":
            cands = sorted((repo / f"tabdiff/result/{name}/seed{seed}").glob("*/samples.csv"), key=lambda q: q.stat().st_mtime)
            csv = cands[-1]
        n, rate = import_samples(csv, part_dir(part), out)
        rec.update(rows=n, positive_rate=round(rate, 5), samples=str(csv.relative_to(ROOT)))
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
    ap.add_argument("--parts", default=None, help="쉼표로 구분한 기간(기본: train,val,test)")
    ap.add_argument("--after-pid", type=int, default=None, help="이 PID(이미 도는 단계)가 끝난 뒤 잇는다")
    ap.add_argument("--start-stage", type=int, default=0, help="외부 생성기 단계 번호(0부터)부터 실행")
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
    {"prepare": lambda: prepare(models, args.seed), "run": lambda: run(models, args.device, args.dry, args.seed, args.parts.split(",") if args.parts else None,
                       args.after_pid, args.start_stage),
     "assemble": lambda: assemble(models, args.seed)}[args.stage]()


if __name__ == "__main__":
    main()
