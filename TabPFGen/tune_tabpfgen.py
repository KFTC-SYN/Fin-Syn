from multiprocessing.sharedctypes import RawValue
import tempfile
import subprocess
import lib
import os
import optuna
import argparse
from pathlib import Path
from train_sample_tabpfgen import train_tabpfgen, sample_tabpfgen
from scripts.eval_catboost import train_catboost

parser = argparse.ArgumentParser()
parser.add_argument('data_path', type=str)
parser.add_argument('train_size', type=int)
parser.add_argument('eval_type', type=str)
parser.add_argument('device', type=str)

args = parser.parse_args()
real_data_path = args.data_path
eval_type = args.eval_type
train_size = args.train_size
device = args.device
assert eval_type in ('merged', 'synthetic')

def objective(trial):

    # n_sgld_steps = trial.suggest_categorical('n_sgld_steps', [500, 1000, 1500, 2000])
    # n_sgld_steps = trial.suggest_categorical('n_sgld_steps', [50])
    # sgld_step_size = trial.suggest_loguniform('sgld_step_size', 0.001, 0.1)
    # sgld_noise_scale = trial.suggest_loguniform('sgld_noise_scale', 0.001, 0.1)

    # TabPFGen hyperparameters to tune
    # 이전 best: n_sgld_steps=2000, sgld_step_size=0.036, sgld_noise_scale=0.055
    # 평가 결과: 클래스 불균형 심각(f1_1≈0), 통계적 유사성 낮음(KL Div=19.47, JSD=0.48)
    # 개선 방향: 더 많은 steps로 품질 향상, step_size/noise_scale 범위 조정
    
    # n_sgld_steps: 이전 best는 2000, 통계적 유사성 개선을 위해 더 많은 steps 탐색
    n_sgld_steps = trial.suggest_categorical('n_sgld_steps', [2000, 2500, 3000])
    
    # sgld_step_size: 이전 best는 0.036 (기본값 0.01의 3.6배)
    # 통계적 유사성 개선을 위해 기본값 주변과 best 값 주변 모두 탐색
    # 범위: 기본값(0.01) ~ best(0.036) ~ 상한(0.1)
    sgld_step_size = trial.suggest_loguniform('sgld_step_size', 0.005, 0.1)
    
    # sgld_noise_scale: 이전 best는 0.055 (기본값 0.01의 5.5배)
    # 너무 큰 noise는 분포 차이를 키울 수 있으므로 상한 조정
    # 범위: 기본값(0.01) ~ best(0.055) ~ 상한(0.08, 이전 best의 약 1.5배)
    sgld_noise_scale = trial.suggest_loguniform('sgld_noise_scale', 0.005, 0.08)
    
    num_samples = int(train_size * (2 ** trial.suggest_int('frac_samples', -1, 1)))
    
    train_params = {
        "n_sgld_steps": n_sgld_steps,
        "sgld_step_size": sgld_step_size,
        "sgld_noise_scale": sgld_noise_scale,
        "balance_classes": True
    }
    
    trial.set_user_attr("train_params", train_params)
    trial.set_user_attr("num_samples", num_samples)

    score = 0.0
    with tempfile.TemporaryDirectory() as dir_:
        dir_ = Path(dir_)
        tabpfgen = train_tabpfgen(
            parent_dir=dir_,
            real_data_path=real_data_path,
            train_params=train_params,
            change_val=True,
            device=device
        )

        for sample_seed in range(5):
            sample_tabpfgen(
                tabpfgen,
                parent_dir=dir_,
                real_data_path=real_data_path,
                num_samples=num_samples,
                train_params=train_params,
                change_val=True,
                seed=sample_seed,
                device=device
            )

            T_dict = {
                "seed": 0,
                "normalization": None,
                "num_nan_policy": None,
                "cat_nan_policy": None,
                "cat_min_frequency": None,
                "cat_encoding": None,
                "y_policy": "default"
            }
            metrics = train_catboost(
                parent_dir=dir_,
                real_data_path=real_data_path, 
                eval_type=eval_type,
                T_dict=T_dict,
                change_val=True,
                seed=0
            )

            score += metrics.get_val_score()
    
    return score / 5


# Study 생성
study = optuna.create_study(
    direction='maximize',
    sampler=optuna.samplers.TPESampler(seed=0),
)

# study.optimize(objective, n_trials=20, show_progress_bar=True)
study.optimize(objective, n_trials=10, show_progress_bar=True)

os.makedirs(f"exp/{Path(real_data_path).name}/tabpfgen/", exist_ok=True)
config = {
    "parent_dir": f"exp/{Path(real_data_path).name}/tabpfgen/",
    "real_data_path": real_data_path,
    "seed": 0,
    "device": args.device,
    "train_params": study.best_trial.user_attrs["train_params"],
    "sample": {"seed": 0, "num_samples": study.best_trial.user_attrs["num_samples"]},
    "eval": {
        "type": {"eval_model": "catboost", "eval_type": eval_type},
        "T": {
            "seed": 0,
            "normalization": None,
            "num_nan_policy": None,
            "cat_nan_policy": None,
            "cat_min_frequency": None,
            "cat_encoding": None,
            "y_policy": "default"
        },
    }
}

train_tabpfgen(
    parent_dir=f"exp/{Path(real_data_path).name}/tabpfgen/",
    real_data_path=real_data_path,
    train_params=study.best_trial.user_attrs["train_params"],
    change_val=False,
    device=device
)

lib.dump_config(config, config["parent_dir"]+"config.toml")

subprocess.run(['python3.11', "scripts/eval_seeds.py", '--config', f'{config["parent_dir"]+"config.toml"}',
                '10', "tabpfgen", eval_type, "catboost", "5"], check=True)

