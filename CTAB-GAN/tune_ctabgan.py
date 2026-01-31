from multiprocessing.sharedctypes import RawValue
import tempfile
import subprocess
import lib
import os
import optuna
import argparse
from pathlib import Path
from train_sample_ctabgan import train_ctabgan, sample_ctabgan
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
    # Learning rate: GAN 안정성을 위한 보수적 범위
    # 2025.11.27 loss 800-900 관련, 기존 0.00001-0.003 범위 축소
    # 2025.12.18 NaN 발산 방지: GAN 특성상 높은 LR은 위험
    # 2025.12.18 로그 분석 결과: 모든 trial에서 97%+ NaN 발생 → LR 범위 대폭 축소
    lr = trial.suggest_loguniform('lr', 0.00001, 0.0003)

    def suggest_dim(name):
        t = trial.suggest_int(name, d_min, d_max)
        return 2 ** t
        
    # construct model
    # 2025.11.24 CUDA out of memory 해결: class_dim 범위 축소
    # min_n_layers, max_n_layers, d_min, d_max = 1, 4, 6, 8
    # 2025.12.18 안정성 개선: 최소 d_min을 5로 상향 (너무 작은 네트워크는 불안정)
    min_n_layers, max_n_layers, d_min, d_max = 2, 4, 5, 6  # class_dim을 32-64로, 최소 레이어 2개
    n_layers = trial.suggest_int('n_layers', min_n_layers, max_n_layers)
    d_first = [suggest_dim('d_first')] if n_layers else []
    d_middle = (
            [suggest_dim('d_middle')] * (n_layers - 2)
            if n_layers > 2
            else []
        )
    d_last = [suggest_dim('d_last')] if n_layers > 1 else []
    d_layers = d_first + d_middle + d_last

    # Epochs: Early Stopping과 함께 사용 시 더 큰 값 허용
    # 2025.11.27
    # steps = trial.suggest_categorical('steps', [1000, 5000, 10000])
    # steps = trial.suggest_categorical('steps', [3000, 5000, 10000])
    steps = trial.suggest_categorical('steps', [1000, 3000, 5000, 10000])

    # Batch size 및 모델 파라미터
    # 2025.11.24 CUDA out of memory 해결: data_dim이 51757로 매우 크므로 배치 크기와 모델 파라미터 축소
    # 2025.11.27 loss 800-900 관련, 기존 범위로 롤백
    # 2025.12.18 안정성 개선: 최소 batch_size를 256으로 상향 (통계량 안정성 확보, NaN 방지)
    batch_size = 2 ** trial.suggest_int('batch_size', 8, 10)  # 256~1024
    random_dim = 2 ** trial.suggest_int('random_dim', 4, 5)  # 16, 32
    num_channels = 2 ** trial.suggest_int('num_channels', 3, 4)  # 8, 16

    # 샘플링 비율
    num_samples = int(train_size * (2 ** trial.suggest_int('frac_samples', -1, 1)))

    train_params = {
        "lr": lr,
        "epochs": steps,
        "class_dim": d_layers,
        "batch_size": batch_size,
        "random_dim": random_dim,
        "num_channels": num_channels
    }
    trial.set_user_attr("train_params", train_params)
    trial.set_user_attr("num_samples", num_samples)

    score = 0.0
    with tempfile.TemporaryDirectory() as dir_:
        dir_ = Path(dir_)
        ctabgan = train_ctabgan(
            parent_dir=dir_,
            real_data_path=real_data_path,
            train_params=train_params,
            change_val=True,
            device=device
        )

        for sample_seed in range(5):
            sample_ctabgan(
                ctabgan,
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
                seed = 0
            )

            current_score = metrics.get_val_score()
            score += current_score
            
            # Optuna에 중간 결과 보고 (조기 종료 판단용)
            trial.report(score / (sample_seed + 1), sample_seed)
            
            # Pruning 체크: 성능이 낮으면 조기 종료
            if trial.should_prune():
                raise optuna.TrialPruned()
                
    return score / 5


study = optuna.create_study(
    direction='maximize',
    sampler=optuna.samplers.TPESampler(seed=0),
    # 성능이 낮은 trial을 조기 종료 (중앙값 기반)
    pruner=optuna.pruners.MedianPruner(
        n_startup_trials=3,      # 처음 3개 trial은 pruning 안 함
        n_warmup_steps=2,        # 각 trial의 처음 2개 샘플은 pruning 안 함
        interval_steps=1         # 매 샘플마다 pruning 체크
    )
)

# study.optimize(objective, n_trials=35, show_progress_bar=True)
study.optimize(objective, n_trials=10, show_progress_bar=True)

os.makedirs(f"exp/{Path(real_data_path).name}/ctabgan/", exist_ok=True)
config = {
    "parent_dir": f"exp/{Path(real_data_path).name}/ctabgan/",
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

train_ctabgan(
    parent_dir=f"exp/{Path(real_data_path).name}/ctabgan/",
    real_data_path=real_data_path,
    train_params=study.best_trial.user_attrs["train_params"],
    change_val=False,
    device=device
)

lib.dump_config(config, config["parent_dir"]+"config.toml")

subprocess.run(['python3.9', "scripts/eval_seeds.py", '--config', f'{config["parent_dir"]+"config.toml"}',
                '10', "ctabgan", eval_type, "catboost", "5"], check=True)