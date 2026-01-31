from multiprocessing.sharedctypes import RawValue
import tempfile
import subprocess
import lib
import os
import optuna
import argparse
from pathlib import Path
from train_sample_ctgan import train_ctgan, sample_ctgan
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
    
    generator_lr = trial.suggest_loguniform('generator_lr', 0.00001, 0.003)
    discriminator_lr = trial.suggest_loguniform('discriminator_lr', 0.00001, 0.003)
    
    def suggest_dim(name, min_val, max_val):
        t = trial.suggest_int(name, min_val, max_val)
        return 2 ** t
    
    # Generator and Discriminator dimensions
    min_n_layers, max_n_layers, d_min, d_max = 1, 3, 6, 9
    n_layers = trial.suggest_int('n_layers', min_n_layers, max_n_layers)
    
    gen_dims = []
    disc_dims = []
    for i in range(n_layers):
        gen_dims.append(suggest_dim(f'gen_dim_{i}', d_min, d_max))
        disc_dims.append(suggest_dim(f'disc_dim_{i}', d_min, d_max))
    
    # If no layers, use default
    if not gen_dims:
        gen_dims = [256, 256]
    if not disc_dims:
        disc_dims = [256, 256]
    
    # 2025.11.29
    # steps = trial.suggest_categorical('steps', [5000, 20000, 30000])
    steps = trial.suggest_categorical('steps', [1000, 3000, 5000])
    
    # PAC을 먼저 선택 (batch_size가 pac의 배수가 되어야 함)
    discriminator_steps = trial.suggest_int('discriminator_steps', 1, 5)
    pac = trial.suggest_categorical('pac', [1, 5, 10])
    
    # batch_size를 선택하고 pac의 배수로 조정
    batch_size_candidates = [256, 500, 1000, 2000, 4096]
    batch_size = trial.suggest_categorical('batch_size', batch_size_candidates)
    
    # batch_size를 pac의 배수로 조정
    if batch_size % pac != 0:
        batch_size = (batch_size // pac) * pac
        # 조정 후에도 최소값 보장
        if batch_size < pac:
            batch_size = pac
    
    num_samples = int(train_size * (2 ** trial.suggest_int('frac_samples', -1, 1)))
    
    embedding_dim = 2 ** trial.suggest_int('embedding_dim', 6, 10)
    
    generator_decay = trial.suggest_loguniform('generator_decay', 1e-7, 1e-5)
    discriminator_decay = trial.suggest_loguniform('discriminator_decay', 1e-7, 1e-5)

    train_params = {
        "generator_lr": generator_lr,
        "discriminator_lr": discriminator_lr,
        "generator_decay": generator_decay,
        "discriminator_decay": discriminator_decay,
        "epochs": steps,
        "embedding_dim": embedding_dim,
        "batch_size": batch_size,
        "discriminator_steps": discriminator_steps,
        "generator_dim": tuple(gen_dims),
        "discriminator_dim": tuple(disc_dims),
        "pac": pac,
        "log_frequency": True
    }

    trial.set_user_attr("train_params", train_params)
    trial.set_user_attr("num_samples", num_samples)

    score = 0.0
    with tempfile.TemporaryDirectory() as dir_:
        dir_ = Path(dir_)
        ctgan = train_ctgan(
            parent_dir=dir_,
            real_data_path=real_data_path,
            train_params=train_params,
            change_val=True,
            device=device
        )

        for sample_seed in range(5):
            sample_ctgan(
                ctgan,
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

            score += metrics.get_val_score()
    return score / 5


study = optuna.create_study(
    direction='maximize',
    sampler=optuna.samplers.TPESampler(seed=0),
)

study.optimize(objective, n_trials=20, show_progress_bar=True)

os.makedirs(f"exp/{Path(real_data_path).name}/ctgan/", exist_ok=True)
config = {
    "parent_dir": f"exp/{Path(real_data_path).name}/ctgan/",
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

train_ctgan(
    parent_dir=f"exp/{Path(real_data_path).name}/ctgan/",
    real_data_path=real_data_path,
    train_params=study.best_trial.user_attrs["train_params"],
    change_val=False,
    device=device
)

lib.dump_config(config, config["parent_dir"]+"config.toml")

subprocess.run(['python3.9', "scripts/eval_seeds.py", '--config', f'{config["parent_dir"]+"config.toml"}',
                '10', "ctgan", eval_type, "catboost", "5"], check=True)

