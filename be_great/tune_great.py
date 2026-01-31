from multiprocessing.sharedctypes import RawValue
import tempfile
import subprocess
import lib
import os
import numpy as np
import optuna
import argparse
from pathlib import Path
from train_sample_great import train_great, sample_great
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
    # GReaT hyperparameters to tune
    llm = trial.suggest_categorical('llm', [
        'distilgpt2',
        'gpt2',
        # 'tabularisai/Qwen3-0.3B-distil',  # Larger model, uncomment if needed
    ])
    
    # Epochs: 클래스 불균형 학습을 위해 더 많은 epoch 필요
    # 기존 [1, 5, 10] → [10, 20, 30]으로 확대
    # 기존 [10, 20, 30] → [10, 15, 20]으로 축소
    epochs = trial.suggest_categorical('epochs', [10, 15, 20])
    batch_size = trial.suggest_categorical('batch_size', [4, 8, 16])
    float_precision = trial.suggest_categorical('float_precision', [None, 2, 3])
    
    # Training hyperparameters (HuggingFace TrainingArguments)
    # learning_rate: 클래스 불균형 학습에 중요
    learning_rate = trial.suggest_loguniform('learning_rate', 1e-5, 5e-4)
    # weight_decay: 정규화를 통한 과적합 방지
    weight_decay = trial.suggest_uniform('weight_decay', 0.0, 0.1)
    # warmup_steps: 학습 안정화 (전체 step의 0-10%)
    warmup_ratio = trial.suggest_uniform('warmup_ratio', 0.0, 0.1)
    # lr_scheduler_type: 학습률 스케줄링 전략
    lr_scheduler_type = trial.suggest_categorical('lr_scheduler_type', ['linear', 'cosine', 'constant'])
    
    # Sampling parameters
    # guided_sampling: 클래스 불균형 해결에 도움이 될 수 있음 (활성화)
    # guided_sampling = True 일 때 속도가 매우 느림, 추후 best 모델에서만 True로 변경
    # guided_sampling = trial.suggest_categorical('guided_sampling', [False, True])
    temperature = trial.suggest_loguniform('temperature', 0.2, 1.0)
    max_length = trial.suggest_categorical('max_length', [100, 150])
    
    num_samples = int(train_size * (2 ** trial.suggest_int('frac_samples', -1, 1)))
    
    train_params = {
        "llm": llm,
        "epochs": epochs,
        "batch_size": batch_size,
        "float_precision": float_precision,
        # 체크포인트 저장 비활성화로 I/O 오류 방지
        "save_strategy": "no",  # 체크포인트 저장 비활성화
        "save_steps": 1000000,  # save_strategy가 'no'이면 무시되지만 명시적으로 큰 값 설정
        "save_total_limit": 0,  # 체크포인트 저장 개수 제한
        "load_best_model_at_end": False,  # 최종 모델 로드 비활성화
        "logging_steps": 10,
        # HuggingFace TrainingArguments 파라미터
        "learning_rate": learning_rate,
        "weight_decay": weight_decay,
        "warmup_ratio": warmup_ratio,
        "lr_scheduler_type": lr_scheduler_type,
        # Sampling parameters
        "guided_sampling": False,
        "temperature": temperature,
        "max_length": max_length,
        "random_feature_order": False,
        "random_conditional_col": True,
    }
    
    trial.set_user_attr("train_params", train_params)
    trial.set_user_attr("num_samples", num_samples)

    score = 0.0
    valid_samples = 0
    
    try:
        with tempfile.TemporaryDirectory() as dir_:
            dir_ = Path(dir_)
            
            # GReaT 모델 학습
            try:
                great_model = train_great(
                    parent_dir=dir_,
                    real_data_path=real_data_path,
                    train_params=train_params,
                    change_val=True,
                    device=device
                )
            except Exception as e:
                print(f"Trial {trial.number} failed during GReaT training: {str(e)}")
                raise optuna.exceptions.TrialPruned()

            for sample_seed in range(5):
                try:
                    # 합성 데이터 생성
                    sample_great(
                        great_model,
                        parent_dir=dir_,
                        real_data_path=real_data_path,
                        num_samples=num_samples,
                        train_params=train_params,
                        change_val=True,
                        seed=sample_seed,
                        device=device
                    )
                    
                    # 생성된 데이터 검증 (타겟 값의 다양성 확인)
                    y_data_path = os.path.join(dir_, 'y_train.npy')
                    if os.path.exists(y_data_path):
                        y_synthetic = np.load(y_data_path)
                        unique_classes = np.unique(y_synthetic)
                        
                        # 타겟이 단일 클래스만 있는 경우 경고
                        if len(unique_classes) < 2:
                            print(f"WARNING: Trial {trial.number}, seed {sample_seed} - "
                                  f"Generated data has only {len(unique_classes)} unique class(es): {unique_classes}")
                            print(f"Skipping this sample and continuing with next seed...")
                            continue

                    T_dict = {
                        "seed": 0,
                        "normalization": None,
                        "num_nan_policy": None,
                        "cat_nan_policy": None,
                        "cat_min_frequency": None,
                        "cat_encoding": None,
                        "y_policy": "default"
                    }
                    
                    # CatBoost 학습 및 평가
                    metrics = train_catboost(
                        parent_dir=dir_,
                        real_data_path=real_data_path, 
                        eval_type=eval_type,
                        T_dict=T_dict,
                        change_val=True,
                        seed=0
                    )

                    score += metrics.get_val_score()
                    valid_samples += 1
                    
                except Exception as e:
                    error_msg = str(e)
                    print(f"Trial {trial.number}, seed {sample_seed} failed: {error_msg}")
                    
                    # CatBoost의 "All train targets are equal" 오류 처리
                    if "All train targets are equal" in error_msg:
                        print(f"Skipping sample {sample_seed} due to single-class target issue")
                        continue
                    # 기타 CatBoost 오류
                    elif "CatBoostError" in error_msg:
                        print(f"CatBoost error in sample {sample_seed}, skipping...")
                        continue
                    # 예상치 못한 오류는 재발생
                    else:
                        raise
        
        # 유효한 샘플이 하나도 없으면 trial prune
        if valid_samples == 0:
            print(f"Trial {trial.number} pruned: No valid samples generated")
            raise optuna.exceptions.TrialPruned()
        
        # 평균 점수 반환 (유효한 샘플 수로 나눔)
        avg_score = score / valid_samples
        print(f"Trial {trial.number} completed: {valid_samples}/5 valid samples, avg score: {avg_score:.4f}")
        return avg_score
        
    except optuna.exceptions.TrialPruned:
        # Trial이 prune되면 Optuna가 처리
        raise
    except Exception as e:
        # 예상치 못한 오류 발생 시 trial prune
        print(f"Trial {trial.number} failed with unexpected error: {str(e)}")
        import traceback
        traceback.print_exc()
        raise optuna.exceptions.TrialPruned()


study = optuna.create_study(
    direction='maximize',
    sampler=optuna.samplers.TPESampler(seed=0),
)

# study.optimize(objective, n_trials=20, show_progress_bar=True)
# 2025.12.14 : 4/10 trials 기준 best trial = 0
study.optimize(objective, n_trials=10, show_progress_bar=True)

os.makedirs(f"exp/{Path(real_data_path).name}/great/", exist_ok=True)
config = {
    "parent_dir": f"exp/{Path(real_data_path).name}/great/",
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

train_great(
    parent_dir=f"exp/{Path(real_data_path).name}/great/",
    real_data_path=real_data_path,
    train_params=study.best_trial.user_attrs["train_params"],
    change_val=False,
    device=device
)

lib.dump_config(config, config["parent_dir"]+"config.toml")

subprocess.run(['python3.9', "scripts/eval_seeds.py", '--config', f'{config["parent_dir"]+"config.toml"}',
                '10', "great", eval_type, "catboost", "5"], check=True)

