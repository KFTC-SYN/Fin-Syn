import warnings
warnings.filterwarnings('ignore', message='.*pkg_resources is deprecated.*')

import subprocess
import lib
import os
import optuna
import shutil
import argparse
import sys
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument('ds_name', type=str)
parser.add_argument('train_size', type=int)
parser.add_argument('eval_type', type=str)
parser.add_argument('eval_model', type=str)
parser.add_argument('prefix', type=str)
parser.add_argument('--eval_seeds', action='store_true',  default=False)

args = parser.parse_args()
train_size = args.train_size
ds_name = args.ds_name
eval_type = args.eval_type 
assert eval_type in ('merged', 'synthetic')
prefix = str(args.prefix)

pipeline = f'scripts/pipeline.py'
base_config_path = f'exp/{ds_name}/config.toml'
parent_path = Path(f'exp/{ds_name}/')
exps_path = Path(f'exp/{ds_name}/many-exps/') # temporary dir. maybe will be replaced with tempdiвdr
eval_seeds = f'scripts/eval_seeds.py'

os.makedirs(exps_path, exist_ok=True)

def _suggest_mlp_layers(trial):
    def suggest_dim(name):
        t = trial.suggest_int(name, d_min, d_max)
        return 2 ** t
    #############################################################################
    # 2026.1.18 모델 구조 안정화: 레이어 수와 차원 범위를 제한하여 불안정성 완화
    # - 원본: min_n_layers, max_n_layers, d_min, d_max = 1, 4, 7, 10
    # - 수정1차: min_n_layers, max_n_layers, d_min, d_max = 1, 3, 7, 9
    # - 수정2차: min_n_layers, max_n_layers, d_min, d_max = 1, 2, 7, 8
    # - 수정3차: min_n_layers, max_n_layers, d_min, d_max = 1, 1, 7, 8
    # - 수정4차: min_n_layers, max_n_layers, d_min, d_max = 1, 1, 6, 7 (차원 범위 축소: 64~128)
    #############################################################################
    min_n_layers, max_n_layers, d_min, d_max = 1, 1, 6, 7
    n_layers = 2 * trial.suggest_int('n_layers', min_n_layers, max_n_layers)
    d_first = [suggest_dim('d_first')] if n_layers else []
    d_middle = (
        [suggest_dim('d_middle')] * (n_layers - 2)
        if n_layers > 2
        else []
    )
    d_last = [suggest_dim('d_last')] if n_layers > 1 else []
    d_layers = d_first + d_middle + d_last
    return d_layers

def objective(trial):
    #############################################################################
    # 2026.1.18 학습률 범위 조정: 높은 학습률로 인한 NaN 손실 방지를 위해 상한을 더 낮춤
    # - 원본: lr = trial.suggest_loguniform('lr', 0.00001, 0.003)
    # - 수정1차: lr = trial.suggest_loguniform('lr', 0.00001, 0.001)
    # - 수정2차: lr = trial.suggest_loguniform('lr', 0.00001, 0.0005)
    # - 수정3차: lr = trial.suggest_loguniform('lr', 0.00001, 0.0003)
    # - 수정4차: lr = trial.suggest_loguniform('lr', 0.00001, 0.0001)
    # - 수정4차: lr = trial.suggest_loguniform('lr', 0.00001, 0.003)
    #############################################################################
    lr = trial.suggest_loguniform('lr', 0.00001, 0.003)
    
    # 2026.1.18 Weight decay 추가: 모델 파라미터 정규화로 과적합 및 불안정성 완화
    # weight_decay = trial.suggest_loguniform('weight_decay', 1e-6, 1e-4)   

    # 2026.1.18 배치 사이즈 축소: 큰 배치 사이즈(4096)로 인한 gradient 분산 증가 및 불안정성 완화
    # - 원본: batch_size = trial.suggest_categorical('batch_size', [256, 4096])
    # - 수정1차: batch_size = trial.suggest_categorical('batch_size', [64, 128])
    # - 수정2차: batch_size = trial.suggest_categorical('batch_size', [128, 256])
    batch_size = trial.suggest_categorical('batch_size', [128, 256])
    
    # 2026.1.18 스텝 추가: 짧은 학습으로 불안정성이 완화되지 못함
    # - 원본: steps = trial.suggest_categorical('steps', [5000, 20000, 30000])
    # - 수정1차: steps = trial.suggest_categorical('steps', [10000, 20000, 30000])
    # - 수정2차: steps = trial.suggest_categorical('steps', [30000])
    # - debug: steps = trial.suggest_categorical('steps', [500]) # for debug
    steps = trial.suggest_categorical('steps', [30000, 40000, 50000])
    
    gaussian_loss_type = 'mse'
    # scheduler = trial.suggest_categorical('scheduler', ['cosine', 'linear'])
    num_timesteps = trial.suggest_categorical('num_timesteps', [100, 1000])
    num_samples = int(train_size * (2 ** trial.suggest_int('num_samples', -2, 1)))

    base_config = lib.load_config(base_config_path)
    
    # model_type 확인 (기본값은 'mlp')
    model_type = base_config.get('model_type', 'mlp')
    
    # MLP용 변수 초기화 (ResNet일 때는 사용 안 함)
    d_layers = None
    
    #############################################################################
    # 2026.1.19 ResNetDiffusion 지원: model_type에 따라 다른 파라미터 설정
    # - MLP: d_layers, dropout 사용
    # - ResNet: n_blocks, d_main, d_hidden, dropout_first, dropout_second 사용
    #############################################################################
    if model_type == 'resnet':
        # ResNet 전용 파라미터
        n_blocks = trial.suggest_int('n_blocks', 2, 3)
        
        # d_main: 각 블록의 입력/출력 차원 (2의 거듭제곱 권장)
        # 1037의 절반 정도인 256~512 범위 권장
        d_main_power = trial.suggest_int('d_main_power', 8, 9)  # 512
        d_main = 2 ** d_main_power
        
        # d_hidden: 각 블록 내부의 hidden 차원 (2의 거듭제곱 권장)
        # d_main의 2배 정도 권장
        d_hidden_power = trial.suggest_int('d_hidden_power', 9, 10)  # 256, 512, 1024
        d_hidden = 2 ** d_hidden_power
        
        # Dropout 파라미터
        dropout_first = trial.suggest_uniform('dropout_first', 0.1, 0.3)
        dropout_second = trial.suggest_uniform('dropout_second', 0.0, 0.2)
        
        # rtdl_params 설정 (ResNet 전용)
        base_config['model_params']['rtdl_params'] = {
            'n_blocks': n_blocks,
            'd_main': d_main,
            'd_hidden': d_hidden,
            'dropout_first': dropout_first,
            'dropout_second': dropout_second
        }
        
        # dim_t는 ResNetDiffusion 기본값이 256
        dim_t = trial.suggest_categorical('dim_t', [256])
    else:
        # MLP용 기존 코드
        d_layers = _suggest_mlp_layers(trial)
        base_config['model_params']['rtdl_params']['d_layers'] = d_layers
        
        #############################################################################
        # 2026.1.18 Dropout 추가: 모델 구조 안정화 및 과적합 방지
        # - 원본: 없음
        # - 수정1차: trial.suggest_uniform('dropout', 0.0, 0.3)
        # - 수정2차: trial.suggest_uniform('dropout', 0.1, 0.3)
        #############################################################################
        base_config['model_params']['rtdl_params']['dropout'] = trial.suggest_uniform('dropout', 0.1, 0.3)
        
        #############################################################################
        # 2026.1.18 dim_t 하이퍼파라미터 추가: 시간 임베딩 차원을 튜닝 가능하게 함
        # - 원본: 하드코딩된 기본값 128
        # - 수정1차: 256
        # - 수정2차: 128
        # - 효과: 정보 보존 개선 및 gradient 안정화
        #############################################################################
        dim_t = trial.suggest_categorical('dim_t', [128])

    base_config['train']['main']['lr'] = lr
    base_config['train']['main']['steps'] = steps
    base_config['train']['main']['batch_size'] = batch_size
    # base_config['train']['main']['weight_decay'] = weight_decay
    base_config['model_params']['dim_t'] = dim_t

    base_config['eval']['type']['eval_type'] = eval_type
    base_config['sample']['num_samples'] = num_samples
    base_config['diffusion_params']['gaussian_loss_type'] = gaussian_loss_type
    base_config['diffusion_params']['num_timesteps'] = num_timesteps
    # base_config['diffusion_params']['scheduler'] = scheduler

    base_config['parent_dir'] = str(exps_path / f"{trial.number}")
    base_config['eval']['type']['eval_model'] = args.eval_model
    if args.eval_model == "mlp":
        base_config['eval']['T']['normalization'] = "quantile"
        base_config['eval']['T']['cat_encoding'] = "one-hot"

    trial.set_user_attr("config", base_config)

    lib.dump_config(base_config, exps_path / 'config.toml')
    
    #############################################################################
    # 2026.1.18 NaN 감지 및 로깅: NaN 발생 시 명시적으로 로그를 남기고 trial 실패 처리
    # check=True를 사용하면 returncode != 0일 때 CalledProcessError가 발생하므로, try-except로 처리
    #############################################################################
    try:
        subprocess.run(['python3.9', f'{pipeline}', '--config', f'{exps_path / "config.toml"}', '--train', '--change_val'], check=True)
    except subprocess.CalledProcessError:
        print(f"\n[Trial {trial.number}] Failed during training!")
        if model_type == 'resnet':
            rtdl_params = base_config['model_params']['rtdl_params']
            print(f"Parameters: lr={lr}, batch_size={batch_size}, steps={steps}, "
                  f"n_blocks={rtdl_params['n_blocks']}, d_main={rtdl_params['d_main']}, "
                  f"d_hidden={rtdl_params['d_hidden']}, dropout_first={rtdl_params['dropout_first']}, "
                  f"dropout_second={rtdl_params['dropout_second']}")
        else:
            print(f"Parameters: lr={lr}, batch_size={batch_size}, steps={steps}, "
                  f"dropout={base_config['model_params']['rtdl_params']['dropout']}, d_layers={d_layers}")
        raise optuna.TrialPruned()

    n_datasets = 5
    score = 0.0

    for sample_seed in range(n_datasets):
        base_config['sample']['seed'] = sample_seed
        lib.dump_config(base_config, exps_path / 'config.toml')
        
        #############################################################################
        # 2026.1.18 NaN 감지 및 로깅: NaN 발생 시 명시적으로 로그를 남기고 trial 실패 처리
        # - check=True를 사용하면 returncode != 0일 때 CalledProcessError가 발생하므로, try-except로 처리
        #############################################################################
        try:
            subprocess.run(['python3.9', f'{pipeline}', '--config', f'{exps_path / "config.toml"}', '--sample', '--eval', '--change_val'], check=True)
        except subprocess.CalledProcessError:
            print(f"\n[Trial {trial.number}] Failed during sampling (seed={sample_seed})!")
            if model_type == 'resnet':
                rtdl_params = base_config['model_params']['rtdl_params']
                print(f"Parameters: lr={lr}, batch_size={batch_size}, steps={steps}, "
                      f"n_blocks={rtdl_params['n_blocks']}, d_main={rtdl_params['d_main']}, "
                      f"d_hidden={rtdl_params['d_hidden']}, dropout_first={rtdl_params['dropout_first']}, "
                      f"dropout_second={rtdl_params['dropout_second']}")
            else:
                print(f"Parameters: lr={lr}, batch_size={batch_size}, steps={steps}, "
                      f"dropout={base_config['model_params']['rtdl_params']['dropout']}, d_layers={d_layers}")
            raise optuna.TrialPruned()

        report_path = str(Path(base_config['parent_dir']) / f'results_{args.eval_model}.json')
        report = lib.load_json(report_path)

        if 'r2' in report['metrics']['val']:
            score += report['metrics']['val']['r2']
        else:
            #############################################################################
            # 2026.1.19 평가 메트릭 변경: Macro Average F1-Score → F1 Positive (클래스 1의 F1-Score)
            # - 클래스 불균형 문제 해결을 위해 소수 클래스(클래스 1)의 F1-Score를 직접 최적화
            # - 원본: report['metrics']['val']['macro avg']['f1-score']
            #############################################################################
            score += report['metrics']['val']['1']['f1-score']

    shutil.rmtree(exps_path / f"{trial.number}")

    return score / n_datasets

study = optuna.create_study(
    direction='maximize',
    sampler=optuna.samplers.TPESampler(seed=0),
)

# study.optimize(objective, n_trials=50, show_progress_bar=True)
study.optimize(objective, n_trials=20, show_progress_bar=True)

best_config_path = parent_path / f'{prefix}_best/config.toml'
best_config = study.best_trial.user_attrs['config']
best_config["parent_dir"] = str(parent_path / f'{prefix}_best/')

os.makedirs(parent_path / f'{prefix}_best', exist_ok=True)
lib.dump_config(best_config, best_config_path)

#############################################################################
# 2026.1.18 NaN 감지 및 로깅: NaN 발생 시 명시적으로 로그를 남기고 trial 실패 처리
# 파라미터 중요도 계산
# 주의: 완료된 trial이 부족하거나 모든 trial이 동일한 점수를 받으면 ZeroDivisionError 발생 가능
#############################################################################
try:
    lib.dump_json(optuna.importance.get_param_importances(study), parent_path / f'{prefix}_best/importance.json')
except Exception:
    # 중요도 계산 실패 시 빈 JSON 저장 (완료된 trial 부족 또는 점수 분산 없음)
    lib.dump_json({}, parent_path / f'{prefix}_best/importance.json')

# subprocess.run(['python3.9', f'{pipeline}', '--config', f'{best_config_path}', '--train', '--sample'], check=True)
subprocess.run(['python3.9', f'{pipeline}', '--config', f'{best_config_path}', '--train', '--sample', '--change_val'], check=True)

if args.eval_seeds:
    best_exp = str(parent_path / f'{prefix}_best/config.toml')
    subprocess.run(['python3.9', f'{eval_seeds}', '--config', f'{best_exp}', '10', "ddpm", eval_type, args.eval_model, '5'], check=True)

