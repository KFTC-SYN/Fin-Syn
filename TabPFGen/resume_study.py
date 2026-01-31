"""
Optuna study 복구 스크립트
로그에서 완료된 trials를 추출하여 새 study에 추가
"""
import optuna
from pathlib import Path
import os

# 완료된 trials 정보 (로그에서 추출)
completed_trials_data = [
    {
        'number': 0,
        'value': 0.4997286840686167,
        'params': {
            'n_sgld_steps': 1000,
            'sgld_step_size': 0.00703573702872215,
            'sgld_noise_scale': 0.019578897201213006,
            'frac_samples': 0
        },
        'user_attrs': {
            'train_params': {
                'n_sgld_steps': 1000,
                'sgld_step_size': 0.00703573702872215,
                'sgld_noise_scale': 0.019578897201213006
            },
            'num_samples': 66528
        }
    },
    {
        'number': 1,
        'value': 0.49817519154794426,
        'params': {
            'n_sgld_steps': 1000,
            'sgld_step_size': 0.011423254155608374,
            'sgld_noise_scale': 0.013680095279726922,
            'frac_samples': 1
        },
        'user_attrs': {
            'train_params': {
                'n_sgld_steps': 1000,
                'sgld_step_size': 0.011423254155608374,
                'sgld_noise_scale': 0.013680095279726922
            },
            'num_samples': 133056
        }
    },
    {
        'number': 2,
        'value': 0.4997286840686167,
        'params': {
            'n_sgld_steps': 2000,
            'sgld_step_size': 0.0360009119291161,
            'sgld_noise_scale': 0.0549571618641162,
            'frac_samples': 1
        },
        'user_attrs': {
            'train_params': {
                'n_sgld_steps': 2000,
                'sgld_step_size': 0.0360009119291161,
                'sgld_noise_scale': 0.0549571618641162
            },
            'num_samples': 133056
        }
    },
    {
        'number': 3,
        'value': 0.4997286840686167,
        'params': {
            'n_sgld_steps': 500,
            'sgld_step_size': 0.019047678084282035,
            'sgld_noise_scale': 0.0019351140885595273,
            'frac_samples': 1
        },
        'user_attrs': {
            'train_params': {
                'n_sgld_steps': 500,
                'sgld_step_size': 0.019047678084282035,
                'sgld_noise_scale': 0.0019351140885595273
            },
            'num_samples': 133056
        }
    },
    {
        'number': 4,
        'value': 0.49647843430565947,
        'params': {
            'n_sgld_steps': 2000,
            'sgld_step_size': 0.008171478915115866,
            'sgld_noise_scale': 0.013704648392849027,
            'frac_samples': -1
        },
        'user_attrs': {
            'train_params': {
                'n_sgld_steps': 2000,
                'sgld_step_size': 0.008171478915115866,
                'sgld_noise_scale': 0.013704648392849027
            },
            'num_samples': 33264
        }
    },
    {
        'number': 5,
        'value': 0.49810355918713806,
        'params': {
            'n_sgld_steps': 2000,
            'sgld_step_size': 0.02310152225018238,
            'sgld_noise_scale': 0.00523619487341915,
            'frac_samples': 0
        },
        'user_attrs': {
            'train_params': {
                'n_sgld_steps': 2000,
                'sgld_step_size': 0.02310152225018238,
                'sgld_noise_scale': 0.00523619487341915
            },
            'num_samples': 66528
        }
    }
]


def create_study_with_completed_trials(study_name, storage_path, completed_trials_data):
    """완료된 trials로 study 생성"""
    # Storage 디렉토리 생성
    storage_path_obj = Path(storage_path)
    os.makedirs(storage_path_obj.parent, exist_ok=True)
    
    # 기존 데이터베이스 파일이 있으면 삭제 (새로 시작)
    if storage_path_obj.exists():
        print(f"Removing existing database: {storage_path}")
        storage_path_obj.unlink()
    
    # 먼저 in-memory study 생성
    study = optuna.create_study(
        direction='maximize',
        sampler=optuna.samplers.TPESampler(seed=0),
    )
    
    # 완료된 trials 추가
    for trial_data in completed_trials_data:
        trial = study.ask()
        
        # 파라미터 공간 정의
        trial.suggest_categorical('n_sgld_steps', [500, 1000, 1500, 2000])
        trial.suggest_loguniform('sgld_step_size', 0.001, 0.1)
        trial.suggest_loguniform('sgld_noise_scale', 0.001, 0.1)
        trial.suggest_int('frac_samples', -1, 1)
        
        # 실제 파라미터 값 설정 (내부 API 사용)
        trial._suggest('n_sgld_steps', trial_data['params']['n_sgld_steps'])
        trial._suggest('sgld_step_size', trial_data['params']['sgld_step_size'])
        trial._suggest('sgld_noise_scale', trial_data['params']['sgld_noise_scale'])
        trial._suggest('frac_samples', trial_data['params']['frac_samples'])
        
        # User attributes 설정
        for attr_name, attr_value in trial_data.get('user_attrs', {}).items():
            trial.set_user_attr(attr_name, attr_value)
        
        # Trial 완료 처리
        study.tell(trial, trial_data['value'], state=optuna.trial.TrialState.COMPLETE)
    
    # Study를 storage에 저장 (JSON 방식으로 시도)
    try:
        import json
        import pickle
        
        # Study 정보를 JSON으로 저장
        study_info = {
            'study_name': study_name,
            'direction': study.direction.name,
            'trials': []
        }
        
        for trial in study.trials:
            if trial.state == optuna.trial.TrialState.COMPLETE:
                trial_info = {
                    'number': trial.number,
                    'value': trial.value,
                    'params': trial.params,
                    'user_attrs': trial.user_attrs,
                    'state': 'COMPLETE'
                }
                study_info['trials'].append(trial_info)
        
        json_path = storage_path_obj.with_suffix('.json')
        with open(json_path, 'w') as f:
            json.dump(study_info, f, indent=2)
        print(f"Study info saved to: {json_path}")
        
        # SQLite storage도 시도
        try:
            study_with_storage = optuna.create_study(
                study_name=study_name,
                storage=f"sqlite:///{storage_path}",
                direction='maximize',
                sampler=optuna.samplers.TPESampler(seed=0),
                load_if_exists=False,
            )
            
            # Trials를 새 study에 추가
            for trial_data in completed_trials_data:
                study_with_storage.enqueue_trial(trial_data['params'])
                trial = study_with_storage.ask()
                for attr_name, attr_value in trial_data.get('user_attrs', {}).items():
                    trial.set_user_attr(attr_name, attr_value)
                study_with_storage.tell(trial, trial_data['value'], state=optuna.trial.TrialState.COMPLETE)
            
            return study_with_storage
        except Exception as e:
            print(f"Warning: Could not create SQLite storage: {e}")
            print("Using in-memory study. You may need to manually resume.")
            return study
            
    except Exception as e:
        print(f"Warning: Could not save study info: {e}")
        return study


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('data_path', type=str)
    args = parser.parse_args()
    
    study_name = f"tabpfgen_{Path(args.data_path).name}"
    storage_path = f"optuna_studies/{study_name}.db"
    
    print(f"Creating study with {len(completed_trials_data)} completed trials...")
    study = create_study_with_completed_trials(study_name, storage_path, completed_trials_data)
    
    print(f"Study created successfully!")
    print(f"Completed trials: {len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])}")
    print(f"Best trial: {study.best_trial.number}, Best value: {study.best_value}")
    print(f"Storage: {storage_path}")
