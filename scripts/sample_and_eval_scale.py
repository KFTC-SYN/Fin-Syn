"""
다양한 크기로 합성 데이터 샘플링 및 평가 스크립트 (다중 모델 지원)

사용법:
    python scripts/sample_and_eval_scale_ddpm.py --config exp/orig-micro/ddpm_cb_best/config.toml --sizes 1.0 1.5 2.0
    python scripts/sample_and_eval_scale_ddpm.py --config exp/orig-micro/ctgan/config.toml --sizes 1.0 1.5 2.0 --model_type ctgan
"""
import argparse
import shutil
import sys
from pathlib import Path
import lib
import torch
from sample import sample
from eval_catboost import train_catboost

# 다른 모델들의 샘플링 함수 import
# lib이 이미 import되어 있으므로 모듈 import 시 의존성 문제가 발생하지 않음
try:
    sys.path.insert(0, 'CTGAN')
    from train_sample_ctgan import sample_ctgan
except Exception:
    try:
        from CTGAN.train_sample_ctgan import sample_ctgan
    except Exception:
        sample_ctgan = None

try:
    sys.path.insert(0, 'CTAB-GAN')
    from train_sample_ctabgan import sample_ctabgan
except Exception:
    try:
        from CTAB_GAN.train_sample_ctabgan import sample_ctabgan
    except Exception:
        sample_ctabgan = None

try:
    sys.path.insert(0, 'CTAB-GAN-Plus')
    from train_sample_ctabganp import sample_ctabgan as sample_ctabganp
except Exception:
    try:
        from CTAB_GAN_Plus.train_sample_ctabganp import sample_ctabgan as sample_ctabganp
    except Exception:
        sample_ctabganp = None

try:
    # be_great를 경로에 추가하기 전에 lib이 이미 import되어 있는지 확인
    # lib이 이미 import되어 있으면 train_sample_great.py의 import lib은 재사용됨
    sys.path.insert(0, 'be_great')
    # train_sample_great.py가 import lib을 시도하지만, lib이 이미 로드되어 있으면 재사용됨
    from train_sample_great import sample_great
except Exception as e:
    # 모든 예외를 잡아서 디버깅 정보 출력
    first_error = e
    # be_great 모듈 경로로 재시도
    try:
        from be_great.train_sample_great import sample_great
    except Exception as e2:
        sample_great = None

try:
    sys.path.insert(0, 'TabPFGen')
    from train_sample_tabpfgen import sample_tabpfgen
except Exception as e:
    print(f"Error importing sample_tabpfgen (first attempt): {e}")
    try:
        from TabPFGen.train_sample_tabpfgen import sample_tabpfgen
    except Exception as e2:
        print(f"Error importing sample_tabpfgen (second attempt): {e2}")
        sample_tabpfgen = None

try:
    sys.path.insert(0, 'CTGAN')
    from train_sample_tvae import sample_tvae
except Exception:
    try:
        from CTGAN.train_sample_tvae import sample_tvae
    except Exception:
        sample_tvae = None

try:
    from smote.sample_smote import sample_smote
except Exception:
    try:
        sys.path.insert(0, 'smote')
        from sample_smote import sample_smote
    except Exception:
        sample_smote = None


def load_config(config_path):
    """설정 파일 로드"""
    # lib.load_config를 사용하여 "__none__" 문자열을 None으로 자동 변환
    return lib.load_config(config_path)


def get_train_size(real_data_path):
    """원본 데이터의 train_size 가져오기"""
    info_path = Path(real_data_path) / 'info.json'
    if info_path.exists():
        info = lib.load_json(info_path)
        return info.get('train_size', None)
    return None


def detect_model_type(base_parent_dir, config=None):
    """
    base_dir 경로나 config에서 모델 타입 자동 감지
    
    Args:
        base_parent_dir: 기본 디렉토리 경로
        config: 설정 딕셔너리 (선택)
    
    Returns:
        모델 타입 문자열 (ddpm, ctgan, ctabgan, ctabganp, great, smote, tabpfgen, tvae)
    """
    base_dir = Path(base_parent_dir)
    
    # config에서 모델 타입 확인
    if config:
        # pipeline 스크립트에서 모델 타입 확인
        if 'model_type' in config:
            model_type = config['model_type']
            if model_type in ['ddpm', 'mlp', 'transformer']:
                return 'ddpm'
            return model_type
    
    # 디렉토리 이름에서 모델 타입 추론
    dir_name = base_dir.name.lower()
    
    if 'ddpm' in dir_name or 'mlp' in dir_name or 'transformer' in dir_name:
        return 'ddpm'
    elif 'ctabganp' in dir_name or 'ctabgan-plus' in dir_name:
        return 'ctabganp'
    elif 'ctabgan' in dir_name:
        return 'ctabgan'
    elif 'ctgan' in dir_name:
        return 'ctgan'
    elif 'great' in dir_name:
        return 'great'
    elif 'smote' in dir_name:
        return 'smote'
    elif 'tabpfgen' in dir_name:
        return 'tabpfgen'
    elif 'tvae' in dir_name:
        return 'tvae'
    
    # 모델 파일로 판단 (디렉토리 이름으로 감지되지 않은 경우 fallback)
    # ctabgan-plus는 ctabgan.obj를 사용하므로 디렉토리 이름을 먼저 확인해야 함
    if (base_dir / "model.pt").exists():
        return 'ddpm'
    elif (base_dir / "ctabgan.obj").exists():
        # ctabgan.obj가 있으면 디렉토리 이름으로 구분
        # ctabgan-plus는 이미 위에서 처리되었으므로 여기서는 ctabgan
        return 'ctabgan'
    elif (base_dir / "ctgan.obj").exists():
        return 'ctgan'
    elif (base_dir / "great.obj").exists():
        return 'great'
    elif (base_dir / "tabpfgen.obj").exists():
        return 'tabpfgen'
    elif (base_dir / "tvae.obj").exists():
        return 'tvae'
    # SMOTE는 모델 파일이 없으므로 디렉토리 이름으로만 감지
    
    # 기본값: ddpm
    return 'ddpm'


def get_sample_function(model_type):
    """
    모델 타입에 따라 적절한 샘플링 함수 반환
    
    Args:
        model_type: 모델 타입 문자열
    
    Returns:
        샘플링 함수
    """
    sample_functions = {
        'ddpm': sample,
        'ctgan': sample_ctgan,
        'ctabgan': sample_ctabgan,
        'ctabganp': sample_ctabganp,
        'great': sample_great,
        'smote': sample_smote,
        'tabpfgen': sample_tabpfgen,
        'tvae': sample_tvae,
    }
    
    func = sample_functions.get(model_type.lower())
    if func is None:
        # 모델 타입이 딕셔너리에 없거나, import가 실패한 경우
        available_models = [k for k, v in sample_functions.items() if v is not None]
        if model_type.lower() in sample_functions:
            raise ValueError(
                f"모델 타입 '{model_type}'의 샘플링 함수를 import할 수 없습니다.\n"
                f"해당 모델의 모듈이 설치되어 있는지 확인하세요.\n"
                f"사용 가능한 모델: {', '.join(available_models)}"
            )
        else:
            raise ValueError(
                f"지원하지 않는 모델 타입: {model_type}\n"
                f"지원하는 모델: {', '.join(available_models)}"
            )
    
    return func


def sample_and_evaluate_size(
    base_parent_dir,
    real_data_path,
    size_multiplier,
    config,
    device,
    seed=0,
    change_val=True,
    model_type=None
):
    """
    특정 크기로 샘플링 및 평가 수행
    
    Args:
        base_parent_dir: 기본 디렉토리 (예: exp/orig-micro/ddpm_cb_best/)
        real_data_path: 원본 데이터 경로
        size_multiplier: 크기 배수 (1.0, 1.5, 2.0 등)
        config: 전체 설정 딕셔너리
        device: 디바이스
        seed: 시드
        change_val: validation split 변경 여부
        model_type: 모델 타입 (None이면 자동 감지)
    """
    # 크기별 디렉토리 생성
    size_dir_name = f"size_{size_multiplier}x"
    size_parent_dir = Path(base_parent_dir) / size_dir_name
    size_parent_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 100)
    print(f"크기 배수 {size_multiplier}x 처리 시작")
    print(f"저장 디렉토리: {size_parent_dir}")
    print("=" * 100)
    
    # 원본 데이터 크기 확인
    train_size = get_train_size(real_data_path)
    
    num_samples = int(train_size * size_multiplier)
    print(f"원본 train_size: {train_size}")
    print(f"샘플링할 크기: {num_samples} (배수: {size_multiplier}x)")
    
    # 모델 타입 감지
    model_type = detect_model_type(base_parent_dir, config)
    print(f"감지된 모델 타입: {model_type}")
    
    # 샘플링 함수 가져오기
    sample_func = get_sample_function(model_type)
    
    # 학습된 모델 경로 확인 및 복사
    base_dir = Path(base_parent_dir)
    model_path = None
    model_dst = None
    
    if model_type == 'ddpm':
        model_path = base_dir / "model.pt"
        if not model_path.exists():
            raise FileNotFoundError(
                f"학습된 모델을 찾을 수 없습니다: {model_path}\n"
                f"먼저 모델을 학습시켜주세요."
            )
        model_dst = size_parent_dir / "model.pt"
        if not model_dst.exists():
            shutil.copy2(model_path, model_dst)
            print(f"모델 파일 복사: {model_path} -> {model_dst}")
    else:
        # 다른 모델들은 모델 파일이 parent_dir에 있어야 함
        model_files = {
            'ctabgan': 'ctabgan.obj',
            'ctabganp': 'ctabgan.obj',  # CTAB-GAN-Plus는 ctabgan.obj 사용
            'ctgan': 'ctgan.obj',
            'great': 'great.obj',  # GReaT는 great.obj 사용
            'tabpfgen': 'tabpfgen.obj',
            'tvae': 'tvae.obj'
        }
        model_file = model_files.get(model_type)
        if model_file:
            model_path = base_dir / model_file
            if not model_path.exists():
                raise FileNotFoundError(
                    f"학습된 모델을 찾을 수 없습니다: {model_path}\n"
                    f"먼저 모델을 학습시켜주세요."
                )
            model_dst = size_parent_dir / model_file
            if not model_dst.exists():
                shutil.copy2(model_path, model_dst)
                print(f"모델 파일 복사: {model_path} -> {model_dst}")
    
    # 1. 샘플링 수행
    print("\n" + "-" * 100)
    print("1단계: 합성 데이터 샘플링")
    print("-" * 100)
    
    # 모델 타입에 따라 다른 파라미터로 샘플링
    if model_type == 'ddpm':
        sample_func(
            parent_dir=str(size_parent_dir),
            real_data_path=real_data_path,
            batch_size=config['sample'].get('batch_size', 10000),
            num_samples=num_samples,
            model_type=config.get('model_type', 'mlp'),
            model_params=config['model_params'],
            model_path=str(model_dst),
            num_timesteps=config['diffusion_params'].get('num_timesteps', 1000),
            gaussian_loss_type=config['diffusion_params'].get('gaussian_loss_type', 'mse'),
            scheduler=config['diffusion_params'].get('scheduler', 'cosine'),
            T_dict=config['train']['T'],
            num_numerical_features=config.get('num_numerical_features', 0),
            disbalance=config['sample'].get('disbalance', None),
            device=device,
            seed=seed,
            change_val=change_val
        )
    else:
        # 다른 모델들은 synthesizer=None으로 호출 (모델은 파일에서 로드)
        train_params = config.get('train_params', {})
        sample_func(
            synthesizer=None,
            parent_dir=str(size_parent_dir),
            real_data_path=real_data_path,
            num_samples=num_samples,
            train_params=train_params,
            change_val=change_val,
            device=device,
            seed=seed
        )
    
    # info.json 복사
    info_src = Path(real_data_path) / 'info.json'
    info_dst = size_parent_dir / 'info.json'
    if info_src.exists():
        shutil.copy2(info_src, info_dst)
    
    # 2. CatBoost 평가
    print("\n" + "-" * 100)
    print("2단계: CatBoost 평가")
    print("-" * 100)
    
    eval_type = config['eval']['type']['eval_type']
    T_dict = config['eval']['T']
    
    train_catboost(
        parent_dir=str(size_parent_dir),
        real_data_path=real_data_path,
        eval_type=eval_type,
        T_dict=T_dict,
        seed=config.get('seed', 0),
        change_val=change_val
    )
    
    print("\n" + "=" * 100)
    print(f"크기 배수 {size_multiplier}x 처리 완료!")
    print(f"결과 저장 위치: {size_parent_dir}")
    print("=" * 100)
    
    return size_parent_dir


def main():
    parser = argparse.ArgumentParser(
        description='다양한 크기로 합성 데이터 샘플링 및 평가 (다중 모델 지원)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
예제:
  # Tab-DDPM
  python scripts/sample_and_eval_scale_ddpm.py \\
      --config exp/orig-micro/ddpm_cb_best/config.toml \\
      --sizes 1.0 1.5 2.0 \\
      --seed 0 \\
      --change_val
  
  # CTGAN
  python scripts/sample_and_eval_scale_ddpm.py \\
      --config exp/orig-micro/ctgan/config.toml \\
      --sizes 1.0 1.5 2.0 \\
      --model_type ctgan \\
      --seed 0 \\
      --change_val
  
  # GReaT
  python scripts/sample_and_eval_scale_ddpm.py \\
      --config exp/orig-micro/great/config.toml \\
      --sizes 1.0 1.5 2.0 \\
      --model_type great \\
      --seed 0 \\
      --change_val
        """
    )
    parser.add_argument('--config', type=str, required=True,
                        help='설정 파일 경로 (config.toml)')
    parser.add_argument('--sizes', type=float, nargs='+', default=[1.0, 1.5, 2.0],
                        help='샘플링할 크기 배수 목록 (기본값: 1.0 1.5 2.0)')
    parser.add_argument('--seed', type=int, default=0,
                        help='랜덤 시드 (기본값: 0)')
    parser.add_argument('--change_val', action='store_true', default=True,
                        help='validation split 변경 여부 (기본값: True)')
    parser.add_argument('--skip_sampling', action='store_true', default=False,
                        help='샘플링 단계 건너뛰기 (이미 샘플링된 경우)')
    parser.add_argument('--skip_catboost', action='store_true', default=False,
                        help='CatBoost 평가 건너뛰기')
    parser.add_argument('--n_seeds', type=int, default=1,
                        help='CatBoost 평가를 수행할 시드 개수 (기본값: 1)')
    parser.add_argument('--model_type', type=str, default=None,
                        choices=['ddpm', 'ctgan', 'ctabgan', 'ctabganp', 'great', 'smote', 'tabpfgen', 'tvae'],
                        help='모델 타입 (None이면 자동 감지, 기본값: None)')
    
    args = parser.parse_args()
    
    # 설정 파일 로드
    config = load_config(args.config)
    
    base_parent_dir = config.get('parent_dir', '')
    real_data_path = config.get('real_data_path', '')
    
    if 'device' in config:
        device = torch.device(config['device'])
    else:
        device = torch.device('cuda:0')
    
    if not base_parent_dir or not real_data_path:
        raise ValueError("config.toml에 parent_dir과 real_data_path가 필요합니다.")
    
    # 모델 타입 감지
    model_type = args.model_type
    if model_type is None:
        model_type = detect_model_type(base_parent_dir, config)
    
    print("=" * 100)
    print(f"다양한 크기로 합성 데이터 샘플링 및 평가 시작 (모델: {model_type.upper()})")
    print("=" * 100)
    print(f"설정 파일: {args.config}")
    print(f"기본 디렉토리: {base_parent_dir}")
    print(f"원본 데이터 경로: {real_data_path}")
    print(f"크기 배수: {args.sizes}")
    print(f"시드: {args.seed}")
    print(f"디바이스: {device}")
    print("=" * 100)
    
    # 원본 데이터 크기 확인
    train_size = get_train_size(real_data_path)
    if train_size is None:
        raise ValueError(f"원본 데이터의 train_size를 확인할 수 없습니다: {real_data_path}")
    
    print(f"\n원본 데이터 train_size: {train_size:,}")
    print(f"샘플링할 크기:")
    for size_mult in args.sizes:
        num_samples = int(train_size * size_mult)
        print(f"  - {size_mult}x: {num_samples:,} 샘플")
    
    # 학습된 모델 확인 (SMOTE는 모델 파일이 없음)
    base_dir = Path(base_parent_dir)
    model_files = {
        'ddpm': 'model.pt',
        'ctabgan': 'ctabgan.obj',
        'ctabganp': 'ctabgan.obj',  # CTAB-GAN-Plus는 ctabgan.obj 사용
        'ctgan': 'ctgan.obj',
        'great': 'great.obj',  # GReaT는 great.obj 사용
        'tabpfgen': 'tabpfgen.obj',
        'tvae': 'tvae.obj'
    }
    model_file = model_files.get(model_type, 'model.pt')
    if model_type != 'smote':
        model_path = base_dir / model_file
        if not model_path.exists():
            raise FileNotFoundError(
                f"학습된 모델을 찾을 수 없습니다: {model_path}\n"
                f"모델 타입: {model_type}\n"
                f"먼저 모델을 학습시켜주세요."
            )
    
    # 각 크기별로 처리
    results = {}
    for size_mult in args.sizes:
        size_dir_name = f"size_{size_mult}x"
        size_parent_dir = Path(base_parent_dir) / size_dir_name
        
        # 샘플링 단계
        if not args.skip_sampling:
            size_parent_dir.mkdir(parents=True, exist_ok=True)
            
            # 모델 파일 복사 (SMOTE는 모델 파일이 없음)
            if model_type != 'smote':
                model_dst = size_parent_dir / model_file
                if not model_dst.exists():
                    shutil.copy2(model_path, model_dst)
                    print(f"\n모델 파일 복사: {model_path} -> {model_dst}")
            
            # 샘플링 함수 가져오기
            sample_func = get_sample_function(model_type)
            
            # 모델 타입에 따라 다른 파라미터로 샘플링
            if model_type == 'ddpm':
                sample_func(
                    parent_dir=str(size_parent_dir),
                    real_data_path=real_data_path,
                    batch_size=config['sample'].get('batch_size', 10000),
                    num_samples=int(train_size * size_mult),
                    model_type=config.get('model_type', 'mlp'),
                    model_params=config['model_params'],
                    model_path=str(model_dst),
                    num_timesteps=config['diffusion_params'].get('num_timesteps', 1000),
                    gaussian_loss_type=config['diffusion_params'].get('gaussian_loss_type', 'mse'),
                    scheduler=config['diffusion_params'].get('scheduler', 'cosine'),
                    T_dict=config['train']['T'],
                    num_numerical_features=config.get('num_numerical_features', 0),
                    disbalance=config['sample'].get('disbalance', None),
                    device=device,
                    seed=args.seed,
                    change_val=args.change_val
                )
            elif model_type == 'smote':
                # SMOTE는 frac_samples를 사용 (원본 대비 추가 생성 비율)
                # size_multiplier = 1.0이면 원본과 같은 크기의 합성 데이터가 필요
                # size_multiplier = 2.0이면 원본의 2배 크기의 합성 데이터가 필요
                # eval_type == "synthetic"이면 원본을 제외한 합성 데이터만 반환하므로
                # frac_samples = size_multiplier로 설정 (원본 + 추가 생성 = size_multiplier * 원본)
                smote_params = config.get('smote_params', {})
                # frac_samples는 원본 대비 추가 생성 비율이므로, size_multiplier와 동일하게 설정
                # 예: size_mult=1.0 -> frac_samples=1.0 (원본 + 원본만큼 더 = 2배, 합성만 = 원본 크기)
                #     size_mult=2.0 -> frac_samples=2.0 (원본 + 원본 2배만큼 더 = 3배, 합성만 = 원본 2배)
                frac_samples = size_mult
                sample_func(
                    parent_dir=str(size_parent_dir),
                    real_data_path=real_data_path,
                    eval_type=smote_params.get('eval_type', 'synthetic'),
                    k_neighbours=smote_params.get('k_neighbours', 5),
                    frac_samples=frac_samples,
                    frac_lam_del=smote_params.get('frac_lam_del', 0.0),
                    change_val=args.change_val,
                    save=True,
                    seed=args.seed
                )
            else:
                # 다른 모델들은 synthesizer=None으로 호출 (모델은 파일에서 로드)
                train_params = config.get('train_params', {})
                sample_func(
                    synthesizer=None,
                    parent_dir=str(size_parent_dir),
                    real_data_path=real_data_path,
                    num_samples=int(train_size * size_mult),
                    train_params=train_params,
                    change_val=args.change_val,
                    device=device,
                    seed=args.seed
                )
            
            # info.json 복사
            info_src = Path(real_data_path) / 'info.json'
            info_dst = size_parent_dir / 'info.json'
            if info_src.exists():
                shutil.copy2(info_src, info_dst)
        else:
            print(f"\n샘플링 단계 건너뛰기: {size_dir_name}")
        
        # CatBoost 평가
        if not args.skip_catboost:
            eval_type = config['eval']['type']['eval_type']
            T_dict = config['eval']['T']
            
            # 여러 시드로 평가 수행
            metrics_seeds_report = lib.SeedsMetricsReport()
            
            print(f"\nCatBoost 평가 시작: {args.n_seeds}회 수행")
            for seed in range(args.n_seeds):
                print(f'\n**Eval Iter: {seed + 1}/{args.n_seeds} (크기: {size_mult}x)**')
                metric_report = train_catboost(
                    parent_dir=str(size_parent_dir),
                    real_data_path=real_data_path,
                    eval_type=eval_type,
                    T_dict=T_dict,
                    seed=seed,
                    change_val=args.change_val
                )
                metrics_seeds_report.add_report(metric_report)
            
            # 평균 및 표준편차 계산
            metrics_seeds_report.get_mean_std()
            res = metrics_seeds_report.print_result()
            
            # 결과를 JSON 파일로 저장
            eval_json_path = size_parent_dir / "eval_catboost.json"
            lib.dump_json({eval_type: res}, eval_json_path)
            print(f"\n평가 결과 저장: {eval_json_path}")
        else:
            print(f"CatBoost 평가 건너뛰기: {size_dir_name}")
        
        results[size_mult] = str(size_parent_dir)
    
    # 결과 요약
    print("\n" + "=" * 100)
    print("모든 작업 완료!")
    print("=" * 100)
    print("\n결과 저장 위치:")
    for size_mult, result_dir in results.items():
        print(f"  {size_mult}x: {result_dir}")
    print("=" * 100)


if __name__ == '__main__':
    main()

