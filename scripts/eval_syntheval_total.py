"""
공식 SynthEval 라이브러리를 사용한 합성 데이터 평가 스크립트
모든 평가 메트릭을 평가합니다.
"""
import numpy as np
import pandas as pd
import os
import json
import argparse
from pathlib import Path
import time
import warnings

# 공식 SynthEval 라이브러리 import
import syntheval  # type: ignore


import lib
from lib import read_pure_data, read_changed_val, load_json

# matplotlib 폰트 관련 경고 억제
warnings.filterwarnings('ignore', category=UserWarning, module='matplotlib')
warnings.filterwarnings('ignore', message='.*Glyph.*missing from font.*')
warnings.filterwarnings('ignore', message='.*HANGUL.*')

# matplotlib 한글 폰트 설정
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

def setup_korean_font():
    """한글을 지원하는 폰트를 찾아 설정"""
    # 한글 폰트 파일 경로 후보 목록 (우선순위 순)
    korean_font_paths = [
        '/usr/share/fonts/truetype/nanum/NanumSquareRoundR.ttf',  # NanumSquareRound Regular
        '/usr/share/fonts/truetype/nanum/NanumSquareR.ttf',       # NanumSquare Regular
        '/usr/share/fonts/truetype/nanum/NanumGothic.ttf',         # NanumGothic
        '/usr/share/fonts/truetype/nanum/NanumBarunGothic.ttf',   # NanumBarunGothic
        '/usr/share/fonts/truetype/nanum/NanumMyeongjo.ttf',      # NanumMyeongjo
    ]
    
    # 폰트 파일 경로 찾기
    font_path = None
    for path in korean_font_paths:
        if os.path.exists(path):
            font_path = path
            break
    
    # 폰트 파일을 찾지 못한 경우, fc-list로 한글 폰트 찾기 시도
    if font_path is None:
        import subprocess
        try:
            result = subprocess.run(['fc-list', ':lang=ko'], 
                                  capture_output=True, text=True, timeout=2)
            if result.returncode == 0 and result.stdout:
                lines = result.stdout.strip().split('\n')
                if lines:
                    # 첫 번째 한글 폰트 경로 사용
                    font_path = lines[0].split(':')[0]
        except Exception:
            pass
    
    # 폰트 설정
    if font_path and os.path.exists(font_path):
        try:
            # 폰트 파일을 matplotlib에 직접 등록
            font_prop = fm.FontProperties(fname=font_path)
            font_name = font_prop.get_name()
            fm.fontManager.addfont(font_path)
            plt.rcParams['font.family'] = font_name
            plt.rcParams['axes.unicode_minus'] = False  # 마이너스 기호 깨짐 방지
            print(f'한글 폰트 설정: {font_name} (경로: {font_path})')
        except Exception as e:
            print(f'폰트 등록 실패: {e}')
            print('기본 폰트를 사용합니다.')
            plt.rcParams['axes.unicode_minus'] = False
    else:
        # 한글 폰트를 찾지 못한 경우 경고 출력
        print('경고: 한글 폰트를 찾을 수 없습니다. 한글이 깨져서 표시될 수 있습니다.')
        print('한글 폰트 설치 방법:')
        print('  Ubuntu/Debian: sudo apt-get install fonts-nanum')
        print('  또는 matplotlib 폰트 캐시 삭제 후 재시작: rm -rf ~/.cache/matplotlib')
        # 기본 폰트 사용 (영문만 표시됨)
        plt.rcParams['axes.unicode_minus'] = False

# 한글 폰트 설정 실행
setup_korean_font()

def numpy_to_dataframe(X_num, X_cat, y, num_cols=None, cat_cols=None, target_col='target'):
    """numpy 배열을 pandas DataFrame으로 변환"""
    dfs = []
    
    if X_num is not None:
        if num_cols is None:
            num_cols = [f'num_{i}' for i in range(X_num.shape[1])]
        df_num = pd.DataFrame(X_num, columns=num_cols)
        dfs.append(df_num)
    
    if X_cat is not None:
        if cat_cols is None:
            cat_cols = [f'cat_{i}' for i in range(X_cat.shape[1])]
        df_cat = pd.DataFrame(X_cat, columns=cat_cols)
        # 범주형 데이터를 문자열로 변환
        for col in df_cat.columns:
            df_cat[col] = df_cat[col].astype(str)
        dfs.append(df_cat)
    
    if len(dfs) > 0:
        df = pd.concat(dfs, axis=1)
    else:
        df = pd.DataFrame()
    
    # 타겟 컬럼 추가
    df[target_col] = y
    return df


def get_column_names(real_data_path):
    """실제 데이터 경로에서 컬럼 이름 정보 가져오기"""
    info_path = Path(real_data_path) / 'info.json'
    if info_path.exists():
        info = load_json(info_path)
        num_cols = info.get('num_cols', None)
        cat_cols = info.get('cat_cols', None)
        return num_cols, cat_cols
    return None, None


def get_target_column_name(real_data_path):
    """타겟 컬럼 이름 가져오기 (info.json에서)"""
    info_path = Path(real_data_path) / 'info.json'
    if info_path.exists():
        info = load_json(info_path)
        # 타겟 컬럼은 보통 'target'이지만, info.json에 명시되어 있을 수 있음
        return info.get('target_col', 'target')
    return 'target'


def train_syntheval_total(
    parent_dir,
    real_data_path,
    eval_type,
    T_dict=None,
    seed=0,
    change_val=True,
    presets_file="full_eval",
    custom_metrics=None,
    exclude_metrics=None,
    device=None
):
    """
    공식 SynthEval 라이브러리를 직접 호출하여 전체 평가 실행
    
    Args:
        parent_dir: 합성 데이터가 저장된 디렉토리
        real_data_path: 실제 데이터 경로
        eval_type: 평가 타입 ('synthetic', 'real', 'merged')
        T_dict: 변환 설정 (사용하지 않음)
        seed: 랜덤 시드
        change_val: validation split 변경 여부
        presets_file: SynthEval preset 파일 ('full_eval', 'fast_eval', 'privacy') 또는 JSON 파일 경로
        custom_metrics: 커스텀 메트릭 설정 딕셔너리 (선택사항)
        device: 디바이스 (사용하지 않음)
    
    Returns:
        dict: 평가 결과
    """
    print('='*100)
    print('SynthEval (공식 라이브러리) 전체 평가 시작')
    print(f'syntheval 버전: {syntheval.__version__ if hasattr(syntheval, "__version__") else "unknown"}')
    print('='*100)
    
    start_time = time.time()
    
    # 실제 데이터 로드
    if change_val:
        X_num_real, X_cat_real, y_real, X_num_val, X_cat_val, y_val = read_changed_val(
            real_data_path, val_size=0.2
        )
        # train과 val을 합침
        if X_num_real is not None and X_num_val is not None:
            X_num_real = np.concatenate([X_num_real, X_num_val], axis=0)
        if X_cat_real is not None and X_cat_val is not None:
            X_cat_real = np.concatenate([X_cat_real, X_cat_val], axis=0)
        y_real = np.concatenate([y_real, y_val], axis=0)
    else:
        X_num_real, X_cat_real, y_real = read_pure_data(real_data_path, 'train')
    
    # Test 데이터 로드 (holdout으로 사용)
    X_num_test, X_cat_test, y_test = read_pure_data(real_data_path, 'test')
    
    # 컬럼 이름 가져오기
    num_cols, cat_cols = get_column_names(real_data_path)
    
    # 타겟 컬럼 이름 가져오기
    target_col = get_target_column_name(real_data_path)
    
    # 실제 데이터를 DataFrame으로 변환
    df_real = numpy_to_dataframe(
        X_num_real, X_cat_real, y_real, 
        num_cols=num_cols, cat_cols=cat_cols, target_col=target_col
    )
    
    # Test 데이터를 DataFrame으로 변환 (holdout)
    df_test = numpy_to_dataframe(
        X_num_test, X_cat_test, y_test,
        num_cols=num_cols, cat_cols=cat_cols, target_col=target_col
    )
    
    print(f'\n실제 데이터: {len(df_real)} 샘플, {len(df_real.columns)} 컬럼')
    print(f'테스트 데이터: {len(df_test)} 샘플, {len(df_test.columns)} 컬럼')
    
    # 카테고리컬 컬럼 자동 감지 (명시되지 않은 경우)
    if cat_cols is None:
        # SynthEval의 기본 동작: unique 값이 10개 미만이면 범주형으로 간주
        cat_cols = []
        for col in df_real.columns:
            if col == target_col:
                # target_col도 categorical인지 확인 (PCA 메트릭 요구사항)
                if df_real[col].dtype == 'object' or df_real[col].nunique() < 10:
                    cat_cols.append(col)
                continue
            if df_real[col].dtype == 'object':
                cat_cols.append(col)
            elif df_real[col].nunique() < 10:
                cat_cols.append(col)
        print(f'자동 감지된 범주형 컬럼: {cat_cols}')
    else:
        # cat_cols가 이미 설정된 경우에도 target_col이 categorical인지 확인
        print(f'설정된 범주형 컬럼: {cat_cols}')
    
    # cat_cols의 원본을 저장 (numpy_to_dataframe 호출 시 사용)
    # target_col은 y로 별도 전달되므로 cat_cols에 포함되지 않아야 함
    cat_cols_for_data = cat_cols.copy() if cat_cols else []
    
    # SynthEval을 위한 cat_cols 준비 (target_col 포함)
    cat_cols_for_syntheval = cat_cols.copy() if cat_cols else []
    
    # target_col이 categorical인지 확인하고 필요시 SynthEval용 cat_cols에 추가
    if target_col not in cat_cols_for_syntheval:
        # target_col이 categorical 조건을 만족하는지 확인
        if df_real[target_col].dtype == 'object' or df_real[target_col].nunique() < 10:
            cat_cols_for_syntheval.append(target_col)
            print(f'target_col ({target_col})을 범주형 컬럼에 추가했습니다.')
        else:
            print(f'경고: target_col ({target_col})이 categorical이 아닙니다 (unique 값: {df_real[target_col].nunique()}).')
            print(f'      PCA 및 cls_acc 메트릭이 실행되지 않을 수 있습니다.')
    
    # SynthEval evaluator 초기화 (syntheval 패키지 직접 호출)
    print(f'\nSynthEval evaluator 초기화 중...')
    evaluator = syntheval.SynthEval(
        df_real,
        holdout_dataframe=df_test,
        cat_cols=cat_cols_for_syntheval if cat_cols_for_syntheval else None,
        verbose=True
    )
    
    # 평가할 데이터 준비
    if eval_type == 'synthetic':
        print(f'\n합성 데이터 로드: {parent_dir}')
        X_num_synt, X_cat_synt, y_synt = read_pure_data(parent_dir)
        
        df_synt = numpy_to_dataframe(
            X_num_synt, X_cat_synt, y_synt,
            num_cols=num_cols, cat_cols=cat_cols_for_data, target_col=target_col
        )
        
        print(f'합성 데이터: {len(df_synt)} 샘플, {len(df_synt.columns)} 컬럼')
        df_to_evaluate = df_synt
        
    elif eval_type == 'real':
        print('\n실제 데이터 평가 (baseline)')
        df_to_evaluate = df_real
        
    elif eval_type == 'merged':
        print('\n병합 데이터 평가')
        X_num_fake, X_cat_fake, y_fake = read_pure_data(parent_dir)
        
        # 실제 데이터와 합성 데이터 병합
        if X_num_real is not None and X_num_fake is not None:
            X_num_merged = np.concatenate([X_num_real, X_num_fake], axis=0)
        else:
            X_num_merged = X_num_real if X_num_real is not None else X_num_fake
            
        if X_cat_real is not None and X_cat_fake is not None:
            X_cat_merged = np.concatenate([X_cat_real, X_cat_fake], axis=0)
        else:
            X_cat_merged = X_cat_real if X_cat_real is not None else X_cat_fake
            
        y_merged = np.concatenate([y_real, y_fake], axis=0)
        
        df_merged = numpy_to_dataframe(
            X_num_merged, X_cat_merged, y_merged,
            num_cols=num_cols, cat_cols=cat_cols_for_data, target_col=target_col
        )
        
        print(f'병합 데이터: {len(df_merged)} 샘플, {len(df_merged.columns)} 컬럼')
        df_to_evaluate = df_merged
    else:
        raise ValueError(f"알 수 없는 eval_type: {eval_type}")
    
    # 결과 저장 경로 설정
    output_path = Path(parent_dir) / 'eval_syntheval_total.json'
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plot_output_dir = output_path.parent
    
    # syntheval 패키지의 evaluate 메서드 직접 호출
    # plot들이 parent_dir에 저장되도록 작업 디렉토리 변경
    original_cwd = os.getcwd()
    try:
        os.chdir(str(plot_output_dir))
        print(f'\n평가 실행 중... (preset: {presets_file})')
        print(f'Plot 저장 경로: {plot_output_dir}')
        
        eval_kwargs = {
            'analysis_target_var': target_col,
            'presets_file': presets_file
        }
        
        # 제외할 메트릭이 있으면 preset 파일을 로드하여 제거
        if exclude_metrics:
            import json
            import os as os_module
            
            # syntheval의 preset 파일 경로 찾기
            syntheval_path = os_module.path.dirname(syntheval.__file__)
            preset_path = os_module.path.join(syntheval_path, 'presets', f'{presets_file}.json')
            
            if os_module.path.exists(preset_path):
                with open(preset_path, 'r') as f:
                    preset_config = json.load(f)
                
                # 제외할 메트릭 제거
                for metric in exclude_metrics:
                    if metric in preset_config:
                        del preset_config[metric]
                        print(f'메트릭 제외: {metric}')
                
                # preset 파일 대신 직접 설정 사용
                eval_kwargs['presets_file'] = None
                eval_kwargs.update(preset_config)
        
        # custom_metrics로 오버라이드 (제외 후에도 적용 가능)
        if custom_metrics:
            eval_kwargs.update(custom_metrics)
        
        # syntheval.SynthEval.evaluate() 직접 호출
        key_results = evaluator.evaluate(df_to_evaluate, **eval_kwargs)
        
        # 생성된 plot 파일 목록 확인
        plot_files = sorted([f for f in os.listdir('.') if f.startswith('SE_') and f.endswith('.png')])
        if plot_files:
            print(f'\n생성된 plot 파일 ({len(plot_files)}개):')
            for plot_file in plot_files:
                print(f'  - {plot_file}')
    finally:
        # 원래 작업 디렉토리로 복귀
        os.chdir(original_cwd)
    
    # 결과를 딕셔너리로 변환
    results_dict = {}
    
    # key_results DataFrame을 딕셔너리로 변환
    if key_results is not None and not key_results.empty:
        # DataFrame을 레코드 리스트로 변환
        results_dict['key_results'] = key_results.to_dict('records')
        # 메트릭별로 그룹화
        metrics_dict = {}
        for _, row in key_results.iterrows():
            metric_name = row['metric']
            if metric_name not in metrics_dict:
                metrics_dict[metric_name] = []
            metrics_dict[metric_name].append({
                'dim': row.get('dim', ''),
                'val': float(row.get('val', 0)) if pd.notna(row.get('val')) else None,
                'err': float(row.get('err', 0)) if pd.notna(row.get('err')) else None,
                'n_val': float(row.get('n_val', 0)) if pd.notna(row.get('n_val')) else None,
                'n_err': float(row.get('n_err', 0)) if pd.notna(row.get('n_err')) else None,
            })
        results_dict['metrics'] = metrics_dict
    
    # evaluator._raw_results에서 추가 정보 추출 (syntheval 패키지의 내부 구조 활용)
    if hasattr(evaluator, '_raw_results') and evaluator._raw_results:
        raw_results_dict = {}
        for metric_name, metric_obj in evaluator._raw_results.items():
            try:
                # syntheval 메트릭 객체의 results 속성 추출
                if hasattr(metric_obj, 'results') and metric_obj.results:
                    # 딕셔너리로 변환 가능한 형태로 처리
                    if isinstance(metric_obj.results, dict):
                        raw_results_dict[metric_name] = metric_obj.results
                    else:
                        raw_results_dict[metric_name] = str(metric_obj.results)
            except Exception as e:
                print(f"Warning: {metric_name} raw_results 추출 실패: {e}")
        
        if raw_results_dict:
            results_dict['raw_results'] = raw_results_dict
    
    # 실행 시간 및 메타데이터 추가
    elapsed_time = time.time() - start_time
    results_dict['_execution_time'] = elapsed_time
    results_dict['_preset_file'] = presets_file
    results_dict['_eval_type'] = eval_type
    results_dict['_target_column'] = target_col
    results_dict['_categorical_columns'] = cat_cols if cat_cols else []
    results_dict['_syntheval_version'] = syntheval.__version__ if hasattr(syntheval, '__version__') else 'unknown'
    
    # JSON 저장
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results_dict, f, indent=2, ensure_ascii=False, default=str)
    
    print(f'\n결과 저장: {output_path}')
    print(f'총 실행 시간: {elapsed_time:.2f}초 ({elapsed_time/60:.2f}분)')
    print('='*100)
    
    return results_dict


def main():
    """메인 함수: 커맨드라인 인터페이스"""
    parser = argparse.ArgumentParser(description='SynthEval 전체 평가 스크립트')
    parser.add_argument('--config', type=str, required=True,
                        help='설정 파일 경로 (config.toml)')
    parser.add_argument('--preset', type=str, default='full_eval',
                        choices=['full_eval', 'fast_eval', 'privacy'],
                        help='SynthEval preset 파일 (default: full_eval)')
    parser.add_argument('--change_val', action='store_true',
                        help='validation split 변경 여부')
    parser.add_argument('--exclude', type=str, nargs='+', default=None,
                        help='제외할 메트릭 목록 (예: --exclude nnaa nndr)')
    parser.add_argument('--sizes', type=float, nargs='+', default=None,
                        help='크기별 평가 (예: --sizes 1.0 1.5 2.0). 지정하면 size_Xx 디렉토리들을 평가')
    
    args = parser.parse_args()
    
    # 설정 파일 로드
    import tomli
    with open(args.config, 'rb') as f:
        config = tomli.load(f)
    
    # 설정 추출
    parent_dir = config.get('parent_dir', '')
    real_data_path = config.get('real_data_path', '')
    
    eval_config = config.get('eval', {})
    eval_type_config = eval_config.get('type', {})
    eval_type = eval_type_config.get('eval_type', 'synthetic')
    
    T_dict = eval_config.get('T', {})
    seed = config.get('seed', 0)
    
    # 크기별 평가 모드
    if args.sizes:
        print('='*100)
        print(f'크기별 SynthEval 평가 시작: {args.sizes}')
        print('='*100)
        
        base_parent_dir = Path(parent_dir)
        results_summary = {}
        
        for size in args.sizes:
            size_dir = base_parent_dir / f'size_{size}x'
            
            if not size_dir.exists():
                print(f'\n경고: 디렉토리를 찾을 수 없습니다: {size_dir}')
                print('건너뜁니다...\n')
                continue
            
            print('\n' + '='*100)
            print(f'평가 중: Size {size}x')
            print(f'디렉토리: {size_dir}')
            print('='*100)
            
            # 평가 실행
            try:
                result = train_syntheval_total(
                    parent_dir=str(size_dir),
                    real_data_path=real_data_path,
                    eval_type=eval_type,
                    T_dict=T_dict,
                    seed=seed,
                    change_val=args.change_val,
                    presets_file=args.preset,
                    exclude_metrics=args.exclude
                )
                results_summary[f'size_{size}x'] = {
                    'status': 'success',
                    'result_file': str(size_dir / f'eval_syntheval_{args.preset}.json')
                }
                print(f'\n 완료: Size {size}x')
            except Exception as e:
                print(f'\n 오류 발생 (Size {size}x): {e}')
                results_summary[f'size_{size}x'] = {
                    'status': 'failed',
                    'error': str(e)
                }
        
        # 요약 출력
        print('\n' + '='*100)
        print('크기별 평가 완료 요약')
        print('='*100)
        for size_name, info in results_summary.items():
            if info['status'] == 'success':
                print(f'  {size_name}: {info["result_file"]}')
            else:
                print(f'  {size_name}: {info["error"]}')
        print('='*100)
    
    else:
        # 단일 디렉토리 평가 (기존 동작)
        train_syntheval_total(
            parent_dir=parent_dir,
            real_data_path=real_data_path,
            eval_type=eval_type,
            T_dict=T_dict,
            seed=seed,
            change_val=args.change_val,
            presets_file=args.preset,
            exclude_metrics=args.exclude
        )


if __name__ == '__main__':
    main()

