"""
합성 데이터를 numpy 배열 형식(.npy)에서 CSV 형식으로 변환하는 유틸리티 스크립트.

각 모델 디렉토리의 npy 파일들을 로드하여 DataFrame으로 변환하고,
크기별 디렉토리(size_1.0x, size_1.5x 등)의 데이터도 독립적으로 통합할 수 있습니다.
"""
import argparse
import json
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd


# 상수 정의
INFO_FILE = "info.json"
NUM_FILE = "X_num_train.npy"
CAT_FILE = "X_cat_train.npy"
TARGET_FILE = "y_train.npy"
DATA_DIR = "data"
BASE_KEY = "base"
TARGET_COLUMN = "y"
DEFAULT_EXP_ROOT = "exp/orig-micro-retry"
DEFAULT_MODEL = "ddpm_cb_best"
DDPM_CB_BEST_MODEL = "ddpm_cb_best"

# 선호 컬럼 순서
PREFERRED_COLUMN_ORDER = [
    "거래일자",
    "거래시간대",
    "출금금융회사일련번호",
    "출금계좌일련번호",
    "입금금융회사일련번호",
    "입금계좌일련번호",
    "거래금액",
    "매체구분",
    "자금구분",
    TARGET_COLUMN,
]


def load_info(info_path: Path) -> dict:
    """JSON 파일을 로드하여 딕셔너리로 반환합니다."""
    with info_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _load_data_info(dataset_name: str) -> dict:
    """데이터셋 이름을 기반으로 info.json 파일을 찾아 로드합니다."""
    candidate_path = Path(DATA_DIR) / dataset_name / INFO_FILE
    if candidate_path.exists():
        return load_info(candidate_path)
    return {}


def _generate_column_names(n_features: Optional[int], prefix: str) -> list[str]:
    """피처 개수를 기반으로 컬럼 이름을 자동 생성합니다."""
    if n_features is None:
        return []
    return [f"{prefix}_{i}" for i in range(int(n_features))]


def resolve_columns(
    info: dict, exp_dir: Path
) -> tuple[list[str], list[str], list[str]]:
    """
    info.json에서 컬럼 정보를 해결합니다.
    
    해결 순서:
    1. info.json에서 직접 추출
    2. data/<dataset_name>/info.json에서 로드
    3. n_num_features, n_cat_features로 자동 생성
    4. column_names가 없으면 num_cols + cat_cols + ["y"]로 생성
    
    Args:
        info: info.json에서 로드한 딕셔너리
        exp_dir: 실험 디렉토리 경로
    
    Returns:
        (num_cols, cat_cols, column_names) 튜플
    """
    num_cols = info.get("num_cols") or []
    cat_cols = info.get("cat_cols") or []
    column_names = info.get("column_names") or []

    # 데이터셋 이름을 기반으로 추가 정보 로드 시도
    data_info = {}
    dataset_name = info.get("name")
    if dataset_name:
        data_info = _load_data_info(dataset_name)

    # 컬럼 정보가 없으면 data_info에서 가져오기
    if not num_cols:
        num_cols = data_info.get("num_cols") or []
    if not cat_cols:
        cat_cols = data_info.get("cat_cols") or []
    if not column_names:
        column_names = data_info.get("column_names") or []

    # 피처 개수로 자동 생성
    if not num_cols:
        n_num = info.get("n_num_features") or data_info.get("n_num_features")
        num_cols = _generate_column_names(n_num, "num")
    if not cat_cols:
        n_cat = info.get("n_cat_features") or data_info.get("n_cat_features")
        cat_cols = _generate_column_names(n_cat, "cat")

    # column_names가 없으면 자동 생성
    if not column_names and (num_cols or cat_cols):
        column_names = list(num_cols) + list(cat_cols) + [TARGET_COLUMN]

    return num_cols, cat_cols, column_names


def _load_single_npy(file_path: Path, allow_pickle: bool) -> Optional[np.ndarray]:
    """
    단일 npy 파일을 로드하고 필요시 reshape합니다.
    
    allow_pickle=False로 로드 실패 시 자동으로 allow_pickle=True로 재시도합니다.
    """
    if not file_path.exists():
        return None
    
    try:
        arr = np.load(file_path, allow_pickle=allow_pickle)
    except ValueError as e:
        # "Object arrays cannot be loaded when allow_pickle=False" 오류 처리
        if "allow_pickle" in str(e) and not allow_pickle:
            # pickle이 필요한 경우 자동으로 재시도
            arr = np.load(file_path, allow_pickle=True)
        else:
            raise
    
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    return arr


def load_npy_files(
    base_dir: Path, allow_pickle_cat: bool = False
) -> tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
    """
    디렉토리에서 npy 파일들을 로드합니다.
    
    Args:
        base_dir: npy 파일들이 있는 디렉토리
        allow_pickle_cat: 범주형 데이터 로드 시 allow_pickle 옵션
    
    Returns:
        (x_num, x_cat, y) 튜플. 파일이 없으면 None 반환
    """
    x_num_path = base_dir / NUM_FILE
    x_cat_path = base_dir / CAT_FILE
    y_path = base_dir / TARGET_FILE

    x_num = _load_single_npy(x_num_path, allow_pickle=True)
    x_cat = _load_single_npy(x_cat_path, allow_pickle=allow_pickle_cat)
    y = _load_single_npy(y_path, allow_pickle=True)

    return x_num, x_cat, y


def create_dataframe(
    x_num: Optional[np.ndarray],
    x_cat: Optional[np.ndarray],
    y: Optional[np.ndarray],
    num_cols: Optional[list[str]],
    cat_cols: Optional[list[str]],
    column_names: Optional[list[str]],
) -> pd.DataFrame:
    """
    numpy 배열들을 DataFrame으로 변환합니다.
    
    Args:
        x_num: 수치형 피처 배열
        x_cat: 범주형 피처 배열
        y: 타겟 변수 배열
        num_cols: 수치형 컬럼 이름 리스트
        cat_cols: 범주형 컬럼 이름 리스트
        column_names: 전체 컬럼 이름 리스트
    
    Returns:
        결합된 DataFrame
    
    Raises:
        ValueError: 모든 배열이 None인 경우
    """
    parts = []

    if x_num is not None:
        if not num_cols:
            num_cols = _generate_column_names(x_num.shape[1], "num")
        parts.append(pd.DataFrame(x_num, columns=num_cols))

    if x_cat is not None:
        if not cat_cols:
            cat_cols = _generate_column_names(x_cat.shape[1], "cat")
        parts.append(pd.DataFrame(x_cat, columns=cat_cols))

    if y is not None:
        target_name = column_names[-1] if column_names else TARGET_COLUMN
        parts.append(pd.DataFrame(y, columns=[target_name]))

    if not parts:
        raise ValueError("데이터가 없습니다. 최소 하나의 npy 파일이 필요합니다.")

    return pd.concat(parts, axis=1)


def reorder_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    DataFrame의 컬럼을 선호 순서로 재정렬합니다.
    
    Args:
        df: 재정렬할 DataFrame
    
    Returns:
        재정렬된 DataFrame
    """
    existing_preferred = [
        col for col in PREFERRED_COLUMN_ORDER if col in df.columns
    ]
    remaining = [col for col in df.columns if col not in existing_preferred]
    return df[existing_preferred + remaining]


def _parse_size_from_dirname(dirname: str) -> Optional[float]:
    """디렉토리 이름에서 크기 배수를 추출합니다."""
    if dirname.startswith("size_") and dirname.endswith("x"):
        try:
            return float(dirname.replace("size_", "").replace("x", ""))
        except ValueError:
            return None
    return None


def find_size_dirs(base_dir: Path) -> list[tuple[float, Path]]:
    """
    크기별 디렉토리들을 찾아서 크기 순으로 정렬합니다.
    
    Args:
        base_dir: 검색할 기본 디렉토리
    
    Returns:
        (크기배수, 디렉토리경로) 튜플 리스트, 크기 순으로 정렬됨
    """
    size_dirs = []
    for item in base_dir.iterdir():
        if not item.is_dir():
            continue
        
        size_mult = _parse_size_from_dirname(item.name)
        if size_mult is not None:
            size_dirs.append((size_mult, item))

    size_dirs.sort(key=lambda x: x[0])
    return size_dirs


def _process_base_directory(
    exp_dir: Path, allow_pickle_cat: bool
) -> tuple[pd.DataFrame, list[str], list[str], list[str]]:
    """
    기본 디렉토리의 데이터를 처리합니다.
    
    Returns:
        (DataFrame, num_cols, cat_cols, column_names) 튜플
    """
    info_path = exp_dir / INFO_FILE
    if not info_path.exists():
        raise FileNotFoundError(f"{INFO_FILE} 파일을 찾을 수 없습니다: {info_path}")

    info = load_info(info_path)
    num_cols, cat_cols, column_names = resolve_columns(info, exp_dir)

    x_num, x_cat, y = load_npy_files(exp_dir, allow_pickle_cat)
    if x_num is None and x_cat is None and y is None:
        raise FileNotFoundError(
            f"기본 디렉토리에 npy 파일이 없습니다: {exp_dir}. "
            f"Expected any of {NUM_FILE}, {CAT_FILE}, {TARGET_FILE}."
        )

    df = create_dataframe(x_num, x_cat, y, num_cols, cat_cols, column_names)
    df = reorder_columns(df)
    return df, num_cols, cat_cols, column_names


def _process_size_directory(
    size_dir: Path,
    allow_pickle_cat: bool,
    default_num_cols: list[str],
    default_cat_cols: list[str],
    default_column_names: list[str],
) -> Optional[pd.DataFrame]:
    """
    크기별 디렉토리의 데이터를 처리합니다.
    
    Returns:
        DataFrame 또는 None (데이터가 없는 경우)
    """
    # 크기별 디렉토리의 info.json 확인 (없으면 기본 디렉토리 것 사용)
    size_info_path = size_dir / INFO_FILE
    if size_info_path.exists():
        size_info = load_info(size_info_path)
        num_cols, cat_cols, column_names = resolve_columns(size_info, size_dir)
    else:
        num_cols, cat_cols, column_names = (
            default_num_cols,
            default_cat_cols,
            default_column_names,
        )

    x_num, x_cat, y = load_npy_files(size_dir, allow_pickle_cat)
    if x_num is None and x_cat is None and y is None:
        return None

    df = create_dataframe(x_num, x_cat, y, num_cols, cat_cols, column_names)
    df = reorder_columns(df)
    return df


def merge_model_data(
    exp_dir: Path, allow_pickle_cat: bool, merge_sizes: bool = False
) -> dict[str, pd.DataFrame]:
    """
    모델 디렉토리의 데이터를 통합합니다.
    
    Args:
        exp_dir: 모델 디렉토리 경로
        allow_pickle_cat: 범주형 데이터 로드 시 allow_pickle 옵션
        merge_sizes: 크기별 디렉토리도 통합할지 여부
    
    Returns:
        {크기명: DataFrame} 딕셔너리. 기본 데이터는 'base' 키로 저장
    """
    # 기본 디렉토리 처리
    base_df, num_cols, cat_cols, column_names = _process_base_directory(
        exp_dir, allow_pickle_cat
    )
    results = {BASE_KEY: base_df}
    print(f"✓ 기본 디렉토리 데이터 통합 완료: {len(base_df)} 행")

    # 크기별 디렉토리 처리
    if merge_sizes:
        size_dirs = find_size_dirs(exp_dir)
        for size_mult, size_dir in size_dirs:
            size_df = _process_size_directory(
                size_dir, allow_pickle_cat, num_cols, cat_cols, column_names
            )
            if size_df is None:
                print(f"Warning: {size_dir}에 npy 파일이 없습니다. 건너뜁니다.")
                continue

            size_key = f"size_{size_mult}x"
            results[size_key] = size_df
            print(f"✓ {size_key} 디렉토리 데이터 통합 완료: {len(size_df)} 행")

    return results


def _generate_output_path(
    size_key: str, exp_root: Path, model: str, custom_out: Optional[str]
) -> Path:
    """출력 CSV 파일 경로를 생성합니다."""
    if custom_out:
        out_path = Path(custom_out)
        if size_key == BASE_KEY:
            return out_path
        return out_path.parent / f"{out_path.stem}_{size_key}{out_path.suffix}"

    # 기본 출력 경로
    if size_key == BASE_KEY:
        return exp_root / f"{model}_syn_data.csv"
    return exp_root / f"{model}_syn_data_{size_key}.csv"


def save_results(
    results: dict[str, pd.DataFrame],
    exp_root: Path,
    model: str,
    custom_out: Optional[str],
) -> None:
    """
    통합된 데이터를 CSV 파일로 저장합니다.
    
    각 크기별로 별도 CSV 파일을 생성합니다.
    
    Args:
        results: {크기명: DataFrame} 딕셔너리
        exp_root: 실험 루트 디렉토리
        model: 모델 이름
        custom_out: 사용자 지정 출력 경로
    """
    for size_key, df in results.items():
        out_path = _generate_output_path(size_key, exp_root, model, custom_out)
        df.to_csv(out_path, index=False)
        print(f"✓ CSV 저장 완료: {out_path} ({len(df)} 행)")


def find_all_models(exp_root: Path) -> list[str]:
    """
    exp_root 디렉토리에서 모든 모델 디렉토리를 찾습니다.
    
    모델 디렉토리 판단 기준:
    - 디렉토리이고
    - info.json 파일이 있거나 npy 파일이 있는 경우
    
    Args:
        exp_root: 실험 루트 디렉토리
    
    Returns:
        모델 이름 리스트
    """
    if not exp_root.exists():
        return []
    
    models = []
    for item in exp_root.iterdir():
        if not item.is_dir():
            continue
        
        # 특정 디렉토리는 제외
        if item.name.startswith(".") or item.name in ["many-exps", "__pycache__"]:
            continue
        
        # info.json 또는 npy 파일이 있으면 모델로 간주
        has_info = (item / INFO_FILE).exists()
        has_npy = any(
            (item / fname).exists()
            for fname in [NUM_FILE, CAT_FILE, TARGET_FILE]
        )
        
        if has_info or has_npy:
            models.append(item.name)
    
    return sorted(models)


def parse_arguments() -> argparse.Namespace:
    """명령줄 인자를 파싱합니다."""
    parser = argparse.ArgumentParser(
        description="합성 데이터를 numpy 배열 형식(.npy)에서 CSV 형식으로 변환합니다."
    )
    parser.add_argument(
        "--exp_root",
        default=DEFAULT_EXP_ROOT,
        help=f"실험 루트 디렉토리 (기본값: {DEFAULT_EXP_ROOT})",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="모델 서브디렉토리 이름 (예: ctgan, ctabgan). --all_models 사용 시 무시됨",
    )
    parser.add_argument(
        "--all_models",
        action="store_true",
        help="exp_root 내의 모든 모델 디렉토리를 자동으로 찾아서 처리",
    )
    parser.add_argument(
        "--out",
        default=None,
        help="출력 CSV 경로. 지정하지 않으면 <exp_root>/<model>_syn_data.csv로 저장",
    )
    parser.add_argument(
        "--merge_sizes",
        action="store_true",
        help="크기별 디렉토리(size_1.0x, size_1.5x 등)의 데이터도 각각 독립적으로 통합",
    )
    return parser.parse_args()


if __name__ == "__main__":
    """메인 함수."""
    args = parse_arguments()
    exp_root = Path(args.exp_root)

    # 모든 모델 처리 모드
    if args.all_models:
        models = find_all_models(exp_root)
        if not models:
            print(f"경고: {exp_root}에서 모델 디렉토리를 찾을 수 없습니다.")
            exit(1)
        
        print(f"총 {len(models)}개 모델 발견: {', '.join(models)}")
        print("=" * 80)
        
        for i, model in enumerate(models, 1):
            print(f"\n[{i}/{len(models)}] 모델 처리 중: {model}")
            print("-" * 80)
            
            exp_dir = exp_root / model
            allow_pickle_cat = model == DDPM_CB_BEST_MODEL
            
            try:
                # 데이터 통합
                results = merge_model_data(exp_dir, allow_pickle_cat, args.merge_sizes)
                
                # CSV 저장
                save_results(
                    results,
                    exp_root,
                    model,
                    None,  # all_models 모드에서는 custom_out 사용 안 함
                )
                print(f"✓ {model} 처리 완료\n")
            except Exception as e:
                print(f"✗ {model} 처리 중 오류 발생: {e}\n")
                continue
        
        print("=" * 80)
        print("모든 모델 처리 완료!")
    
    # 단일 모델 처리 모드
    else:
        model = args.model or DEFAULT_MODEL
        exp_dir = exp_root / model
        
        if not exp_dir.exists():
            raise FileNotFoundError(f"모델 디렉토리를 찾을 수 없습니다: {exp_dir}")

        allow_pickle_cat = model == DDPM_CB_BEST_MODEL

        print(f"데이터 통합 시작: {exp_dir}")
        print(f"크기별 디렉토리 통합: {'예' if args.merge_sizes else '아니오'}")

        # 데이터 통합
        results = merge_model_data(exp_dir, allow_pickle_cat, args.merge_sizes)

        # CSV 저장
        save_results(
            results,
            exp_root,
            model,
            args.out,
        )
