"""
Fraud Label Fidelity Analysis

합성데이터에서 fraud label 분포 보존 여부 및 조건부 패턴 비교 분석
- exp/orig-micro 디렉토리의 각 생성 모델별 합성데이터와 원본데이터 비교
"""

import os
import json
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from pandas.api.types import is_numeric_dtype

# 한글 폰트 설정
plt.rcParams["font.family"] = "NanumGothic"
plt.rcParams["axes.unicode_minus"] = False

# lib 모듈 import
import sys
sys.path.append(str(Path(__file__).parent.parent))
from lib.data import read_pure_data


def numpy_to_dataframe(X_num, X_cat, y, num_cols=None, cat_cols=None, target_col='y'):
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


def load_json(path):
    """JSON 파일 로드"""
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def get_column_names(data_path):
    """데이터 경로에서 컬럼 이름 정보 가져오기"""
    info_path = Path(data_path) / 'info.json'
    if info_path.exists():
        info = load_json(info_path)
        num_cols = info.get('num_cols', None)
        cat_cols = info.get('cat_cols', None)
        return num_cols, cat_cols
    return None, None


# =========================================================
# Plotting functions
# =========================================================

def plot_label_distribution_comparison(
    df_orig: pd.DataFrame,
    df_syn: pd.DataFrame,
    label_col: str,
    *,
    figsize: tuple,
    out_path: str,
    color_map: dict,
    label_map: dict,
    model_name: str,
):
    """
    전체 레이블 분포 비교 (Fraud label 분포 보존 여부)
    """
    orig_counts = df_orig[label_col].value_counts().sort_index()
    syn_counts = df_syn[label_col].value_counts().sort_index()
    
    orig_props = orig_counts / len(df_orig)
    syn_props = syn_counts / len(df_syn)
    
    labels = sorted(set(orig_counts.index) | set(syn_counts.index))
    x = np.arange(len(labels))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # 비율 표시
    orig_vals = [orig_props.get(l, 0) for l in labels]
    syn_vals = [syn_props.get(l, 0) for l in labels]
    
    bars1 = ax.bar(x - width/2, orig_vals, width, label='Original', 
                   color='#4A90E2', alpha=0.8, edgecolor='black', linewidth=1)
    bars2 = ax.bar(x + width/2, syn_vals, width, label='Synthetic', 
                   color='#E24A4A', alpha=0.8, edgecolor='black', linewidth=1, hatch='//')
    
    # 값 표시
    for i, (orig_val, syn_val) in enumerate(zip(orig_vals, syn_vals)):
        ax.text(i - width/2, orig_val + 0.01, f'{orig_val:.3f}', 
                ha='center', va='bottom', fontsize=9)
        ax.text(i + width/2, syn_val + 0.01, f'{syn_val:.3f}', 
                ha='center', va='bottom', fontsize=9)
    
    ax.set_xlabel('Label', fontsize=12)
    ax.set_ylabel('Proportion', fontsize=12)
    # ax.set_title(f'Fraud Label Distribution Comparison ({model_name})', fontsize=13, fontweight='bold')
    # ax.set_xticks(x)
    # ax.set_xticklabels([label_map.get(l, str(l)) for l in labels])
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis='y', linestyle=':')
    ax.set_ylim([0, max(max(orig_vals), max(syn_vals)) * 1.2])
    
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()

    # CSV 파일로 저장
    csv_path = out_path.replace('.png', '.csv')
    csv_data = []
    for label in labels:
        orig_count = orig_counts.get(label, 0)
        syn_count = syn_counts.get(label, 0)
        orig_prop = orig_props.get(label, 0)
        syn_prop = syn_props.get(label, 0)
        diff = abs(orig_prop - syn_prop)
        csv_data.append({
            'Label': label_map.get(label, str(label)),
            'Label_Value': label,
            'Original_Count': int(orig_count),
            'Original_Proportion': float(orig_prop),
            'Synthetic_Count': int(syn_count),
            'Synthetic_Proportion': float(syn_prop),
            'Difference': float(diff),
        })
    
    df_csv = pd.DataFrame(csv_data)
    df_csv.to_csv(csv_path, index=False, encoding='utf-8-sig')
    
    # 통계 출력
    print(f"\n[{model_name}] Fraud Label Distribution Comparison")
    print("="*60)
    for label in labels:
        orig_count = orig_counts.get(label, 0)
        syn_count = syn_counts.get(label, 0)
        orig_prop = orig_props.get(label, 0)
        syn_prop = syn_props.get(label, 0)
        diff = abs(orig_prop - syn_prop)
        print(f"{label_map.get(label, str(label))}:")
        print(f"  Original: {orig_count:,} ({orig_prop:.4f})")
        print(f"  Synthetic: {syn_count:,} ({syn_prop:.4f})")
        print(f"  Difference: {diff:.4f}")
    print("="*60 + "\n")
    print(f"CSV 파일 저장: {csv_path}")


def plot_label_conditioned_feature_orig_vs_syn(
    df_orig_0: pd.DataFrame,
    df_orig_1: pd.DataFrame,
    df_syn_0: pd.DataFrame,
    df_syn_1: pd.DataFrame,
    feature: str,
    *,
    bins: int,
    figsize: tuple,
    alpha: float,
    top_k: int,
    clip_q: float,
    force_categorical: set,
    feature_alias: dict,
    out_path: str,
    color_map: dict,
    label_map: dict,
    model_name: str,
):
    """
    원본 vs 합성 데이터의 레이블별 조건부 분포 비교
    
    조건부 패턴(금액, 시간대, 매체 등) 비교
    """
    # 각 그룹의 데이터 추출
    orig_0 = df_orig_0[feature].dropna()
    orig_1 = df_orig_1[feature].dropna()
    syn_0 = df_syn_0[feature].dropna()
    syn_1 = df_syn_1[feature].dropna()

    # 빈 데이터 체크
    if len(orig_0) == 0 or len(orig_1) == 0 or len(syn_0) == 0 or len(syn_1) == 0:
        print(f"[SKIP] {feature}: one of the groups is empty")
        return

    fig, ax = plt.subplots(figsize=figsize)

    # ---------------- Numeric ----------------
    if is_numeric_dtype(df_orig_0[feature]) and feature not in force_categorical:
        from scipy.stats import gaussian_kde
        
        # 모든 데이터를 합쳐서 범위 결정
        all_data = pd.concat([orig_0, orig_1, syn_0, syn_1])
        upper = np.percentile(all_data, clip_q)
        
        orig_0_clipped = orig_0.clip(upper=upper)
        orig_1_clipped = orig_1.clip(upper=upper)
        syn_0_clipped = syn_0.clip(upper=upper)
        syn_1_clipped = syn_1.clip(upper=upper)

        min_val = min(orig_0_clipped.min(), orig_1_clipped.min(), syn_0_clipped.min(), syn_1_clipped.min())
        max_val = max(orig_0_clipped.max(), orig_1_clipped.max(), syn_0_clipped.max(), syn_1_clipped.max())

        if min_val == max_val:
            plt.close()
            return

        # KDE 플롯을 위한 x 범위 설정
        x_range = np.linspace(min_val, max_val, 200)
        
        # KDE 계산 및 플롯
        try:
            # 원본 데이터 (실선)
            if len(orig_0_clipped) > 1:
                kde_orig_0 = gaussian_kde(orig_0_clipped)
                kde_orig_0_values = kde_orig_0(x_range)
                ax.plot(x_range, kde_orig_0_values, 
                       label=f"Original {label_map[0]}", 
                       color=color_map[0], linestyle='-', linewidth=1, alpha=alpha)
                ax.fill_between(x_range, kde_orig_0_values, alpha=0.2, color=color_map[0])
            
            if len(orig_1_clipped) > 1:
                kde_orig_1 = gaussian_kde(orig_1_clipped)
                kde_orig_1_values = kde_orig_1(x_range)
                ax.plot(x_range, kde_orig_1_values, 
                       label=f"Original {label_map[1]}", 
                       color=color_map[1], linestyle='-', linewidth=1, alpha=alpha)
                ax.fill_between(x_range, kde_orig_1_values, alpha=0.2, color=color_map[1])
            
            # 합성 데이터 (점선)
            if len(syn_0_clipped) > 1:
                kde_syn_0 = gaussian_kde(syn_0_clipped)
                kde_syn_0_values = kde_syn_0(x_range)
                ax.plot(x_range, kde_syn_0_values, 
                       label=f"Synthetic {label_map[0]}", 
                       color=color_map[0], linestyle='--', linewidth=1, alpha=alpha*0.8)
            
            if len(syn_1_clipped) > 1:
                kde_syn_1 = gaussian_kde(syn_1_clipped)
                kde_syn_1_values = kde_syn_1(x_range)
                ax.plot(x_range, kde_syn_1_values, 
                       label=f"Synthetic {label_map[1]}", 
                       color=color_map[1], linestyle='--', linewidth=1, alpha=alpha*0.8)
        except Exception as e:
            # KDE 실패 시 히스토그램으로 대체
            print(f"[WARNING] {feature}: KDE 계산 실패, 히스토그램 사용 - {e}")
            bins_arr = np.linspace(min_val, max_val, bins)
            ax.hist(orig_0_clipped, bins=bins_arr, density=True, alpha=alpha, 
                    label=f"Original {label_map[0]}", 
                    color=color_map[0], linestyle='-', linewidth=1, histtype='step')
            ax.hist(orig_1_clipped, bins=bins_arr, density=True, alpha=alpha, 
                    label=f"Original {label_map[1]}", 
                    color=color_map[1], linestyle='-', linewidth=1, histtype='step')
            ax.hist(syn_0_clipped, bins=bins_arr, density=True, alpha=alpha*0.7, 
                    label=f"Synthetic {label_map[0]}", 
                    color=color_map[0], linestyle='--', linewidth=1, histtype='step')
            ax.hist(syn_1_clipped, bins=bins_arr, density=True, alpha=alpha*0.7, 
                    label=f"Synthetic {label_map[1]}", 
                    color=color_map[1], linestyle='--', linewidth=1, histtype='step')

    # ---------------- Categorical ----------------
    else:
        # anal_orig_hist.py와 동일한 방식: 원본 데이터의 레이블 0에서 상위 k개 카테고리 선택
        vc = orig_0.value_counts()
        top_cats = vc.head(top_k).index

        orig_0_processed = orig_0.where(orig_0.isin(top_cats), "OTHER")
        orig_1_processed = orig_1.where(orig_1.isin(top_cats), "OTHER")
        syn_0_processed = syn_0.where(syn_0.isin(top_cats), "OTHER")
        syn_1_processed = syn_1.where(syn_1.isin(top_cats), "OTHER")
        
        # 모든 값을 문자열로 변환 (where()로 인해 문자열과 숫자가 섞일 수 있음)
        orig_0_processed = orig_0_processed.astype(str)
        orig_1_processed = orig_1_processed.astype(str)
        syn_0_processed = syn_0_processed.astype(str)
        syn_1_processed = syn_1_processed.astype(str)
        
        # NaN을 문자열로 변환된 경우 처리
        orig_0_processed = orig_0_processed.replace('nan', 'OTHER')
        orig_1_processed = orig_1_processed.replace('nan', 'OTHER')
        syn_0_processed = syn_0_processed.replace('nan', 'OTHER')
        syn_1_processed = syn_1_processed.replace('nan', 'OTHER')

        cats = sorted(set(orig_0_processed.unique()) | set(orig_1_processed.unique()) | 
                     set(syn_0_processed.unique()) | set(syn_1_processed.unique()))
        x = np.arange(len(cats))

        p_orig_0 = orig_0_processed.value_counts(normalize=True).reindex(cats, fill_value=0)
        p_orig_1 = orig_1_processed.value_counts(normalize=True).reindex(cats, fill_value=0)
        p_syn_0 = syn_0_processed.value_counts(normalize=True).reindex(cats, fill_value=0)
        p_syn_1 = syn_1_processed.value_counts(normalize=True).reindex(cats, fill_value=0)

        width = 0.2
        x_pos = x - width * 1.5
        
        # 원본 데이터
        ax.bar(x_pos, p_orig_0, width=width, label=f"Original {label_map[0]}", 
               color=color_map[0], alpha=alpha, edgecolor='black', linewidth=0.5)
        ax.bar(x_pos + width, p_orig_1, width=width, label=f"Original {label_map[1]}", 
               color=color_map[1], alpha=alpha, edgecolor='black', linewidth=0.5)
        
        # 합성 데이터
        ax.bar(x_pos + width * 2, p_syn_0, width=width, label=f"Synthetic {label_map[0]}", 
               color=color_map[0], alpha=alpha*0.7, edgecolor='black', linewidth=0.5, hatch='//')
        ax.bar(x_pos + width * 3, p_syn_1, width=width, label=f"Synthetic {label_map[1]}", 
               color=color_map[1], alpha=alpha*0.7, edgecolor='black', linewidth=0.5, hatch='//')

        # ax.set_xticks(x)
        # ax.set_xticklabels(cats, rotation=45, ha="right", fontsize=8)

    alias = feature_alias.get(feature, feature)
    ax.set_xlabel(alias, fontsize=11)
    ax.set_ylabel("Probability", fontsize=12)
    # ax.set_title(f"{alias} Distribution by Label ({model_name})", fontsize=12, fontweight='bold')
    ax.legend(fontsize=9, loc='best')
    ax.grid(True, alpha=0.3, linestyle=':')

    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_all_features_orig_vs_syn(
    df_orig, df_syn, features, label_col, out_dir, model_name, **config
):
    """
    원본과 합성 데이터의 레이블별 조건부 분포 비교
    """
    df_orig_0 = df_orig[df_orig[label_col] == 0]
    df_orig_1 = df_orig[df_orig[label_col] == 1]
    df_syn_0 = df_syn[df_syn[label_col] == 0]
    df_syn_1 = df_syn[df_syn[label_col] == 1]

    for feature in features:
        if feature not in df_orig.columns or feature not in df_syn.columns:
            print(f"[SKIP] {feature}: not found in data")
            continue
            
        alias = config['feature_alias'].get(feature, feature)
        out_path = os.path.join(out_dir, f"label_conditional_{alias}.png")
        
        plot_label_conditioned_feature_orig_vs_syn(
            df_orig_0=df_orig_0,
            df_orig_1=df_orig_1,
            df_syn_0=df_syn_0,
            df_syn_1=df_syn_1,
            feature=feature,
            out_path=out_path,
            model_name=model_name,
            **config,
        )


# =========================================================
# Main evaluation function
# =========================================================

def evaluate_model(
    model_dir: Path,
    real_data_path: Path,
    output_dir: Path = None,
    config: dict = None,
):
    """
    단일 모델에 대한 평가 수행
    
    Args:
        model_dir: 모델 디렉토리 경로
        real_data_path: 실제 데이터 경로 (info.json이 있는 디렉토리, 원본 데이터도 여기서 로드)
        output_dir: 출력 디렉토리 (None이면 model_dir/fraud_fidelity 사용)
        config: 설정 딕셔너리
    """
    model_name = model_dir.name
    print(f"\n{'='*80}")
    print(f"Evaluating model: {model_name}")
    print(f"{'='*80}")
    
    # 합성 데이터 로드
    if not (model_dir / 'y_train.npy').exists():
        print(f"[SKIP] {model_name}: 합성 데이터 파일이 없습니다 (y_train.npy)")
        return
    
    try:
        X_num_syn, X_cat_syn, y_syn = read_pure_data(str(model_dir))
    except Exception as e:
        print(f"[ERROR] {model_name}: 합성 데이터 로드 실패 - {e}")
        return
    
    # 컬럼 이름 가져오기
    num_cols, cat_cols = get_column_names(real_data_path)
    
    # 합성 데이터를 DataFrame으로 변환
    df_syn = numpy_to_dataframe(
        X_num_syn, X_cat_syn, y_syn,
        num_cols=num_cols, cat_cols=cat_cols, target_col='y'
    )
    
    # 원본 데이터 로드 (real_data_path에서 train 데이터 사용)
    try:
        X_num_orig, X_cat_orig, y_orig = read_pure_data(str(real_data_path), 'train')
        df_orig = numpy_to_dataframe(
            X_num_orig, X_cat_orig, y_orig,
            num_cols=num_cols, cat_cols=cat_cols, target_col='y'
        )
    except Exception as e:
        print(f"[ERROR] {model_name}: 원본 데이터 로드 실패 - {e}")
        return
    
    # 컬럼 매칭 확인 및 필터링
    common_cols = set(df_orig.columns) & set(df_syn.columns)
    common_cols.discard('y')
    
    if 'y' not in df_syn.columns:
        print(f"[ERROR] {model_name}: 합성 데이터에 레이블 컬럼 'y'가 없습니다")
        return
    
    print(f"원본 데이터: {len(df_orig):,} 샘플, {len(df_orig.columns)} 컬럼")
    print(f"합성 데이터: {len(df_syn):,} 샘플, {len(df_syn.columns)} 컬럼")
    print(f"공통 컬럼: {len(common_cols)}개")
    
    # 출력 디렉토리 생성 (기본값: model_dir/fraud_fidelity)
    if output_dir is None:
        model_output_dir = model_dir / 'fraud_fidelity'
    else:
        model_output_dir = output_dir / model_name
    model_output_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. 레이블 분포 비교
    plot_label_distribution_comparison(
        df_orig=df_orig,
        df_syn=df_syn,
        label_col='y',
        figsize=(6, 4),
        out_path=str(model_output_dir / 'label_distribution.png'),
        color_map=config['color_map'],
        label_map=config['label_map'],
        model_name=model_name,
    )
    
    # 2. 조건부 패턴 비교 (모든 피처)
    key_features = [
        '거래금액',      # 조건부 패턴: 금액
        '거래일자',      # 조건부 패턴: 날짜
        '거래시간대',    # 조건부 패턴: 시간대
        '매체구분',      # 조건부 패턴: 매체
        '자금구분',      # 조건부 패턴: 자금구분
        '입금계좌일련번호',    # 계좌 정보
        '출금계좌일련번호',    # 계좌 정보
        '출금금융회사일련번호',  # 금융회사 정보
        '입금금융회사일련번호',  # 금융회사 정보
    ]
    
    # 공통 컬럼 중에서만 선택
    available_features = [f for f in key_features if f in common_cols]
    
    if available_features:
        plot_all_features_orig_vs_syn(
            df_orig=df_orig,
            df_syn=df_syn,
            features=available_features,
            label_col='y',
            out_dir=str(model_output_dir),
            model_name=model_name,
            **config,
        )
        print(f"✅ {model_name} 평가 완료: {len(available_features)}개 피처 분석")
    else:
        print(f"[WARNING] {model_name}: 분석할 피처가 없습니다")


# =========================================================
# Main
# =========================================================

def main():
    parser = argparse.ArgumentParser(description='Fraud Label Fidelity Analysis')
    parser.add_argument('--exp_dir', type=str, default='exp/orig-micro-retry',
                        help='실험 디렉토리 경로')
    parser.add_argument('--real_data', type=str, default='data/orig-micro-retry',
                        help='실제 데이터 경로 (info.json이 있는 디렉토리, 원본 및 합성 데이터 모두 여기서 로드)')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='결과 출력 디렉토리 (지정하지 않으면 각 모델 디렉토리 내 fraud_fidelity 사용)')
    parser.add_argument('--models', type=str, nargs='+', default=None,
                        help='평가할 모델 리스트 (지정하지 않으면 모든 모델 평가)')
    parser.add_argument('--sizes', type=str, nargs='+', default=['1.0x', '1.5x', '2.0x'],
                        help='평가할 사이즈 리스트 (기본값: 1.0x 1.5x 2.0x)')
    args = parser.parse_args()
    
    # 설정
    config = {
        "bins": 30,
        "figsize": (6, 4),
        "alpha": 0.8,
        "top_k": 20,
        "clip_q": 99.5,
        "force_categorical": {
            "입금계좌일련번호",
            "출금계좌일련번호",
            "출금금융회사일련번호",
            "입금금융회사일련번호",
        },
        "feature_alias": {
            "입금계좌일련번호": "deposit_account",
            "출금계좌일련번호": "withdraw_account",
            "출금금융회사일련번호": "withdraw_bank",
            "입금금융회사일련번호": "deposit_bank",
            "거래금액": "amount",
            "거래일자": "date",
            "거래시간대": "timeband",
            "매체구분": "mediatype",
            "자금구분": "fundtype",
        },
        "color_map": {
            0: "#8EC6E8",      # Benign (파란색)
            1: "#F6BD60",      # Suspicious (주황색)
        },
        "label_map": {
            0: "Benign",
            1: "Suspicious",
        }
    }
    
    exp_dir = Path(args.exp_dir)
    real_data_path = Path(args.real_data)
    output_dir = Path(args.output_dir) if args.output_dir else None
    
    # 경로 확인
    if not exp_dir.exists():
        raise FileNotFoundError(f"실험 디렉토리가 없습니다: {exp_dir}")
    if not real_data_path.exists():
        raise FileNotFoundError(f"실제 데이터 경로가 없습니다: {real_data_path}")
    
    # 모델 디렉토리 찾기
    if args.models:
        model_dirs = [exp_dir / model for model in args.models]
    else:
        # exp_dir의 모든 서브디렉토리 찾기 (y_train.npy 또는 size_* 서브디렉토리가 있는 경우)
        model_dirs = []
        for d in exp_dir.iterdir():
            if d.is_dir():
                # y_train.npy가 있거나 size_* 디렉토리가 있으면 모델 디렉토리로 간주
                has_data = (d / 'y_train.npy').exists()
                has_size_dirs = any((d / f'size_{size}').exists() for size in args.sizes)
                if has_data or has_size_dirs:
                    model_dirs.append(d)
    
    model_dirs = [d for d in model_dirs if d.exists()]
    
    if not model_dirs:
        print("평가할 모델이 없습니다.")
        return
    
    print(f"총 {len(model_dirs)}개 모델 평가 예정:")
    for d in model_dirs:
        print(f"  - {d.name}")
    
    # 각 모델과 사이즈별 평가
    total_evaluated = 0
    for model_dir in sorted(model_dirs):
        model_name = model_dir.name
        print(f"\n{'='*80}")
        print(f"모델: {model_name}")
        print(f"{'='*80}")
        
        # 1. 모델 루트 디렉토리 평가 (y_train.npy가 있는 경우)
        if (model_dir / 'y_train.npy').exists():
            try:
                print(f"\n[{model_name}] 루트 디렉토리 평가 중...")
                evaluate_model(
                    model_dir=model_dir,
                    real_data_path=real_data_path,
                    output_dir=output_dir,
                    config=config,
                )
                total_evaluated += 1
            except Exception as e:
                print(f"[ERROR] {model_name} (루트) 평가 중 오류 발생: {e}")
                import traceback
                traceback.print_exc()
        
        # 2. 각 사이즈별 디렉토리 평가
        for size in args.sizes:
            size_dir = model_dir / f'size_{size}'
            if not size_dir.exists():
                print(f"[SKIP] {model_name}/size_{size}: 디렉토리가 존재하지 않습니다")
                continue
            
            if not (size_dir / 'y_train.npy').exists():
                print(f"[SKIP] {model_name}/size_{size}: y_train.npy 파일이 없습니다")
                continue
            
            try:
                print(f"\n[{model_name}/size_{size}] 평가 중...")
                evaluate_model(
                    model_dir=size_dir,
                    real_data_path=real_data_path,
                    output_dir=output_dir,
                    config=config,
                )
                total_evaluated += 1
            except Exception as e:
                print(f"[ERROR] {model_name}/size_{size} 평가 중 오류 발생: {e}")
                import traceback
                traceback.print_exc()
    
    print(f"\n{'='*80}")
    print(f"✅ 모든 평가 완료! (총 {total_evaluated}개 데이터셋 평가)")
    if output_dir:
        print(f"   결과 저장 위치: {output_dir}")
    else:
        print(f"   결과는 각 모델 디렉토리 내 fraud_fidelity 폴더에 저장되었습니다.")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()

