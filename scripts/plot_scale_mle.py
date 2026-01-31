"""
Size별 메트릭 plot 생성 스크립트

사용법:
    python scripts/plot_size_metrics.py --base_dir exp/orig-micro/ddpm_cb_best --eval_type synthetic
"""
import argparse
import json
import matplotlib.pyplot as plt
from pathlib import Path


def plot_size_metrics(base_parent_dir, eval_type='synthetic'):
    """
    size별 eval_catboost.json에서 test 결과를 읽어서 메트릭 변화를 plot으로 그리기
    
    Args:
        base_parent_dir: 기본 디렉토리 (예: exp/orig-micro/ddpm_cb_best/)
        eval_type: 평가 타입 ('synthetic' 또는 'real', 기본값: 'synthetic')
    """
    base_dir = Path(base_parent_dir)
    
    # base_dir에서 마지막 디렉토리 이름 추출 (예: ddpm_cb_best)
    dir_suffix = base_dir.name
    
    # size별 디렉토리 찾기
    size_dirs = []
    for item in base_dir.iterdir():
        if item.is_dir() and item.name.startswith('size_') and item.name.endswith('x'):
            try:
                size_mult = float(item.name.replace('size_', '').replace('x', ''))
                size_dirs.append((size_mult, item))
            except ValueError:
                continue
    
    if not size_dirs:
        print(f"Warning: size_*x 디렉토리를 찾을 수 없습니다: {base_dir}")
        return
    
    # 크기 순으로 정렬
    size_dirs.sort(key=lambda x: x[0])
    
    # 메트릭 데이터 수집
    sizes = []
    metrics = {
        'accuracy': [],
        'f1_neg': [],
        'f1_pos': [],
        'roc_auc': []
    }
    metric_stds = {
        'accuracy': [],
        'f1_neg': [],
        'f1_pos': [],
        'roc_auc': []
    }
    
    for size_mult, size_dir in size_dirs:
        eval_json_path = size_dir / "eval_catboost.json"
        
        if not eval_json_path.exists():
            print(f"Warning: {eval_json_path} 파일을 찾을 수 없습니다. 건너뜁니다.")
            continue
        
        # JSON 파일 읽기
        with open(eval_json_path, 'r') as f:
            eval_data = json.load(f)
        
        # eval_type과 test 데이터 추출
        if eval_type not in eval_data:
            print(f"Warning: {eval_type} 키를 찾을 수 없습니다: {eval_json_path}")
            continue
        
        if 'test' not in eval_data[eval_type]:
            print(f"Warning: test 키를 찾을 수 없습니다: {eval_json_path}")
            continue
        
        test_metrics = eval_data[eval_type]['test']
        
        # 메트릭 추출
        sizes.append(size_mult)
        metrics['accuracy'].append(test_metrics.get('acc-mean', 0.0))
        metrics['f1_neg'].append(test_metrics.get('f1_0-mean', 0.0))
        metrics['f1_pos'].append(test_metrics.get('f1_1-mean', 0.0))
        metrics['roc_auc'].append(test_metrics.get('roc_auc-mean', 0.0))
        
        # 표준편차 추출
        metric_stds['accuracy'].append(test_metrics.get('acc-std', 0.0))
        metric_stds['f1_neg'].append(test_metrics.get('f1_0-std', 0.0))
        metric_stds['f1_pos'].append(test_metrics.get('f1_1-std', 0.0))
        metric_stds['roc_auc'].append(test_metrics.get('roc_auc-std', 0.0))
    
    if not sizes:
        print("Warning: 유효한 메트릭 데이터를 찾을 수 없습니다.")
        return
    
    # 각 메트릭별로 별도의 plot 생성 및 저장
    plot_configs = [
        {
            'key': 'accuracy',
            'title': 'Accuracy',
            'ylabel': 'Accuracy',
            'color': 'blue',
            'label': 'Accuracy'
        },
        {
            'key': 'f1_neg',
            'title': 'F1-score (Negative Class)',
            'ylabel': 'F1-score (Negative)',
            'color': 'green',
            'label': 'F1-score (Neg)'
        },
        {
            'key': 'f1_pos',
            'title': 'F1-score (Positive Class)',
            'ylabel': 'F1-score (Positive)',
            'color': 'red',
            'label': 'F1-score (Pos)'
        },
        {
            'key': 'roc_auc',
            'title': 'ROC-AUC',
            'ylabel': 'ROC-AUC',
            'color': 'purple',
            'label': 'ROC-AUC'
        }
    ]
    
    saved_paths = []
    for config in plot_configs:
        key = config['key']
        fig, ax = plt.subplots(figsize=(4, 3))
        
        ax.plot(sizes, metrics[key], 'o-', linewidth=2, markersize=8, 
               color=config['color'], label=config['label'])
        
        if any(metric_stds[key]):
            ax.errorbar(sizes, metrics[key], yerr=metric_stds[key], 
                       fmt='none', capsize=5, capthick=2, alpha=0.5, color=config['color'])
        
        # 각 점 위에 수치 텍스트 추가
        for i, (size, value) in enumerate(zip(sizes, metrics[key])):
            # 표준편차가 있으면 함께 표시
            if metric_stds[key][i] > 0:
                text = f'{value:.4f}\n±{metric_stds[key][i]:.4f}'
            else:
                text = f'{value:.4f}'
            
            # 점 위에 텍스트 추가 (y 오프셋으로 위치 조정)
            ax.text(size, value, text, 
                   ha='center', va='bottom', fontsize=9,
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7, edgecolor='none'))
        
        ax.set_xlabel('Synthetic Data Scale', fontsize=10)
        ax.set_ylabel(config['ylabel'], fontsize=10)
        # ax.set_title(f'Size별 {config["title"]} 변화 ({eval_type})', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        # ax.legend(fontsize=11)
        
        plt.tight_layout()
        
        # 파일명에서 특수문자 제거
        filename_key = key.replace('_', '_').replace('-', '_')
        output_path = base_dir / f'scale_mle_{filename_key}_{eval_type}_{dir_suffix}.png'
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        saved_paths.append(output_path)
        print(f"메트릭 plot 저장: {output_path}")
        
        plt.close()
    
    print(f"\n총 {len(saved_paths)}개의 plot 파일이 저장되었습니다.")
    
    # 데이터 출력
    print("\n" + "=" * 100)
    print(f"Size별 Test 메트릭 ({eval_type}):")
    print("=" * 100)
    print(f"{'Size':<10} {'Accuracy':<12} {'F1-Neg':<12} {'F1-Pos':<12} {'ROC-AUC':<12}")
    print("-" * 100)
    for i, size in enumerate(sizes):
        print(f"{size:<10.1f} {metrics['accuracy'][i]:<12.4f} {metrics['f1_neg'][i]:<12.4f} "
              f"{metrics['f1_pos'][i]:<12.4f} {metrics['roc_auc'][i]:<12.4f}")
    print("=" * 100)
    
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description='Size별 메트릭 plot 생성',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
예제:
  python scripts/plot_size_metrics.py \\
      --base_dir exp/orig-micro/ddpm_cb_best \\
      --eval_type synthetic
        """
    )
    parser.add_argument('--base_dir', type=str, required=True,
                        help='기본 디렉토리 경로 (예: exp/orig-micro/ddpm_cb_best)')
    parser.add_argument('--eval_type', type=str, default='synthetic',
                        choices=['synthetic', 'real'],
                        help='평가 타입 (synthetic 또는 real, 기본값: synthetic)')
    
    args = parser.parse_args()
    
    print("=" * 100)
    print("Size별 메트릭 plot 생성")
    print("=" * 100)
    print(f"기본 디렉토리: {args.base_dir}")
    print(f"평가 타입: {args.eval_type}")
    print("=" * 100)
    
    plot_size_metrics(args.base_dir, eval_type=args.eval_type)


if __name__ == '__main__':
    main()

