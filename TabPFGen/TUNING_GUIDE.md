# TabPFGen 하이퍼파라미터 튜닝 가이드

## 개요

TabPFGen의 품질 결과를 분석한 결과, Downstream Task 성능은 우수하지만 통계적 유사성에서 개선이 필요합니다. 이를 위해 하이퍼파라미터 튜닝 전략을 개선했습니다.

## 현재 튜닝 중인 하이퍼파라미터

### 1. 핵심 SGLD 파라미터

#### `n_sgld_steps` (SGLD 반복 횟수)
- **의미**: 합성 데이터를 생성하기 위한 refinement step의 횟수
- **기본값**: 1000 (tutorial/코드 기본값)
- **현재 범위**: [500, 1000, 1500, 2000, 2500]
- **영향**: 
  - 높을수록: 더 정제된 데이터, 하지만 시간 증가
  - 낮을수록: 빠른 생성, 하지만 품질 저하 가능
- **참고**: 기본값 1000을 포함하여 튜닝 권장

#### `sgld_step_size` (SGLD Step Size)
- **의미**: 각 refinement step의 크기
- **기본값**: 0.01 (tutorial/코드 기본값)
- **현재 범위**: [0.0001, 0.2] (log-uniform)
- **영향**:
  - 높을수록: 빠른 수렴, 하지만 불안정할 수 있음
  - 낮을수록: 안정적이지만 느린 수렴
- **참고**: 기본값 0.01을 중심으로 튜닝 권장

#### `sgld_noise_scale` (SGLD Noise Scale)
- **의미**: 각 step에 추가되는 노이즈의 크기
- **기본값**: 0.01 (tutorial/코드 기본값)
- **현재 범위**: [0.0001, 0.2] (log-uniform)
- **영향**:
  - 높을수록: 더 다양한 샘플, 하지만 노이즈 증가
  - 낮을수록: 더 일관된 샘플, 하지만 다양성 감소
- **참고**: 기본값 0.01을 중심으로 튜닝 권장

### 2. 샘플링 파라미터

#### `frac_samples` (샘플 수 배율)
- **의미**: 원본 데이터 대비 생성할 샘플 수의 배율
- **현재 범위**: [-1, 2] (정수)
- **계산**: `num_samples = train_size * (2 ** frac_samples)`
- **영향**:
  - 높을수록: 더 많은 샘플 생성, 시간 증가
  - 낮을수록: 적은 샘플, 빠른 생성
- **권장**: 0-1 (원본과 동일 또는 2배)

### 3. Task-Specific 파라미터 (새로 추가)

#### `balance_classes` (Classification 전용)
- **의미**: 합성 데이터에서 클래스를 균형있게 생성할지 여부
- **현재 범위**: [True, False]
- **영향**:
  - `True`: 클래스 불균형 문제 완화, 하지만 원본 분포와 다를 수 있음
  - `False`: 원본 분포 유지, 하지만 불균형 문제 지속
- **권장**: 클래스 불균형이 심하면 `True`, 원본 분포 보존이 중요하면 `False`

#### `use_quantiles` (Regression 전용)
- **의미**: Regression에서 quantile 예측을 사용할지 여부
- **현재 범위**: [True, False]
- **영향**:
  - `True`: 더 다양한 target 값 생성, 통계적 유사성 향상 가능
  - `False`: 중앙값 예측만 사용, 빠르지만 다양성 감소
- **권장**: 통계적 유사성이 중요하면 `True`

## 추가로 발견한 튜닝 가능한 파라미터 (코드 수정 필요)

다음 파라미터들은 현재 코드에서 하드코딩되어 있어 튜닝하려면 코드 수정이 필요합니다:

### 1. 초기화 노이즈 스케일 (`init_noise_scale`)
- **현재 값**: 0.01 (하드코딩)
- **위치**: `tabpfgen.py` line 408, 419
- **의미**: 초기 합성 샘플 생성 시 추가하는 노이즈의 크기
- **튜닝 가능 범위**: [0.001, 0.1]
- **영향**: 초기 샘플의 다양성에 영향

### 2. Regression Stratified Sampling 파라미터

#### `n_strata` (Stratum 개수)
- **현재 값**: 10 (하드코딩)
- **위치**: `tabpfgen.py` line 504
- **의미**: Target 값을 나누는 구간의 개수
- **튜닝 가능 범위**: [5, 20]
- **영향**: Target 값의 분포 커버리지에 영향

#### `stratum_noise_scale` (Stratum 노이즈 스케일)
- **현재 값**: 0.1 (하드코딩)
- **위치**: `tabpfgen.py` line 524
- **의미**: 각 stratum 내에서 추가하는 노이즈의 크기
- **튜닝 가능 범위**: [0.01, 0.5]
- **영향**: Stratum 내 다양성에 영향

### 3. Adaptive Step Size 파라미터 (Regression 전용)

#### `adaptive_decay_rate` (Step Size 감소율)
- **현재 값**: 0.9 (하드코딩)
- **위치**: `tabpfgen.py` line 535
- **의미**: 매 100 step마다 step size를 곱하는 값
- **튜닝 가능 범위**: [0.85, 0.99]
- **영향**: 학습 후반부 안정성에 영향

#### `adaptive_decay_freq` (Decay 주기)
- **현재 값**: 100 (하드코딩)
- **위치**: `tabpfgen.py` line 534
- **의미**: Step size를 감소시키는 주기
- **튜닝 가능 범위**: [50, 200]
- **영향**: Step size 감소 속도에 영향

## 개선된 튜닝 전략

### 1. Multi-Objective 최적화

기존에는 Downstream Task 성능만 최적화했지만, 이제는 다음 세 가지 옵션을 제공합니다:

- **`utility`**: Downstream Task 성능만 최적화 (기존 방식)
- **`statistical`**: 통계적 유사성만 최적화
- **`composite`**: 두 목표를 가중합으로 최적화 (권장)

### 2. 통계적 유사성 메트릭 통합

다음 메트릭들을 종합하여 통계적 유사성 점수를 계산합니다:

- DWM (Dimension-Wise Means)
- pMSE (Propensity MSE)
- Hellinger Distance
- JSD (Jensen-Shannon Divergence)
- KL Divergence
- qMSE (Quantile MSE)
- Mutual Information Diff
- Entropy Diff
- Conditional Entropy Diff
- Theil's U Diff

각 메트릭은 0~1 범위로 정규화되고, 중요도에 따라 가중 평균됩니다.

### 3. 평가 메트릭 개선

- F1 Score 대신 **ROC-AUC** 사용 (클래스 불균형에 더 강건)
- 여러 seed로 반복 평가하여 안정성 확보

## 사용 방법

### 기본 사용 (Composite Objective)

```bash
python tune_tabpfgen_improved.py data/orig-micro/ 66528 synthetic cuda:0 \
    --objective composite --utility_weight 0.6
```

### Utility만 최적화 (기존 방식)

```bash
python tune_tabpfgen_improved.py data/orig-micro/ 66528 synthetic cuda:0 \
    --objective utility
```

### Statistical만 최적화

```bash
python tune_tabpfgen_improved.py data/orig-micro/ 66528 synthetic cuda:0 \
    --objective statistical
```

### Utility Weight 조정

```bash
# 통계적 유사성에 더 중점
python tune_tabpfgen_improved.py data/orig-micro/ 66528 synthetic cuda:0 \
    --objective composite --utility_weight 0.4

# Downstream Task에 더 중점
python tune_tabpfgen_improved.py data/orig-micro/ 66528 synthetic cuda:0 \
    --objective composite --utility_weight 0.7
```

## 예상 효과

### 개선 전 (현재 결과)
- **Downstream Task**: ROC-AUC 0.83 (우수)
- **통계적 유사성**: KL Divergence 19.47, JSD 0.48 (개선 필요)

### 개선 후 (예상)
- **Downstream Task**: ROC-AUC 0.80-0.85 (유지)
- **통계적 유사성**: KL Divergence < 10, JSD < 0.3 (개선)

## 향후 개선 사항

1. **초기화 파라미터 튜닝**: `init_noise_scale` 추가
2. **Regression 전용 파라미터**: `n_strata`, `stratum_noise_scale` 추가
3. **Adaptive Step Size**: `adaptive_decay_rate`, `adaptive_decay_freq` 추가
4. **Early Stopping**: 통계적 유사성 기반 early stopping 추가
5. **Pruning**: Optuna의 pruning 기능 활용

## 참고 자료

- TabPFGen 튜토리얼: `tutorial/` 디렉토리
- TabPFGen 소스 코드: `src/tabpfgen/tabpfgen.py`
- 평가 결과: `exp/orig-micro/tabpfgen/eval_*.json`

