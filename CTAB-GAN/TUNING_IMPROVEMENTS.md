# CTAB-GAN 하이퍼파라미터 튜닝 개선 사항

## 📋 현재 코드 (`tune_ctabgan.py`) 분석 결과

### 발견된 문제점

#### 1. **평가 메트릭 문제** ⚠️
- **현재**: `get_val_score()` → macro F1-score 사용
- **문제**: 불균형 데이터에서 macro F1은 부적절할 수 있음
- **실제 성능**: ROC-AUC 0.8181, F1 0.5521 (매우 낮음)
- **영향**: 최적화 메트릭과 실제 관심 메트릭(ROC-AUC)이 다름

#### 2. **Pruning 미사용** ⚠️
- **현재**: Optuna pruning 기능 미사용
- **문제**: 나쁜 trial도 전체 평가 완료까지 실행 (시간 낭비)
- **영향**: Trial당 3시간 × 10개 = 30시간 소요

#### 3. **에러 핸들링 부재** ⚠️
- **현재**: Trial 실패 시 처리 로직 없음
- **문제**: 하나의 seed 평가 실패 시 전체 trial 실패 가능
- **영향**: 불안정한 튜닝 프로세스

#### 4. **평가 Seed 수** ⚠️
- **현재**: 고정 5개 seed
- **문제**: 시간이 많이 소요됨 (5개 × 평가 시간)
- **영향**: 탐색 효율 저하

#### 5. **파라미터 범위 제한** ⚠️
- **현재**: Learning rate 0.0001~0.001 (매우 좁음)
- **문제**: 더 좋은 파라미터를 놓칠 수 있음
- **영향**: 탐색 공간 축소

#### 6. **Trial 수 부족** ⚠️
- **현재**: 10개 trial (주석 처리된 35개 있음)
- **문제**: 파라미터 공간이 매우 큰데 탐색이 부족
- **영향**: 최적 파라미터를 찾지 못할 가능성

#### 7. **진행 상황 모니터링 부족** ⚠️
- **현재**: 기본적인 진행 상황만 표시
- **문제**: Pruned/Failed trial 수, 중간 결과 확인 불가
- **영향**: 디버깅 및 분석 어려움

---

## ✅ 개선 사항 (`tune_ctabgan_improved.py`)

### 1. **다양한 평가 메트릭 지원** ✨

```python
--metric roc_auc  # ROC-AUC (기본값, 불균형 데이터에 적합)
--metric balanced_acc  # Balanced Accuracy
--metric f1  # Macro F1-score (기존)
```

**효과**: 실제 관심 메트릭(ROC-AUC)으로 최적화 가능

### 2. **Optuna Pruning 추가** ✨

```python
--use_pruning  # Pruning 활성화
--pruning_type median  # 또는 halving
```

**효과**:
- 나쁜 trial 조기 종료
- 시간 절약: 30시간 → 약 15~20시간 (예상)
- 더 많은 trial 탐색 가능

### 3. **에러 핸들링 강화** ✨

```python
try:
    metrics = train_catboost(...)
    seed_score = get_score(metrics, metric)
except Exception as e:
    print(f"Warning: Seed {sample_seed} evaluation failed: {e}")
    scores_list.append(0.0)  # 실패한 seed는 0점 처리
    continue
```

**효과**: 일부 seed 실패해도 trial 계속 진행

### 4. **평가 Seed 수 조정 가능** ✨

```python
--n_seeds 3  # 기본값: 3개 (기존 5개에서 감소)
```

**효과**:
- 시간 절약: 5개 → 3개 (40% 단축)
- Pruning과 함께 사용 시 더 효율적

### 5. **파라미터 범위 확대** ✨

```python
# Learning rate: 0.00005 ~ 0.002 (기존 0.0001 ~ 0.001에서 확대)
lr = trial.suggest_loguniform('lr', 0.00005, 0.002)

# Epochs: 10000 추가
steps = trial.suggest_categorical('steps', [1000, 3000, 5000, 10000])
```

**효과**: 더 넓은 탐색 공간

### 6. **Trial 수 조정 가능** ✨

```python
--n_trials 20  # 기본값: 20개 (기존 10개에서 증가)
```

**효과**: 더 많은 탐색으로 최적 파라미터 발견 가능성 증가

### 7. **상세한 진행 상황 모니터링** ✨

```python
print(f"완료된 trial 수: {len([...])}")
print(f"Pruned trial 수: {len([...])}")
print(f"실패한 trial 수: {len([...])}")
print(f"최고 점수 ({metric}): {study.best_value:.4f}")
```

**효과**: 튜닝 진행 상황 파악 용이

---

## 🚀 사용 방법

### 기본 사용 (개선된 버전)

```bash
python CTAB-GAN/tune_ctabgan_improved.py \
    data/orig-micro/ \
    133056 \
    synthetic \
    cuda:0 \
    --n_trials 20 \
    --n_seeds 3 \
    --metric roc_auc \
    --use_pruning \
    --pruning_type median
```

### 빠른 탐색 (시간 절약)

```bash
python CTAB-GAN/tune_ctabgan_improved.py \
    data/orig-micro/ \
    133056 \
    synthetic \
    cuda:0 \
    --n_trials 30 \
    --n_seeds 2 \
    --metric roc_auc \
    --use_pruning \
    --pruning_type halving
```

### 정밀 탐색 (성능 최우선)

```bash
python CTAB-GAN/tune_ctabgan_improved.py \
    data/orig-micro/ \
    133056 \
    synthetic \
    cuda:0 \
    --n_trials 50 \
    --n_seeds 5 \
    --metric roc_auc \
    --use_pruning \
    --pruning_type median
```

---

## 📊 예상 효과

| 항목 | 기존 | 개선 후 | 개선율 |
|------|------|---------|--------|
| **최적화 메트릭** | Macro F1 | ROC-AUC | ✅ 실제 성능 반영 |
| **Trial 수** | 10개 | 20~50개 (조정 가능) | ✅ 2~5배 증가 |
| **평가 Seed 수** | 5개 (고정) | 2~5개 (조정 가능) | ✅ 유연성 증가 |
| **Pruning** | 없음 | Median/Halving | ✅ 시간 절약 |
| **Trial당 시간** | ~3시간 | ~1.5~2시간 (pruning 시) | ✅ 33~50% 단축 |
| **전체 시간** | 30시간 (10 trials) | 30~60시간 (20 trials, pruning) | ✅ 동일 시간, 더 많은 탐색 |
| **에러 처리** | 없음 | 강화됨 | ✅ 안정성 증가 |

---

## 🔧 추가 개선 제안

### 1. **Early Stopping 추가** (중요도: 높음)

현재 `ctabgan_synthesizer.py`의 `fit()` 메서드에 early stopping이 없습니다.
모든 epochs를 다 실행하므로 시간이 많이 소요됩니다.

**구현 필요**:
- Loss 기반 early stopping
- Patience 설정
- 최소 학습 step 보장

**예상 효과**: Trial당 시간 50~67% 단축

### 2. **학습률 스케줄링** (중요도: 중간)

현재 고정 learning rate 사용.
학습률 스케줄링으로 더 빠른 수렴 가능.

**구현 필요**:
- Cosine annealing
- ReduceLROnPlateau

### 3. **Gradient Clipping** (중요도: 중간)

코드에 주석 처리된 gradient clipping 활성화.

**현재 코드**:
```python
# torch.nn.utils.clip_grad_norm_(self.generator.parameters(), max_norm=1.0)
```

**활성화 필요**: 학습 안정성 향상

### 4. **다중 메트릭 최적화** (중요도: 낮음)

현재 단일 메트릭만 최적화.
다중 목표 최적화로 trade-off 고려 가능.

**예시**:
- ROC-AUC 최대화 + Privacy 메트릭 고려
- Pareto optimal solutions

---

## 📝 마이그레이션 가이드

### 기존 코드에서 개선된 코드로 전환

1. **파일 교체**:
   ```bash
   mv CTAB-GAN/tune_ctabgan.py CTAB-GAN/tune_ctabgan_old.py
   mv CTAB-GAN/tune_ctabgan_improved.py CTAB-GAN/tune_ctabgan.py
   ```

2. **기존 스크립트 수정**:
   - 인자 추가: `--n_trials`, `--n_seeds`, `--metric`, `--use_pruning`
   - 또는 기본값 사용

3. **결과 비교**:
   - 기존 결과와 새 결과 비교
   - 성능 향상 확인

---

## 🎯 권장 설정

### 빠른 프로토타이핑
```bash
--n_trials 10 --n_seeds 2 --use_pruning --pruning_type halving
```

### 일반적인 사용
```bash
--n_trials 20 --n_seeds 3 --metric roc_auc --use_pruning --pruning_type median
```

### 최고 성능 추구
```bash
--n_trials 50 --n_seeds 5 --metric roc_auc --use_pruning --pruning_type median
```

---

## ⚠️ 주의사항

1. **Pruning 사용 시**: 처음 몇 개 trial은 pruning되지 않으므로 초기 시간은 비슷함
2. **Seed 수 감소**: 평가 정확도가 약간 감소할 수 있음 (trade-off)
3. **메모리**: 더 많은 trial 실행 시 메모리 사용량 증가 가능
4. **Early Stopping**: `ctabgan_synthesizer.py` 수정 필요 (별도 작업)

---

## 📈 성능 향상 예상

현재 성능:
- ROC-AUC: 0.8181
- F1-score: 0.5521

개선 후 예상:
- ROC-AUC: 0.85~0.90 (목표)
- F1-score: 0.60~0.70

**개선 방법**:
1. ROC-AUC로 최적화
2. 더 많은 trial 탐색
3. Pruning으로 효율적 탐색
4. Early Stopping 추가 (추후)

