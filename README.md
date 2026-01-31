# Fin-Syn: Financial Synthetic Data Generation Framework

## 프로젝트 개요

Fin-Syn은 금융 데이터를 포함한 테이블 형식 데이터에 대한 합성 데이터 생성 및 평가를 위한 통합 프레임워크입니다. 다양한 생성 모델을 지원하며, 종합적인 평가 메트릭을 제공합니다.

## 주요 기능

### 지원하는 합성 데이터 생성 모델

1. **TabDDPM** (`ddpm`) - Diffusion Model 기반 테이블 데이터 생성
2. **CTGAN** (`ctgan`) - Conditional GAN 기반 생성
3. **CTAB-GAN** (`ctabgan`) - 테이블 데이터 특화 GAN
4. **CTAB-GAN-Plus** (`ctabganp`) - CTAB-GAN의 개선 버전
5. **GReaT** (`great`) - Transformer 기반 생성 모델
6. **SMOTE** (`smote`) - 오버샘플링 기법
7. **TVAE** (`tvae`) - Variational Autoencoder 기반 생성
8. **TabPFGen** (`tabpfgen`) - TabPFN 기반 에너지 모델 생성
9. **OURS** (`ours`) - 커스텀 Transformer 기반 생성 모델 (이분 그래프 구조)

### 평가 방법

- **MLE (Maximum Likelihood Estimation) 평가**: CatBoost/MLP 분류기를 사용한 다운스트림 태스크 성능 평가
- **SynthEval 평가**: 통계적 유사성, 분포 유사성 등 종합 평가 메트릭
- **Privacy 평가**: 개인정보 보호 수준 평가 (IR, DCR, NNDR)
- **크기별 스케일 평가**: 다양한 크기의 합성 데이터에 대한 성능 분석

## 프로젝트 구조

```
Fin-Syn/
├── scripts/                    # 주요 실행 스크립트
│   ├── pipeline.py            # TabDDPM 파이프라인 (학습/생성/평가)
│   ├── tune_ddpm.py           # TabDDPM 하이퍼파라미터 튜닝
│   ├── sample.py               # TabDDPM 샘플링
│   ├── eval_seeds.py           # 다중 시드 평가
│   ├── eval_catboost.py        # CatBoost 기반 평가
│   ├── eval_mlp.py             # MLP 기반 평가
│   ├── eval_syntheval.py       # SynthEval 스타일 평가
│   ├── eval_syntheval_total.py # SynthEval 공식 라이브러리 평가
│   ├── eval_privacy.py         # Privacy 평가
│   ├── sample_and_eval_scale.py # 크기별 스케일 평가
│   └── plot_scale_mle.py      # 크기별 평가 결과 시각화
│
├── run_orig_*.sh               # 각 모델별 실행 스크립트
│   ├── run_orig_ddpm.sh        # TabDDPM 실행
│   ├── run_orig_ctgan.sh       # CTGAN 실행
│   ├── run_orig_ctab.sh        # CTAB-GAN 실행
│   ├── run_orig_ctabp.sh       # CTAB-GAN-Plus 실행
│   ├── run_orig_great.sh       # GReaT 실행
│   ├── run_orig_smote.sh       # SMOTE 실행
│   ├── run_orig_tvae.sh        # TVAE 실행
│   ├── run_orig_tabpfgen.sh    # TabPFGen 실행
│   └── run_orig_ours.sh        # OURS 실행
│
├── exp/                        # 실험 결과 저장 디렉토리
│   └── [dataset_name]/         # 데이터셋별 디렉토리
│       └── [model_name]/       # 모델별 디렉토리
│           ├── config.toml      # 설정 파일
│           ├── model.pt         # 학습된 모델
│           └── eval_*.json      # 평가 결과
│
├── data/                       # 데이터셋 디렉토리
│   └── [dataset_name]/         # 데이터셋별 디렉토리
│       ├── info.json           # 데이터셋 메타정보
│       ├── X_num_train.npy     # 수치형 피처
│       ├── X_cat_train.npy     # 범주형 피처
│       └── y_train.npy         # 타겟 변수
│
├── lib/                        # 공통 유틸리티 라이브러리
├── CTGAN/                      # CTGAN 구현
├── CTAB-GAN/                   # CTAB-GAN 구현
├── CTAB-GAN-Plus/              # CTAB-GAN-Plus 구현
├── be_great/                   # GReaT 구현
├── smote/                      # SMOTE 구현
├── TabPFGen/                   # TabPFGen 구현
├── ours/                       # OURS 커스텀 모델 구현
└── tab_ddpm/                   # TabDDPM 구현
```

## 워크플로우

### 1. 평가 모델 튜닝

먼저 평가에 사용할 CatBoost 또는 MLP 모델의 하이퍼파라미터를 튜닝합니다.

```bash
# CatBoost 튜닝
python scripts/tune_evaluation_model.py [dataset] catboost cv [device]

# MLP 튜닝
python scripts/tune_evaluation_model.py [dataset] mlp cv [device]
```

### 2. 생성 모델 튜닝

각 생성 모델의 하이퍼파라미터를 튜닝합니다.

```bash
# TabDDPM 튜닝
python scripts/tune_ddpm.py [dataset] [train_size] synthetic [catboost|mlp] [exp_name]

# 다른 모델들은 각 디렉토리의 tune_*.py 스크립트 사용
# 예: python CTGAN/tune_ctgan.py [data_path] [train_size] synthetic [device]
```

### 3. 모델 학습/생성/평가

튜닝된 하이퍼파라미터로 모델을 학습하고 합성 데이터를 생성한 후 평가합니다.

```bash
# TabDDPM 파이프라인
python scripts/pipeline.py --config exp/[dataset]/[model]/config.toml --train --sample --eval

# 다른 모델들은 각 디렉토리의 pipeline_*.py 스크립트 사용
```

### 4. 다중 시드 평가

여러 샘플링 시드와 평가 시드로 안정적인 성능을 측정합니다.

```bash
python scripts/eval_seeds.py --config exp/[dataset]/[model]/config.toml [n_eval_seeds] [model_type] synthetic [catboost|mlp] [n_sample_seeds]
```

### 5. 크기별 스케일 평가

다양한 크기의 합성 데이터에 대한 성능을 평가합니다.

```bash
python scripts/sample_and_eval_scale.py --config exp/[dataset]/[model]/config.toml --model_type [model_type] --sizes 1.0 1.5 2.0 --seed 0 --change_val --n_seeds 5

# 결과 시각화
python scripts/plot_scale_mle.py --base_dir exp/[dataset]/[model]
```

### 6. SynthEval 평가

통계적 유사성 등 종합적인 평가 메트릭을 계산합니다.

```bash
# 전체 메트릭 평가
python scripts/eval_syntheval_total.py --config exp/[dataset]/[model]/config.toml --preset full_eval --change_val --exclude nnaa

# 특정 메트릭만 평가
python scripts/eval_syntheval.py --config exp/[dataset]/[model]/config.toml --metrics dwm p_mse corr_diff ks_test
```

### 7. Privacy 평가

개인정보 보호 수준을 평가합니다.

```bash
python scripts/eval_privacy.py --config exp/[dataset]/[model]/config.toml --metrics ir dcr nndr
```

## 모델별 실행 예시

### TabDDPM

```bash
# 실행 스크립트 사용
bash run_orig_ddpm.sh

# 또는 개별 명령어
python scripts/tune_ddpm.py orig-micro 66528 synthetic catboost ddpm_cb
python scripts/pipeline.py --config exp/orig-micro/ddpm_cb_best/config.toml --train --sample --eval
python scripts/eval_seeds.py --config exp/orig-micro/ddpm_cb_best/config.toml 10 ddpm synthetic catboost 5
python scripts/sample_and_eval_scale.py --config exp/orig-micro/ddpm_cb_best/config.toml --model_type ddpm --sizes 1.0 1.5 2.0 --seed 0 --change_val --n_seeds 5
```

### CTGAN

```bash
bash run_orig_ctgan.sh

# 또는 개별 명령어
python CTGAN/tune_ctgan.py data/orig-micro/ 66528 synthetic cuda:0
python CTGAN/pipeline_ctgan.py --config exp/orig-micro/ctgan/config.toml --train --sample --eval
python scripts/sample_and_eval_scale.py --config exp/orig-micro/ctgan/config.toml --model_type ctgan --sizes 1.0 1.5 2.0 --seed 0 --change_val --n_seeds 5
```

### TabPFGen

```bash
bash run_orig_tabpfgen.sh

# 또는 개별 명령어
python TabPFGen/tune_tabpfgen.py data/orig-micro/ 66528 synthetic cuda:0
python TabPFGen/pipeline_tabpfgen.py --config exp/orig-micro/tabpfgen/config.toml --train --sample --eval
python scripts/sample_and_eval_scale.py --config exp/orig-micro/tabpfgen/config.toml --model_type tabpfgen --sizes 1.0 1.5 2.0 --seed 0 --change_val --n_seeds 5
```

### OURS (커스텀 모델)

```bash
bash run_orig_ours.sh

# 또는 개별 명령어
python ours_20251125/tune_ours.py data/orig-micro/ 66528 synthetic cuda:0
python ours_20251125/pipeline_ours.py --config exp/orig-micro/ours/config.toml --train --sample --eval
python scripts/sample_and_eval_scale.py --config exp/orig-micro/ours/config.toml --model_type ours --sizes 1.0 1.5 2.0 --seed 0 --change_val --n_seeds 5
```

## 환경 설정

### Conda 환경

프로젝트는 주로 다음 Conda 환경을 사용합니다:

- **tddpm**: TabDDPM 및 대부분의 모델 실행 환경
- **tddpm2**: GReaT 모델 실행 환경
- **finsyn**: SynthEval 라이브러리 기반 평가 환경

### 주요 의존성

- Python 3.9+
- PyTorch 1.10.1+ (CUDA 지원)
- NumPy < 2.0 (PyArrow 호환성)
- pandas, scikit-learn
- CatBoost
- 각 모델별 추가 의존성 (각 디렉토리의 requirements.txt 참조)

### 환경별 특수 요구사항

- **GReaT**: `tddpm2` 환경 필요
- **SynthEval 평가**: `finsyn` 환경 필요
- **TabPFGen**: `tabpfn` 패키지 설치 필요 (`pip install tabpfn>=2.0.1`)

## 설정 파일 구조

각 실험은 `config.toml` 파일로 관리됩니다. 주요 설정 항목:

- `parent_dir`: 결과 저장 디렉토리
- `real_data_path`: 원본 데이터 경로
- `train_params`: 모델 학습 파라미터
- `sample`: 샘플링 설정 (샘플 수, 시드 등)
- `eval`: 평가 설정 (평가 모델 타입, 평가 타입 등)
- `device`: 사용할 디바이스 (cpu 또는 cuda:X)

## 평가 메트릭

### MLE 평가 메트릭

- `acc`: 정확도
- `f1`: F1 스코어
- `f1_0`, `f1_1`: 클래스별 F1 스코어
- `f1_weighted`: 가중 F1 스코어
- `balanced_acc`: 균형 정확도
- `mcc`: Matthews 상관계수
- `kappa`: Cohen's Kappa
- `roc_auc`: ROC AUC

### SynthEval 메트릭

- `dwm`: Distance to Closest Record
- `p_mse`: Predictive Mean Squared Error
- `corr_diff`: Correlation Difference
- `ks_test`: Kolmogorov-Smirnov Test
- `h_dist`: Hellinger Distance
- `cio`: Categorical IO
- `mi_diff`: Mutual Information Difference
- `q_mse`: Quantile Mean Squared Error
- `jsd`: Jensen-Shannon Divergence
- `kl_div`: Kullback-Leibler Divergence
- `theils_u`: Theil's U
- `cond_entropy`: Conditional Entropy
- `entropy`: Entropy
- `pearson_corr`: Pearson Correlation

### Privacy 메트릭

- `ir`: Identity Risk
- `dcr`: Distance to Closest Record
- `nndr`: Nearest Neighbor Distance Ratio

## 주의사항

1. **환경별 실행**: 일부 모델은 특정 환경에서만 실행 가능합니다 (예: GReaT는 tddpm2 환경 필요)

2. **메모리 제한**: 큰 데이터셋의 경우 메모리 부족 문제가 발생할 수 있습니다. 특히 SynthEval의 `ks_test` 메트릭은 CPU/메모리 오버플로우 이슈가 있어 제외하는 것을 권장합니다.

3. **NumPy 버전**: PyArrow 호환성을 위해 NumPy < 2.0을 사용해야 합니다.

4. **의존성 충돌**: 일부 모델은 서로 다른 버전의 의존성을 요구할 수 있습니다. 각 모델의 requirements.txt를 확인하세요.

## 참고 자료

- TabDDPM 원본 논문: [TabDDPM: Modelling Tabular Data with Diffusion Models](https://arxiv.org/abs/2209.15421)
- TabPFGen: [TabPFGen Documentation](https://github.com/sebhaan/TabPFGen)
- SynthEval: [SynthEval Library](https://github.com/synthcity-ai/synthcity)

## 라이선스

각 모델의 라이선스는 해당 디렉토리의 LICENSE 파일을 참조하세요.
