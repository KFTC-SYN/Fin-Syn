# Fin-Syn: Financial Synthetic Data Generation Framework

금융 데이터를 포함한 테이블 형식 데이터에 대한 합성 데이터 생성 및 평가를 위한 통합 프레임워크입니다.

## 목차

- [주요 기능](#주요-기능)
- [프로젝트 구조](#프로젝트-구조)
- [설치](#설치)
- [빠른 시작](#빠른-시작)
- [워크플로우](#워크플로우)
- [모델별 사용법](#모델별-사용법)
- [평가 메트릭](#평가-메트릭)
- [설정 파일](#설정-파일)
- [주의사항](#주의사항)
- [참고 자료](#참고-자료)

---

## 주요 기능

### 지원하는 합성 데이터 생성 모델

| 모델 | 타입 | 설명 | 디렉토리 |
|------|------|------|----------|
| **TabDDPM** | Diffusion | Denoising Diffusion 기반 테이블 데이터 생성 | `tab_ddpm/`, `scripts/` |
| **CTGAN** | GAN | Conditional GAN 기반 생성 | `CTGAN/` |
| **TVAE** | VAE | Variational Autoencoder 기반 생성 | `CTGAN/` |
| **CTAB-GAN** | GAN | 테이블 데이터 특화 GAN | `CTAB-GAN/` |
| **CTAB-GAN-Plus** | GAN | CTAB-GAN의 개선 버전 (프라이버시 기능 포함) | `CTAB-GAN-Plus/` |
| **GReaT** | Transformer | GPT-2 기반 생성 모델 | `be_great/` |
| **TabPFGen** | Energy Model | TabPFN 기반 에너지 모델 생성 | `TabPFGen/` |
| **SMOTE** | Oversampling | 오버샘플링 기반 데이터 증강 | `smote/` |

### 평가 방법

- **MLE (Machine Learning Efficacy)**: CatBoost/MLP 분류기를 사용한 다운스트림 태스크 성능 평가
- **SynthEval**: 통계적 유사성, 분포 유사성 등 종합 평가 메트릭
- **Privacy**: 개인정보 보호 수준 평가 (IR, DCR, NNDR, TCAP)
- **Fidelity**: 레이블 분포 및 조건부 패턴 보존 분석
- **Scale**: 다양한 크기의 합성 데이터에 대한 성능 분석

---

## 프로젝트 구조

```
Fin-Syn/
├── scripts/                    # 주요 실행 스크립트
│   ├── pipeline.py             # TabDDPM 파이프라인 (학습/생성/평가)
│   ├── train.py                # TabDDPM 학습
│   ├── sample.py               # TabDDPM 샘플링
│   ├── tune_ddpm.py            # TabDDPM 하이퍼파라미터 튜닝
│   ├── tune_evaluation_model.py # 평가 모델 튜닝
│   ├── eval_catboost.py        # CatBoost 기반 평가
│   ├── eval_mlp.py             # MLP 기반 평가
│   ├── eval_simple.py          # 간단한 모델 평가 (tree, rf, lr, mlp)
│   ├── eval_seeds.py           # 다중 시드 평가 (CatBoost/MLP)
│   ├── eval_seeds_simple.py    # 다중 시드 평가 (간단한 모델)
│   ├── eval_syntheval_style.py # SynthEval 스타일 평가 (직접 구현)
│   ├── eval_syntheval_total.py # SynthEval 공식 라이브러리 평가
│   ├── eval_privacy.py         # Privacy 평가 (IR, DCR, NNDR, TCAP)
│   ├── eval_fraud_fidelity.py  # Fraud 레이블 충실도 분석
│   ├── sample_and_eval_scale.py # 크기별 스케일 평가
│   ├── plot_scale_mle.py       # 크기별 평가 결과 시각화
│   ├── resample_privacy.py     # 프라이버시 리샘플링
│   └── utils_train.py          # 학습 유틸리티 함수
│
├── lib/                        # 공통 유틸리티 라이브러리
│   ├── data.py                 # 데이터 전처리 및 로딩
│   ├── deep.py                 # 딥러닝 모델 유틸리티
│   ├── metrics.py              # 평가 지표 계산
│   ├── util.py                 # 설정/파일 관리 유틸리티
│   └── env.py                  # 환경 설정
│
├── tab_ddpm/                   # TabDDPM 모델 구현
│   ├── gaussian_multinomial_diffsuion.py  # Diffusion 모델
│   ├── modules.py              # MLP/ResNet 모듈
│   └── utils.py                # 유틸리티
│
├── CTGAN/                      # CTGAN/TVAE 구현
│   ├── pipeline_ctgan.py       # CTGAN 파이프라인
│   ├── pipeline_tvae.py        # TVAE 파이프라인
│   ├── tune_ctgan.py           # CTGAN 튜닝
│   ├── tune_tvae.py            # TVAE 튜닝
│   ├── train_sample_ctgan.py   # CTGAN 학습/샘플링
│   └── train_sample_tvae.py    # TVAE 학습/샘플링
│
├── CTAB-GAN/                   # CTAB-GAN 구현
│   ├── pipeline_ctabgan.py     # 파이프라인
│   └── model/                  # 모델 구현
│
├── CTAB-GAN-Plus/              # CTAB-GAN-Plus 구현
│   ├── pipeline_ctabganp.py    # 파이프라인
│   └── model/                  # 모델 구현 (privacy_utils 포함)
│
├── be_great/                   # GReaT 구현
│   ├── pipeline_great.py       # 파이프라인
│   ├── tune_great.py           # 튜닝
│   └── be_great/               # 모델 구현
│
├── TabPFGen/                   # TabPFGen 구현
│   ├── pipeline_tabpfgen.py    # 파이프라인
│   ├── tune_tabpfgen.py        # 튜닝
│   └── src/                    # 모델 구현
│
├── smote/                      # SMOTE 구현
│   ├── pipeline_smote.py       # 파이프라인
│   ├── tune_smote.py           # 튜닝
│   └── sample_smote.py         # 샘플링
│
├── exp/                        # 실험 결과 저장 디렉토리
│   └── [dataset_name]/         # 데이터셋별 디렉토리
│       └── [model_name]/       # 모델별 디렉토리
│           ├── config.toml     # 설정 파일
│           ├── *.obj / *.pt    # 학습된 모델
│           ├── X_*.npy, y_*.npy # 합성 데이터
│           ├── eval_*.json     # 평가 결과
│           └── size_*/         # 크기별 결과
│
├── tuned_models/               # 튜닝된 평가 모델 저장
│   ├── catboost/               # CatBoost 하이퍼파라미터
│   └── mlp/                    # MLP 하이퍼파라미터
│
├── run_orig_*.sh               # 모델별 실행 스크립트
│   ├── run_orig_ddpm.sh        # TabDDPM
│   ├── run_orig_ctgan.sh       # CTGAN
│   ├── run_orig_tvae.sh        # TVAE
│   ├── run_orig_ctab.sh        # CTAB-GAN
│   ├── run_orig_ctabp.sh       # CTAB-GAN-Plus
│   ├── run_orig_great.sh       # GReaT
│   ├── run_orig_tabpfgen.sh    # TabPFGen
│   └── run_orig_smote.sh       # SMOTE
│
├── requirements.txt            # Python 의존성
└── LICENSE.md                  # MIT 라이선스
```

---

## 설치

### 요구사항

- Python 3.9+
- PyTorch 1.10.1+ (CUDA 지원 권장)
- NumPy < 2.0 (PyArrow 호환성)

### Conda 환경 설정

프로젝트는 여러 Conda 환경을 사용합니다:

```bash
# 메인 환경 (TabDDPM, CTGAN, TVAE, CTAB-GAN, TabPFGen 등)
conda create -n tddpm python=3.9
conda activate tddpm
pip install -r requirements.txt

# GReaT 모델용 환경 (transformers 버전 충돌로 별도 환경 필요)
conda create -n tddpm2 python=3.9
conda activate tddpm2
pip install -r be_great/requirements.txt

# SynthEval 라이브러리 평가용 환경
conda create -n finsyn python=3.9
conda activate finsyn
pip install syntheval
```

### 주요 의존성

```
catboost==1.0.3
numpy==1.21.4
pandas==1.3.4
scikit-learn==1.0.2
torch>=1.10.1
optuna==2.10.1
imbalanced-learn==0.7.0  # SMOTE
rdt==0.6.4               # TVAE
```

---

## 빠른 시작

### 1. 데이터 준비

데이터는 다음 형식으로 `exp/[dataset_name]/` 디렉토리에 준비합니다:

```
exp/[dataset_name]/
├── info.json           # 데이터셋 메타정보
├── X_num_train.npy     # 수치형 피처 (optional)
├── X_cat_train.npy     # 범주형 피처 (optional)
├── y_train.npy         # 타겟 변수
├── X_num_val.npy       # 검증 수치형 피처
├── X_cat_val.npy       # 검증 범주형 피처
├── y_val.npy           # 검증 타겟
├── X_num_test.npy      # 테스트 수치형 피처
├── X_cat_test.npy      # 테스트 범주형 피처
└── y_test.npy          # 테스트 타겟
```

`info.json` 예시:
```json
{
    "task_type": "binclass",
    "n_num_features": 1,
    "n_cat_features": 8,
    "train_size": 63703,
    "val_size": 10617,
    "test_size": 10617
}
```

### 2. 평가 모델 튜닝

```bash
# CatBoost 튜닝
python scripts/tune_evaluation_model.py [dataset_name] catboost cv cuda:0

# MLP 튜닝
python scripts/tune_evaluation_model.py [dataset_name] mlp cv cuda:0
```

### 3. 합성 데이터 생성 및 평가

```bash
# TabDDPM 예시
python scripts/tune_ddpm.py [dataset_name] [train_size] synthetic catboost ddpm_cb
python scripts/pipeline.py --config exp/[dataset_name]/ddpm_cb_best/config.toml --train --sample --eval

# CTGAN 예시
python CTGAN/tune_ctgan.py exp/[dataset_name]/ [train_size] synthetic cuda:0
python CTGAN/pipeline_ctgan.py --config exp/[dataset_name]/ctgan/config.toml --train --sample --eval
```

---

## 워크플로우

### 전체 파이프라인

```
1. 데이터 준비
       ↓
2. 평가 모델 튜닝 (CatBoost/MLP)
       ↓
3. 생성 모델 하이퍼파라미터 튜닝
       ↓
4. 모델 학습 (--train)
       ↓
5. 합성 데이터 생성 (--sample)
       ↓
6. 평가 (--eval)
       ↓
7. 다중 시드 평가 (eval_seeds.py)
       ↓
8. 크기별 스케일 평가 (sample_and_eval_scale.py)
       ↓
9. SynthEval/Privacy 평가
```

### 공통 명령어 인자

모든 파이프라인 스크립트는 다음 인자를 지원합니다:

| 인자 | 설명 |
|------|------|
| `--config` | 설정 파일 경로 (config.toml) |
| `--train` | 모델 학습 실행 |
| `--sample` | 합성 데이터 생성 |
| `--eval` | 평가 실행 |
| `--change_val` | 검증 데이터 분할 변경 |

---

## 모델별 사용법

### TabDDPM (Diffusion Model)

```bash
# 튜닝
python scripts/tune_ddpm.py orig-micro-retry 63703 synthetic catboost ddpm_cb

# 학습/생성/평가
python scripts/pipeline.py --config exp/orig-micro-retry/ddpm_cb_best/config.toml --train --sample --eval

# 다중 시드 평가
python scripts/eval_seeds.py --config exp/orig-micro-retry/ddpm_cb_best/config.toml 10 ddpm synthetic catboost 5

# 크기별 평가
python scripts/sample_and_eval_scale.py --config exp/orig-micro-retry/ddpm_cb_best/config.toml \
    --model_type ddpm --sizes 1.0 1.5 2.0 --seed 0 --change_val --n_seeds 5
```

### CTGAN

```bash
# 튜닝
python CTGAN/tune_ctgan.py exp/orig-micro-retry/ 63703 synthetic cuda:0

# 학습/생성/평가
python CTGAN/pipeline_ctgan.py --config exp/orig-micro-retry/ctgan/config.toml --train --sample --eval

# 크기별 평가
python scripts/sample_and_eval_scale.py --config exp/orig-micro-retry/ctgan/config.toml \
    --model_type ctgan --sizes 1.0 1.5 2.0 --seed 0 --change_val --n_seeds 5
```

### TVAE

```bash
python CTGAN/tune_tvae.py exp/orig-micro-retry/ 63703 synthetic cuda:0
python CTGAN/pipeline_tvae.py --config exp/orig-micro-retry/tvae/config.toml --train --sample --eval
```

### CTAB-GAN / CTAB-GAN-Plus

```bash
# CTAB-GAN
python CTAB-GAN/tune_ctabgan.py exp/orig-micro-retry/ 63703 synthetic cuda:0
python CTAB-GAN/pipeline_ctabgan.py --config exp/orig-micro-retry/ctabgan/config.toml --train --sample --eval

# CTAB-GAN-Plus
python CTAB-GAN-Plus/tune_ctabgan.py exp/orig-micro-retry/ 63703 synthetic cuda:0
python CTAB-GAN-Plus/pipeline_ctabganp.py --config exp/orig-micro-retry/ctabgan-plus/config.toml --train --sample --eval
```

### GReaT (Transformer)

> **주의**: GReaT는 별도의 conda 환경(`tddpm2`)에서 실행해야 합니다.

```bash
conda activate tddpm2
python be_great/tune_great.py exp/orig-micro-retry/ 63703 synthetic cuda:0
python be_great/pipeline_great.py --config exp/orig-micro-retry/great/config.toml --train --sample --eval
```

### TabPFGen

```bash
python TabPFGen/tune_tabpfgen.py exp/orig-micro-retry/ 63703 synthetic cuda:0
python TabPFGen/pipeline_tabpfgen.py --config exp/orig-micro-retry/tabpfgen/config.toml --train --sample --eval
```

### SMOTE

```bash
# SMOTE는 학습 단계 없이 바로 샘플링
python smote/tune_smote.py exp/orig-micro-retry/ synthetic
python smote/pipeline_smote.py --config exp/orig-micro-retry/smote/config.toml --sample --eval
```

---

## 평가 메트릭

### MLE (Machine Learning Efficacy) 메트릭

| 메트릭 | 설명 |
|--------|------|
| `acc` | 정확도 |
| `f1` | F1 스코어 |
| `f1_0`, `f1_1` | 클래스별 F1 스코어 |
| `f1_weighted` | 가중 F1 스코어 |
| `balanced_acc` | 균형 정확도 |
| `mcc` | Matthews 상관계수 |
| `kappa` | Cohen's Kappa |
| `roc_auc` | ROC AUC |

### SynthEval 메트릭

| 메트릭 | 설명 |
|--------|------|
| `dwm` | Distance to Closest Record (DCR) |
| `p_mse` | Predictive Mean Squared Error |
| `corr_diff` | Correlation Difference |
| `ks_test` | Kolmogorov-Smirnov Test |
| `h_dist` | Hellinger Distance |
| `cio` | Categorical IO |
| `mi_diff` | Mutual Information Difference |
| `jsd` | Jensen-Shannon Divergence |
| `kl_div` | Kullback-Leibler Divergence |
| `theils_u` | Theil's U |

```bash
# SynthEval 스타일 평가 (직접 구현)
python scripts/eval_syntheval_style.py --config exp/[dataset]/[model]/config.toml \
    --metrics dwm p_mse corr_diff ks_test

# SynthEval 공식 라이브러리 평가 (finsyn 환경 필요)
conda activate finsyn
python scripts/eval_syntheval_total.py --config exp/[dataset]/[model]/config.toml \
    --preset full_eval --change_val --exclude nnaa
```

### Privacy 메트릭

| 메트릭 | 설명 |
|--------|------|
| `ir` | Identity Risk |
| `dcr` | Distance to Closest Record |
| `nndr` | Nearest Neighbor Distance Ratio |
| `tcap` | Target Correct Attribution Probability |

```bash
python scripts/eval_privacy.py --config exp/[dataset]/[model]/config.toml \
    --metrics ir dcr nndr tcap
```

### Fraud Fidelity 분석

레이블 분포 보존 및 조건부 패턴 비교 분석:

```bash
python scripts/eval_fraud_fidelity.py --exp_dir exp/[dataset] \
    --real_data exp/[dataset]/ \
    --models ctgan tvae great ddpm_cb_best \
    --sizes 1.0 1.5 2.0
```

---

## 설정 파일

각 실험은 `config.toml` 파일로 관리됩니다.

### TabDDPM config.toml 예시

```toml
seed = 0
parent_dir = "exp/orig-micro-retry/ddpm_cb_best"
real_data_path = "data/orig-micro-retry/"
model_type = "resnet"
num_numerical_features = 1
device = "cuda:0"

[model_params]
num_classes = 2
is_y_cond = true
dim_t = 256

[model_params.rtdl_params]
n_blocks = 3
d_main = 512
d_hidden = 1024
dropout_first = 0.185
dropout_second = 0.121

[diffusion_params]
num_timesteps = 100
gaussian_loss_type = "mse"
scheduler = "cosine"

[train.main]
steps = 40000
lr = 0.00054
weight_decay = 1e-05
batch_size = 256

[sample]
num_samples = 63703
batch_size = 10000
seed = 0

[eval.type]
eval_model = "catboost"
eval_type = "synthetic"
```

### CTGAN config.toml 예시

```toml
parent_dir = "exp/orig-micro-retry/ctgan/"
real_data_path = "data/orig-micro-retry/"
seed = 0
device = "cuda:0"

[train_params]
generator_lr = 0.000556
discriminator_lr = 1.2e-05
epochs = 3000
embedding_dim = 128
batch_size = 2000
discriminator_steps = 3
generator_dim = [256]
discriminator_dim = [256]
pac = 5

[sample]
seed = 0
num_samples = 63703

[eval.type]
eval_model = "catboost"
eval_type = "synthetic"
```

---

## 주의사항

### 환경별 실행

| 모델/평가 | 환경 |
|-----------|------|
| TabDDPM, CTGAN, TVAE, CTAB-GAN, TabPFGen, SMOTE | `tddpm` |
| GReaT | `tddpm2` |
| SynthEval 공식 라이브러리 평가 | `finsyn` |

### 알려진 이슈

1. **NVIDIA L40 GPU**: JIT 컴파일 오류로 인해 일부 학습 실패 가능
   - 오류: `nvrtc: error: invalid value for --gpu-architecture`

2. **SynthEval ks_test**: CPU/메모리 오버플로우 이슈
   - 해결: `--exclude nnaa` 옵션 사용 또는 `eval_syntheval_style.py`로 별도 평가

3. **NumPy 버전**: PyArrow 호환성을 위해 NumPy < 2.0 사용 필요

4. **메모리 제한**: 큰 데이터셋의 경우 배치 크기 조정 필요

### 결과 시각화

```bash
# 크기별 MLE 평가 결과 플롯
python scripts/plot_scale_mle.py --base_dir exp/[dataset]/[model]
```

생성되는 플롯:
- `scale_mle_accuracy.png`
- `scale_mle_f1_neg.png`
- `scale_mle_f1_pos.png`
- `scale_mle_roc_auc.png`

---

## 참고 자료

### 논문

- **TabDDPM**: [TabDDPM: Modelling Tabular Data with Diffusion Models](https://arxiv.org/abs/2209.15421)
- **CTGAN/TVAE**: [Modeling Tabular data using Conditional GAN](https://arxiv.org/abs/1907.00503)
- **CTAB-GAN**: [CTAB-GAN: Effective Table Data Synthesizing](https://arxiv.org/abs/2102.08369)
- **GReaT**: [Language Models are Realistic Tabular Data Generators](https://arxiv.org/abs/2210.06280)

### 라이브러리

- [SynthEval](https://github.com/schneiderkamplab/syntheval) - 합성 데이터 평가 라이브러리
- [SDV (Synthetic Data Vault)](https://github.com/sdv-dev/SDV) - CTGAN/TVAE 구현

---

## 라이선스

MIT License - 자세한 내용은 [LICENSE.md](LICENSE.md)를 참조하세요.

각 서브모듈의 라이선스는 해당 디렉토리의 LICENSE 파일을 확인하세요.
