# Fin-Syn: Financial Synthetic Data Generation Framework

A unified framework for synthetic data generation and evaluation for tabular data, including financial datasets.

## Table of Contents

- [Key Features](#key-features)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Workflow](#workflow)
- [Model-Specific Usage](#model-specific-usage)
- [Evaluation Metrics](#evaluation-metrics)
- [Configuration Files](#configuration-files)
- [Notes and Caveats](#notes-and-caveats)
- [References](#references)

---

## Key Features

### Supported Synthetic Data Generation Models

| Model | Type | Description | Directory |
|-------|------|-------------|-----------|
| **TabDDPM** | Diffusion | Denoising Diffusion-based tabular data generation | `tab_ddpm/`, `scripts/` |
| **CTGAN** | GAN | Conditional GAN-based generation | `CTGAN/` |
| **TVAE** | VAE | Variational Autoencoder-based generation | `CTGAN/` |
| **CTAB-GAN** | GAN | GAN specialized for tabular data | `CTAB-GAN/` |
| **CTAB-GAN-Plus** | GAN | Improved version of CTAB-GAN (with privacy features) | `CTAB-GAN-Plus/` |
| **GReaT** | Transformer | GPT-2 based generation model | `be_great/` |
| **TabPFGen** | Energy Model | TabPFN-based energy model generation | `TabPFGen/` |
| **SMOTE** | Oversampling | Oversampling-based data augmentation | `smote/` |

### Evaluation Methods

- **MLE (Machine Learning Efficacy)**: Downstream task performance evaluation using CatBoost/MLP classifiers
- **SynthEval**: Comprehensive evaluation metrics including statistical similarity and distribution similarity
- **Privacy**: Privacy protection level evaluation (IR, DCR, NNDR, TCAP)
- **Fidelity**: Label distribution and conditional pattern preservation analysis
- **Scale**: Performance analysis across various synthetic data sizes

---

## Project Structure

```
Fin-Syn/
├── scripts/                    # Main execution scripts
│   ├── pipeline.py             # TabDDPM pipeline (train/sample/eval)
│   ├── train.py                # TabDDPM training
│   ├── sample.py               # TabDDPM sampling
│   ├── tune_ddpm.py            # TabDDPM hyperparameter tuning
│   ├── tune_evaluation_model.py # Evaluation model tuning
│   ├── eval_catboost.py        # CatBoost-based evaluation
│   ├── eval_mlp.py             # MLP-based evaluation
│   ├── eval_simple.py          # Simple model evaluation (tree, rf, lr, mlp)
│   ├── eval_seeds.py           # Multi-seed evaluation (CatBoost/MLP)
│   ├── eval_seeds_simple.py    # Multi-seed evaluation (simple models)
│   ├── eval_syntheval_style.py # SynthEval-style evaluation (custom impl.)
│   ├── eval_syntheval_total.py # Official SynthEval library evaluation
│   ├── eval_privacy.py         # Privacy evaluation (IR, DCR, NNDR, TCAP)
│   ├── eval_fraud_fidelity.py  # Fraud label fidelity analysis
│   ├── sample_and_eval_scale.py # Scale-based evaluation
│   ├── plot_scale_mle.py       # Scale evaluation result visualization
│   ├── resample_privacy.py     # Privacy resampling
│   └── utils_train.py          # Training utility functions
│
├── lib/                        # Common utility library
│   ├── data.py                 # Data preprocessing and loading
│   ├── deep.py                 # Deep learning model utilities
│   ├── metrics.py              # Evaluation metric computation
│   ├── util.py                 # Config/file management utilities
│   └── env.py                  # Environment settings
│
├── tab_ddpm/                   # TabDDPM model implementation
│   ├── gaussian_multinomial_diffsuion.py  # Diffusion model
│   ├── modules.py              # MLP/ResNet modules
│   └── utils.py                # Utilities
│
├── CTGAN/                      # CTGAN/TVAE implementation
│   ├── pipeline_ctgan.py       # CTGAN pipeline
│   ├── pipeline_tvae.py        # TVAE pipeline
│   ├── tune_ctgan.py           # CTGAN tuning
│   ├── tune_tvae.py            # TVAE tuning
│   ├── train_sample_ctgan.py   # CTGAN train/sample
│   └── train_sample_tvae.py    # TVAE train/sample
│
├── CTAB-GAN/                   # CTAB-GAN implementation
│   ├── pipeline_ctabgan.py     # Pipeline
│   └── model/                  # Model implementation
│
├── CTAB-GAN-Plus/              # CTAB-GAN-Plus implementation
│   ├── pipeline_ctabganp.py    # Pipeline
│   └── model/                  # Model implementation (incl. privacy_utils)
│
├── be_great/                   # GReaT implementation
│   ├── pipeline_great.py       # Pipeline
│   ├── tune_great.py           # Tuning
│   └── be_great/               # Model implementation
│
├── TabPFGen/                   # TabPFGen implementation
│   ├── pipeline_tabpfgen.py    # Pipeline
│   ├── tune_tabpfgen.py        # Tuning
│   └── src/                    # Model implementation
│
├── smote/                      # SMOTE implementation
│   ├── pipeline_smote.py       # Pipeline
│   ├── tune_smote.py           # Tuning
│   └── sample_smote.py         # Sampling
│
├── exp/                        # Experiment results directory
│   └── [dataset_name]/         # Per-dataset directory
│       └── [model_name]/       # Per-model directory
│           ├── config.toml     # Configuration file
│           ├── *.obj / *.pt    # Trained model
│           ├── X_*.npy, y_*.npy # Synthetic data
│           ├── eval_*.json     # Evaluation results
│           └── size_*/         # Scale-specific results
│
├── tuned_models/               # Tuned evaluation model storage
│   ├── catboost/               # CatBoost hyperparameters
│   └── mlp/                    # MLP hyperparameters
│
├── run_orig_*.sh               # Per-model execution scripts
│   ├── run_orig_ddpm.sh        # TabDDPM
│   ├── run_orig_ctgan.sh       # CTGAN
│   ├── run_orig_tvae.sh        # TVAE
│   ├── run_orig_ctab.sh        # CTAB-GAN
│   ├── run_orig_ctabp.sh       # CTAB-GAN-Plus
│   ├── run_orig_great.sh       # GReaT
│   ├── run_orig_tabpfgen.sh    # TabPFGen
│   └── run_orig_smote.sh       # SMOTE
│
├── requirements.txt            # Python dependencies
└── LICENSE.md                  # MIT License
```

---

## Installation

### Requirements

- Python 3.9+
- PyTorch 1.10.1+ (CUDA support recommended)
- NumPy < 2.0 (PyArrow compatibility)

### Conda Environment Setup

The project uses multiple Conda environments:

```bash
# Main environment (TabDDPM, CTGAN, TVAE, CTAB-GAN, TabPFGen, etc.)
conda create -n tddpm python=3.9
conda activate tddpm
pip install -r requirements.txt

# GReaT model environment (separate due to transformers version conflicts)
conda create -n tddpm2 python=3.9
conda activate tddpm2
pip install -r be_great/requirements.txt

# SynthEval library evaluation environment
conda create -n finsyn python=3.9
conda activate finsyn
pip install syntheval
```

### Key Dependencies

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

## Quick Start

### 1. Data Preparation

Prepare data in the following format under `exp/[dataset_name]/` directory:

```
exp/[dataset_name]/
├── info.json           # Dataset metadata
├── X_num_train.npy     # Numerical features (optional)
├── X_cat_train.npy     # Categorical features (optional)
├── y_train.npy         # Target variable
├── X_num_val.npy       # Validation numerical features
├── X_cat_val.npy       # Validation categorical features
├── y_val.npy           # Validation target
├── X_num_test.npy      # Test numerical features
├── X_cat_test.npy      # Test categorical features
└── y_test.npy          # Test target
```

`info.json` example:
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

### 2. Evaluation Model Tuning

```bash
# CatBoost tuning
python scripts/tune_evaluation_model.py [dataset_name] catboost cv cuda:0

# MLP tuning
python scripts/tune_evaluation_model.py [dataset_name] mlp cv cuda:0
```

### 3. Synthetic Data Generation and Evaluation

```bash
# TabDDPM example
python scripts/tune_ddpm.py [dataset_name] [train_size] synthetic catboost ddpm_cb
python scripts/pipeline.py --config exp/[dataset_name]/ddpm_cb_best/config.toml --train --sample --eval

# CTGAN example
python CTGAN/tune_ctgan.py exp/[dataset_name]/ [train_size] synthetic cuda:0
python CTGAN/pipeline_ctgan.py --config exp/[dataset_name]/ctgan/config.toml --train --sample --eval
```

---

## Workflow

### Full Pipeline

```
1. Data Preparation
       ↓
2. Evaluation Model Tuning (CatBoost/MLP)
       ↓
3. Generation Model Hyperparameter Tuning
       ↓
4. Model Training (--train)
       ↓
5. Synthetic Data Generation (--sample)
       ↓
6. Evaluation (--eval)
       ↓
7. Multi-seed Evaluation (eval_seeds.py)
       ↓
8. Scale-based Evaluation (sample_and_eval_scale.py)
       ↓
9. SynthEval/Privacy Evaluation
```

### Common Command Arguments

All pipeline scripts support the following arguments:

| Argument | Description |
|----------|-------------|
| `--config` | Configuration file path (config.toml) |
| `--train` | Execute model training |
| `--sample` | Generate synthetic data |
| `--eval` | Execute evaluation |
| `--change_val` | Change validation data split |

---

## Model-Specific Usage

### TabDDPM (Diffusion Model)

```bash
# Tuning
python scripts/tune_ddpm.py orig-micro-retry 63703 synthetic catboost ddpm_cb

# Train/Sample/Eval
python scripts/pipeline.py --config exp/orig-micro-retry/ddpm_cb_best/config.toml --train --sample --eval

# Multi-seed evaluation
python scripts/eval_seeds.py --config exp/orig-micro-retry/ddpm_cb_best/config.toml 10 ddpm synthetic catboost 5

# Scale-based evaluation
python scripts/sample_and_eval_scale.py --config exp/orig-micro-retry/ddpm_cb_best/config.toml \
    --model_type ddpm --sizes 1.0 1.5 2.0 --seed 0 --change_val --n_seeds 5
```

### CTGAN

```bash
# Tuning
python CTGAN/tune_ctgan.py exp/orig-micro-retry/ 63703 synthetic cuda:0

# Train/Sample/Eval
python CTGAN/pipeline_ctgan.py --config exp/orig-micro-retry/ctgan/config.toml --train --sample --eval

# Scale-based evaluation
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

> **Note**: GReaT must be run in a separate conda environment (`tddpm2`).

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
# SMOTE samples directly without a training step
python smote/tune_smote.py exp/orig-micro-retry/ synthetic
python smote/pipeline_smote.py --config exp/orig-micro-retry/smote/config.toml --sample --eval
```

---

## Evaluation Metrics

### MLE (Machine Learning Efficacy) Metrics

| Metric | Description |
|--------|-------------|
| `acc` | Accuracy |
| `f1` | F1 Score |
| `f1_0`, `f1_1` | Per-class F1 Score |
| `f1_weighted` | Weighted F1 Score |
| `balanced_acc` | Balanced Accuracy |
| `mcc` | Matthews Correlation Coefficient |
| `kappa` | Cohen's Kappa |
| `roc_auc` | ROC AUC |

### SynthEval Metrics

| Metric | Description |
|--------|-------------|
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
# SynthEval-style evaluation (custom implementation)
python scripts/eval_syntheval_style.py --config exp/[dataset]/[model]/config.toml \
    --metrics dwm p_mse corr_diff ks_test

# Official SynthEval library evaluation (requires finsyn environment)
conda activate finsyn
python scripts/eval_syntheval_total.py --config exp/[dataset]/[model]/config.toml \
    --preset full_eval --change_val --exclude nnaa
```

### Privacy Metrics

| Metric | Description |
|--------|-------------|
| `ir` | Identity Risk |
| `dcr` | Distance to Closest Record |
| `nndr` | Nearest Neighbor Distance Ratio |
| `tcap` | Target Correct Attribution Probability |

```bash
python scripts/eval_privacy.py --config exp/[dataset]/[model]/config.toml \
    --metrics ir dcr nndr tcap
```

### Fraud Fidelity Analysis

Label distribution preservation and conditional pattern comparison analysis:

```bash
python scripts/eval_fraud_fidelity.py --exp_dir exp/[dataset] \
    --real_data exp/[dataset]/ \
    --models ctgan tvae great ddpm_cb_best \
    --sizes 1.0 1.5 2.0
```

---

## Configuration Files

Each experiment is managed via a `config.toml` file.

### TabDDPM config.toml Example

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

### CTGAN config.toml Example

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

## Result Visualization

```bash
# Scale-based MLE evaluation result plots
python scripts/plot_scale_mle.py --base_dir exp/[dataset]/[model]
```

Generated plots:
- `scale_mle_accuracy.png`
- `scale_mle_f1_neg.png`
- `scale_mle_f1_pos.png`
- `scale_mle_roc_auc.png`

---

## References

### Papers

- **TabDDPM**: [TabDDPM: Modelling Tabular Data with Diffusion Models](https://arxiv.org/abs/2209.15421)
- **CTGAN/TVAE**: [Modeling Tabular data using Conditional GAN](https://arxiv.org/abs/1907.00503)
- **CTAB-GAN**: [CTAB-GAN: Effective Table Data Synthesizing](https://arxiv.org/abs/2102.08369)
- **GReaT**: [Language Models are Realistic Tabular Data Generators](https://arxiv.org/abs/2210.06280)

### Libraries

- [SynthEval](https://github.com/schneiderkamplab/syntheval) - Synthetic data evaluation library
- [SDV (Synthetic Data Vault)](https://github.com/sdv-dev/SDV) - CTGAN/TVAE implementation

---

## License

MIT License - See [LICENSE.md](LICENSE.md) for details.

For submodule licenses, please refer to the LICENSE file in each directory.
