# TabPFGen: Synthetic Tabular Data Generation with TabPFN

[![PyPI version](https://badge.fury.io/py/tabpfgen.svg)](https://badge.fury.io/py/tabpfgen)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

![TabPFGen Overview](docs/images/tabpfgen_featureimage.jpg)

TabPFGen is a Python library for generating high-quality synthetic tabular data using energy-based modeling and stochastic gradient Langevin dynamics (SGLD). It supports both classification and regression tasks with built-in visualization capabilities.

*Integration with TabPFN Extensions: TabPFGen is being integrated into the tabpfn-extensions ecosystem as a separate module ([TabPFGen Data Synthesizer Extension](https://github.com/sebhaan/tabpfn-extensions/tree/tabpfgen-datasynthesizer/src/tabpfn_extensions/tabpfgen_datasynthesizer), [PR #83](https://github.com/PriorLabs/tabpfn-extensions/pull/83)), which will enable seamless integration with other TabPFN tools and extensions.*

## Motivation

While there are many tools available for generating synthetic images or text, creating realistic tabular data that preserves the statistical properties and relationships of the original dataset has been more challenging.

Generating synthetic tabular data is particularly useful in scenarios where:

1. You have limited real data but need more samples for training
2. You can't share real data due to privacy concerns
3. You need to balance an imbalanced dataset
4. You want to test how your models would perform with more data

What makes TabPFGen interesting is that it's built on the TabPFN transformer architecture and doesn't require additional training. It includes built-in visualization tools to help you verify the quality of the generated data by comparing distributions, feature correlations, and other important metrics between the real and synthetic datasets.


## Key Features

- Energy-based synthetic data generation
- Support for both classification and regression tasks
- Automatic dataset balancing for imbalanced classes
- Class-balanced sampling option
- Comprehensive visualization tools
- Built on TabPFN transformer architecture
- No additional training required

## Installation

```bash
pip install tabpfgen
```

Verify installation
```bash
python -c 'from tabpfgen import TabPFGen; print("Installation successful!")'
```

## Quick Start

### Classification Example

```python
from tabpfgen import TabPFGen
from tabpfgen.visuals import visualize_classification_results
from sklearn.datasets import load_breast_cancer

# Load data
X, y = load_breast_cancer(return_X_y=True)

# Initialize generator
generator = TabPFGen(n_sgld_steps=500)

# Generate synthetic data
X_synth, y_synth = generator.generate_classification(
    X, y,
    n_samples=100,
    balance_classes=True
)

# Visualize results
visualize_classification_results(
    X, y, X_synth, y_synth,
    feature_names=load_breast_cancer().feature_names
)
```

### Dataset Balancing Example

```python
from tabpfgen import TabPFGen
from tabpfgen.visuals import visualize_classification_results
from sklearn.datasets import make_classification

# Create imbalanced dataset
X, y = make_classification(n_samples=1000, n_classes=3, 
                         n_informative=3, n_redundant=1,
                         weights=[0.7, 0.2, 0.1], random_state=42)

# Initialize generator
generator = TabPFGen(n_sgld_steps=500)

# Balance dataset automatically (balances to majority class size)
X_synth, y_synth, X_combined, y_combined = generator.balance_dataset(X, y)

# Or specify custom target per class:
X_synth, y_synth, X_combined, y_combined = generator.balance_dataset(
    X, y, target_per_class=1000
)

print(f"Original dataset: {len(X)} samples")
print(f"Synthetic samples: {len(X_synth)} samples") 
print(f"Combined dataset: {len(X_combined)} samples")

visualize_classification_results(
    X, y, X_synth, y_synth,
    feature_names=[f'feature_{i}' for i in range(X.shape[1])]
)
```

**Note on Balancing Results**: The final class distribution may be approximately balanced rather than perfectly balanced. This is due to TabPFN's label refinement process, which prioritizes data quality and realism over exact class counts. The method ensures significant improvement in class balance while maintaining high-quality synthetic samples.

### Regression Example

```python
from tabpfgen import TabPFGen
from tabpfgen.visuals import visualize_regression_results
from sklearn.datasets import load_diabetes

# Load regression dataset
X, y = load_diabetes(return_X_y=True)

# Initialize generator
generator = TabPFGen(n_sgld_steps=500)

# Generate synthetic regression data
X_synth, y_synth = generator.generate_regression(
    X, y,
    n_samples=100,
    use_quantiles=True
)

# Visualize results
visualize_regression_results(
    X, y, X_synth, y_synth,
    feature_names=load_diabetes().feature_names
)
```

## Visualization Features

The package includes comprehensive visualization tools:

### Classification Visualizations
- Class distribution comparison
- t-SNE visualization of feature space
- Feature importance analysis
- Feature distribution comparisons
- Feature correlation matrices

### Regression Visualizations
- Target value distribution comparison
- Q-Q plots for distribution analysis
- Box plot comparisons
- Feature importance analysis
- Scatter plots of important features
- t-SNE visualization with target value mapping
- Residuals analysis
- Feature correlation matrices

## Parameters

### TabPFGen
- `n_sgld_steps`: Number of SGLD iterations (default: 1000)
- `sgld_step_size`: Step size for SGLD updates (default: 0.01)
- `sgld_noise_scale`: Scale of noise in SGLD (default: 0.01)
- `device`: Computing device ('cpu' or 'cuda', default: 'auto')

### Classification Generation
- `n_samples`: Number of synthetic samples to generate
- `balance_classes`: Whether to generate balanced class distributions (default: True)

### Dataset Balancing
- `target_per_class`: Target number of samples per class (default: None, uses majority class size)  
- `min_class_size`: Minimum class size to include in balancing (default: 5)

### Regression Generation
- `n_samples`: Number of synthetic samples to generate
- `use_quantiles`: Whether to use quantile-based sampling (default: True)

### Tests

```bash
python -m unittest tests/test_tabpfgen.py
```

## Documentation

For detailed documentation and tutorials, visit our [tutorial pages](https://github.com/sebhaan/TabPFGen/blob/main/tutorial/index.md).

## How It Works

1. **Energy-Based Modeling**: Uses a distance-based energy function that combines:
   - Feature space distances between synthetic and real samples
   - Class-conditional information for classification tasks

2. **SGLD Sampling**: Generates synthetic samples through iterative updates:
   ```
   x_new = x - step_size * gradient + noise_scale * random_noise
   ```

3. **Quality Assurance**:
   - Automatic feature scaling
   - Class balance maintenance
   - Distribution matching through energy minimization
   - Quantile-based sampling for regression

## Limitations

- Memory usage scales with dataset size
- SGLD convergence can be sensitive to step size parameters
- Computation time increases with `n_sgld_steps`
- Dataset balancing produces approximate rather than perfect balance due to TabPFN's quality-focused label refinement process


## References

This project is inspired by the TabPFGen method described in Ma, Junwei, et al. This is an independent implementation and may not strictly follow all aspects 
of the original approach. We are not affiliated with the original authors.

Ma, Junwei, et al. "TabPFGen--Tabular Data Generation with TabPFN." arXiv preprint arXiv:2406.05216 (2024).

Hollmann, Noah, et al. "Accurate predictions on small data with a tabular foundation model." Nature 637.8045 (2025): 319-326.
