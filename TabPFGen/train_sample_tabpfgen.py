import lib
import os
import numpy as np
import argparse
from pathlib import Path
import torch
import pickle
import warnings
from sklearn.preprocessing import OrdinalEncoder
from sklearn.model_selection import train_test_split
import sys

# TabPFGen import - use local version (has ignore_pretraining_limits=True fix)
from src.tabpfgen.tabpfgen import TabPFGen

warnings.filterwarnings("ignore")

def train_tabpfgen(
    parent_dir,
    real_data_path,
    train_params={"n_sgld_steps": 1000},
    change_val=False,
    device="cpu"
):
    """
    Initialize TabPFGen model (no actual training needed as it uses pre-trained TabPFN).
    
    Args:
        parent_dir: Directory to save the model
        real_data_path: Path to the real data
        train_params: Dictionary containing TabPFGen parameters:
            - n_sgld_steps: Number of SGLD steps (default: 1000)
            - sgld_step_size: Step size for SGLD (default: 0.01)
            - sgld_noise_scale: Noise scale for SGLD (default: 0.01)
        change_val: Whether to use changed validation split
        device: Device to use ('cpu' or 'cuda:X')
    
    Returns:
        TabPFGen generator instance
    """
    real_data_path = Path(real_data_path)
    parent_dir = Path(parent_dir)
    parent_dir.mkdir(parents=True, exist_ok=True)
    
    # Convert device string to TabPFGen format
    if isinstance(device, str) and device.startswith("cuda"):
        device_str = device
    elif device == "cpu":
        device_str = "cpu"
    else:
        device_str = str(device)
    
    # Initialize TabPFGen with parameters
    n_sgld_steps = train_params.get("n_sgld_steps", 1000)
    sgld_step_size = train_params.get("sgld_step_size", 0.01)
    sgld_noise_scale = train_params.get("sgld_noise_scale", 0.01)
    
    generator = TabPFGen(
        n_sgld_steps=n_sgld_steps,
        sgld_step_size=sgld_step_size,
        sgld_noise_scale=sgld_noise_scale,
        device=device_str
    )
    
    # Save generator (optional, TabPFGen doesn't require saving but we save params)
    with open(parent_dir / "tabpfgen.obj", "wb") as f:
        pickle.dump({
            "n_sgld_steps": n_sgld_steps,
            "sgld_step_size": sgld_step_size,
            "sgld_noise_scale": sgld_noise_scale,
            "device": device_str
        }, f)
    
    print(f"TabPFGen initialized with parameters:")
    print(f"  n_sgld_steps: {n_sgld_steps}")
    print(f"  sgld_step_size: {sgld_step_size}")
    print(f"  sgld_noise_scale: {sgld_noise_scale}")
    print(f"  device: {device_str}")
    
    return generator


def sample_tabpfgen(
    synthesizer,
    parent_dir,
    real_data_path,
    num_samples,
    train_params={"n_sgld_steps": 1000},
    change_val=False,
    device="cpu",
    seed=0
):
    """
    Generate synthetic samples using TabPFGen.
    
    Args:
        synthesizer: TabPFGen generator instance (can be None, will be loaded)
        parent_dir: Directory to save synthetic data
        real_data_path: Path to the real data
        num_samples: Number of synthetic samples to generate
        train_params: Dictionary containing TabPFGen parameters
        change_val: Whether to use changed validation split
        device: Device to use
        seed: Random seed for reproducibility
    """
    real_data_path = Path(real_data_path)
    parent_dir = Path(parent_dir)
    parent_dir.mkdir(parents=True, exist_ok=True)
    
    # Set random seeds
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    # Load data
    if change_val:
        X_num_train, X_cat_train, y_train, _, _, _ = lib.read_changed_val(real_data_path)
    else:
        X_num_train, X_cat_train, y_train = lib.read_pure_data(real_data_path, 'train')
    
    # Get task type
    task_type = lib.load_json(real_data_path / "info.json")["task_type"]
    is_regression = (task_type == "regression")
    
    # Limit data to 10,000 samples for TabPFN compatibility
    MAX_SAMPLES = 10000
    n_samples = len(y_train)
    if n_samples > MAX_SAMPLES:
        print(f"Warning: Dataset has {n_samples} samples, limiting to {MAX_SAMPLES} for TabPFN compatibility")
        
        # Prepare indices for sampling
        indices = np.arange(n_samples)
        
        if is_regression:
            # For regression: random sampling
            indices_sampled, _ = train_test_split(
                indices, 
                train_size=MAX_SAMPLES, 
                random_state=seed,
                shuffle=True
            )
        else:
            # For classification: stratified sampling to maintain class distribution
            indices_sampled, _ = train_test_split(
                indices,
                train_size=MAX_SAMPLES,
                random_state=seed,
                shuffle=True,
                stratify=y_train
            )
        
        # Sample the data
        X_num_train = X_num_train[indices_sampled]
        X_cat_train = X_cat_train[indices_sampled]
        y_train = y_train[indices_sampled]
        
        print(f"Sampled {len(y_train)} samples (original: {n_samples})")
    
    # Convert device string
    if isinstance(device, str) and device.startswith("cuda"):
        device_str = device
    elif device == "cpu":
        device_str = "cpu"
    else:
        device_str = str(device)
    
    # Initialize generator if not provided
    if synthesizer is None:
        n_sgld_steps = train_params.get("n_sgld_steps", 1000)
        sgld_step_size = train_params.get("sgld_step_size", 0.01)
        sgld_noise_scale = train_params.get("sgld_noise_scale", 0.01)
        
        synthesizer = TabPFGen(
            n_sgld_steps=n_sgld_steps,
            sgld_step_size=sgld_step_size,
            sgld_noise_scale=sgld_noise_scale,
            device=device_str
        )
    
    # Prepare input data: combine numerical and categorical features
    # TabPFGen works with numerical data, so we need to encode categorical features
    if X_cat_train is not None:
        # Encode categorical features to numerical
        encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
        X_cat_encoded = encoder.fit_transform(X_cat_train)
        
        # Combine numerical and encoded categorical features
        if X_num_train is not None:
            X_train = np.hstack([X_num_train, X_cat_encoded])
        else:
            X_train = X_cat_encoded
    else:
        X_train = X_num_train
    
    # Ensure X_train is 2D
    if X_train is None:
        raise ValueError("No features available (both X_num and X_cat are None)")
    
    # Generate synthetic data
    print(f"Generating {num_samples} synthetic samples...")
    print(f"Task type: {task_type}")
    print(f"Input shape: {X_train.shape}")
    
    # Get task-specific parameters from train_params
    use_quantiles = train_params.get("use_quantiles", True)
    balance_classes = train_params.get("balance_classes", True)
    
    if is_regression:
        # Regression task
        X_synth, y_synth = synthesizer.generate_regression(
            X_train, y_train,
            n_samples=num_samples,
            use_quantiles=use_quantiles
        )
    else:
        # Classification task
        X_synth, y_synth = synthesizer.generate_classification(
            X_train, y_train,
            n_samples=num_samples,
            balance_classes=balance_classes
        )
    
    print(f"Generated synthetic data shape: {X_synth.shape}")
    print(f"Generated labels shape: {y_synth.shape}")
    
    # Split back into numerical and categorical features if needed
    # Since TabPFGen generates continuous values, we need to handle categorical features
    if X_cat_train is not None and X_num_train is not None:
        # Split synthetic features back
        n_num_features = X_num_train.shape[1]
        X_num_synth = X_synth[:, :n_num_features]
        X_cat_synth_encoded = X_synth[:, n_num_features:]
        
        # Round categorical features to nearest integer and clip to valid range
        X_cat_synth_encoded = np.round(X_cat_synth_encoded).astype(int)
        # Clip to valid range [0, max_category_value]
        for i in range(X_cat_synth_encoded.shape[1]):
            max_val = int(X_cat_train[:, i].max()) if len(np.unique(X_cat_train[:, i])) > 0 else 0
            X_cat_synth_encoded[:, i] = np.clip(X_cat_synth_encoded[:, i], 0, max_val)
        
        # Convert back to original categorical format (as strings)
        # This is approximate - TabPFGen generates continuous values
        X_cat_synth = X_cat_synth_encoded.astype(str)
        
        # Save synthetic data
        np.save(parent_dir / 'X_num_train', X_num_synth.astype(float))
        np.save(parent_dir / 'X_cat_train', X_cat_synth.astype(str))
    elif X_num_train is not None:
        # Only numerical features
        np.save(parent_dir / 'X_num_train', X_synth.astype(float))
    elif X_cat_train is not None:
        # Only categorical features (encoded)
        X_cat_synth_encoded = np.round(X_synth).astype(int)
        # Clip to valid range
        for i in range(X_cat_synth_encoded.shape[1]):
            max_val = int(X_cat_train[:, i].max()) if len(np.unique(X_cat_train[:, i])) > 0 else 0
            X_cat_synth_encoded[:, i] = np.clip(X_cat_synth_encoded[:, i], 0, max_val)
        X_cat_synth = X_cat_synth_encoded.astype(str)
        np.save(parent_dir / 'X_cat_train', X_cat_synth.astype(str))
    
    # Save labels
    y_synth = y_synth.astype(float)
    if not is_regression:
        y_synth = y_synth.astype(int)
    np.save(parent_dir / 'y_train', y_synth)
    
    print("Synthetic data saved successfully!")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('real_data_path', type=str)
    parser.add_argument('parent_dir', type=str)
    parser.add_argument('train_size', type=int)
    args = parser.parse_args()

    generator = train_tabpfgen(args.parent_dir, args.real_data_path, change_val=True)
    sample_tabpfgen(
        generator, args.parent_dir, args.real_data_path, 
        args.train_size, change_val=True
    )


if __name__ == '__main__':
    main()

