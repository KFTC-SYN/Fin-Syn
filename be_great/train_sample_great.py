import lib
import os
import numpy as np
import pandas as pd
import argparse
from pathlib import Path
import torch
import pickle
import warnings
import sys

# Add be_great to path
sys.path.insert(0, str(Path(__file__).parent))
from be_great.great import GReaT

warnings.filterwarnings("ignore")


def train_great(
    parent_dir,
    real_data_path,
    train_params={"epochs": 100, "batch_size": 8},
    change_val=False,
    device="cpu"
):
    """
    Train GReaT model on tabular data.
    
    Args:
        parent_dir: Directory to save the model
        real_data_path: Path to the real data
        train_params: Dictionary containing GReaT parameters:
            - llm: HuggingFace model name (default: 'distilgpt2')
            - epochs: Number of training epochs (default: 100)
            - batch_size: Batch size (default: 8)
            - float_precision: Number of decimal places for floats (default: None)
            - experiment_dir: Directory for training checkpoints (default: 'trainer_great')
            - random_conditional_col: Whether to use random conditional column (default: True)
        change_val: Whether to use changed validation split
        device: Device to use ('cpu' or 'cuda:X')
    
    Returns:
        GReaT model instance
    """
    real_data_path = Path(real_data_path)
    parent_dir = Path(parent_dir)
    parent_dir.mkdir(parents=True, exist_ok=True)
    
    # Load data
    if change_val:
        X_num_train, X_cat_train, y_train, _, _, _ = lib.read_changed_val(real_data_path)
    else:
        X_num_train, X_cat_train, y_train = lib.read_pure_data(real_data_path, 'train')
    
    # Combine data into DataFrame
    X = lib.concat_to_pd(X_num_train, X_cat_train, y_train)
    X.columns = [str(_) for _ in X.columns]
    
    # Get GReaT parameters
    llm = train_params.get("llm", "distilgpt2")
    epochs = train_params.get("epochs", 100)
    batch_size = train_params.get("batch_size", 8)
    float_precision = train_params.get("float_precision", None)
    experiment_dir = train_params.get("experiment_dir", str(parent_dir / "trainer_great"))
    random_conditional_col = train_params.get("random_conditional_col", True)
    
    # Extract parameters for GReaT initialization (excluding fit-specific and sampling params)
    # Training hyperparameters like save_steps, logging_steps go to train_kwargs
    excluded_params = ["llm", "epochs", "batch_size", "float_precision", 
                       "experiment_dir", "random_conditional_col",
                       "guided_sampling", "temperature", "max_length", "random_feature_order"]
    great_init_params = {k: v for k, v in train_params.items() if k not in excluded_params}
    
    # Convert device string
    if isinstance(device, str) and device.startswith("cuda"):
        device_str = device
    elif device == "cpu":
        device_str = "cpu"
    else:
        device_str = str(device)
    
    # Initialize GReaT model
    print(f"Initializing GReaT with parameters:")
    print(f"  llm: {llm}")
    print(f"  epochs: {epochs}")
    print(f"  batch_size: {batch_size}")
    print(f"  float_precision: {float_precision}")
    print(f"  experiment_dir: {experiment_dir}")
    if great_init_params:
        print(f"  Additional training params: {great_init_params}")
    
    # Initialize GReaT with all parameters
    # Parameters like save_steps, logging_steps will be passed as train_kwargs
    model = GReaT(
        llm=llm,
        experiment_dir=experiment_dir,
        epochs=epochs,
        batch_size=batch_size,
        float_precision=float_precision,
        **great_init_params  # Includes save_steps, logging_steps, etc.
    )
    
    # Train the model
    print(f"Training GReaT on {len(X)} samples...")
    print(f"  Data shape: {X.shape}")
    print(f"  Columns: {list(X.columns)}")
    # Note: column_names parameter is only needed for numpy arrays, not DataFrames
    trainer = model.fit(
        X,
        conditional_col=None,  # Use last column ('y') as conditional
        resume_from_checkpoint=False,
        random_conditional_col=random_conditional_col
    )
    
    # Save model
    model.save(str(parent_dir / "great_model"))
    
    # Also save as pickle for compatibility
    with open(parent_dir / "great.obj", "wb") as f:
        pickle.dump(model, f)
    
    print("GReaT training completed!")
    return model


def sample_great(
    synthesizer,
    parent_dir,
    real_data_path,
    num_samples,
    train_params={"epochs": 100, "batch_size": 8},
    change_val=False,
    device="cpu",
    seed=0
):
    """
    Generate synthetic samples using GReaT.
    
    Args:
        synthesizer: GReaT model instance (can be None, will be loaded)
        parent_dir: Directory to save synthetic data
        real_data_path: Path to the real data
        num_samples: Number of synthetic samples to generate
        train_params: Dictionary containing GReaT parameters
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
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    
    # Load data to get column information
    if change_val:
        X_num_train, X_cat_train, y_train, _, _, _ = lib.read_changed_val(real_data_path)
    else:
        X_num_train, X_cat_train, y_train = lib.read_pure_data(real_data_path, 'train')
    
    # Get task type
    task_type = lib.load_json(real_data_path / "info.json")["task_type"]
    is_regression = (task_type == "regression")
    
    # Load model if not provided
    if synthesizer is None:
        model_path = parent_dir / "great_model"
        if model_path.exists():
            print(f"Loading GReaT model from {model_path}")
            synthesizer = GReaT.load_from_dir(str(model_path))
        else:
            # Try loading from pickle
            pickle_path = parent_dir / "great.obj"
            if pickle_path.exists():
                print(f"Loading GReaT model from pickle {pickle_path}")
                with open(pickle_path, "rb") as f:
                    synthesizer = pickle.load(f)
            else:
                raise FileNotFoundError(
                    f"GReaT model not found. Expected either {model_path} or {pickle_path}"
                )
    
    # Convert device string
    if isinstance(device, str) and device.startswith("cuda"):
        device_str = device
    elif device == "cpu":
        device_str = "cpu"
    else:
        device_str = str(device)
    
    # Get sampling parameters
    guided_sampling = train_params.get("guided_sampling", False)
    temperature = train_params.get("temperature", 0.7)
    max_length = train_params.get("max_length", 200)
    random_feature_order = train_params.get("random_feature_order", True)
    
    # Generate synthetic data
    print(f"Generating {num_samples} synthetic samples...")
    print(f"Task type: {task_type}")
    print(f"Guided sampling: {guided_sampling}")
    
    synthetic_df = synthesizer.sample(
        n_samples=num_samples,
        temperature=temperature,
        max_length=max_length,
        device=device_str,
        guided_sampling=guided_sampling,
        random_feature_order=random_feature_order,
        drop_nan=False
    )
    
    print(f"Generated synthetic data shape: {synthetic_df.shape}")
    print(f"Generated columns: {list(synthetic_df.columns)}")
    
    # Get original column structure
    original_X = lib.concat_to_pd(X_num_train, X_cat_train, y_train)
    original_X.columns = [str(_) for _ in original_X.columns]
    
    # Get column information
    n_num_features = X_num_train.shape[1] if X_num_train is not None else 0
    n_cat_features = X_cat_train.shape[1] if X_cat_train is not None else 0
    
    # Extract target column (should be 'y')
    if 'y' in synthetic_df.columns:
        y_synth = synthetic_df['y'].values
        X_synth = synthetic_df.drop(columns=['y'])
        
        # 디버깅: y 값의 범위 확인
        unique_y_synth = np.unique(y_synth)
        if len(unique_y_synth) > 0:
            y_min, y_max = unique_y_synth.min(), unique_y_synth.max()
            unique_y_train = np.unique(y_train)
            train_min, train_max = unique_y_train.min(), unique_y_train.max()
            if y_min < train_min or y_max > train_max:
                print(f"Warning: Synthetic y values out of range. "
                      f"Train range: [{train_min}, {train_max}], "
                      f"Synthetic range: [{y_min}, {y_max}], "
                      f"Unique synthetic values: {unique_y_synth}")
    else:
        # If 'y' is not found, try to infer from original data
        print("Warning: 'y' column not found in synthetic data. Using last column as target.")
        y_synth = synthetic_df.iloc[:, -1].values
        X_synth = synthetic_df.iloc[:, :-1]
    
    # Ensure synthetic data has the same column structure as original
    # GReaT should generate columns in the same order as training data
    expected_cols = [str(i) for i in range(len(original_X.columns) - 1)]  # Excluding 'y'
    
    if list(X_synth.columns) != expected_cols:
        # Reorder columns to match original structure
        if len(X_synth.columns) == len(expected_cols):
            # Columns match in count but may be in different order
            # Try to match by name first
            if set(X_synth.columns) == set(expected_cols):
                X_synth = X_synth[expected_cols]
            else:
                # If column names don't match, assume same order and rename
                X_synth.columns = expected_cols
        else:
            # Column count mismatch - use positional mapping
            print(f"Warning: Column count mismatch. Expected {len(expected_cols)}, got {len(X_synth.columns)}")
            if len(X_synth.columns) >= len(expected_cols):
                X_synth = X_synth.iloc[:, :len(expected_cols)]
                X_synth.columns = expected_cols
            else:
                # Pad with NaN if needed
                missing_cols = len(expected_cols) - len(X_synth.columns)
                for i in range(missing_cols):
                    X_synth[f'{len(X_synth.columns)}'] = np.nan
                X_synth.columns = expected_cols
    
    # Split into numerical and categorical
    if n_num_features > 0 and n_cat_features > 0:
        X_num_synth = X_synth.iloc[:, :n_num_features]
        X_cat_synth = X_synth.iloc[:, n_num_features:]
        
        # Convert numerical columns to float
        for col in X_num_synth.columns:
            X_num_synth[col] = pd.to_numeric(X_num_synth[col], errors='coerce')
        
        # Convert categorical columns to string
        for col in X_cat_synth.columns:
            X_cat_synth[col] = X_cat_synth[col].astype(str)
        
        # Save synthetic data
        np.save(parent_dir / 'X_num_train', X_num_synth.values.astype(float))
        np.save(parent_dir / 'X_cat_train', X_cat_synth.values.astype(str))
    elif n_num_features > 0:
        # Only numerical features
        for col in X_synth.columns:
            X_synth[col] = pd.to_numeric(X_synth[col], errors='coerce')
        np.save(parent_dir / 'X_num_train', X_synth.values.astype(float))
    elif n_cat_features > 0:
        # Only categorical features
        for col in X_synth.columns:
            X_synth[col] = X_synth[col].astype(str)
        np.save(parent_dir / 'X_cat_train', X_synth.values.astype(str))
    
    # Save labels
    y_synth = pd.to_numeric(y_synth, errors='coerce').astype(float)
    if not is_regression:
        y_synth = y_synth.astype(int)
        
        # 2025.12.02 _catboost.CatBoostError: Unknown class label: "157" 관련 추가
        # 유효한 클래스 범위로 제한
        # 학습 데이터의 유효한 클래스 값 가져오기
        unique_classes = np.unique(y_train)
        min_class = int(unique_classes.min())
        max_class = int(unique_classes.max())
        # 범위를 벗어나는 값을 클리핑
        y_synth = np.clip(y_synth, min_class, max_class)

    np.save(parent_dir / 'y_train', y_synth)
    
    print("Synthetic data saved successfully!")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('real_data_path', type=str)
    parser.add_argument('parent_dir', type=str)
    parser.add_argument('train_size', type=int)
    args = parser.parse_args()

    model = train_great(args.parent_dir, args.real_data_path, change_val=True)
    sample_great(
        model, args.parent_dir, args.real_data_path, 
        args.train_size, change_val=True
    )


if __name__ == '__main__':
    main()

