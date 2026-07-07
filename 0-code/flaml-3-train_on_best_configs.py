#!/usr/bin/env python3
"""
Train models using best configurations extracted from FLAML optimization.

This script:
1. Loads best configs from JSON file
2. Selects top-N methods by validation loss
3. Trains each method on training set and evaluates on test set
4. Trains on combined train+test and predicts on global challenge test set
5. Saves predictions and performance metrics
"""

import json
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
import lightgbm as lgb
import xgboost as xgb
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.linear_model import SGDClassifier, LogisticRegression
import warnings
warnings.filterwarnings('ignore')

try:
    from catboost import CatBoostClassifier
    CATBOOST_AVAILABLE = True
except ImportError:
    CATBOOST_AVAILABLE = False
    print("Warning: CatBoost not available, will skip catboost models")


def load_best_configs(config_file: Path) -> dict:
    """Load best configurations from JSON file."""
    with open(config_file, 'r') as f:
        return json.load(f)


def select_top_methods(configs: dict, n_top: int = 3) -> list:
    """
    Select top-N methods by validation loss.
    
    Returns:
        List of tuples: (learner_name, config_data)
    """
    all_methods = []
    for learner, config_list in configs.items():
        if config_list:  # Check if list is not empty
            # Take the best config for this learner
            all_methods.append((learner, config_list[0]))
    
    # Sort by validation loss (lower is better)
    all_methods.sort(key=lambda x: x[1]['validation_loss'])
    
    # Return top N
    return all_methods[:n_top]


def create_model(learner: str, config: dict, random_state: int = 42):
    """
    Create a model instance from learner name and config.
    
    Args:
        learner: Name of the learner (lgbm, xgboost, etc.)
        config: Configuration dictionary
        random_state: Random seed for reproducibility
    
    Returns:
        Initialized model instance
    """
    # Make a copy to avoid modifying original
    params = config.copy()
    
    if learner == 'lgbm':
        # Convert log_max_bin to max_bin
        if 'log_max_bin' in params:
            params['max_bin'] = 2 ** params.pop('log_max_bin')
        
        # Remove FLAML-specific parameters
        params.pop('FLAML_sample_size', None)
        
        model = lgb.LGBMClassifier(
            **params,
            random_state=random_state,
            verbose=-1,
            n_jobs=-1
        )
    
    elif learner == 'xgboost' or learner == 'xgb_limitdepth':
        # Remove FLAML-specific parameters
        params.pop('FLAML_sample_size', None)
        
        model = xgb.XGBClassifier(
            **params,
            random_state=random_state,
            eval_metric='logloss',
            n_jobs=-1,
            verbosity=0
        )
    
    elif learner == 'extra_tree':
        # Convert max_leaves to max_leaf_nodes for sklearn
        if 'max_leaves' in params:
            params['max_leaf_nodes'] = params.pop('max_leaves')
        
        # Remove FLAML-specific parameters
        params.pop('FLAML_sample_size', None)
        
        model = ExtraTreesClassifier(
            **params,
            random_state=random_state,
            n_jobs=-1,
            verbose=0
        )
    
    elif learner == 'rf':
        # Convert max_leaves to max_leaf_nodes for sklearn
        if 'max_leaves' in params:
            params['max_leaf_nodes'] = params.pop('max_leaves')
        
        # Remove FLAML-specific parameters
        params.pop('FLAML_sample_size', None)
        
        model = RandomForestClassifier(
            **params,
            random_state=random_state,
            n_jobs=-1,
            verbose=0
        )
    
    elif learner == 'sgd':
        # Remove FLAML-specific parameters
        params.pop('FLAML_sample_size', None)
        
        model = SGDClassifier(
            **params,
            random_state=random_state,
            max_iter=1000,
            tol=1e-3
        )
    
    elif learner == 'catboost':
        if not CATBOOST_AVAILABLE:
            raise ValueError("CatBoost is not available")
        
        # Remove FLAML-specific parameters
        params.pop('FLAML_sample_size', None)
        
        model = CatBoostClassifier(
            **params,
            random_state=random_state,
            verbose=False,
            thread_count=-1
        )
    
    elif learner == 'lrl1':
        # Remove FLAML-specific parameters
        params.pop('FLAML_sample_size', None)
        
        model = LogisticRegression(
            penalty='l1',
            solver='saga',
            **params,
            random_state=random_state,
            max_iter=1000,
            n_jobs=-1,
            verbose=0
        )
    
    else:
        raise ValueError(f"Unknown learner: {learner}")
    
    return model


def load_data(data_path: Path) -> tuple:
    """
    Load CSV data file.
    
    Returns:
        X, y (or just X if no activity/target column)
    """
    df = pd.read_csv(data_path)
    
    # Replace NaN values with 0 (consistent with FLAML training)
    nan_count = df.isna().sum().sum()
    if nan_count > 0:
        df = df.fillna(0)
    
    # Check if activity column exists (standard for this project)
    if 'activity' in df.columns:
        X = df.drop('activity', axis=1)
        y = df['activity']
        return X, y
    # Fallback: check for 'target' column
    elif 'target' in df.columns:
        X = df.drop('target', axis=1)
        y = df['target']
        return X, y
    else:
        # No target column - this is a prediction-only dataset
        return df, None


def sanitize_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """
    Sanitize column names to be compatible with all models (especially LGBM).
    LGBM doesn't accept special JSON characters like: [ ] { } " : ,
    
    Args:
        df: DataFrame with potentially problematic column names
    
    Returns:
        DataFrame with sanitized column names
    """
    # Create a copy to avoid modifying original
    df = df.copy()
    
    # Replace special characters with underscores
    new_columns = []
    for col in df.columns:
        # Replace problematic characters
        new_col = str(col)
        for char in '[]{}",:':
            new_col = new_col.replace(char, '_')
        # Remove multiple consecutive underscores
        while '__' in new_col:
            new_col = new_col.replace('__', '_')
        # Remove leading/trailing underscores
        new_col = new_col.strip('_')
        new_columns.append(new_col)
    
    df.columns = new_columns
    return df


def infer_file_paths(config_file: Path) -> dict:
    """
    Infer train, test, and global test file paths from config file name.

    DEPRECATED: This function is kept for backward compatibility only.
    The shell script should always pass --train-file, --test-file, and --global-test-file explicitly.

    Returns:
        Dictionary with None values (explicit paths should be provided)
    """
    print("WARNING: Using deprecated path inference. Please pass file paths explicitly:")
    print("         --train-file, --test-file, --global-test-file")
    print("         The shell script flaml-0-run_them_all.sh should handle this automatically.")

    return {
        'train': None,
        'test': None,
        'global_test': None
    }


def main():
    parser = argparse.ArgumentParser(
        description='Train models using best FLAML configurations',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Use top 3 methods, auto-infer file paths
  python train_best_configs.py best_configs.json
  
  # Use top 5 methods
  python train_best_configs.py best_configs.json -n 5
  
  # Specify data files explicitly
  python train_best_configs.py best_configs.json \\
    --train-file train.csv.gz \\
    --test-file test.csv.gz \\
    --global-test-file global_test.csv.gz
  
  # Custom output directory
  python train_best_configs.py best_configs.json -o predictions/
        """
    )
    
    parser.add_argument('config_file', type=Path,
                        help='Path to best_configs.json file')
    parser.add_argument('-n', '--n-top', type=int, default=3,
                        help='Number of top methods to use (default: 3)')
    parser.add_argument('--train-file', type=Path, default=None,
                        help='Training data file (auto-inferred if not provided)')
    parser.add_argument('--test-file', type=Path, default=None,
                        help='Test data file (auto-inferred if not provided)')
    parser.add_argument('--global-test-file', type=Path, default=None,
                        help='Global challenge test file (auto-inferred if not provided)')
    parser.add_argument('-o', '--output-dir', type=Path, default=None,
                        help='Output directory (default: same as config file)')
    parser.add_argument('--random-state', type=int, default=42,
                        help='Random seed (default: 42)')
    parser.add_argument('--skip-test-eval', action='store_true',
                        help='Skip evaluation on test set (only train on full data)')
    
    args = parser.parse_args()
    
    # Validate config file
    if not args.config_file.exists():
        print(f"Error: Config file not found: {args.config_file}")
        return 1
    
    # Set output directory
    if args.output_dir is None:
        output_dir = args.config_file.parent
    else:
        output_dir = args.output_dir
        output_dir.mkdir(parents=True, exist_ok=True)
    
    # Use simplified output names (no encoding of dataset/descriptor names)
    output_base = "model"  # Will create model__test_results.csv, etc.
    
    print("=" * 80)
    print("TRAINING MODELS FROM BEST FLAML CONFIGURATIONS")
    print("=" * 80)
    print(f"Config file: {args.config_file}")
    print(f"Top N methods: {args.n_top}")
    print(f"Random state: {args.random_state}")
    print(f"Output directory: {output_dir}")
    print()
    
    # Load best configs
    print("Loading best configurations...")
    configs = load_best_configs(args.config_file)
    print(f"Loaded {len(configs)} learner types")
    
    # Select top methods
    top_methods = select_top_methods(configs, args.n_top)
    print(f"\nSelected top {len(top_methods)} methods:")
    for i, (learner, cfg) in enumerate(top_methods, 1):
        val_loss = cfg['validation_loss']
        print(f"  {i}. {learner}: validation_loss = {val_loss:.6f}")
    print()
    
    # Infer or use provided file paths
    if args.train_file is None or args.test_file is None:
        print("Inferring file paths from config filename...")
        paths = infer_file_paths(args.config_file)
        train_file = args.train_file or paths['train']
        test_file = args.test_file or paths['test']
        global_test_file = args.global_test_file or paths['global_test']
    else:
        train_file = args.train_file
        test_file = args.test_file
        global_test_file = args.global_test_file
    
    print(f"Training file: {train_file}")
    print(f"Test file: {test_file}")
    print(f"Global test file: {global_test_file}")
    print()
    
    # Validate train file
    if not train_file.exists():
        print(f"Error: Training file not found: {train_file}")
        return 1
    
    # Load training data
    print("Loading training data...")
    X_train, y_train = load_data(train_file)
    print(f"Training set: {X_train.shape[0]} samples, {X_train.shape[1]} features")
    
    if y_train is None:
        print("Error: No target column ('activity' or 'target') found in training data")
        print("Available columns:", X_train.columns.tolist()[:10], "...")
        return 1
    
    # Sanitize column names for compatibility with all models (especially LGBM)
    X_train = sanitize_column_names(X_train)
    
    print(f"Target distribution: {np.bincount(y_train.astype(int))}")
    print()
    
    # Load test data if exists and not skipping
    X_test, y_test = None, None
    if not args.skip_test_eval and test_file.exists():
        print("Loading test data...")
        X_test, y_test = load_data(test_file)
        print(f"Test set: {X_test.shape[0]} samples, {X_test.shape[1]} features")
        
        # Sanitize column names to match training data
        X_test = sanitize_column_names(X_test)
        
        if y_test is not None:
            print(f"Target distribution: {np.bincount(y_test.astype(int))}")
        print()
    
    # Results storage
    results = []
    test_predictions = {}
    
    # Train and evaluate each method
    print("=" * 80)
    print("PHASE 1: TRAIN ON TRAINING SET, EVALUATE ON TEST SET")
    print("=" * 80)
    print()

    if y_test is not None:
         test_predictions["activity"] = y_test

    
    for i, (learner, cfg_data) in enumerate(top_methods, 1):
        print(f"[{i}/{len(top_methods)}] Training {learner}...")
        print(f"  Validation loss (from FLAML): {cfg_data['validation_loss']:.6f}")
        
        try:
            # Create model
            model = create_model(learner, cfg_data['config'], args.random_state)
            
            # Train on training set
            print(f"  Training on {X_train.shape[0]} samples...")
            model.fit(X_train, y_train)
            
            # Evaluate on test set if available
            if X_test is not None and y_test is not None:
                print(f"  Evaluating on test set...")
                y_pred_proba = model.predict_proba(X_test)[:, 1]
                test_auroc = roc_auc_score(y_test, y_pred_proba)
                print(f"  Test AUROC: {test_auroc:.6f}")
                
                # Store predictions
                test_predictions[learner] = y_pred_proba
                
                # Store results
                results.append({
                    'method': learner,
                    'flaml_validation_loss': cfg_data['validation_loss'],
                    'test_auroc': test_auroc,
                    'config': cfg_data['config']
                })
            else:
                print(f"  Test set not available, skipping evaluation")
                results.append({
                    'method': learner,
                    'flaml_validation_loss': cfg_data['validation_loss'],
                    'test_auroc': None,
                    'config': cfg_data['config']
                })
            
            print(f"  Done")
            
        except Exception as e:
            print(f"  — Error training {learner}: {e}")
            results.append({
                'method': learner,
                'flaml_validation_loss': cfg_data['validation_loss'],
                'test_auroc': None,
                'error': str(e),
                'config': cfg_data['config']
            })
        
        print()
    
    # Create subdirectories for organized output
    predictions_dir = output_dir / "predictions"
    predictions_dir.mkdir(parents=True, exist_ok=True)

    # Save test set results
    if results:
        results_df = pd.DataFrame(results)
        results_file = output_dir / "results.csv"
        results_df.to_csv(results_file, index=False)
        print(f"Test results saved to: {results_file}")
        print()

        # Print summary
        print("=" * 80)
        print("TEST SET EVALUATION SUMMARY")
        print("=" * 80)
        print(results_df[['method', 'flaml_validation_loss', 'test_auroc']].to_string(index=False))
        print()

    # Save test predictions
    if test_predictions:
        pred_file = predictions_dir / "test_predictions.csv"
        pred_df = pd.DataFrame(test_predictions)
        pred_df.to_csv(pred_file, index=False)
        print(f"Test predictions saved to: {pred_file}")
        print()
    
    # Phase 2: Train on combined data and predict on global test
    print("=" * 80)
    print("PHASE 2: TRAIN ON FULL DATA, PREDICT ON CHALLENGE TEST SET")
    print("=" * 80)
    print()
    
    if global_test_file is None:
        print("— Global test file not provided (use --global-test-file)")
        print("   Skipping Phase 2 - no challenge test predictions will be generated")
        print()
    elif not global_test_file.exists():
        print(f"— Global test file not found: {global_test_file}")
        print("   Skipping Phase 2 - no challenge test predictions will be generated")
        print()
    else:
        print(f"Global test file found: {global_test_file}")
        print()
        
        # Combine train and test data
        if X_test is not None:
            print("Combining training and test data...")
            X_full = pd.concat([X_train, X_test], axis=0, ignore_index=True)
            y_full = pd.concat([pd.Series(y_train), pd.Series(y_test)], axis=0, ignore_index=True)
            print(f"Combined dataset: {X_full.shape[0]} samples")
        else:
            X_full = X_train
            y_full = y_train
            print(f"Using training data only: {X_full.shape[0]} samples")
        print()
        
        # Load global test data
        print("Loading global test data...")
        X_global_test, _ = load_data(global_test_file)
        print(f"Global test set: {X_global_test.shape[0]} samples")
        
        # Sanitize column names to match training data
        X_global_test = sanitize_column_names(X_global_test)
        print()
        
        # Train and predict with each method
        global_predictions = {}
        
        for i, (learner, cfg_data) in enumerate(top_methods, 1):
            print(f"[{i}/{len(top_methods)}] Training {learner} on full data...")
            
            try:
                # Create model
                model = create_model(learner, cfg_data['config'], args.random_state)
                
                # Train on full data
                print(f"  Training on {X_full.shape[0]} samples...")
                model.fit(X_full, y_full)
                
                # Predict on global test
                print(f"  Predicting on global test set...")
                y_global_pred = model.predict_proba(X_global_test)[:, 1]
                global_predictions[learner] = y_global_pred
                
                print(f"  Done")
                
            except Exception as e:
                print(f"  — Error: {e}")
            
            print()
        
        # Save global test predictions
        if global_predictions:
            global_pred_file = predictions_dir / "global_test_predictions.csv"
            global_pred_df = pd.DataFrame(global_predictions)
            global_pred_df.to_csv(global_pred_file, index=False)
            print(f"Global test predictions saved to: {global_pred_file}")
            print()

            # Also save in EUOS challenge submission format (one file per method)
            submissions_dir = output_dir / "submissions"
            submissions_dir.mkdir(parents=True, exist_ok=True)

            for learner, predictions in global_predictions.items():
                submission_file = submissions_dir / f"{learner}.txt"
                np.savetxt(submission_file, predictions, fmt='%.10f')
                print(f"Submission file for {learner}: {submission_file}")
            print()
        else:
            print("=" * 80)
            print("Global test file not found, skipping Phase 2")
            print("=" * 80)
            print()
    
    print("=" * 80)
    print("DONE!")
    print("=" * 80)
    
    return 0


if __name__ == '__main__':
    exit(main())