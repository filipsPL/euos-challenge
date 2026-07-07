#!/usr/bin/env python3
"""
Generate out-of-fold predictions for training dataset using best FLAML methods.

This script:
1. Loads best configs from JSON file
2. Selects top-N methods by validation loss
3. For each method:
   - Performs 5-fold cross-validation on the training set
   - Generates out-of-fold predictions (maintains sample order)
   - Calculates AUROC for each fold
4. Saves all predictions to predictions/train_predictions.csv (one column per method)
5. Saves detailed report to predictions/train_predictions.txt with fold-level AUROC

The predictions maintain the exact same order as the training dataset, allowing
them to be used as meta-features for ensemble models.
"""

import json
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
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


def select_top_methods(configs: dict, n_top: int = 4) -> list:
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
    Infer train file path from config file location.

    DEPRECATED: This function is kept for backward compatibility only.
    The shell script should always pass --train-file explicitly.

    Returns:
        Dictionary with 'train' path (will be None if not found)
    """
    print("WARNING: Using deprecated path inference. Please pass --train-file explicitly.")
    print("         The shell script flaml-0-run_them_all.sh should handle this automatically.")

    return {'train': None}


def generate_oof_predictions(X: pd.DataFrame, y: pd.Series,
                            learner: str, config: dict,
                            n_folds: int = 5, random_state: int = 42) -> tuple:
    """
    Generate out-of-fold predictions using stratified K-fold cross-validation.

    Args:
        X: Feature matrix
        y: Target labels
        learner: Learner name
        config: Model configuration
        n_folds: Number of folds (default: 5)
        random_state: Random seed

    Returns:
        Tuple of (oof_predictions, fold_aurocs)
        - oof_predictions: Array of predictions in original order
        - fold_aurocs: List of AUROC scores for each fold
    """
    # Initialize array to store out-of-fold predictions
    oof_predictions = np.zeros(len(X))
    fold_aurocs = []

    # Create stratified K-fold splitter
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_state)

    # Iterate through folds
    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X, y), 1):
        # Split data
        X_fold_train = X.iloc[train_idx]
        y_fold_train = y.iloc[train_idx]
        X_fold_val = X.iloc[val_idx]
        y_fold_val = y.iloc[val_idx]

        # Create and train model
        model = create_model(learner, config, random_state)
        model.fit(X_fold_train, y_fold_train)

        # Generate predictions for validation fold
        y_pred_proba = model.predict_proba(X_fold_val)[:, 1]

        # Store predictions in original positions
        oof_predictions[val_idx] = y_pred_proba

        # Calculate AUROC for this fold
        fold_auroc = roc_auc_score(y_fold_val, y_pred_proba)
        fold_aurocs.append(fold_auroc)

        print(f"    Fold {fold_idx}/{n_folds}: AUROC = {fold_auroc:.6f}")

    return oof_predictions, fold_aurocs


def main():
    parser = argparse.ArgumentParser(
        description='Generate out-of-fold predictions for training set using best FLAML configs',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Use top 4 methods (default)
  python flaml-3-make_predictions_on_training_with_best_config.py \\
    ./5-predictions/.../euos25_challenge_train_transmittance340/configs/best_configs.json

  # Use top 6 methods
  python flaml-3-make_predictions_on_training_with_best_config.py \\
    best_configs.json -n 6

  # Specify training file explicitly
  python flaml-3-make_predictions_on_training_with_best_config.py \\
    best_configs.json --train-file train.csv.gz
        """
    )

    parser.add_argument('config_file', type=Path,
                        help='Path to best_configs.json file')
    parser.add_argument('-n', '--n-top', type=int, default=4,
                        help='Number of top methods to use (default: 4)')
    parser.add_argument('--train-file', type=Path, default=None,
                        help='Training data file (auto-inferred if not provided)')
    parser.add_argument('--n-folds', type=int, default=5,
                        help='Number of CV folds (default: 5)')
    parser.add_argument('--random-state', type=int, default=42,
                        help='Random seed (default: 42)')

    args = parser.parse_args()

    # Validate config file
    if not args.config_file.exists():
        print(f"Error: Config file not found: {args.config_file}")
        return 1

    # Determine output directory (parent of config file, which should be target dir)
    target_dir = args.config_file.parent.parent
    predictions_dir = target_dir / "predictions"
    predictions_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("GENERATING OUT-OF-FOLD PREDICTIONS FOR TRAINING SET")
    print("=" * 80)
    print(f"Config file: {args.config_file}")
    print(f"Top N methods: {args.n_top}")
    print(f"Number of folds: {args.n_folds}")
    print(f"Random state: {args.random_state}")
    print(f"Output directory: {predictions_dir}")
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

    # Infer or use provided training file path
    if args.train_file is None:
        print("Inferring training file path from config location...")
        paths = infer_file_paths(args.config_file)
        train_file = paths['train']
    else:
        train_file = args.train_file

    if train_file is None or not train_file.exists():
        print(f"Error: Training file not found: {train_file}")
        return 1

    print(f"Training file: {train_file}")
    print()

    # Load training data
    print("Loading training data...")
    X_train, y_train = load_data(train_file)
    print(f"Training set: {X_train.shape[0]} samples, {X_train.shape[1]} features")

    if y_train is None:
        print("Error: No target column ('activity' or 'target') found in training data")
        return 1

    # Sanitize column names
    X_train = sanitize_column_names(X_train)

    print(f"Target distribution: {np.bincount(y_train.astype(int))}")
    print()

    # Generate out-of-fold predictions for each method
    print("=" * 80)
    print("GENERATING OUT-OF-FOLD PREDICTIONS")
    print("=" * 80)
    print()

    all_predictions = {}
    all_results = []

    for i, (learner, cfg_data) in enumerate(top_methods, 1):
        print(f"[{i}/{len(top_methods)}] Generating OOF predictions for {learner}...")
        print(f"  FLAML validation loss: {cfg_data['validation_loss']:.6f}")

        try:
            # Generate out-of-fold predictions
            oof_preds, fold_aurocs = generate_oof_predictions(
                X_train, y_train, learner, cfg_data['config'],
                n_folds=args.n_folds, random_state=args.random_state
            )

            # Calculate average AUROC
            avg_auroc = np.mean(fold_aurocs)
            std_auroc = np.std(fold_aurocs)

            print(f"  Average AUROC: {avg_auroc:.6f} ± {std_auroc:.6f}")
            print(f"  Done")
            print()

            # Store predictions
            all_predictions[learner] = oof_preds

            # Store results
            result = {
                'method': learner,
                'flaml_validation_loss': cfg_data['validation_loss'],
                'cv_mean_auroc': avg_auroc,
                'cv_std_auroc': std_auroc,
            }
            # Add individual fold AUROCs
            for fold_idx, auroc in enumerate(fold_aurocs, 1):
                result[f'fold_{fold_idx}_auroc'] = auroc

            all_results.append(result)

        except Exception as e:
            print(f"  ✗ Error: {e}")
            import traceback
            traceback.print_exc()
            print()

    # Save predictions to CSV
    if all_predictions:
        print("=" * 80)
        print("SAVING RESULTS")
        print("=" * 80)
        print()

        # Create DataFrame with predictions (one column per method)
        pred_df = pd.DataFrame(all_predictions)

        # Add activity column as first column
        pred_df.insert(0, 'activity', y_train.values)

        # Save predictions
        pred_file = predictions_dir / "train_predictions.csv"
        pred_df.to_csv(pred_file, index=False, float_format='%.10f')
        print(f"✓ Predictions saved to: {pred_file}")
        print(f"  Dimensions: {pred_df.shape[0]} samples × {pred_df.shape[1]} columns")
        print(f"  Columns: activity, {', '.join(all_predictions.keys())}")
        print()

        # Create detailed text report
        report_file = predictions_dir / "train_predictions.txt"
        with open(report_file, 'w') as f:
            f.write("=" * 80 + "\n")
            f.write("OUT-OF-FOLD PREDICTIONS REPORT\n")
            f.write("=" * 80 + "\n\n")

            f.write("Configuration:\n")
            f.write(f"  Training file: {train_file}\n")
            f.write(f"  Training samples: {len(y_train)}\n")
            f.write(f"  Number of folds: {args.n_folds}\n")
            f.write(f"  Random state: {args.random_state}\n")
            f.write(f"  Number of methods: {len(all_predictions)}\n")
            f.write("\n")

            f.write("=" * 80 + "\n")
            f.write("PERFORMANCE BY METHOD\n")
            f.write("=" * 80 + "\n\n")

            for result in all_results:
                f.write(f"Method: {result['method']}\n")
                f.write("-" * 80 + "\n")
                f.write(f"  FLAML validation loss: {result['flaml_validation_loss']:.6f}\n")
                f.write(f"  CV Mean AUROC: {result['cv_mean_auroc']:.6f}\n")
                f.write(f"  CV Std AUROC:  {result['cv_std_auroc']:.6f}\n")
                f.write("\n")
                f.write("  Fold-level AUROC:\n")
                for fold_idx in range(1, args.n_folds + 1):
                    fold_key = f'fold_{fold_idx}_auroc'
                    if fold_key in result:
                        f.write(f"    Fold {fold_idx}: {result[fold_key]:.6f}\n")
                f.write("\n")

            f.write("=" * 80 + "\n")
            f.write("SUMMARY TABLE\n")
            f.write("=" * 80 + "\n\n")

            # Create summary table
            summary_df = pd.DataFrame(all_results)
            col_order = ['method', 'flaml_validation_loss', 'cv_mean_auroc', 'cv_std_auroc']
            fold_cols = [f'fold_{i}_auroc' for i in range(1, args.n_folds + 1)]
            col_order.extend(fold_cols)
            summary_df = summary_df[col_order]

            f.write(summary_df.to_string(index=False))
            f.write("\n\n")

            # Add ranking by CV AUROC
            f.write("=" * 80 + "\n")
            f.write("RANKING BY CV MEAN AUROC\n")
            f.write("=" * 80 + "\n\n")

            ranked = summary_df.sort_values('cv_mean_auroc', ascending=False)
            f.write(f"{'Rank':<6} {'Method':<20} {'CV Mean AUROC':<15} {'CV Std AUROC':<15}\n")
            f.write("-" * 80 + "\n")
            for rank, (_, row) in enumerate(ranked.iterrows(), 1):
                f.write(f"{rank:<6} {row['method']:<20} {row['cv_mean_auroc']:<15.6f} "
                       f"{row['cv_std_auroc']:<15.6f}\n")
            f.write("\n")

        print(f"✓ Report saved to: {report_file}")
        print()

        # Print summary to console
        print("=" * 80)
        print("SUMMARY")
        print("=" * 80)
        print()
        summary_df = pd.DataFrame(all_results)
        print(summary_df[['method', 'flaml_validation_loss', 'cv_mean_auroc', 'cv_std_auroc']].to_string(index=False))
        print()
    else:
        print("No predictions generated")
        return 1

    print("=" * 80)
    print("COMPLETED SUCCESSFULLY")
    print("=" * 80)

    return 0


if __name__ == '__main__':
    exit(main())
