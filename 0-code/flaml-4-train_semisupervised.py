#!/usr/bin/env python3
"""
Semi-supervised learning using FLAML-optimized configurations.

NEW CLEAN DIRECTORY STRUCTURE:
Input:  {target}/configs/best_configs.json
        {target}/predictions/  (for data file inference)
Output: {target}/semisupervised/results.csv
        {target}/semisupervised/test_predictions.csv
        {target}/semisupervised/global_test_predictions.csv
        {target}/semisupervised/submissions/{strategy}_{learner}.txt

This script:
1. Loads best configs from {target}/configs/best_configs.json
2. Creates models with optimal hyperparameters
3. Applies semi-supervised learning strategies
4. Saves results to {target}/semisupervised/

Strategies:
  - simple: Simple pseudo-labeling with confidence threshold
  - iterative: Iterative self-training with progressive labeling
  - ensemble: Ensemble-based pseudo-labeling with agreement
  - cotraining: Co-training with feature splits
  - all: Run all strategies

Usage:
    python flaml-4-train_semisupervised.py <target_directory> [options]

Example:
    python flaml-4-train_semisupervised.py \\
        ./5-predictions/5-flaml-descriptors/euos25_challenge_train_fluorescence340_450 \\
        --strategy ensemble --top-k 3
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
from datetime import datetime
import logging
import sys
import traceback
from copy import deepcopy
warnings.filterwarnings('ignore')

try:
    from catboost import CatBoostClassifier
    CATBOOST_AVAILABLE = True
except ImportError:
    CATBOOST_AVAILABLE = False
    print("Warning: CatBoost not available, will skip catboost models")


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def setup_logging(output_dir):
    """Setup logging to both file and console."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_filename = output_dir / f"semisup_{timestamp}.log"

    logger = logging.getLogger('semi_supervised_flaml')
    logger.setLevel(logging.INFO)
    logger.handlers = []

    file_handler = logging.FileHandler(log_filename, mode='w')
    file_handler.setLevel(logging.INFO)
    file_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(file_formatter)

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_formatter = logging.Formatter('%(message)s')
    console_handler.setFormatter(console_formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger, log_filename


def log_separator(logger, char='=', length=80):
    """Log a separator line."""
    logger.info(char * length)


# ============================================================================
# DATA LOADING & PREPARATION
# ============================================================================

def load_data(filepath, logger):
    """Load CSV file and handle NaN values."""
    try:
        logger.info(f"  Loading: {filepath}")
        df = pd.read_csv(filepath, compression='gzip')
        logger.info(f"    Shape: {df.shape}")
        return df
    except Exception as e:
        logger.error(f"    ERROR loading data: {str(e)}")
        raise


def prepare_data(df, logger, has_labels=True):
    """Separate features and target."""
    try:
        if has_labels:
            if 'activity' not in df.columns:
                raise ValueError("'activity' column not found in dataframe")
            X = df.drop('activity', axis=1)
            y = df['activity']

            # Sanitize column names to remove special JSON characters that LightGBM doesn't support
            original_cols = X.columns.tolist()
            sanitized_cols = []
            for col in original_cols:
                # Replace special JSON characters: " ' [ ] { } : , \ /
                sanitized = str(col)
                for char in ['"', "'", '[', ']', '{', '}', ':', ',', '\\', '/']:
                    sanitized = sanitized.replace(char, '_')
                sanitized_cols.append(sanitized)

            X.columns = sanitized_cols

            # Check if any columns were changed
            changed_count = sum(1 for orig, san in zip(original_cols, sanitized_cols) if orig != san)
            if changed_count > 0:
                logger.info(f"    Sanitized {changed_count} column names to remove special JSON characters")

            class_counts = y.value_counts()
            logger.info(f"    Class distribution: {dict(class_counts)}")
            return X, y
        else:
            if 'activity' in df.columns:
                df = df.drop('activity', axis=1)

            # Sanitize column names for unlabeled data too
            original_cols = df.columns.tolist()
            sanitized_cols = []
            for col in original_cols:
                sanitized = str(col)
                for char in ['"', "'", '[', ']', '{', '}', ':', ',', '\\', '/']:
                    sanitized = sanitized.replace(char, '_')
                sanitized_cols.append(sanitized)

            df.columns = sanitized_cols

            return df, None
    except Exception as e:
        logger.error(f"    ERROR preparing data: {str(e)}")
        raise


def sanitize_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """
    Sanitize column names to be compatible with all models (especially LGBM).
    LGBM doesn't accept special JSON characters like: [ ] { } " : ,
    """
    df = df.copy()

    new_columns = []
    for col in df.columns:
        new_col = str(col)
        for char in '[]{}",:':
            new_col = new_col.replace(char, '_')
        while '__' in new_col:
            new_col = new_col.replace('__', '_')
        new_col = new_col.strip('_')
        new_columns.append(new_col)

    df.columns = new_columns
    return df


def infer_data_files(target_dir: Path) -> dict:
    """
    Infer data file locations from target directory structure.

    DEPRECATED: This function is kept for backward compatibility only.
    The shell script should always pass file paths explicitly via command-line arguments:
      --train-file, --test-file, --global-test-file

    Returns:
        dict with None values (explicit paths should be provided)
    """
    print("WARNING: Using deprecated path inference for semi-supervised learning.")
    print("         Please pass --train-file, --test-file, and --global-test-file explicitly.")
    print("         The shell script flaml-0-run_them_all.sh should handle this automatically.")

    return {
        'train': None,
        'test': None,
        'global_test': None
    }


# ============================================================================
# FLAML CONFIG LOADING & MODEL CREATION
# ============================================================================

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
        if config_list:
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
        if 'log_max_bin' in params:
            params['max_bin'] = 2 ** params.pop('log_max_bin')
        params.pop('FLAML_sample_size', None)

        model = lgb.LGBMClassifier(
            **params,
            random_state=random_state,
            verbose=-1,
            n_jobs=-1
        )

    elif learner == 'xgboost' or learner == 'xgb_limitdepth':
        params.pop('FLAML_sample_size', None)

        model = xgb.XGBClassifier(
            **params,
            random_state=random_state,
            eval_metric='logloss',
            n_jobs=-1,
            verbosity=0
        )

    elif learner == 'extra_tree':
        if 'max_leaves' in params:
            params['max_leaf_nodes'] = params.pop('max_leaves')
        params.pop('FLAML_sample_size', None)

        model = ExtraTreesClassifier(
            **params,
            random_state=random_state,
            n_jobs=-1,
            verbose=0
        )

    elif learner == 'rf':
        if 'max_leaves' in params:
            params['max_leaf_nodes'] = params.pop('max_leaves')
        params.pop('FLAML_sample_size', None)

        model = RandomForestClassifier(
            **params,
            random_state=random_state,
            n_jobs=-1,
            verbose=0
        )

    elif learner == 'sgd':
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
        params.pop('FLAML_sample_size', None)

        model = CatBoostClassifier(
            **params,
            random_state=random_state,
            verbose=False,
            thread_count=-1
        )

    elif learner == 'lrl1':
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


# ============================================================================
# BASELINE: SUPERVISED LEARNING ONLY
# ============================================================================

def train_baseline_models(top_methods, X_train, y_train, X_test, y_test, logger, random_state=42):
    """
    Train baseline models (supervised only, no semi-supervised learning).
    Returns results for comparison.
    """
    logger.info("\n" + "="*80)
    logger.info("BASELINE: SUPERVISED LEARNING ONLY")
    logger.info("="*80)
    logger.info(f"\nTraining {len(top_methods)} FLAML-optimized models on labeled data only...")
    logger.info(f"Training set: {len(X_train)} samples")
    logger.info(f"Test set: {len(X_test)} samples\n")

    baseline_results = []
    baseline_models = {}
    baseline_predictions = {}

    for idx, (learner, cfg_data) in enumerate(top_methods, 1):
        logger.info(f"  [{idx}/{len(top_methods)}] {learner}")
        logger.info(f"    FLAML validation loss: {cfg_data['validation_loss']:.6f}")

        try:
            # Create model with FLAML-optimized config
            model = create_model(learner, cfg_data['config'], random_state)

            # Train on labeled data only
            logger.info(f"    Training on {len(X_train)} labeled samples...")
            model.fit(X_train, y_train)

            # Evaluate on test set
            y_pred_proba = model.predict_proba(X_test)[:, 1]
            test_auroc = roc_auc_score(y_test, y_pred_proba)

            logger.info(f"    Test AUROC: {test_auroc:.6f}")

            baseline_results.append({
                'method': learner,
                'strategy': 'baseline_supervised',
                'flaml_validation_loss': cfg_data['validation_loss'],
                'test_auroc': test_auroc,
                'n_labeled': len(X_train),
                'n_pseudo_labeled': 0
            })

            baseline_models[learner] = model
            baseline_predictions[learner] = y_pred_proba

        except Exception as e:
            logger.error(f"    Error: {e}")
            baseline_results.append({
                'method': learner,
                'strategy': 'baseline_supervised',
                'flaml_validation_loss': cfg_data['validation_loss'],
                'test_auroc': None,
                'error': str(e)
            })

        logger.info("")

    # Summary
    logger.info("-"*80)
    logger.info("BASELINE SUMMARY:")
    for result in baseline_results:
        if result['test_auroc'] is not None:
            logger.info(f"  {result['method']}: AUROC = {result['test_auroc']:.6f}")
    logger.info("")

    return baseline_results, baseline_models, baseline_predictions


# ============================================================================
# SEMI-SUPERVISED LEARNING STRATEGIES
# ============================================================================

def simple_pseudo_labeling(models, X_train, y_train, X_unlabeled,
                           confidence_threshold=0.9, logger=None):
    """
    Simple pseudo-labeling: Label high-confidence predictions once.

    Returns:
        X_combined, y_combined, n_pseudo_labeled
    """
    if logger:
        logger.info(f"\n  Simple pseudo-labeling (confidence >= {confidence_threshold})")

    # Get predictions from all models
    all_predictions = []
    for name, model in models.items():
        pred_proba = model.predict_proba(X_unlabeled)[:, 1]
        all_predictions.append(pred_proba)

    # Average predictions
    avg_predictions = np.mean(all_predictions, axis=0)

    # Select high-confidence samples
    confident_mask = (avg_predictions >= confidence_threshold) | (avg_predictions <= (1 - confidence_threshold))

    if not any(confident_mask):
        if logger:
            logger.info(f"    No samples above confidence threshold")
        return X_train.copy(), y_train.copy(), 0

    X_pseudo = X_unlabeled[confident_mask]
    y_pseudo = (avg_predictions[confident_mask] >= 0.5).astype(int)

    # Combine with original training data
    X_combined = pd.concat([X_train, X_pseudo], axis=0, ignore_index=True)
    y_combined = pd.concat([pd.Series(y_train), pd.Series(y_pseudo)], axis=0, ignore_index=True)

    if logger:
        logger.info(f"    Added {len(y_pseudo)} pseudo-labeled samples")
        logger.info(f"    New training size: {len(y_combined)}")

    return X_combined, y_combined, len(y_pseudo)


def iterative_self_training(models, X_train, y_train, X_unlabeled,
                            n_iterations=5, samples_per_iter=1000,
                            confidence_threshold=0.9, logger=None):
    """
    Iterative self-training: Progressively add high-confidence samples.

    Returns:
        X_combined, y_combined, n_pseudo_labeled
    """
    if logger:
        logger.info(f"\n  Iterative self-training ({n_iterations} iterations)")

    X_current = X_train.copy()
    y_current = y_train.copy()
    X_remaining = X_unlabeled.copy()
    total_added = 0

    for iteration in range(n_iterations):
        if len(X_remaining) == 0:
            if logger:
                logger.info(f"    Iteration {iteration+1}: No unlabeled samples remaining")
            break

        # Retrain models on current labeled set
        current_models = {}
        for name, base_model in models.items():
            model = deepcopy(base_model)
            model.fit(X_current, y_current)
            current_models[name] = model

        # Get predictions for remaining unlabeled data
        all_predictions = []
        for model in current_models.values():
            pred_proba = model.predict_proba(X_remaining)[:, 1]
            all_predictions.append(pred_proba)

        avg_predictions = np.mean(all_predictions, axis=0)

        # Calculate confidence (distance from 0.5)
        confidence = np.abs(avg_predictions - 0.5)

        # Select top samples_per_iter most confident samples above threshold
        threshold_mask = (avg_predictions >= confidence_threshold) | (avg_predictions <= (1 - confidence_threshold))

        if not any(threshold_mask):
            if logger:
                logger.info(f"    Iteration {iteration+1}: No samples above threshold")
            break

        # Get indices sorted by confidence
        confident_indices = np.where(threshold_mask)[0]
        sorted_indices = confident_indices[np.argsort(confidence[confident_indices])[::-1]]
        selected_indices = sorted_indices[:samples_per_iter]

        # Add pseudo-labeled samples
        X_pseudo = X_remaining.iloc[selected_indices]
        y_pseudo = (avg_predictions[selected_indices] >= 0.5).astype(int)

        X_current = pd.concat([X_current, X_pseudo], axis=0, ignore_index=True)
        y_current = pd.concat([pd.Series(y_current), pd.Series(y_pseudo)], axis=0, ignore_index=True)

        # Remove selected samples from remaining
        X_remaining = X_remaining.drop(X_remaining.index[selected_indices]).reset_index(drop=True)

        total_added += len(selected_indices)

        if logger:
            logger.info(f"    Iteration {iteration+1}: Added {len(selected_indices)} samples, "
                       f"total labeled: {len(y_current)}, remaining: {len(X_remaining)}")

    if logger:
        logger.info(f"    Total pseudo-labeled samples added: {total_added}")

    return X_current, y_current, total_added


def ensemble_pseudo_labeling(models, X_train, y_train, X_unlabeled,
                             agreement_threshold=0.8, logger=None):
    """
    Ensemble pseudo-labeling: Only label when models agree.

    Returns:
        X_combined, y_combined, n_pseudo_labeled
    """
    if logger:
        logger.info(f"\n  Ensemble pseudo-labeling (agreement >= {agreement_threshold})")

    # Get predictions from all models
    all_predictions = []
    for name, model in models.items():
        pred_proba = model.predict_proba(X_unlabeled)[:, 1]
        pred_class = (pred_proba >= 0.5).astype(int)
        all_predictions.append(pred_class)

    # Calculate agreement (fraction of models that agree on majority class)
    predictions_array = np.array(all_predictions)
    majority_vote = (np.mean(predictions_array, axis=0) >= 0.5).astype(int)

    # Calculate agreement score
    agreement = np.mean(predictions_array == majority_vote, axis=0)

    # Select samples with high agreement
    high_agreement_mask = agreement >= agreement_threshold

    if not any(high_agreement_mask):
        if logger:
            logger.info(f"    No samples with sufficient agreement")
        return X_train.copy(), y_train.copy(), 0

    X_pseudo = X_unlabeled[high_agreement_mask]
    y_pseudo = majority_vote[high_agreement_mask]

    # Combine with original training data
    X_combined = pd.concat([X_train, X_pseudo], axis=0, ignore_index=True)
    y_combined = pd.concat([pd.Series(y_train), pd.Series(y_pseudo)], axis=0, ignore_index=True)

    if logger:
        logger.info(f"    Added {len(y_pseudo)} pseudo-labeled samples")
        logger.info(f"    Average agreement: {np.mean(agreement[high_agreement_mask]):.3f}")
        logger.info(f"    New training size: {len(y_combined)}")

    return X_combined, y_combined, len(y_pseudo)


def cotraining(models, X_train, y_train, X_unlabeled,
               n_iterations=5, samples_per_iter=1000,
               confidence_threshold=0.9, logger=None):
    """
    Co-training: Split features and train complementary models.

    Returns:
        X_combined, y_combined, n_pseudo_labeled
    """
    if logger:
        logger.info(f"\n  Co-training with feature splits ({n_iterations} iterations)")

    # Split features into two views
    n_features = X_train.shape[1]
    mid_point = n_features // 2

    features_view1 = X_train.columns[:mid_point].tolist()
    features_view2 = X_train.columns[mid_point:].tolist()

    if logger:
        logger.info(f"    View 1: {len(features_view1)} features")
        logger.info(f"    View 2: {len(features_view2)} features")

    # Prepare data views
    X_train_v1 = X_train[features_view1]
    X_train_v2 = X_train[features_view2]
    X_unlabeled_v1 = X_unlabeled[features_view1]
    X_unlabeled_v2 = X_unlabeled[features_view2]

    # Initialize with original training data
    X_current_v1 = X_train_v1.copy()
    X_current_v2 = X_train_v2.copy()
    y_current = y_train.copy()
    X_remaining_v1 = X_unlabeled_v1.copy()
    X_remaining_v2 = X_unlabeled_v2.copy()
    total_added = 0

    # Use first model for each view (could use different models)
    model_v1_base = list(models.values())[0] if len(models) > 0 else None
    model_v2_base = list(models.values())[1] if len(models) > 1 else model_v1_base

    if model_v1_base is None:
        if logger:
            logger.error("    No models available for co-training")
        return X_train.copy(), y_train.copy(), 0

    for iteration in range(n_iterations):
        if len(X_remaining_v1) == 0:
            if logger:
                logger.info(f"    Iteration {iteration+1}: No unlabeled samples remaining")
            break

        # Train models on each view
        model_v1 = deepcopy(model_v1_base)
        model_v2 = deepcopy(model_v2_base)

        model_v1.fit(X_current_v1, y_current)
        model_v2.fit(X_current_v2, y_current)

        # Get predictions from each view
        pred_v1 = model_v1.predict_proba(X_remaining_v1)[:, 1]
        pred_v2 = model_v2.predict_proba(X_remaining_v2)[:, 1]

        # Find samples where both views are confident (and agree)
        confident_v1 = (pred_v1 >= confidence_threshold) | (pred_v1 <= (1 - confidence_threshold))
        confident_v2 = (pred_v2 >= confidence_threshold) | (pred_v2 <= (1 - confidence_threshold))

        # Both models must agree on the class
        class_v1 = (pred_v1 >= 0.5).astype(int)
        class_v2 = (pred_v2 >= 0.5).astype(int)
        agree = class_v1 == class_v2

        # Samples must be confident in both views AND agree
        candidate_mask = confident_v1 & confident_v2 & agree

        if not any(candidate_mask):
            if logger:
                logger.info(f"    Iteration {iteration+1}: No samples meeting criteria")
            break

        # Select top samples_per_iter samples
        candidate_indices = np.where(candidate_mask)[0]
        confidence_scores = np.minimum(np.abs(pred_v1 - 0.5), np.abs(pred_v2 - 0.5))
        sorted_indices = candidate_indices[np.argsort(confidence_scores[candidate_indices])[::-1]]
        selected_indices = sorted_indices[:samples_per_iter]

        # Add pseudo-labeled samples
        X_pseudo_v1 = X_remaining_v1.iloc[selected_indices]
        X_pseudo_v2 = X_remaining_v2.iloc[selected_indices]
        y_pseudo = class_v1[selected_indices]

        X_current_v1 = pd.concat([X_current_v1, X_pseudo_v1], axis=0, ignore_index=True)
        X_current_v2 = pd.concat([X_current_v2, X_pseudo_v2], axis=0, ignore_index=True)
        y_current = pd.concat([pd.Series(y_current), pd.Series(y_pseudo)], axis=0, ignore_index=True)

        # Remove selected samples
        X_remaining_v1 = X_remaining_v1.drop(X_remaining_v1.index[selected_indices]).reset_index(drop=True)
        X_remaining_v2 = X_remaining_v2.drop(X_remaining_v2.index[selected_indices]).reset_index(drop=True)

        total_added += len(selected_indices)

        if logger:
            logger.info(f"    Iteration {iteration+1}: Added {len(selected_indices)} samples, "
                       f"total labeled: {len(y_current)}, remaining: {len(X_remaining_v1)}")

    # Reconstruct full feature space
    X_combined = pd.concat([X_current_v1, X_current_v2], axis=1)
    # Reorder columns to match original
    X_combined = X_combined[X_train.columns]

    if logger:
        logger.info(f"    Total pseudo-labeled samples added: {total_added}")

    return X_combined, y_current, total_added


# ============================================================================
# SEMI-SUPERVISED TRAINING ORCHESTRATION
# ============================================================================

def train_semi_supervised_models(X_train, y_train, X_test, y_test, X_unlabeled,
                                 strategy, config, top_methods, logger, random_state=42):
    """
    Train models using semi-supervised learning with FLAML-optimized configs.

    Args:
        strategy: 'simple', 'iterative', 'ensemble', 'cotraining'
        config: Dictionary with strategy-specific parameters
        top_methods: List of (learner_name, config_data) tuples from FLAML

    Returns:
        results, trained_models, test_predictions
    """
    logger.info(f"\n  Strategy: {strategy}")
    logger.info(f"  Number of models: {len(top_methods)}")
    logger.info(f"  Unlabeled samples: {len(X_unlabeled)}")

    # Step 1: Train initial models on labeled data
    logger.info("\n  Step 1: Training initial models on labeled data...")
    initial_models = {}

    for learner, cfg_data in top_methods:
        try:
            model = create_model(learner, cfg_data['config'], random_state)
            model.fit(X_train, y_train)
            initial_models[learner] = model
            logger.info(f"    Trained {learner}")
        except Exception as e:
            logger.error(f"    Error training {learner}: {e}")

    if not initial_models:
        logger.error("  No models successfully trained!")
        return [], {}, {}

    # Step 2: Apply semi-supervised strategy
    logger.info("\n  Step 2: Applying semi-supervised strategy...")

    if strategy == 'simple':
        X_combined, y_combined, n_pseudo = simple_pseudo_labeling(
            initial_models, X_train, y_train, X_unlabeled,
            confidence_threshold=config['confidence_threshold'],
            logger=logger
        )

    elif strategy == 'iterative':
        X_combined, y_combined, n_pseudo = iterative_self_training(
            initial_models, X_train, y_train, X_unlabeled,
            n_iterations=config['n_iterations'],
            samples_per_iter=config['samples_per_iter'],
            confidence_threshold=config['confidence_threshold'],
            logger=logger
        )

    elif strategy == 'ensemble':
        X_combined, y_combined, n_pseudo = ensemble_pseudo_labeling(
            initial_models, X_train, y_train, X_unlabeled,
            agreement_threshold=config['agreement_threshold'],
            logger=logger
        )

    elif strategy == 'cotraining':
        X_combined, y_combined, n_pseudo = cotraining(
            initial_models, X_train, y_train, X_unlabeled,
            n_iterations=config['n_iterations'],
            samples_per_iter=config['samples_per_iter'],
            confidence_threshold=config['confidence_threshold'],
            logger=logger
        )

    else:
        raise ValueError(f"Unknown strategy: {strategy}")

    # Step 3: Retrain models on combined data
    logger.info("\n  Step 3: Retraining models on combined labeled + pseudo-labeled data...")
    logger.info(f"    Combined training set size: {len(y_combined)}")

    results = []
    trained_models = {}
    test_predictions = {}

    for idx, (learner, cfg_data) in enumerate(top_methods, 1):
        logger.info(f"\n    [{idx}/{len(top_methods)}] {learner}")

        try:
            # Create and train model on combined data
            model = create_model(learner, cfg_data['config'], random_state)
            model.fit(X_combined, y_combined)

            # Evaluate on test set
            y_pred_proba = model.predict_proba(X_test)[:, 1]
            test_auroc = roc_auc_score(y_test, y_pred_proba)

            logger.info(f"      Test AUROC: {test_auroc:.6f}")
            logger.info(f"      Improvement: +{n_pseudo} pseudo-labeled samples")

            results.append({
                'method': learner,
                'strategy': strategy,
                'flaml_validation_loss': cfg_data['validation_loss'],
                'test_auroc': test_auroc,
                'n_labeled': len(y_train),
                'n_pseudo_labeled': n_pseudo,
                'n_total': len(y_combined)
            })

            trained_models[f"{learner}_{strategy}"] = model
            test_predictions[f"{learner}_{strategy}"] = y_pred_proba

        except Exception as e:
            logger.error(f"      Error: {e}")
            results.append({
                'method': learner,
                'strategy': strategy,
                'test_auroc': None,
                'error': str(e)
            })

    return results, trained_models, test_predictions


# ============================================================================
# RESULTS SAVING
# ============================================================================

def save_results(all_results, output_dir, logger):
    """Save results to CSV file."""
    try:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Create results DataFrame
        df_results = pd.DataFrame(all_results)

        # Save to CSV with simple filename
        csv_path = output_dir / "results.csv"
        df_results.to_csv(csv_path, index=False, float_format='%.6f')
        logger.info(f"\n  Results saved to CSV: {csv_path}")

        return csv_path

    except Exception as e:
        logger.error(f"  ERROR saving results: {str(e)}")
        logger.error(f"  Traceback: {traceback.format_exc()}")
        return None


def save_test_predictions(test_predictions, y_test, output_dir, logger):
    """Save test set predictions for all models/strategies."""
    try:
        output_dir = Path(output_dir)

        # Create DataFrame with actual labels
        pred_df = pd.DataFrame({'activity': y_test})

        # Add predictions from all models
        for name, preds in test_predictions.items():
            pred_df[name] = preds

        # Save with simple filename
        output_file = output_dir / "test_predictions.csv"
        pred_df.to_csv(output_file, index=False, float_format='%.10f')

        logger.info(f"  Test predictions saved: {output_file}")

    except Exception as e:
        logger.error(f"  ERROR saving test predictions: {str(e)}")


def predict_global_test(trained_models, output_dir, global_test_path,
                       training_features, target_name, logger):
    """Make predictions on global test set using all trained models."""
    try:
        if not global_test_path or not Path(global_test_path).exists():
            logger.info("  Global test file not found, skipping global predictions")
            return

        logger.info(f"\n  Loading global test data: {global_test_path}")
        df_global = pd.read_csv(global_test_path, compression='gzip')

        # Remove activity column if present
        if 'activity' in df_global.columns:
            df_global = df_global.drop('activity', axis=1)

        # Sanitize column names
        df_global = sanitize_column_names(df_global)

        # Align features with training data
        logger.info(f"    Aligning features...")
        logger.info(f"    Training features: {len(training_features)}")
        logger.info(f"    Global test features: {len(df_global.columns)}")

        # Keep only common features in same order as training
        X_global = df_global[training_features]

        logger.info(f"    Global test shape: {X_global.shape}")

        # Make predictions with all models
        global_predictions = {}

        for name, model in trained_models.items():
            logger.info(f"    Predicting with {name}...")
            try:
                y_global_pred = model.predict_proba(X_global)[:, 1]
                global_predictions[name] = y_global_pred
            except Exception as e:
                logger.error(f"      Error: {e}")

        if not global_predictions:
            logger.warning("  No successful predictions on global test set")
            return

        # Save predictions CSV
        output_file = output_dir / "global_test_predictions.csv"
        pred_df = pd.DataFrame(global_predictions)
        pred_df.to_csv(output_file, index=False, float_format='%.10f')

        logger.info(f"  Global test predictions saved: {output_file}")

        # Also save individual submission files
        submissions_dir = output_dir / "submissions"
        submissions_dir.mkdir(parents=True, exist_ok=True)

        for name, preds in global_predictions.items():
            # Extract strategy and learner from name (format: learner_strategy)
            parts = name.split('_')
            if len(parts) >= 2:
                learner = parts[0]
                strategy = '_'.join(parts[1:])
                submission_file = submissions_dir / f"{strategy}_{learner}.txt"
            else:
                submission_file = submissions_dir / f"{name}.txt"

            np.savetxt(submission_file, preds, fmt='%.10f')

        logger.info(f"  Individual submission files saved to: {submissions_dir}")

    except Exception as e:
        logger.error(f"  ERROR predicting on global test: {str(e)}")
        logger.error(f"  Traceback: {traceback.format_exc()}")


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Semi-supervised learning using FLAML-optimized configurations',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Strategies:
  simple      - Simple pseudo-labeling with confidence threshold
  iterative   - Iterative self-training with progressive labeling
  ensemble    - Ensemble-based pseudo-labeling with agreement
  cotraining  - Co-training with feature splits
  all         - Run all strategies

Examples:
  # Use top 3 FLAML models with ensemble strategy
  python flaml-4-train_semisupervised.py \\
      ./5-predictions/5-flaml-descriptors/euos25_challenge_train_fluorescence340_450 \\
      --strategy ensemble --top-k 3

  # Run all strategies with custom parameters
  python flaml-4-train_semisupervised.py \\
      ./5-predictions/5-flaml-descriptors/euos25_challenge_train_fluorescence340_450 \\
      --strategy all --top-k 5 --confidence 0.9 --agreement 0.8 --iterations 5

  # Explicitly specify data files
  python flaml-4-train_semisupervised.py \\
      ./5-predictions/5-flaml-descriptors/euos25_challenge_train_fluorescence340_450 \\
      --strategy ensemble --train-file ./4-datasets/.../train.csv.gz \\
      --test-file ./4-datasets/.../test.csv.gz \\
      --global-test-file ./4-datasets/euos25_challenge_test/.../train.csv.gz

Expected directory structure:
  {target}/
  ├── configs/
  │   └── best_configs.json             # Input
  └── semisupervised/
      ├── results.csv                   # Output
      ├── test_predictions.csv          # Output
      ├── global_test_predictions.csv   # Output
      └── submissions/                  # Output
          ├── ensemble_lgbm.txt
          └── ...
        """
    )

    parser.add_argument('target_dir', type=Path,
                        help='Target directory (e.g., .../euos25_challenge_train_fluorescence340_450)')
    parser.add_argument('--strategy', type=str, default='ensemble',
                        choices=['simple', 'iterative', 'ensemble', 'cotraining', 'all'],
                        help='Semi-supervised strategy (default: ensemble)')
    parser.add_argument('--top-k', type=int, default=3,
                        help='Number of top FLAML models to use (default: 3)')
    parser.add_argument('--confidence', type=float, default=0.9,
                        help='Confidence threshold (0.5-1.0, default: 0.9)')
    parser.add_argument('--agreement', type=float, default=0.8,
                        help='Agreement threshold for ensemble (0.0-1.0, default: 0.8)')
    parser.add_argument('--iterations', type=int, default=5,
                        help='Number of iterations for iterative/co-training (default: 5)')
    parser.add_argument('--samples-per-iter', type=int, default=1000,
                        help='Samples to add per iteration (default: 1000)')
    parser.add_argument('--train-file', type=Path, default=None,
                        help='Training data file (auto-inferred if not provided)')
    parser.add_argument('--test-file', type=Path, default=None,
                        help='Test data file (auto-inferred if not provided)')
    parser.add_argument('--global-test-file', type=Path, default=None,
                        help='Global challenge test file (auto-inferred if not provided)')
    parser.add_argument('--random-state', type=int, default=42,
                        help='Random seed (default: 42)')

    args = parser.parse_args()

    # Validate target directory
    target_dir = Path(args.target_dir)
    if not target_dir.exists():
        print(f"Error: Target directory not found: {target_dir}")
        return 1

    target_name = target_dir.name

    # Define input and output directories
    config_file = target_dir / "configs" / "best_configs.json"
    semisup_dir = target_dir / "semisupervised"
    semisup_dir.mkdir(parents=True, exist_ok=True)

    # Validate config file
    if not config_file.exists():
        print(f"Error: Config file not found: {config_file}")
        return 1

    # Setup logging
    logger, log_filename = setup_logging(semisup_dir)

    logger.info("="*80)
    logger.info("SEMI-SUPERVISED LEARNING WITH FLAML-OPTIMIZED MODELS")
    logger.info("="*80)
    logger.info(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"Target: {target_name}")
    logger.info(f"Config file: {config_file}")
    logger.info(f"Log file: {log_filename}")
    logger.info("")

    try:
        # Load FLAML best configs
        logger.info("="*80)
        logger.info("LOADING FLAML CONFIGURATIONS")
        log_separator(logger)

        configs = load_best_configs(config_file)
        logger.info(f"Loaded {len(configs)} learner types")

        # Select top methods
        top_methods = select_top_methods(configs, args.top_k)
        logger.info(f"\nSelected top {len(top_methods)} methods:")
        for i, (learner, cfg) in enumerate(top_methods, 1):
            val_loss = cfg['validation_loss']
            logger.info(f"  {i}. {learner}: validation_loss = {val_loss:.6f}")
        logger.info("")

        # Infer or use provided file paths
        logger.info("="*80)
        logger.info("DATA FILES")
        log_separator(logger)

        if args.train_file is None or args.test_file is None:
            logger.info("Inferring file paths from target directory...")
            paths = infer_data_files(target_dir)
            train_file = args.train_file or paths['train']
            test_file = args.test_file or paths['test']
            global_test_file = args.global_test_file or paths['global_test']
        else:
            train_file = args.train_file
            test_file = args.test_file
            global_test_file = args.global_test_file

        logger.info(f"Training file: {train_file}")
        logger.info(f"Test file: {test_file}")
        logger.info(f"Global test file: {global_test_file}")
        logger.info("")

        # Validate files
        if not train_file or not train_file.exists():
            logger.error(f"Error: Training file not found: {train_file}")
            return 1

        if not test_file or not test_file.exists():
            logger.error(f"Error: Test file not found: {test_file}")
            return 1

        if global_test_file and not global_test_file.exists():
            logger.warning(f"Warning: Global test file not found: {global_test_file}")
            logger.warning("Will skip global test predictions")
            global_test_file = None

        # Load data
        logger.info("="*80)
        logger.info("LOADING DATA")
        log_separator(logger)

        logger.info("\n  Loading training data...")
        df_train = load_data(train_file, logger)

        logger.info("\n  Loading test data...")
        df_test = load_data(test_file, logger)

        if global_test_file:
            logger.info("\n  Loading global test data (unlabeled)...")
            df_unlabeled = load_data(global_test_file, logger)
        else:
            logger.warning("\n  No global test file available")
            logger.error("Semi-supervised learning requires unlabeled data!")
            return 1

        # Prepare data
        logger.info("\n" + "="*80)
        logger.info("PREPARING DATA")
        log_separator(logger)

        X_train, y_train = prepare_data(df_train, logger, has_labels=True)
        X_test, y_test = prepare_data(df_test, logger, has_labels=True)
        X_unlabeled, _ = prepare_data(df_unlabeled, logger, has_labels=False)

        logger.info(f"\n  Training set:   {X_train.shape[0]} samples, {X_train.shape[1]} features")
        logger.info(f"  Test set:       {X_test.shape[0]} samples, {X_test.shape[1]} features")
        logger.info(f"  Unlabeled set:  {X_unlabeled.shape[0]} samples, {X_unlabeled.shape[1]} features")

        # Sanitize column names
        logger.info("\n  Sanitizing column names...")
        X_train = sanitize_column_names(X_train)
        X_test = sanitize_column_names(X_test)
        X_unlabeled = sanitize_column_names(X_unlabeled)

        # Check for constant features
        logger.info("\n  Checking for constant features...")
        constant_cols = X_train.columns[X_train.nunique() <= 1].tolist()
        if constant_cols:
            logger.info(f"    Found {len(constant_cols)} constant columns")
            logger.info(f"    Removing from all datasets...")
            X_train = X_train.drop(constant_cols, axis=1)
            X_test = X_test.drop(constant_cols, axis=1, errors='ignore')
            X_unlabeled = X_unlabeled.drop(constant_cols, axis=1, errors='ignore')
            logger.info(f"    After removal: {X_train.shape[1]} features")

        # Align features across datasets
        logger.info("\n  Aligning features across datasets...")
        common_features = list(set(X_train.columns) & set(X_test.columns) & set(X_unlabeled.columns))
        logger.info(f"    Common features: {len(common_features)}")

        X_train = X_train[common_features]
        X_test = X_test[common_features]
        X_unlabeled = X_unlabeled[common_features]

        # PHASE 1: Train baseline models (supervised only)
        logger.info("\n" + "="*80)
        logger.info("PHASE 1: BASELINE (SUPERVISED LEARNING ONLY)")
        log_separator(logger)

        baseline_results, baseline_models, baseline_predictions = train_baseline_models(
            top_methods, X_train, y_train, X_test, y_test, logger, args.random_state
        )

        # PHASE 2: Semi-supervised learning
        logger.info("\n" + "="*80)
        logger.info("PHASE 2: SEMI-SUPERVISED LEARNING")
        log_separator(logger)

        config = {
            'confidence_threshold': args.confidence,
            'agreement_threshold': args.agreement,
            'n_iterations': args.iterations,
            'samples_per_iter': args.samples_per_iter,
            'top_k': args.top_k,
        }

        logger.info(f"\nConfiguration:")
        logger.info(f"  Strategy: {args.strategy}")
        logger.info(f"  Confidence threshold: {config['confidence_threshold']}")
        logger.info(f"  Agreement threshold: {config['agreement_threshold']}")
        logger.info(f"  Iterations: {config['n_iterations']}")
        logger.info(f"  Samples per iteration: {config['samples_per_iter']}")

        # Determine strategies to run
        strategies_to_run = ['simple', 'iterative', 'ensemble', 'cotraining'] if args.strategy == 'all' else [args.strategy]

        all_results = baseline_results.copy()
        all_trained_models = baseline_models.copy()
        all_test_predictions = baseline_predictions.copy()

        for strategy in strategies_to_run:
            logger.info(f"\n{'='*80}")
            logger.info(f"STRATEGY: {strategy.upper()}")
            logger.info(f"{'='*80}")

            results, trained_models, test_predictions = train_semi_supervised_models(
                X_train, y_train, X_test, y_test, X_unlabeled,
                strategy, config, top_methods, logger, args.random_state
            )

            all_results.extend(results)
            all_trained_models.update(trained_models)
            all_test_predictions.update(test_predictions)

        # PHASE 3: Save results
        logger.info("\n" + "="*80)
        logger.info("PHASE 3: SAVING RESULTS")
        log_separator(logger)

        save_results(all_results, semisup_dir, logger)
        save_test_predictions(all_test_predictions, y_test, semisup_dir, logger)

        if global_test_file:
            predict_global_test(
                all_trained_models, semisup_dir, global_test_file,
                common_features, target_name, logger
            )

        # Final summary
        logger.info("\n" + "="*80)
        logger.info("SUMMARY")
        log_separator(logger)

        df_results = pd.DataFrame(all_results)

        # Compare baseline vs semi-supervised
        logger.info("\nBaseline (Supervised Only):")
        baseline_df = df_results[df_results['strategy'] == 'baseline_supervised']
        for _, row in baseline_df.iterrows():
            if pd.notna(row['test_auroc']):
                logger.info(f"  {row['method']}: AUROC = {row['test_auroc']:.6f}")

        if len(baseline_df) > 0 and baseline_df['test_auroc'].notna().any():
            logger.info(f"  Average: {baseline_df['test_auroc'].mean():.6f}")

        logger.info("\nBest Semi-Supervised Results per Strategy:")
        for strategy in strategies_to_run:
            strategy_df = df_results[df_results['strategy'] == strategy]
            if len(strategy_df) > 0 and strategy_df['test_auroc'].notna().any():
                best_auroc = strategy_df['test_auroc'].max()
                best_method = strategy_df.loc[strategy_df['test_auroc'].idxmax(), 'method']
                logger.info(f"  {strategy}: {best_auroc:.6f} ({best_method})")

        log_separator(logger)
        logger.info("COMPLETED SUCCESSFULLY")
        log_separator(logger)
        logger.info(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info(f"All results saved to: {semisup_dir}")
        logger.info("")

        return 0

    except Exception as e:
        logger.error("\n" + "="*80)
        logger.error("FATAL ERROR")
        logger.error("="*80)
        logger.error(f"An unexpected error occurred: {str(e)}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        logger.error("="*80)
        return 1


if __name__ == '__main__':
    exit(main())
