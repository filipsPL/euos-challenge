#!/usr/bin/env python3
"""
Calculate consensus predictions from FLAML model predictions.

NEW CLEAN DIRECTORY STRUCTURE:
Input:  {target}/predictions/test_predictions.csv
        {target}/predictions/global_test_predictions.csv
Output: {target}/consensus/test_predictions.csv
        {target}/consensus/global_test_predictions.csv
        {target}/consensus/results.csv

This script:
1. Reads individual model predictions from {target}/predictions/
2. Creates consensus predictions for all combinations of 2 and 3 models
3. Evaluates AUROC scores for individual and consensus predictions (test set)
4. Saves results to {target}/consensus/

Usage:
    python flaml-4-consensus.py <target_directory>

Example:
    python flaml-4-consensus.py ./5-predictions/5-flaml-descriptors/euos25_challenge_train_fluorescence340_450
"""

import argparse
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import logging
import sys
import traceback
from itertools import combinations
from sklearn.metrics import roc_auc_score

import warnings
warnings.filterwarnings('ignore')


def setup_logging(output_dir):
    """Setup logging to both file and console."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_filename = output_dir / f"consensus_{timestamp}.log"

    logger = logging.getLogger('flaml_consensus')
    logger.setLevel(logging.INFO)
    logger.handlers = []

    # File handler
    file_handler = logging.FileHandler(log_filename, mode='w')
    file_handler.setLevel(logging.INFO)
    file_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(file_formatter)

    # Console handler
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


def create_consensus_predictions(df, has_activity=True, logger=None):
    """
    Create consensus predictions for all combinations of 2 and 3 models.

    Args:
        df: DataFrame with predictions (and optionally activity column)
        has_activity: Whether the DataFrame has an activity column
        logger: Logger instance

    Returns:
        DataFrame with original predictions and consensus predictions,
        list of all prediction columns (original + consensus),
        list of consensus-only prediction columns
    """
    if logger:
        logger.info("  Creating consensus predictions...")

    # Separate activity column if present
    if has_activity:
        if 'activity' not in df.columns:
            raise ValueError("Expected 'activity' column not found")
        activity = df['activity']
        pred_cols = [col for col in df.columns if col != 'activity']
        df_pred = df[pred_cols]
    else:
        activity = None
        pred_cols = df.columns.tolist()
        df_pred = df

    if logger:
        logger.info(f"    Found {len(pred_cols)} prediction columns: {pred_cols}")

    # Start with original predictions
    result_df = df_pred.copy()

    # Calculate consensus for all combinations
    all_pred_cols = []
    consensus_cols = []  # Track which columns are consensus

    # Single models (already in df, but we track them)
    for col in pred_cols:
        all_pred_cols.append(col)

    # Pairwise combinations (2 models)
    if len(pred_cols) >= 2:
        n_pairs = len(list(combinations(pred_cols, 2)))
        if logger:
            logger.info(f"    Creating {n_pairs} pairwise consensus predictions...")

        for combo in combinations(pred_cols, 2):
            consensus_name = '+'.join(combo)
            result_df[consensus_name] = df_pred[list(combo)].mean(axis=1)
            all_pred_cols.append(consensus_name)
            consensus_cols.append(consensus_name)

    # Triple combinations (3 models)
    if len(pred_cols) >= 3:
        n_triples = len(list(combinations(pred_cols, 3)))
        if logger:
            logger.info(f"    Creating {n_triples} triple consensus predictions...")

        for combo in combinations(pred_cols, 3):
            consensus_name = '+'.join(combo)
            result_df[consensus_name] = df_pred[list(combo)].mean(axis=1)
            all_pred_cols.append(consensus_name)
            consensus_cols.append(consensus_name)

    # Add activity column at the beginning if present
    if has_activity:
        result_df.insert(0, 'activity', activity)

    if logger:
        logger.info(f"    Total prediction columns: {len(all_pred_cols)} "
                   f"({len(pred_cols)} individual + {len(consensus_cols)} consensus)")

    return result_df, all_pred_cols, consensus_cols


def calculate_auroc_scores(df, pred_cols, consensus_cols, target_name, logger):
    """
    Calculate AUROC scores for all prediction columns.

    Args:
        df: DataFrame with activity and prediction columns
        pred_cols: List of all prediction column names
        consensus_cols: List of consensus-only column names
        target_name: Name of the target/dataset
        logger: Logger instance

    Returns:
        List of dictionaries with results
    """
    if 'activity' not in df.columns:
        logger.warning("    No activity column found, cannot calculate AUROC")
        return []

    logger.info("  Calculating AUROC scores...")

    y_true = df['activity']
    results = []

    for col in pred_cols:
        try:
            y_pred = df[col]
            auroc = roc_auc_score(y_true, y_pred)

            # Determine if this is a consensus model
            is_consensus = col in consensus_cols
            model_type = 'consensus' if is_consensus else 'individual'

            # Count number of models in consensus
            n_models = len(col.split('+'))

            results.append({
                'target': target_name,
                'model': col,
                'model_type': model_type,
                'n_models': n_models,
                'test_auroc': auroc,
                'status': 'success'
            })

        except Exception as e:
            logger.error(f"    Error calculating AUROC for {col}: {str(e)}")
            results.append({
                'target': target_name,
                'model': col,
                'model_type': 'error',
                'n_models': 0,
                'test_auroc': np.nan,
                'status': f'failed: {str(e)[:100]}'
            })

    # Log summary statistics
    successful = [r for r in results if r['status'] == 'success']
    if successful:
        aurocs = [r['test_auroc'] for r in successful]
        logger.info(f"    Successfully evaluated {len(successful)} models")
        logger.info(f"    AUROC range: {min(aurocs):.6f} - {max(aurocs):.6f}")
        logger.info(f"    Mean AUROC: {np.mean(aurocs):.6f}")

    return results


def save_results(results_df, output_dir, logger):
    """Save AUROC results to CSV file."""
    if results_df.empty:
        logger.warning("No results to save")
        return None

    try:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save to CSV with simple filename
        csv_path = output_dir / "results.csv"
        results_df.to_csv(csv_path, index=False, float_format='%.6f')
        logger.info(f"  Results saved to: {csv_path}")

        return csv_path

    except Exception as e:
        logger.error(f"  ERROR saving results: {str(e)}")
        logger.error(f"  Traceback: {traceback.format_exc()}")
        return None


def main():
    parser = argparse.ArgumentParser(
        description='Calculate consensus predictions from FLAML models',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example:
    python flaml-4-consensus.py ./5-predictions/5-flaml-descriptors/euos25_challenge_train_fluorescence340_450

Expected directory structure:
  {target}/
  ├── predictions/
  │   ├── test_predictions.csv          # Input
  │   └── global_test_predictions.csv   # Input
  └── consensus/
      ├── test_predictions.csv          # Output
      ├── global_test_predictions.csv   # Output
      └── results.csv                   # Output
        """
    )
    parser.add_argument('target_dir', type=str,
                        help='Target directory (e.g., .../euos25_challenge_train_fluorescence340_450)')

    args = parser.parse_args()

    target_dir = Path(args.target_dir)
    target_name = target_dir.name

    # Define input and output directories
    predictions_dir = target_dir / "predictions"
    consensus_dir = target_dir / "consensus"
    consensus_dir.mkdir(parents=True, exist_ok=True)

    # Setup logging
    logger, log_filename = setup_logging(consensus_dir)

    logger.info("="*80)
    logger.info("FLAML CONSENSUS PREDICTIONS")
    logger.info("="*80)
    logger.info(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"Target: {target_name}")
    logger.info(f"Log file: {log_filename}")
    logger.info("")

    try:
        # Check if directories exist
        if not target_dir.exists():
            logger.error(f"ERROR: Target directory does not exist: {target_dir}")
            return 1

        if not predictions_dir.exists():
            logger.error(f"ERROR: Predictions directory does not exist: {predictions_dir}")
            logger.error(f"  Expected: {predictions_dir}")
            return 1

        log_separator(logger)
        logger.info("CONFIGURATION")
        log_separator(logger)
        logger.info(f"Target directory: {target_dir}")
        logger.info(f"Predictions directory: {predictions_dir}")
        logger.info(f"Consensus output directory: {consensus_dir}")

        # Define input files
        test_pred_file = predictions_dir / "test_predictions.csv"
        global_test_pred_file = predictions_dir / "global_test_predictions.csv"

        logger.info("\n" + "="*80)
        logger.info("CHECKING INPUT FILES")
        log_separator(logger)
        logger.info(f"Test predictions: {test_pred_file}")
        logger.info(f"  Exists: {test_pred_file.exists()}")
        logger.info(f"Global test predictions: {global_test_pred_file}")
        logger.info(f"  Exists: {global_test_pred_file.exists()}")

        if not test_pred_file.exists():
            logger.error("ERROR: Test predictions file not found!")
            logger.error(f"  Expected: {test_pred_file}")
            return 1

        all_results = []

        # ====================================================================
        # Process test predictions (has activity column)
        # ====================================================================
        logger.info("\n" + "="*80)
        logger.info("PROCESSING TEST PREDICTIONS")
        log_separator(logger)

        logger.info(f"  Loading: {test_pred_file.name}")
        df_test = pd.read_csv(test_pred_file)
        logger.info(f"    Shape: {df_test.shape}")
        logger.info(f"    Columns: {list(df_test.columns)}")

        # Create consensus predictions
        df_test_consensus, test_pred_cols, test_consensus_cols = create_consensus_predictions(
            df_test, has_activity=True, logger=logger
        )

        # Save consensus test predictions
        output_path = consensus_dir / "test_predictions.csv"
        logger.info(f"\n  Saving consensus test predictions...")
        df_test_consensus.to_csv(output_path, index=False, float_format='%.10f')
        logger.info(f"    Saved to: {output_path}")
        logger.info(f"    Shape: {df_test_consensus.shape}")

        # Calculate AUROC scores
        logger.info(f"\n  Evaluating test predictions...")
        test_results = calculate_auroc_scores(
            df_test_consensus, test_pred_cols, test_consensus_cols,
            target_name, logger
        )
        all_results.extend(test_results)

        # ====================================================================
        # Process global test predictions (no activity column)
        # ====================================================================
        if global_test_pred_file.exists():
            logger.info("\n" + "="*80)
            logger.info("PROCESSING GLOBAL TEST PREDICTIONS")
            log_separator(logger)

            logger.info(f"  Loading: {global_test_pred_file.name}")
            df_global = pd.read_csv(global_test_pred_file)
            logger.info(f"    Shape: {df_global.shape}")
            logger.info(f"    Columns: {list(df_global.columns)}")

            # Create consensus predictions
            df_global_consensus, global_pred_cols, global_consensus_cols = create_consensus_predictions(
                df_global, has_activity=False, logger=logger
            )

            # Save consensus global test predictions
            output_path = consensus_dir / "global_test_predictions.csv"
            logger.info(f"\n  Saving consensus global test predictions...")
            df_global_consensus.to_csv(output_path, index=False, float_format='%.10f')
            logger.info(f"    Saved to: {output_path}")
            logger.info(f"    Shape: {df_global_consensus.shape}")
        else:
            logger.info("\n  Global test predictions file not found, skipping")

        # ====================================================================
        # Save aggregated results
        # ====================================================================
        logger.info("\n" + "="*80)
        logger.info("SAVING RESULTS")
        log_separator(logger)

        if all_results:
            results_df = pd.DataFrame(all_results)

            # Reorder columns
            cols = ['target', 'model', 'model_type', 'n_models', 'test_auroc', 'status']
            results_df = results_df[cols]

            # Sort by AUROC (descending)
            results_df = results_df.sort_values('test_auroc', ascending=False, na_position='last')

            save_results(results_df, consensus_dir, logger)

            # Print summary
            logger.info("\n  Summary:")
            successful = results_df[results_df['status'] == 'success']
            if len(successful) > 0:
                logger.info(f"    Total models evaluated: {len(successful)}")
                logger.info(f"    Best AUROC: {successful['test_auroc'].max():.6f} ({successful.iloc[0]['model']})")
                logger.info(f"    Mean AUROC: {successful['test_auroc'].mean():.6f}")

                individual = successful[successful['model_type'] == 'individual']
                consensus = successful[successful['model_type'] == 'consensus']

                if len(individual) > 0:
                    logger.info(f"\n    Individual models:")
                    logger.info(f"      Count: {len(individual)}")
                    logger.info(f"      Best AUROC: {individual['test_auroc'].max():.6f}")

                if len(consensus) > 0:
                    logger.info(f"\n    Consensus models:")
                    logger.info(f"      Count: {len(consensus)}")
                    logger.info(f"      Best AUROC: {consensus['test_auroc'].max():.6f}")

        log_separator(logger)
        logger.info("COMPLETED SUCCESSFULLY")
        log_separator(logger)
        logger.info(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info(f"Results saved to: {consensus_dir}")
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


if __name__ == "__main__":
    exit(main())
