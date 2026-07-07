#!/usr/bin/env python3
"""
Summarize FLAML results and create submission file.

NEW CLEAN DIRECTORY STRUCTURE:
Input:  {output_base}/{target}/consensus/results.csv
        {output_base}/{target}/semisupervised/results.csv
        {output_base}/{target}/consensus/global_test_predictions.csv
        {output_base}/{target}/semisupervised/global_test_predictions.csv
Output: {output_base}/final_submission.csv
        {output_base}/final_summary.txt

This script:
1. Scans all target directories in output_base
2. Reads consensus results from {target}/consensus/results.csv
3. Reads semi-supervised results from {target}/semisupervised/results.csv
4. Selects the best performing method for each target (by test AUROC)
5. Generates final submission file with predictions for all four targets

Usage:
    python flaml-9-summarize.py <output_base_directory>

Example:
    python flaml-9-summarize.py ./5-predictions/5-flaml-descriptors
"""

import argparse
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import logging
import sys
import warnings
warnings.filterwarnings('ignore')


# Mapping from target names to submission column names
TARGET_TO_COLUMN = {
    'euos25_challenge_train_transmittance340': 'Transmittance(340)',
    'euos25_challenge_train_transmittance450plus': 'Transmittance(450)',
    'euos25_challenge_train_fluorescence340_450': 'Fluorescence(340/480)',
    'euos25_challenge_train_fluorescence480plus': 'Fluorescence(multiple)',
}

# Required column order for submission
SUBMISSION_COLUMNS = [
    'Transmittance(340)',
    'Transmittance(450)',
    'Fluorescence(340/480)',
    'Fluorescence(multiple)',
]


def setup_logging(output_dir):
    """Setup logging to both file and console."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_filename = output_dir / f"summary_{timestamp}.log"

    logger = logging.getLogger('flaml_summarize')
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


def find_target_directories(base_dir, logger):
    """
    Find all target directories in the base directory.

    Returns:
        List of (target_name, target_path) tuples
    """
    base_dir = Path(base_dir)
    targets = []

    for item in base_dir.iterdir():
        if item.is_dir() and item.name in TARGET_TO_COLUMN:
            targets.append((item.name, item))

    logger.info(f"  Found {len(targets)} target directories")
    for target_name, target_path in sorted(targets):
        logger.info(f"    - {target_name}")

    return targets


def load_results_from_target(target_name, target_path, logger):
    """
    Load all results (consensus + semisupervised) for a single target.

    Returns:
        DataFrame with combined results
    """
    all_results = []

    # Load consensus results
    consensus_file = target_path / "consensus" / "results.csv"
    if consensus_file.exists():
        try:
            logger.info(f"    Loading consensus results: {consensus_file.name}")
            df = pd.read_csv(consensus_file)
            df['source_type'] = 'consensus'
            df['target'] = target_name
            all_results.append(df)
            logger.info(f"      Loaded {len(df)} consensus results")
        except Exception as e:
            logger.error(f"      Error loading consensus results: {str(e)}")
    else:
        logger.info(f"    No consensus results found")

    # Load semi-supervised results
    semisup_file = target_path / "semisupervised" / "results.csv"
    if semisup_file.exists():
        try:
            logger.info(f"    Loading semi-supervised results: {semisup_file.name}")
            df = pd.read_csv(semisup_file)
            df['source_type'] = 'semisupervised'
            df['target'] = target_name

            # Add strategy prefix for clarity if not already present
            if 'strategy' in df.columns:
                # Check if strategy already has semisup_ prefix
                if not df['strategy'].str.startswith('semisup_').any():
                    df['strategy'] = 'semisup_' + df['strategy'].astype(str)

            all_results.append(df)
            logger.info(f"      Loaded {len(df)} semi-supervised results")
        except Exception as e:
            logger.error(f"      Error loading semi-supervised results: {str(e)}")
    else:
        logger.info(f"    No semi-supervised results found")

    if all_results:
        combined = pd.concat(all_results, ignore_index=True)
        return combined
    else:
        return pd.DataFrame()


def load_all_results(base_dir, logger):
    """
    Load all results from all target directories.

    Returns:
        DataFrame with all combined results
    """
    logger.info("\n  Scanning target directories...")

    targets = find_target_directories(base_dir, logger)

    if not targets:
        logger.error("  No target directories found!")
        return pd.DataFrame()

    all_results = []

    for target_name, target_path in targets:
        logger.info(f"\n  Processing {target_name}...")
        df = load_results_from_target(target_name, target_path, logger)
        if not df.empty:
            all_results.append(df)

    if all_results:
        combined = pd.concat(all_results, ignore_index=True)
        logger.info(f"\n  Total results loaded: {len(combined)}")
        logger.info(f"  Targets: {combined['target'].nunique()}")
        return combined
    else:
        return pd.DataFrame()


def select_best_per_target(df, logger):
    """Select best performing method for each target."""
    logger.info("\n  Selecting best method per target...")

    if df.empty:
        logger.error("  No results to process!")
        return {}

    # Filter to only valid AUROC scores
    valid_df = df[df['test_auroc'].notna()].copy()

    if valid_df.empty:
        logger.error("  No valid AUROC scores found!")
        return {}

    best_methods = {}

    for target in TARGET_TO_COLUMN.keys():
        target_df = valid_df[valid_df['target'] == target]

        if target_df.empty:
            logger.warning(f"  No results for target: {target}")
            continue

        # Sort by AUROC (descending)
        target_df = target_df.sort_values('test_auroc', ascending=False)

        # Get best result
        best = target_df.iloc[0]

        # Determine method name (could be in 'model' or 'method' column)
        # Note: need to check for NaN since concat creates both columns
        model_val = best.get('model')
        method_val = best.get('method')

        # Use model if it exists and is not NaN, otherwise use method
        if pd.notna(model_val):
            method_name = model_val
        elif pd.notna(method_val):
            method_name = method_val
        else:
            method_name = 'unknown'

        best_methods[target] = {
            'target': target,
            'method_name': method_name,
            'strategy': best.get('strategy', 'unknown'),
            'test_auroc': best['test_auroc'],
            'source_type': best.get('source_type', 'unknown'),
        }

        column_name = TARGET_TO_COLUMN[target]
        logger.info(f"\n  {column_name}:")
        logger.info(f"    Method: {best_methods[target]['method_name']}")
        logger.info(f"    Strategy: {best_methods[target]['strategy']}")
        logger.info(f"    Test AUROC: {best_methods[target]['test_auroc']:.6f}")
        logger.info(f"    Source: {best_methods[target]['source_type']}")

    return best_methods


def load_predictions_for_target(target_path, source_type, method_name, logger):
    """
    Load global test predictions for a specific target and method.

    Returns:
        numpy array of predictions, or None if not found
    """
    # Determine which predictions file to use based on source_type
    if source_type == 'consensus':
        pred_file = target_path / "consensus" / "global_test_predictions.csv"
    elif source_type == 'semisupervised':
        pred_file = target_path / "semisupervised" / "global_test_predictions.csv"
    else:
        logger.error(f"    Unknown source type: {source_type}")
        return None

    if not pred_file.exists():
        logger.error(f"    Prediction file not found: {pred_file}")
        return None

    try:
        logger.info(f"    Loading predictions from: {pred_file.name}")
        df = pd.read_csv(pred_file)

        logger.info(f"      File shape: {df.shape}")
        logger.info(f"      Available columns: {list(df.columns)[:10]}...")

        # Try to find the method column
        # For consensus files, look for the exact model name or consensus combination
        # For semisup files, look for method_strategy format

        possible_columns = [method_name]

        # For semi-supervised, try variations with strategy suffix
        if source_type == 'semisupervised':
            # Method name might be just the learner (e.g., 'lgbm')
            # Column might be 'lgbm_ensemble', 'lgbm_simple', etc.
            for col in df.columns:
                col_str = str(col)  # Ensure column name is string
                if method_name in col_str or col_str.startswith(method_name):
                    possible_columns.append(col)

        # For consensus, the column might be the full consensus name
        # e.g., 'lgbm+catboost+xgboost'
        if source_type == 'consensus' and '+' in method_name:
            possible_columns.append(method_name)

        # Try to find matching column
        for col in possible_columns:
            if col in df.columns:
                logger.info(f"      Using column: {col}")
                return df[col].values

        # If exact match not found, try to find best match
        logger.warning(f"      Method '{method_name}' not found in columns")
        logger.info(f"      Trying to find best match...")

        # For consensus, if we can't find exact match, use first column
        if source_type == 'consensus':
            # Try to find any consensus column (contains '+')
            consensus_cols = [col for col in df.columns if '+' in str(col)]
            if consensus_cols:
                best_col = consensus_cols[0]
                logger.info(f"      Using first consensus column: {best_col}")
                return df[best_col].values

        # For semisup, try to find column with matching base method
        if source_type == 'semisupervised':
            base_method = method_name.replace('semisup_', '').split('_')[0]
            matching_cols = [col for col in df.columns if base_method in col]
            if matching_cols:
                best_col = matching_cols[0]
                logger.info(f"      Using first matching column: {best_col}")
                return df[best_col].values

        # Last resort: use first non-index column
        if len(df.columns) > 0:
            best_col = df.columns[0]
            logger.warning(f"      Using fallback column: {best_col}")
            return df[best_col].values

        logger.error(f"      Could not find any suitable predictions column")
        return None

    except Exception as e:
        logger.error(f"      Error loading predictions: {str(e)}")
        return None


def create_submission_file(best_methods, base_dir, output_dir, logger):
    """
    Create submission file with predictions from best methods for each target.
    """
    logger.info("\n  Creating submission file...")

    base_dir = Path(base_dir)
    submission_data = {}

    for target, info in best_methods.items():
        column_name = TARGET_TO_COLUMN[target]
        logger.info(f"\n  Processing {column_name}...")

        target_path = base_dir / target

        # Load predictions
        predictions = load_predictions_for_target(
            target_path, info['source_type'], info['method_name'], logger
        )

        if predictions is not None:
            submission_data[column_name] = predictions
            logger.info(f"    ✓ Loaded {len(predictions)} predictions")
        else:
            logger.error(f"    ✗ Failed to load predictions")

    # Check if we have all required columns
    missing_columns = [col for col in SUBMISSION_COLUMNS if col not in submission_data]
    if missing_columns:
        logger.error(f"\n  ERROR: Missing required columns: {missing_columns}")
        logger.error(f"  Cannot create submission file")
        return None

    # Create DataFrame with required column order
    submission_df = pd.DataFrame(submission_data)[SUBMISSION_COLUMNS]

    # Save submission file with simple filename
    output_path = output_dir / "final_submission.csv"

    submission_df.to_csv(output_path, index=False, float_format='%.10f')
    logger.info(f"\n  ✓ Submission file saved: {output_path}")
    logger.info(f"    Shape: {submission_df.shape}")
    logger.info(f"    Columns: {list(submission_df.columns)}")

    return output_path


def save_summary(combined_df, best_methods, output_dir, logger):
    """Save comprehensive summary of all results."""

    # Save detailed text report with simple filename
    txt_path = output_dir / "final_summary.txt"

    with open(txt_path, 'w') as f:
        f.write("="*80 + "\n")
        f.write("FLAML COMPLETE RESULTS SUMMARY\n")
        f.write("="*80 + "\n\n")
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        f.write(f"Generated: {timestamp}\n")
        f.write(f"Total evaluations: {len(combined_df)}\n")
        f.write(f"Valid AUROC scores: {combined_df['test_auroc'].notna().sum()}\n\n")

        # Best methods section
        f.write("="*80 + "\n")
        f.write("BEST METHODS PER TARGET (SELECTED FOR SUBMISSION)\n")
        f.write("="*80 + "\n\n")

        for col in SUBMISSION_COLUMNS:
            target = [k for k, v in TARGET_TO_COLUMN.items() if v == col][0]
            if target in best_methods:
                info = best_methods[target]
                f.write(f"{col}:\n")
                f.write(f"  Method:      {info['method_name']}\n")
                f.write(f"  Strategy:    {info['strategy']}\n")
                f.write(f"  Test AUROC:  {info['test_auroc']:.6f}\n")
                f.write(f"  Source:      {info['source_type']}\n\n")

        # Results by target
        f.write("\n" + "="*80 + "\n")
        f.write("TOP 10 METHODS PER TARGET\n")
        f.write("="*80 + "\n\n")

        valid_df = combined_df[combined_df['test_auroc'].notna()].copy()

        for col in SUBMISSION_COLUMNS:
            target = [k for k, v in TARGET_TO_COLUMN.items() if v == col][0]
            target_df = valid_df[valid_df['target'] == target]

            if target_df.empty:
                continue

            f.write(f"\n{col}:\n")
            f.write("-"*80 + "\n")
            f.write(f"{'Rank':<6} {'AUROC':<10} {'Strategy':<25} {'Method':<40}\n")
            f.write("-"*80 + "\n")

            # Sort and show top 10
            sorted_df = target_df.sort_values('test_auroc', ascending=False).head(10)

            for rank, (_, row) in enumerate(sorted_df.iterrows(), 1):
                auroc = f"{row['test_auroc']:.6f}"
                strategy = str(row.get('strategy', 'unknown'))[:25]
                method = str(row.get('model', row.get('method', 'unknown')))[:40]
                f.write(f"{rank:<6} {auroc:<10} {strategy:<25} {method:<40}\n")

        # Statistics by source type
        f.write("\n\n" + "="*80 + "\n")
        f.write("STATISTICS BY SOURCE TYPE\n")
        f.write("="*80 + "\n\n")

        for source_type in valid_df['source_type'].unique():
            source_df = valid_df[valid_df['source_type'] == source_type]
            f.write(f"{source_type.upper()}:\n")
            f.write(f"  Count:       {len(source_df)}\n")
            f.write(f"  Best AUROC:  {source_df['test_auroc'].max():.6f}\n")
            f.write(f"  Mean AUROC:  {source_df['test_auroc'].mean():.6f}\n")
            f.write(f"  Median:      {source_df['test_auroc'].median():.6f}\n\n")

        f.write("\n" + "="*80 + "\n")
        f.write("SUMMARY STATISTICS\n")
        f.write("="*80 + "\n\n")

        f.write(f"Overall best AUROC: {valid_df['test_auroc'].max():.6f}\n")
        f.write(f"Overall mean AUROC: {valid_df['test_auroc'].mean():.6f}\n")
        f.write(f"Overall median AUROC: {valid_df['test_auroc'].median():.6f}\n")

        # Count of results per target
        f.write(f"\nResults per target:\n")
        for target in TARGET_TO_COLUMN.keys():
            count = len(valid_df[valid_df['target'] == target])
            col_name = TARGET_TO_COLUMN[target]
            f.write(f"  {col_name}: {count}\n")

    logger.info(f"  Detailed summary saved: {txt_path}")

    return txt_path


def main():
    parser = argparse.ArgumentParser(
        description='Summarize FLAML results and create submission file',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
This script aggregates results from the new clean directory structure:
  - {target}/consensus/results.csv (consensus predictions)
  - {target}/semisupervised/results.csv (semi-supervised learning)

It selects the best performing method for each target and creates a submission file.

Example:
    python flaml-9-summarize.py ./5-predictions/5-flaml-descriptors

Expected directory structure:
  {output_base}/
  ├── euos25_challenge_train_fluorescence340_450/
  │   ├── consensus/
  │   │   ├── results.csv
  │   │   └── global_test_predictions.csv
  │   └── semisupervised/
  │       ├── results.csv
  │       └── global_test_predictions.csv
  ├── euos25_challenge_train_fluorescence480plus/
  ├── euos25_challenge_train_transmittance340/
  ├── euos25_challenge_train_transmittance450plus/
  ├── final_submission.csv          # Output
  └── final_summary.txt             # Output
        """
    )

    parser.add_argument('directory', type=Path,
                        help='Base directory containing target subdirectories')

    args = parser.parse_args()

    base_dir = Path(args.directory)

    if not base_dir.exists():
        print(f"Error: Directory does not exist: {base_dir}")
        return 1

    # Setup logging
    logger, log_filename = setup_logging(base_dir)

    logger.info("="*80)
    logger.info("FLAML RESULTS SUMMARY AND SUBMISSION GENERATOR")
    logger.info("="*80)
    logger.info(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"Base directory: {base_dir}")
    logger.info(f"Log file: {log_filename}")
    logger.info("")

    try:
        # Step 1: Load all results
        log_separator(logger)
        logger.info("STEP 1: LOADING ALL RESULTS")
        log_separator(logger)

        combined_df = load_all_results(base_dir, logger)

        if combined_df.empty:
            logger.error("\nNo results found!")
            logger.error("Make sure target directories exist with consensus/ or semisupervised/ subdirectories")
            return 1

        # Step 2: Select best methods
        log_separator(logger)
        logger.info("STEP 2: SELECTING BEST METHODS PER TARGET")
        log_separator(logger)

        best_methods = select_best_per_target(combined_df, logger)

        if len(best_methods) != len(TARGET_TO_COLUMN):
            logger.warning(f"\nWarning: Expected {len(TARGET_TO_COLUMN)} targets, found {len(best_methods)}")
            missing = set(TARGET_TO_COLUMN.keys()) - set(best_methods.keys())
            if missing:
                logger.warning(f"Missing targets: {missing}")

        # Step 3: Create submission file
        log_separator(logger)
        logger.info("STEP 3: CREATING SUBMISSION FILE")
        log_separator(logger)

        submission_path = create_submission_file(best_methods, base_dir, base_dir, logger)

        # Step 4: Save summary
        log_separator(logger)
        logger.info("STEP 4: SAVING SUMMARY")
        log_separator(logger)

        summary_path = save_summary(combined_df, best_methods, base_dir, logger)

        # Final summary
        log_separator(logger)
        logger.info("COMPLETED SUCCESSFULLY")
        log_separator(logger)
        logger.info(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info(f"\nAll outputs saved to: {base_dir}")

        if submission_path:
            logger.info(f"\n✓ Submission file ready: {submission_path.name}")
        if summary_path:
            logger.info(f"✓ Summary file ready: {summary_path.name}")

        logger.info("")

        return 0

    except Exception as e:
        logger.error("\n" + "="*80)
        logger.error("FATAL ERROR")
        logger.error("="*80)
        logger.error(f"An unexpected error occurred: {str(e)}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        logger.error("="*80)
        return 1


if __name__ == "__main__":
    exit(main())
