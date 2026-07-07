#!/usr/bin/env python3
"""
Extract best configurations from FLAML optimization logs.

This script:
1. Parses FLAML log files (JSON lines format)
2. Extracts the top-n best configurations for each learner
3. Analyzes optimization progress over time
4. Generates plots showing performance and convergence
5. Saves best configs in a format ready for model training
"""

import json
import argparse
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from collections import defaultdict


def parse_flaml_log(log_file: Path) -> List[Dict]:
    """Parse FLAML log file (JSON lines format)."""
    records = []
    with open(log_file, 'r') as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    record = json.loads(line)
                    records.append(record)
                except json.JSONDecodeError as e:
                    print(f"Warning: Failed to parse line: {line[:100]}... Error: {e}")
    return records


def extract_best_configs(records: List[Dict], n_best: int = 1) -> Dict[str, List[Dict]]:
    """
    Extract top-n best configurations for each learner.
    
    Args:
        records: List of FLAML log records
        n_best: Number of best configurations to extract per learner
        
    Returns:
        Dictionary mapping learner name to list of best configs
    """
    # Group records by learner
    learner_records = defaultdict(list)
    for record in records:
        learner = record.get('learner')
        if learner:
            learner_records[learner].append(record)
    
    # Extract top-n for each learner
    best_configs = {}
    for learner, recs in learner_records.items():
        # Sort by validation_loss (lower is better)
        sorted_recs = sorted(recs, key=lambda x: x.get('validation_loss', float('inf')))
        best_configs[learner] = sorted_recs[:n_best]
    
    return best_configs


def analyze_optimization_progress(records: List[Dict]) -> Dict:
    """
    Analyze optimization progress across all trials.
    
    Returns:
        Dictionary with analysis results
    """
    analysis = {
        'total_trials': len(records),
        'learners': defaultdict(lambda: {'count': 0, 'best_loss': float('inf')}),
        'cumulative_best': [],
        'wall_clock_times': [],
        'validation_losses': [],
        'train_losses': [],
        'trial_times': [],
        'learner_sequence': [],
        'best_overall': None
    }
    
    current_best = float('inf')
    
    for record in records:
        learner = record.get('learner', 'unknown')
        val_loss = record.get('validation_loss', float('inf'))
        train_loss = record.get('logged_metric', {}).get('train_loss', float('inf'))
        trial_time = record.get('trial_time', 0)
        wall_time = record.get('wall_clock_time', 0)
        
        # Update learner stats
        analysis['learners'][learner]['count'] += 1
        if val_loss < analysis['learners'][learner]['best_loss']:
            analysis['learners'][learner]['best_loss'] = val_loss
        
        # Track cumulative best
        if val_loss < current_best:
            current_best = val_loss
        analysis['cumulative_best'].append(current_best)
        
        # Track other metrics
        analysis['wall_clock_times'].append(wall_time)
        analysis['validation_losses'].append(val_loss)
        analysis['train_losses'].append(train_loss)
        analysis['trial_times'].append(trial_time)
        analysis['learner_sequence'].append(learner)
        
        # Track best overall
        if val_loss == current_best:
            analysis['best_overall'] = record
    
    return analysis


def plot_optimization_progress(analysis: Dict, output_file: Path):
    """
    Create comprehensive visualization of optimization progress.
    """
    fig = plt.figure(figsize=(16, 12))
    gs = gridspec.GridSpec(3, 2, figure=fig, hspace=0.3, wspace=0.3)
    
    # 1. Cumulative Best Loss over Trials
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(range(len(analysis['cumulative_best'])), 
             analysis['cumulative_best'], 
             linewidth=2, color='#2E86AB')
    ax1.set_xlabel('Trial Number', fontsize=11)
    ax1.set_ylabel('Best Validation Loss (AUROC)', fontsize=11)
    ax1.set_title('Optimization Progress: Cumulative Best', fontsize=12, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    
    # 2. All Validation Losses (scatter by learner)
    ax2 = fig.add_subplot(gs[0, 1])
    learners_unique = list(set(analysis['learner_sequence']))
    colors = plt.cm.tab10(np.linspace(0, 1, len(learners_unique)))
    color_map = dict(zip(learners_unique, colors))
    
    for learner in learners_unique:
        indices = [i for i, l in enumerate(analysis['learner_sequence']) if l == learner]
        losses = [analysis['validation_losses'][i] for i in indices]
        ax2.scatter(indices, losses, label=learner, alpha=0.6, s=30, color=color_map[learner])
    
    ax2.set_xlabel('Trial Number', fontsize=11)
    ax2.set_ylabel('Validation Loss (AUROC)', fontsize=11)
    ax2.set_title('All Trials by Learner', fontsize=12, fontweight='bold')
    ax2.legend(loc='best', fontsize=9)
    ax2.grid(True, alpha=0.3)
    
    # 3. Train vs Validation Loss
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.scatter(analysis['train_losses'], analysis['validation_losses'], 
                alpha=0.5, s=30, color='#A23B72')
    ax3.plot([0, max(analysis['train_losses'])], 
             [0, max(analysis['train_losses'])], 
             'k--', alpha=0.3, label='Perfect fit')
    ax3.set_xlabel('Train Loss', fontsize=11)
    ax3.set_ylabel('Validation Loss', fontsize=11)
    ax3.set_title('Train vs Validation Loss', fontsize=12, fontweight='bold')
    ax3.legend(fontsize=9)
    ax3.grid(True, alpha=0.3)
    
    # 4. Trial Times
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.plot(range(len(analysis['trial_times'])), 
             analysis['trial_times'], 
             linewidth=1, alpha=0.7, color='#F18F01')
    ax4.set_xlabel('Trial Number', fontsize=11)
    ax4.set_ylabel('Trial Time (seconds)', fontsize=11)
    ax4.set_title('Computational Cost per Trial', fontsize=12, fontweight='bold')
    ax4.grid(True, alpha=0.3)
    
    # 5. Learner Performance Distribution (Box plot)
    ax5 = fig.add_subplot(gs[2, 0])
    learner_losses = defaultdict(list)
    for learner, loss in zip(analysis['learner_sequence'], analysis['validation_losses']):
        learner_losses[learner].append(loss)
    
    bp = ax5.boxplot([learner_losses[l] for l in learners_unique], 
                      labels=learners_unique,
                      patch_artist=True)
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)
    
    ax5.set_xlabel('Learner', fontsize=11)
    ax5.set_ylabel('Validation Loss (AUROC)', fontsize=11)
    ax5.set_title('Performance Distribution by Learner', fontsize=12, fontweight='bold')
    ax5.grid(True, alpha=0.3, axis='y')
    plt.setp(ax5.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    # 6. Learner Trial Counts
    ax6 = fig.add_subplot(gs[2, 1])
    learner_counts = [analysis['learners'][l]['count'] for l in learners_unique]
    bars = ax6.bar(learners_unique, learner_counts, color=colors, alpha=0.7)
    ax6.set_xlabel('Learner', fontsize=11)
    ax6.set_ylabel('Number of Trials', fontsize=11)
    ax6.set_title('Trial Distribution by Learner', fontsize=12, fontweight='bold')
    ax6.grid(True, alpha=0.3, axis='y')
    plt.setp(ax6.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    plt.suptitle('FLAML Optimization Analysis', fontsize=14, fontweight='bold', y=0.995)
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Plot saved to: {output_file}")
    plt.close()


def generate_text_summary(analysis: Dict, best_configs: Dict[str, List[Dict]], 
                          output_file: Path):
    """
    Generate text summary of optimization results.
    """
    with open(output_file, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("FLAML OPTIMIZATION SUMMARY\n")
        f.write("=" * 80 + "\n\n")
        
        # Overall Statistics
        f.write("OVERALL STATISTICS\n")
        f.write("-" * 80 + "\n")
        f.write(f"Total trials: {analysis['total_trials']}\n")
        f.write(f"Best validation loss: {min(analysis['validation_losses']):.6f}\n")
        f.write(f"Worst validation loss: {max(analysis['validation_losses']):.6f}\n")
        f.write(f"Mean validation loss: {np.mean(analysis['validation_losses']):.6f}\n")
        f.write(f"Std validation loss: {np.std(analysis['validation_losses']):.6f}\n")
        f.write(f"Total wall clock time: {max(analysis['wall_clock_times']):.2f} seconds "
                f"({max(analysis['wall_clock_times'])/60:.2f} minutes)\n")
        f.write(f"Mean trial time: {np.mean(analysis['trial_times']):.2f} seconds\n\n")
        
        # Learner Statistics
        f.write("LEARNER STATISTICS\n")
        f.write("-" * 80 + "\n")
        f.write(f"{'Learner':<15} {'Trials':<10} {'Best Loss':<15} {'Mean Loss':<15}\n")
        f.write("-" * 80 + "\n")
        
        for learner in sorted(analysis['learners'].keys()):
            count = analysis['learners'][learner]['count']
            best_loss = analysis['learners'][learner]['best_loss']
            learner_losses = [loss for l, loss in zip(analysis['learner_sequence'], 
                                                       analysis['validation_losses']) 
                              if l == learner]
            mean_loss = np.mean(learner_losses)
            f.write(f"{learner:<15} {count:<10} {best_loss:<15.6f} {mean_loss:<15.6f}\n")
        
        f.write("\n")
        
        # Best Overall Configuration
        f.write("BEST OVERALL CONFIGURATION\n")
        f.write("-" * 80 + "\n")
        if analysis['best_overall']:
            best = analysis['best_overall']
            f.write(f"Learner: {best.get('learner')}\n")
            f.write(f"Validation Loss: {best.get('validation_loss'):.6f}\n")
            f.write(f"Train Loss: {best.get('logged_metric', {}).get('train_loss', 'N/A')}\n")
            f.write(f"Trial Time: {best.get('trial_time', 'N/A'):.2f} seconds\n")
            f.write(f"Configuration:\n")
            config = best.get('config', {})
            for key, value in sorted(config.items()):
                if key != 'FLAML_sample_size':
                    f.write(f"  {key}: {value}\n")
        f.write("\n")
        
        # Top Configurations per Learner
        f.write("TOP CONFIGURATIONS BY LEARNER\n")
        f.write("=" * 80 + "\n\n")
        
        for learner in sorted(best_configs.keys()):
            f.write(f"Learner: {learner.upper()}\n")
            f.write("-" * 80 + "\n")
            
            for idx, record in enumerate(best_configs[learner], 1):
                f.write(f"\nRank #{idx}:\n")
                f.write(f"  Validation Loss: {record.get('validation_loss'):.6f}\n")
                f.write(f"  Train Loss: {record.get('logged_metric', {}).get('train_loss', 'N/A')}\n")
                f.write(f"  Trial Time: {record.get('trial_time', 'N/A'):.2f} seconds\n")
                f.write(f"  Configuration:\n")
                config = record.get('config', {})
                for key, value in sorted(config.items()):
                    if key != 'FLAML_sample_size':
                        f.write(f"    {key}: {value}\n")
            
            f.write("\n")
    
    print(f"Summary saved to: {output_file}")


def save_best_configs(best_configs: Dict[str, List[Dict]], output_file: Path):
    """
    Save best configurations in JSON format for easy loading.
    
    Format:
    {
        "learner_name": [
            {
                "config": {...},
                "validation_loss": float,
                "train_loss": float,
                "metadata": {...}
            },
            ...
        ],
        ...
    }
    """
    output_data = {}
    
    for learner, records in best_configs.items():
        output_data[learner] = []
        for record in records:
            config_data = {
                'config': {k: v for k, v in record.get('config', {}).items() 
                          if k != 'FLAML_sample_size'},
                'validation_loss': record.get('validation_loss'),
                'train_loss': record.get('logged_metric', {}).get('train_loss'),
                'metadata': {
                    'record_id': record.get('record_id'),
                    'trial_time': record.get('trial_time'),
                    'wall_clock_time': record.get('wall_clock_time'),
                    'sample_size': record.get('sample_size')
                }
            }
            output_data[learner].append(config_data)
    
    with open(output_file, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    print(f"Best configurations saved to: {output_file}")


def main():
    parser = argparse.ArgumentParser(
        description='Extract best configurations from FLAML optimization logs',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Extract top 1 config per learner
  python extract_best_configs.py flaml_log.txt
  
  # Extract top 3 configs per learner
  python extract_best_configs.py flaml_log.txt -n 3
  
  # Custom output directory
  python extract_best_configs.py flaml_log.txt -o results/
        """
    )
    
    parser.add_argument('log_file', type=Path,
                        help='Path to FLAML log file (JSON lines format)')
    parser.add_argument('-n', '--n-best', type=int, default=1,
                        help='Number of best configs to extract per learner (default: 1)')
    parser.add_argument('-o', '--output-dir', type=Path, default=None,
                        help='Output directory (default: same as log file)')
    
    args = parser.parse_args()
    
    # Validate input
    if not args.log_file.exists():
        print(f"Error: Log file not found: {args.log_file}")
        return 1
    
    # Set output directory
    if args.output_dir is None:
        output_dir = args.log_file.parent
    else:
        output_dir = args.output_dir
        output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate output filenames (simplified names without dataset/descriptor info)
    configs_file = output_dir / "best_configs.json"
    summary_file = output_dir / "optimization_summary.txt"
    plot_file = output_dir / "optimization_analysis.png"
    
    print(f"Processing: {args.log_file}")
    print(f"Extracting top {args.n_best} config(s) per learner...")
    print()
    
    # Parse log file
    records = parse_flaml_log(args.log_file)
    if not records:
        print("Error: No valid records found in log file")
        return 1
    
    print(f"Parsed {len(records)} records")
    
    # Extract best configs
    best_configs = extract_best_configs(records, args.n_best)
    print(f"Found configurations for {len(best_configs)} learners")
    
    # Analyze optimization progress
    analysis = analyze_optimization_progress(records)
    
    # Generate outputs
    print("\nGenerating outputs...")
    save_best_configs(best_configs, configs_file)
    generate_text_summary(analysis, best_configs, summary_file)
    plot_optimization_progress(analysis, plot_file)
    
    print("\n" + "=" * 80)
    print("QUICK SUMMARY")
    print("=" * 80)
    print(f"Total trials: {analysis['total_trials']}")
    print(f"Best validation loss: {min(analysis['validation_losses']):.6f}")
    print(f"Best learner: {analysis['best_overall'].get('learner') if analysis['best_overall'] else 'N/A'}")
    print(f"Total time: {max(analysis['wall_clock_times'])/60:.2f} minutes")
    print("\nLearner trial counts:")
    for learner in sorted(analysis['learners'].keys()):
        count = analysis['learners'][learner]['count']
        best = analysis['learners'][learner]['best_loss']
        print(f"  {learner}: {count} trials, best loss = {best:.6f}")
    
    print("\n" + "=" * 80)
    print("Done!")
    return 0


if __name__ == '__main__':
    exit(main())


