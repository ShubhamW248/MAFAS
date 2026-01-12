"""Analyze MAFAS evaluation results with precision, recall, F1-score, and visualizations.

This script reads MAFAS results CSV files, calculates classification metrics,
and generates comprehensive visualizations including confusion matrices,
precision/recall charts, and performance comparisons.

Usage (from repo root):
    python -m evaluation.analyze_mafas_results --input results.csv
    python -m evaluation.analyze_mafas_results --input results.csv --output-dir ./reports
"""

from __future__ import annotations

import os
import argparse
from typing import Dict, List, Tuple
import warnings

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    accuracy_score,
    confusion_matrix,
    classification_report
)

warnings.filterwarnings('ignore')

# Set style for better-looking plots
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10


def calculate_metrics(y_true: List[str], y_pred: List[str], labels: List[str]) -> Dict[str, float]:
    """Calculate precision, recall, F1-score, and accuracy.
    
    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels
        labels: List of all possible labels (e.g., ['BUY', 'HOLD', 'SELL'])
    
    Returns:
        Dictionary with metrics
    """
    # Filter out invalid predictions/ground truth
    valid_mask = pd.Series(y_true).isin(labels) & pd.Series(y_pred).isin(labels)
    y_true_valid = [y for i, y in enumerate(y_true) if valid_mask.iloc[i]]
    y_pred_valid = [y for i, y in enumerate(y_pred) if valid_mask.iloc[i]]
    
    if len(y_true_valid) == 0 or len(y_pred_valid) == 0:
        return {
            'precision': 0.0,
            'recall': 0.0,
            'f1': 0.0,
            'accuracy': 0.0,
            'support': 0
        }
    
    # Calculate metrics
    precision = precision_score(y_true_valid, y_pred_valid, labels=labels, average='weighted', zero_division=0)
    recall = recall_score(y_true_valid, y_pred_valid, labels=labels, average='weighted', zero_division=0)
    f1 = f1_score(y_true_valid, y_pred_valid, labels=labels, average='weighted', zero_division=0)
    accuracy = accuracy_score(y_true_valid, y_pred_valid)
    
    return {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'accuracy': accuracy,
        'support': len(y_true_valid)
    }


def calculate_per_class_metrics(y_true: List[str], y_pred: List[str], labels: List[str]) -> pd.DataFrame:
    """Calculate precision, recall, and F1 for each class.
    
    Returns:
        DataFrame with metrics per class
    """
    # Filter valid predictions
    valid_mask = pd.Series(y_true).isin(labels) & pd.Series(y_pred).isin(labels)
    y_true_valid = [y for i, y in enumerate(y_true) if valid_mask.iloc[i]]
    y_pred_valid = [y for i, y in enumerate(y_pred) if valid_mask.iloc[i]]
    
    if len(y_true_valid) == 0:
        return pd.DataFrame()
    
    metrics_per_class = []
    for label in labels:
        # Precision: TP / (TP + FP)
        tp = sum(1 for t, p in zip(y_true_valid, y_pred_valid) if t == label and p == label)
        fp = sum(1 for t, p in zip(y_true_valid, y_pred_valid) if t != label and p == label)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        
        # Recall: TP / (TP + FN)
        fn = sum(1 for t, p in zip(y_true_valid, y_pred_valid) if t == label and p != label)
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        
        # F1-score
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        
        # Support
        support = sum(1 for t in y_true_valid if t == label)
        
        metrics_per_class.append({
            'Class': label,
            'Precision': precision,
            'Recall': recall,
            'F1-Score': f1,
            'Support': support
        })
    
    return pd.DataFrame(metrics_per_class)


def plot_confusion_matrix(y_true: List[str], y_pred: List[str], labels: List[str], 
                          title: str, ax=None) -> None:
    """Plot confusion matrix."""
    # Filter valid predictions
    valid_mask = pd.Series(y_true).isin(labels) & pd.Series(y_pred).isin(labels)
    y_true_valid = [y for i, y in enumerate(y_true) if valid_mask.iloc[i]]
    y_pred_valid = [y for i, y in enumerate(y_pred) if valid_mask.iloc[i]]
    
    if len(y_true_valid) == 0:
        if ax:
            ax.text(0.5, 0.5, 'No valid predictions', ha='center', va='center')
        return
    
    cm = confusion_matrix(y_true_valid, y_pred_valid, labels=labels)
    
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))
    
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, 
                yticklabels=labels, ax=ax, cbar_kws={'label': 'Count'})
    ax.set_xlabel('Predicted', fontsize=12, fontweight='bold')
    ax.set_ylabel('Actual', fontsize=12, fontweight='bold')
    ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
    plt.tight_layout()


def analyze_results(df: pd.DataFrame, output_dir: str = None) -> Dict:
    """Analyze MAFAS results and generate visualizations.
    
    Args:
        df: DataFrame with MAFAS results
        output_dir: Directory to save plots (if None, uses current directory)
    
    Returns:
        Dictionary with all metrics
    """
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    else:
        output_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Check if ground truth exists and determine evaluation type
    ground_truth_col = None
    evaluation_type = None
    
    # First, try to get evaluation type from the DataFrame if it exists
    if 'Evaluation_Type' in df.columns:
        evaluation_type = df['Evaluation_Type'].iloc[0] if len(df) > 0 else None
    
    # Check for short-term ground truth
    if 'Correct Signal (Short-Term 2-6M)' in df.columns:
        ground_truth_col = 'Correct Signal (Short-Term 2-6M)'
        evaluation_type = 'shortterm'
    # Check for long-term ground truth
    elif 'Correct Signal (Long-Term 3Y)' in df.columns:
        ground_truth_col = 'Correct Signal (Long-Term 3Y)'
        evaluation_type = 'longterm'
    # Check for Ground_Truth_Signal (from evaluation results)
    elif 'Ground_Truth_Signal' in df.columns:
        ground_truth_col = 'Ground_Truth_Signal'
        # If evaluation_type not set, try to infer from filename or column name
        if evaluation_type is None:
            # Try to infer from other context
            if 'Short' in str(df.columns) or 'shortterm' in str(df.columns):
                evaluation_type = 'shortterm'
            elif 'Long' in str(df.columns) or 'longterm' in str(df.columns):
                evaluation_type = 'longterm'
    # Fallback to any ground truth column
    else:
        for col in df.columns:
            if 'Ground_Truth' in col or 'Correct_Signal' in col:
                ground_truth_col = col
                # Try to infer type from column name
                if 'Short' in col or 'short' in col or '2-6M' in col:
                    evaluation_type = 'shortterm'
                elif 'Long' in col or 'long' in col or '3Y' in col:
                    evaluation_type = 'longterm'
                break
    
    if ground_truth_col is None or df[ground_truth_col].isna().all():
        print("⚠️  Warning: No ground truth signals found. Cannot calculate accuracy metrics.")
        print("   Generating signal distribution charts only...")
        plot_signal_distributions(df, output_dir)
        return {}
    
    # Filter rows with valid ground truth
    valid_df = df[df[ground_truth_col].notna() & (df[ground_truth_col] != '')].copy()
    
    if len(valid_df) == 0:
        print("⚠️  No valid ground truth data found.")
        return {}
    
    print(f"\n{'='*70}")
    eval_type_str = evaluation_type.upper() if evaluation_type else "UNKNOWN"
    print(f"ANALYZING {len(valid_df)} STOCKS WITH GROUND TRUTH ({eval_type_str})")
    print(f"{'='*70}\n")
    
    labels = ['BUY', 'HOLD', 'SELL']
    y_true = valid_df[ground_truth_col].tolist()
    
    # Get prediction columns based on evaluation type
    if evaluation_type == 'shortterm':
        # Only evaluate short-term predictions
        prediction_cols = {
            'Judge (Short-Term)': 'Judge_Short_Term',
            'Cautious Value (Short-Term)': 'Cautious Value_Short_Term',
            'Aggressive Growth (Short-Term)': 'Aggressive Growth_Short_Term',
            'Technical Trader (Short-Term)': 'Technical Trader_Short_Term',
        }
    elif evaluation_type == 'longterm':
        # Only evaluate long-term predictions
        prediction_cols = {
            'Judge (Long-Term)': 'Judge_Long_Term',
            'Cautious Value (Long-Term)': 'Cautious Value_Long_Term',
            'Aggressive Growth (Long-Term)': 'Aggressive Growth_Long_Term',
            'Technical Trader (Long-Term)': 'Technical Trader_Long_Term',
        }
    else:
        # Fallback: evaluate all (for backwards compatibility)
        prediction_cols = {
            'Judge (Short-Term)': 'Judge_Short_Term',
            'Judge (Long-Term)': 'Judge_Long_Term',
            'Cautious Value (Short-Term)': 'Cautious Value_Short_Term',
            'Cautious Value (Long-Term)': 'Cautious Value_Long_Term',
            'Aggressive Growth (Short-Term)': 'Aggressive Growth_Short_Term',
            'Aggressive Growth (Long-Term)': 'Aggressive Growth_Long_Term',
            'Technical Trader (Short-Term)': 'Technical Trader_Short_Term',
            'Technical Trader (Long-Term)': 'Technical Trader_Long_Term',
        }
    
    # Calculate metrics for each predictor
    all_metrics = {}
    per_class_metrics = {}
    
    for name, col in prediction_cols.items():
        if col not in valid_df.columns:
            continue
        
        y_pred = valid_df[col].tolist()
        # Filter out "N/A" values (signals not evaluated for this timeframe)
        y_pred = [p for p in y_pred if p != "N/A" and p != ""]
        # Also filter corresponding ground truth
        y_true_filtered = [t for i, t in enumerate(y_true) if valid_df[col].iloc[i] != "N/A" and valid_df[col].iloc[i] != ""]
        
        if len(y_pred) == 0 or len(y_true_filtered) == 0:
            print(f"{name}: No valid predictions (all N/A)")
            continue
        
        metrics = calculate_metrics(y_true_filtered, y_pred, labels)
        all_metrics[name] = metrics
        
        per_class = calculate_per_class_metrics(y_true, y_pred, labels)
        per_class_metrics[name] = per_class
        
        print(f"{name}:")
        print(f"  Accuracy: {metrics['accuracy']:.3f}")
        print(f"  Precision: {metrics['precision']:.3f}")
        print(f"  Recall: {metrics['recall']:.3f}")
        print(f"  F1-Score: {metrics['f1']:.3f}")
        print(f"  Support: {metrics['support']}")
        print()
    
    # Create visualizations
    print("Generating visualizations...")
    
    # 1. Overall metrics comparison
    plot_overall_metrics_comparison(all_metrics, output_dir)
    
    # 2. Per-class metrics
    plot_per_class_metrics(per_class_metrics, output_dir)
    
    # 3. Confusion matrices
    plot_all_confusion_matrices(valid_df, ground_truth_col, prediction_cols, labels, output_dir)
    
    # 4. Signal distributions
    plot_signal_distributions(valid_df, output_dir)
    
    # 5. Comparison chart
    plot_agent_comparison(all_metrics, output_dir)
    
    print(f"\n✓ All visualizations saved to: {output_dir}")
    
    # Save metrics to CSV
    metrics_df = pd.DataFrame(all_metrics).T
    metrics_file = os.path.join(output_dir, 'mafas_metrics_summary.csv')
    metrics_df.to_csv(metrics_file)
    print(f"✓ Metrics summary saved to: {metrics_file}")
    
    return all_metrics


def plot_overall_metrics_comparison(all_metrics: Dict, output_dir: str) -> None:
    """Plot overall metrics comparison across all agents."""
    if not all_metrics:
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('MAFAS Performance Metrics Comparison', fontsize=16, fontweight='bold', y=1.02)
    
    names = list(all_metrics.keys())
    metrics_to_plot = ['accuracy', 'precision', 'recall', 'f1']
    metric_labels = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
    
    for idx, (metric, label) in enumerate(zip(metrics_to_plot, metric_labels)):
        ax = axes[idx // 2, idx % 2]
        values = [all_metrics[name][metric] for name in names]
        
        bars = ax.barh(names, values, color=sns.color_palette("husl", len(names)))
        ax.set_xlabel(label, fontweight='bold')
        ax.set_xlim(0, 1.0)
        ax.grid(axis='x', alpha=0.3)
        
        # Add value labels on bars
        for i, (bar, val) in enumerate(zip(bars, values)):
            ax.text(val + 0.01, i, f'{val:.3f}', va='center', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'overall_metrics_comparison.png'), dpi=300, bbox_inches='tight')
    plt.close()


def plot_per_class_metrics(per_class_metrics: Dict, output_dir: str) -> None:
    """Plot precision, recall, F1 per class for each agent."""
    if not per_class_metrics:
        return
    
    # Focus on Judge and key agents
    key_agents = [k for k in per_class_metrics.keys() if 'Judge' in k or 'Short_Term' in k]
    
    fig, axes = plt.subplots(len(key_agents), 1, figsize=(14, 4 * len(key_agents)))
    if len(key_agents) == 1:
        axes = [axes]
    
    fig.suptitle('Per-Class Metrics (Precision, Recall, F1-Score)', fontsize=16, fontweight='bold')
    
    for idx, agent_name in enumerate(key_agents):
        df = per_class_metrics[agent_name]
        if df.empty:
            continue
        
        ax = axes[idx]
        x = np.arange(len(df))
        width = 0.25
        
        ax.bar(x - width, df['Precision'], width, label='Precision', alpha=0.8)
        ax.bar(x, df['Recall'], width, label='Recall', alpha=0.8)
        ax.bar(x + width, df['F1-Score'], width, label='F1-Score', alpha=0.8)
        
        ax.set_xlabel('Class', fontweight='bold')
        ax.set_ylabel('Score', fontweight='bold')
        ax.set_title(agent_name, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(df['Class'])
        ax.set_ylim(0, 1.1)
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'per_class_metrics.png'), dpi=300, bbox_inches='tight')
    plt.close()


def plot_all_confusion_matrices(df: pd.DataFrame, ground_truth_col: str, 
                                prediction_cols: Dict, labels: List[str], output_dir: str) -> None:
    """Plot confusion matrices for all agents."""
    # Focus on Judge and short-term predictions
    key_cols = {k: v for k, v in prediction_cols.items() 
                if 'Judge' in k or ('Short_Term' in k and 'Judge' not in k)}
    
    n_plots = len(key_cols)
    if n_plots == 0:
        return
    
    cols = 2
    rows = (n_plots + 1) // 2
    
    fig, axes = plt.subplots(rows, cols, figsize=(16, 8 * rows))
    if rows == 1:
        axes = axes.reshape(1, -1)
    fig.suptitle('Confusion Matrices', fontsize=16, fontweight='bold', y=1.0)
    
    y_true = df[ground_truth_col].tolist()
    
    for idx, (name, col) in enumerate(key_cols.items()):
        if col not in df.columns:
            continue
        
        row = idx // cols
        col_idx = idx % cols
        ax = axes[row, col_idx]
        
        y_pred = df[col].tolist()
        plot_confusion_matrix(y_true, y_pred, labels, name, ax=ax)
    
    # Hide unused subplots
    for idx in range(n_plots, rows * cols):
        row = idx // cols
        col_idx = idx % cols
        axes[row, col_idx].axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'confusion_matrices.png'), dpi=300, bbox_inches='tight')
    plt.close()


def plot_signal_distributions(df: pd.DataFrame, output_dir: str) -> None:
    """Plot signal distribution charts."""
    prediction_cols = {
        'Judge_Short_Term': 'Judge (Short-Term)',
        'Judge_Long_Term': 'Judge (Long-Term)',
        'Cautious Value_Short_Term': 'Cautious Value (ST)',
        'Aggressive Growth_Short_Term': 'Aggressive Growth (ST)',
        'Technical Trader_Short_Term': 'Technical Trader (ST)',
    }
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle('Signal Distribution Comparison', fontsize=16, fontweight='bold')
    axes = axes.flatten()
    
    for idx, (col, name) in enumerate(prediction_cols.items()):
        if col not in df.columns:
            continue
        
        ax = axes[idx]
        signals = df[col].value_counts()
        colors = {'BUY': 'green', 'HOLD': 'orange', 'SELL': 'red'}
        signal_colors = [colors.get(s, 'gray') for s in signals.index]
        
        bars = ax.bar(signals.index, signals.values, color=signal_colors, alpha=0.7)
        ax.set_title(name, fontweight='bold')
        ax.set_ylabel('Count', fontweight='bold')
        ax.grid(axis='y', alpha=0.3)
        
        # Add count labels on bars
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{int(height)}', ha='center', va='bottom', fontweight='bold')
    
    # Hide unused subplots
    for idx in range(len(prediction_cols), len(axes)):
        axes[idx].axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'signal_distributions.png'), dpi=300, bbox_inches='tight')
    plt.close()


def plot_agent_comparison(all_metrics: Dict, output_dir: str) -> None:
    """Create a comprehensive comparison chart."""
    if not all_metrics:
        return
    
    # Focus on short-term predictions
    short_term_agents = {k: v for k, v in all_metrics.items() if 'Short' in k}
    
    if not short_term_agents:
        return
    
    fig, ax = plt.subplots(figsize=(14, 8))
    
    names = list(short_term_agents.keys())
    x = np.arange(len(names))
    width = 0.25
    
    accuracy = [short_term_agents[n]['accuracy'] for n in names]
    precision = [short_term_agents[n]['precision'] for n in names]
    recall = [short_term_agents[n]['recall'] for n in names]
    f1 = [short_term_agents[n]['f1'] for n in names]
    
    ax.bar(x - 1.5*width, accuracy, width, label='Accuracy', alpha=0.8)
    ax.bar(x - 0.5*width, precision, width, label='Precision', alpha=0.8)
    ax.bar(x + 0.5*width, recall, width, label='Recall', alpha=0.8)
    ax.bar(x + 1.5*width, f1, width, label='F1-Score', alpha=0.8)
    
    ax.set_xlabel('Agent', fontweight='bold', fontsize=12)
    ax.set_ylabel('Score', fontweight='bold', fontsize=12)
    ax.set_title('Short-Term Prediction Performance Comparison', fontweight='bold', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=45, ha='right')
    ax.set_ylim(0, 1.1)
    ax.legend(loc='upper left')
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'agent_comparison.png'), dpi=300, bbox_inches='tight')
    plt.close()


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Analyze MAFAS evaluation results with metrics and visualizations"
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Path to MAFAS results CSV file"
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory to save plots and metrics (default: same as input file directory)"
    )
    
    args = parser.parse_args()
    
    # Load CSV
    if not os.path.exists(args.input):
        print(f"✗ Error: File not found: {args.input}")
        return
    
    print(f"Loading results from: {args.input}")
    df = pd.read_csv(args.input)
    print(f"Loaded {len(df)} rows")
    
    # Determine output directory
    if args.output_dir:
        output_dir = args.output_dir
    else:
        input_dir = os.path.dirname(os.path.abspath(args.input))
        # If input is in evaluation folder, use results subfolder
        if "evaluation" in input_dir or "results" in input_dir:
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) if "evaluation" in os.path.dirname(os.path.abspath(__file__)) else os.path.dirname(os.path.abspath(__file__))
            results_dir = os.path.join(base_dir, "evaluation", "results")
            output_dir = results_dir
        else:
            output_dir = input_dir
        if not output_dir:
            output_dir = os.getcwd()
        os.makedirs(output_dir, exist_ok=True)
    
    # Analyze and generate visualizations
    metrics = analyze_results(df, output_dir)
    
    print(f"\n{'='*70}")
    print("ANALYSIS COMPLETE!")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()

