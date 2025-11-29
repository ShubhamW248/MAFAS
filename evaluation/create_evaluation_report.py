"""
Create evaluation report with visualizations and metrics
Run this after evaluation is complete and ground truth is filled

Usage:
    python create_evaluation_report.py --file longterm_evaluation_results_YYYYMMDD_HHMMSS.csv
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
import os
from datetime import datetime

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 10)


def load_results(csv_file):
    """Load evaluation results"""
    if not os.path.exists(csv_file):
        print(f"Error: {csv_file} not found!")
        return None
    
    df = pd.read_csv(csv_file)
    return df


def calculate_metrics(df):
    """Calculate evaluation metrics"""
    
    # Filter out rows with missing correct signals
    valid_df = df[df['correct_signal'].notna() & (df['correct_signal'] != '')].copy()
    
    if len(valid_df) == 0:
        print("No ground truth signals found. Fill 'Correct Signal' column first.")
        return None
    
    metrics = {
        'total_evaluated': len(valid_df),
        'mafas_correct': (valid_df['mafas_correct'] == 1).sum(),
        'gemini_correct': (valid_df['gemini_correct'] == 1).sum(),
        'mafas_accuracy': (valid_df['mafas_correct'] == 1).sum() / len(valid_df),
        'gemini_accuracy': (valid_df['gemini_correct'] == 1).sum() / len(valid_df),
    }
    
    # Per-signal accuracy
    for signal in ['BUY', 'HOLD', 'SELL']:
        signal_rows = valid_df[valid_df['correct_signal'] == signal]
        if len(signal_rows) > 0:
            mafas_correct_sig = (signal_rows['mafas_correct'] == 1).sum()
            gemini_correct_sig = (signal_rows['gemini_correct'] == 1).sum()
            
            metrics[f'mafas_{signal.lower()}_accuracy'] = mafas_correct_sig / len(signal_rows)
            metrics[f'gemini_{signal.lower()}_accuracy'] = gemini_correct_sig / len(signal_rows)
            metrics[f'{signal.lower()}_count'] = len(signal_rows)
    
    metrics['advantage'] = metrics['mafas_accuracy'] - metrics['gemini_accuracy']
    
    return metrics


def print_metrics(metrics):
    """Print evaluation metrics"""
    
    if metrics is None:
        return
    
    print("\n" + "="*60)
    print("EVALUATION METRICS")
    print("="*60)
    
    print(f"\nTotal Stocks with Ground Truth: {metrics['total_evaluated']}")
    print(f"\nAccuracy:")
    print(f"  MAFAS:       {metrics['mafas_correct']}/{metrics['total_evaluated']} ({metrics['mafas_accuracy']*100:.1f}%)")
    print(f"  Raw Gemini:  {metrics['gemini_correct']}/{metrics['total_evaluated']} ({metrics['gemini_accuracy']*100:.1f}%)")
    print(f"  Advantage:   {metrics['advantage']*100:+.1f}%")
    
    # Per-signal breakdown
    print(f"\nPer-Signal Accuracy:")
    for signal in ['BUY', 'HOLD', 'SELL']:
        signal_key = signal.lower()
        count = metrics.get(f'{signal_key}_count', 0)
        if count > 0:
            mafas_acc = metrics.get(f'mafas_{signal_key}_accuracy', 0)
            gemini_acc = metrics.get(f'gemini_{signal_key}_accuracy', 0)
            print(f"\n  {signal} ({count} stocks):")
            print(f"    MAFAS:  {mafas_acc*100:.1f}%")
            print(f"    Gemini: {gemini_acc*100:.1f}%")


def create_accuracy_comparison(df, output_dir):
    """Create accuracy comparison chart"""
    
    valid_df = df[df['correct_signal'].notna() & (df['correct_signal'] != '')].copy()
    
    if len(valid_df) == 0:
        print("Skipping accuracy chart: No ground truth signals")
        return
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Overall accuracy
    accuracies = [
        (valid_df['mafas_correct'] == 1).sum() / len(valid_df),
        (valid_df['gemini_correct'] == 1).sum() / len(valid_df),
    ]
    
    ax = axes[0]
    bars = ax.bar(['MAFAS', 'Raw Gemini'], accuracies, color=['#2ecc71', '#3498db'], alpha=0.8)
    ax.set_ylabel('Accuracy', fontsize=12)
    ax.set_title('Overall Accuracy Comparison', fontsize=14, fontweight='bold')
    ax.set_ylim([0, 1])
    ax.axhline(y=0.5, color='red', linestyle='--', alpha=0.5, label='Random Chance')
    
    # Add value labels on bars
    for bar, acc in zip(bars, accuracies):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{acc*100:.1f}%', ha='center', va='bottom', fontweight='bold')
    
    # Per-signal accuracy
    ax = axes[1]
    signals = []
    mafas_per_signal = []
    gemini_per_signal = []
    
    for signal in ['BUY', 'HOLD', 'SELL']:
        signal_rows = valid_df[valid_df['correct_signal'] == signal]
        if len(signal_rows) > 0:
            signals.append(signal)
            mafas_per_signal.append((signal_rows['mafas_correct'] == 1).sum() / len(signal_rows))
            gemini_per_signal.append((signal_rows['gemini_correct'] == 1).sum() / len(signal_rows))
    
    x = range(len(signals))
    width = 0.35
    
    bars1 = ax.bar([i - width/2 for i in x], mafas_per_signal, width, label='MAFAS', color='#2ecc71', alpha=0.8)
    bars2 = ax.bar([i + width/2 for i in x], gemini_per_signal, width, label='Raw Gemini', color='#3498db', alpha=0.8)
    
    ax.set_ylabel('Accuracy', fontsize=12)
    ax.set_title('Accuracy by Signal Type', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(signals)
    ax.set_ylim([0, 1])
    ax.legend()
    
    plt.tight_layout()
    filepath = os.path.join(output_dir, 'accuracy_comparison.png')
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {filepath}")
    plt.close()


def create_signal_distribution(df, output_dir):
    """Create signal distribution charts"""
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Correct signals distribution
    ax = axes[0, 0]
    valid_df = df[df['correct_signal'].notna() & (df['correct_signal'] != '')].copy()
    correct_counts = valid_df['correct_signal'].value_counts()
    ax.bar(correct_counts.index, correct_counts.values, color=['#e74c3c', '#2ecc71', '#f39c12'], alpha=0.7)
    ax.set_title('Ground Truth Signal Distribution', fontsize=12, fontweight='bold')
    ax.set_ylabel('Count', fontsize=11)
    ax.set_xlabel('Signal', fontsize=11)
    
    # MAFAS signals
    ax = axes[0, 1]
    mafas_counts = df['mafas_signal'].value_counts()
    ax.bar(mafas_counts.index, mafas_counts.values, color=['#e74c3c', '#2ecc71', '#f39c12'], alpha=0.7)
    ax.set_title('MAFAS Signal Distribution', fontsize=12, fontweight='bold')
    ax.set_ylabel('Count', fontsize=11)
    ax.set_xlabel('Signal', fontsize=11)
    
    # Gemini signals
    ax = axes[1, 0]
    gemini_counts = df['raw_gemini_signal'].value_counts()
    ax.bar(gemini_counts.index, gemini_counts.values, color=['#e74c3c', '#2ecc71', '#f39c12'], alpha=0.7)
    ax.set_title('Raw Gemini Signal Distribution', fontsize=12, fontweight='bold')
    ax.set_ylabel('Count', fontsize=11)
    ax.set_xlabel('Signal', fontsize=11)
    
    # Comparison matrix
    ax = axes[1, 1]
    ax.axis('off')
    
    # Create comparison text
    summary_text = f"""
    EVALUATION SUMMARY
    
    Total Stocks: {len(df)}
    With Ground Truth: {len(valid_df)}
    
    MAFAS Signals:
    • BUY:  {mafas_counts.get('BUY', 0)}
    • HOLD: {mafas_counts.get('HOLD', 0)}
    • SELL: {mafas_counts.get('SELL', 0)}
    
    Gemini Signals:
    • BUY:  {gemini_counts.get('BUY', 0)}
    • HOLD: {gemini_counts.get('HOLD', 0)}
    • SELL: {gemini_counts.get('SELL', 0)}
    """
    
    ax.text(0.1, 0.5, summary_text, fontsize=11, family='monospace',
            verticalalignment='center', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    filepath = os.path.join(output_dir, 'signal_distribution.png')
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {filepath}")
    plt.close()


def create_confusion_matrix(df, output_dir):
    """Create confusion matrix for MAFAS vs Gemini"""
    
    valid_df = df[df['correct_signal'].notna() & (df['correct_signal'] != '')].copy()
    
    if len(valid_df) == 0:
        print("Skipping confusion matrix: No ground truth signals")
        return
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    signals = ['BUY', 'HOLD', 'SELL']
    
    # MAFAS confusion matrix
    ax = axes[0]
    mafas_matrix = pd.crosstab(valid_df['correct_signal'], valid_df['mafas_signal'], 
                                rownames=['Ground Truth'], colnames=['MAFAS'])
    # Reindex to ensure all signals are present
    mafas_matrix = mafas_matrix.reindex(signals, fill_value=0)
    mafas_matrix = mafas_matrix.reindex(columns=signals, fill_value=0)
    
    sns.heatmap(mafas_matrix, annot=True, fmt='d', cmap='YlGn', ax=ax, cbar=True)
    ax.set_title('MAFAS Confusion Matrix', fontsize=12, fontweight='bold')
    ax.set_ylabel('Ground Truth', fontsize=11)
    ax.set_xlabel('MAFAS Prediction', fontsize=11)
    
    # Gemini confusion matrix
    ax = axes[1]
    gemini_matrix = pd.crosstab(valid_df['correct_signal'], valid_df['raw_gemini_signal'],
                                 rownames=['Ground Truth'], colnames=['Gemini'])
    gemini_matrix = gemini_matrix.reindex(signals, fill_value=0)
    gemini_matrix = gemini_matrix.reindex(columns=signals, fill_value=0)
    
    sns.heatmap(gemini_matrix, annot=True, fmt='d', cmap='YlGn', ax=ax, cbar=True)
    ax.set_title('Raw Gemini Confusion Matrix', fontsize=12, fontweight='bold')
    ax.set_ylabel('Ground Truth', fontsize=11)
    ax.set_xlabel('Gemini Prediction', fontsize=11)
    
    plt.tight_layout()
    filepath = os.path.join(output_dir, 'confusion_matrices.png')
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {filepath}")
    plt.close()


def create_return_analysis(df, output_dir):
    """Analyze returns by signal"""
    
    # Try to get return columns
    return_col = None
    for col in df.columns:
        if 'return' in col.lower() and '%' in col:
            return_col = col
            break
    
    if return_col is None:
        print("Skipping return analysis: No return column found")
        return
    
    valid_df = df[df['correct_signal'].notna() & (df['correct_signal'] != '')].copy()
    valid_df[return_col] = pd.to_numeric(valid_df[return_col], errors='coerce')
    valid_df = valid_df.dropna(subset=[return_col])
    
    if len(valid_df) == 0:
        print("Skipping return analysis: No valid return data")
        return
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Returns by ground truth signal
    ax = axes[0]
    buy_returns = valid_df[valid_df['correct_signal'] == 'BUY'][return_col]
    hold_returns = valid_df[valid_df['correct_signal'] == 'HOLD'][return_col]
    sell_returns = valid_df[valid_df['correct_signal'] == 'SELL'][return_col]
    
    box_data = [buy_returns.dropna(), hold_returns.dropna(), sell_returns.dropna()]
    ax.boxplot(box_data, labels=['BUY', 'HOLD', 'SELL'])
    ax.set_ylabel(f'Return {return_col}', fontsize=11)
    ax.set_title('Return Distribution by Ground Truth Signal', fontsize=12, fontweight='bold')
    ax.axhline(y=0, color='r', linestyle='--', alpha=0.5)
    ax.grid(True, alpha=0.3)
    
    # Average returns by signal
    ax = axes[1]
    avg_returns = valid_df.groupby('correct_signal')[return_col].mean()
    avg_returns = avg_returns.reindex(['BUY', 'HOLD', 'SELL'], fill_value=0)
    
    colors = ['#2ecc71', '#f39c12', '#e74c3c']
    bars = ax.bar(avg_returns.index, avg_returns.values, color=colors, alpha=0.7)
    ax.set_ylabel(f'Average Return {return_col}', fontsize=11)
    ax.set_title('Average Return by Signal Type', fontsize=12, fontweight='bold')
    ax.axhline(y=0, color='k', linestyle='-', alpha=0.3)
    
    # Add value labels
    for bar, val in zip(bars, avg_returns.values):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.1f}%', ha='center', va='bottom' if val > 0 else 'top', fontweight='bold')
    
    plt.tight_layout()
    filepath = os.path.join(output_dir, 'return_analysis.png')
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {filepath}")
    plt.close()


def create_report(csv_file, output_dir="evaluation_reports"):
    """Create complete evaluation report"""
    
    print(f"\nLoading results from: {csv_file}")
    df = load_results(csv_file)
    
    if df is None:
        return
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Calculate metrics
    metrics = calculate_metrics(df)
    
    # Print metrics
    print_metrics(metrics)
    
    # Create visualizations
    print(f"\nGenerating visualizations in {output_dir}/...")
    
    if metrics is not None:
        create_accuracy_comparison(df, output_dir)
        create_confusion_matrix(df, output_dir)
    
    create_signal_distribution(df, output_dir)
    create_return_analysis(df, output_dir)
    
    # Save metrics to JSON
    import json
    metrics_file = os.path.join(output_dir, 'metrics.json')
    with open(metrics_file, 'w') as f:
        json.dump({k: float(v) if isinstance(v, (int, float)) else v for k, v in metrics.items() if metrics}, f, indent=2)
    print(f"✓ Saved: {metrics_file}")
    
    print(f"\n✓ Report complete! Check {output_dir}/ for visualizations")


def main():
    parser = argparse.ArgumentParser(description="Create evaluation report with visualizations")
    parser.add_argument("--file", required=True, help="Path to evaluation results CSV")
    parser.add_argument("--output", default="evaluation_reports", help="Output directory for visualizations")
    
    args = parser.parse_args()
    
    create_report(args.file, args.output)


if __name__ == "__main__":
    main()
