"""Run MAFAS system evaluation on CSV datasets and save BUY/HOLD/SELL signals.

This script reads evaluation CSV files (original or comprehensive news versions),
runs the MAFAS multi-agent system for each stock, and saves the recommendations
from all agents plus the judge to a results CSV file.

Usage (from repo root):
    python -m evaluation.run_mafas_evaluation --type shortterm
    python -m evaluation.run_mafas_evaluation --type longterm
    python -m evaluation.run_mafas_evaluation --input path/to/custom.csv

Requirements:
    - Environment variables:
        GEMINI_API_KEY : Google Gemini API key
"""

from __future__ import annotations

import os
import sys
import time
import argparse
from datetime import datetime
from typing import Dict, Any

import pandas as pd
import numpy as np

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents.agent_prompts import get_agent_analysis, get_judge_analysis
from dotenv import load_dotenv

load_dotenv()


def extract_metrics_from_row(row: pd.Series) -> Dict[str, Any]:
    """Extract metrics from CSV row organized by investor type."""
    
    metrics = {
        "Cautious Value": {
            "P/E Ratio": row.get("CV_P/E Ratio"),
            "Debt-to-Equity": row.get("CV_Debt-to-Equity"),
            "Profit Margin": row.get("CV_Profit Margin"),
            "Dividend Yield": row.get("CV_Dividend Yield"),
            "Current Price": row.get("Snapshot Price"),
            "52w High": row.get("CV_52w High"),
            "52w Low": row.get("CV_52w Low"),
            "5-Year Return (%)": row.get("CV_5-Year Return (%)"),
        },
        "Aggressive Growth": {
            "Revenue Growth": row.get("AG_Revenue Growth"),
            "Earnings Growth": row.get("AG_Earnings Growth"),
            "Forward P/E": row.get("AG_Forward P/E"),
            "Trailing P/E": row.get("AG_Trailing P/E"),
            "1-Year Return (%)": row.get("AG_1-Year Return (%)"),
            "News Article Count": row.get("AG_News Article Count"),
            "News Sentiment Score": row.get("AG_News Sentiment Score") or row.get("News_Sentiment_Score"),
        },
        "Technical Trader": {
            "RSI(14)": row.get("TT_RSI(14)"),
            "MACD": row.get("TT_MACD"),
            "MACD Signal": row.get("TT_MACD Signal"),
            "SMA50": row.get("TT_SMA50"),
            "SMA200": row.get("TT_SMA200"),
            "Price vs SMA50": row.get("TT_Price vs SMA50"),
            "Recent Volume": row.get("TT_Recent Volume"),
            "Current Price": row.get("Snapshot Price"),
        },
    }
    
    return metrics


def normalize_metrics(metrics: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize numeric inputs and handle NaNs before sending to agents.
    
    Args:
        metrics: Raw metrics dictionary
    
    Returns:
        Normalized metrics with NaNs replaced and numeric coercion
    """
    import numpy as np
    
    normalized = {}
    
    for agent_type, agent_metrics in metrics.items():
        normalized[agent_type] = {}
        for key, value in agent_metrics.items():
            if value is None or (isinstance(value, float) and (np.isnan(value) or np.isinf(value))):
                normalized[agent_type][key] = "MISSING"
            elif isinstance(value, (int, float)):
                # Keep numeric values as-is
                normalized[agent_type][key] = value
            else:
                normalized[agent_type][key] = value
    
    return normalized


def extract_signal(text: str, timeframe: str = "short") -> str:
    """Extract BUY/HOLD/SELL signal from LLM response.
    
    Args:
        text: The analysis text from the agent/judge
        timeframe: "short" or "long" to extract the appropriate signal
    """
    """Extract BUY/HOLD/SELL signal with robust regex + JSON fallback.
    
    Priority:
    1. Look for "RECOMMENDATION:" line (most reliable)
    2. Try to find JSON block
    3. Regex for standalone BUY/SELL/HOLD words
    4. Check for contradictory signals
    """
    import re
    import json
    
    if not text or len(text.strip()) == 0:
        return "UNCLEAR"
    
    text_upper = text.upper()
    
    # Method 1: Look for "RECOMMENDATION:" line (highest priority - most reliable format)
    rec_pattern = r'RECOMMENDATION:\s*(BUY|SELL|HOLD)'
    match = re.search(rec_pattern, text_upper, re.IGNORECASE)
    if match:
        return match.group(1).upper()
    
    # Method 2: Try to find JSON block
    json_pattern = r'\{[^{}]*"recommendation"[^{}]*"([^"]+)"[^{}]*\}'
    json_match = re.search(json_pattern, text_upper, re.IGNORECASE)
    if json_match:
        signal = json_match.group(1).upper()
        if signal in ["BUY", "SELL", "HOLD"]:
            return signal
    
    # Try to parse as JSON if it looks like JSON
    try:
        # Find JSON-like blocks
        json_start = text.find('{')
        json_end = text.rfind('}')
        if json_start != -1 and json_end > json_start:
            json_text = text[json_start:json_end+1]
            data = json.loads(json_text)
            if 'recommendation' in data:
                signal = str(data['recommendation']).upper()
                if signal in ["BUY", "SELL", "HOLD"]:
                    return signal
    except:
        pass
    
    # Method 3: Look for timeframe-specific lines with colon
    lines = text.split('\n')
    for line in lines:
        line_upper = line.upper().strip()
        if timeframe == "short":
            if ("SHORT-TERM" in line_upper or "SHORT TERM" in line_upper) and ":" in line_upper:
                parts = line_upper.split(":")
                if len(parts) > 1:
                    signal_part = parts[1].strip()
                    if "BUY" in signal_part:
                        return "BUY"
                    elif "SELL" in signal_part:
                        return "SELL"
                    elif "HOLD" in signal_part:
                        return "HOLD"
        else:  # long
            if ("LONG-TERM" in line_upper or "LONG TERM" in line_upper or "MEDIUM-TERM" in line_upper) and ":" in line_upper:
                parts = line_upper.split(":")
                if len(parts) > 1:
                    signal_part = parts[1].strip()
                    if "BUY" in signal_part:
                        return "BUY"
                    elif "SELL" in signal_part:
                        return "SELL"
                    elif "HOLD" in signal_part:
                        return "HOLD"
    
    # Method 4: Regex for standalone words (but check for contradictions)
    buy_matches = len(re.findall(r'\bBUY\b', text_upper))
    sell_matches = len(re.findall(r'\bSELL\b', text_upper))
    hold_matches = len(re.findall(r'\bHOLD\b', text_upper))
    
    # Check for contradictions - if multiple signals appear, be cautious
    signal_counts = {"BUY": buy_matches, "SELL": sell_matches, "HOLD": hold_matches}
    non_zero = [s for s, count in signal_counts.items() if count > 0]
    
    if len(non_zero) == 1:
        # Only one signal type found - use it
        return non_zero[0]
    elif len(non_zero) > 1:
        # Multiple signals - prefer the one with timeframe context
        if timeframe == "short":
            if "SHORT-TERM" in text_upper or "SHORT TERM" in text_upper:
                # Find signal near timeframe mention
                for line in lines:
                    if ("SHORT-TERM" in line.upper() or "SHORT TERM" in line.upper()):
                        if "BUY" in line.upper():
                            return "BUY"
                        elif "SELL" in line.upper():
                            return "SELL"
                        elif "HOLD" in line.upper():
                            return "HOLD"
        else:
            if "LONG-TERM" in text_upper or "LONG TERM" in text_upper or "MEDIUM-TERM" in text_upper:
                for line in lines:
                    if ("LONG-TERM" in line.upper() or "LONG TERM" in line.upper() or "MEDIUM-TERM" in line.upper()):
                        if "BUY" in line.upper():
                            return "BUY"
                        elif "SELL" in line.upper():
                            return "SELL"
                        elif "HOLD" in line.upper():
                            return "HOLD"
        
        # If still unclear, use the most frequent signal
        max_signal = max(signal_counts.items(), key=lambda x: x[1])
        if max_signal[1] > 0:
            return max_signal[0]
    
    return "UNCLEAR"


def extract_both_signals(text: str) -> Dict[str, str]:
    """Extract both short-term and long-term signals from judge's analysis."""
    text_upper = text.upper()
    
    short_signal = "UNCLEAR"
    long_signal = "UNCLEAR"
    
    # Try to find explicit short-term and long-term signals
    lines = text.split("\n")
    for i, line in enumerate(lines):
        line_upper = line.upper()
        if "SHORT" in line_upper or "1-6 MONTH" in line_upper or "6-12 MONTH" in line_upper:
            if "BUY" in line_upper:
                short_signal = "BUY"
            elif "SELL" in line_upper:
                short_signal = "SELL"
            elif "HOLD" in line_upper:
                short_signal = "HOLD"
        elif "LONG" in line_upper or "3 YEAR" in line_upper or "1-3 YEAR" in line_upper:
            if "BUY" in line_upper:
                long_signal = "BUY"
            elif "SELL" in line_upper:
                long_signal = "SELL"
            elif "HOLD" in line_upper:
                long_signal = "HOLD"
    
    # Fallback: if we didn't find explicit signals, use general extraction
    if short_signal == "UNCLEAR":
        short_signal = extract_signal(text, "short")
    if long_signal == "UNCLEAR":
        long_signal = extract_signal(text, "long")
    
    return {
        "short_term": short_signal,
        "long_term": long_signal
    }


def run_mafas_on_stock(row: pd.Series, evaluation_type: str = "shortterm") -> Dict[str, Any]:
    """Run MAFAS system on a single stock and extract all signals.
    
    Returns dict with ticker info and all agent/judge signals.
    """
    ticker = str(row.get("Ticker", "")).strip()
    snapshot_date = str(row.get("Snapshot Date", "")).strip()
    
    print(f"\n[{ticker}] Processing...")
    
    result = {
        "Ticker": ticker,
        "Snapshot_Date": snapshot_date,
        "Snapshot_Price": row.get("Snapshot Price"),
        "Current_Price": row.get("Current Price"),
        "Evaluation_Type": evaluation_type,  # Add evaluation type for analysis
    }
    
    # Extract metrics from CSV row
    metrics = extract_metrics_from_row(row)
    
    # Normalize metrics (handle NaNs, missing values)
    metrics = normalize_metrics(metrics)
    
    # Run each agent
    agents = ["Cautious Value", "Aggressive Growth", "Technical Trader"]
    agent_results = {}
    
    for agent_name in agents:
        print(f"  Running {agent_name} agent...", end=" ", flush=True)
        try:
            agent_result = get_agent_analysis(metrics, agent_name, evaluation_type)
            agent_text = agent_result.get("analysis", "")
            
            if not agent_text:
                print(f"✗ (No analysis text)")
                signal = "ERROR"
            else:
                # Extract only the relevant signal based on evaluation type
                if evaluation_type == "shortterm":
                    signal = extract_signal(agent_text, "short")
                    agent_results[agent_name] = {
                        "short_term": signal,
                        "analysis": agent_text[:500]  # First 500 chars
                    }
                    result[f"{agent_name}_Short_Term"] = signal
                    result[f"{agent_name}_Long_Term"] = "N/A"  # Not evaluated
                else:  # longterm
                    signal = extract_signal(agent_text, "long")
                    agent_results[agent_name] = {
                        "long_term": signal,
                        "analysis": agent_text[:500]  # First 500 chars
                    }
                    result[f"{agent_name}_Short_Term"] = "N/A"  # Not evaluated
                    result[f"{agent_name}_Long_Term"] = signal
                
                if signal == "UNCLEAR":
                    print(f"⚠ (Signal unclear, text: {agent_text[:100]}...)")
                else:
                    print(f"✓ ({signal})")
            time.sleep(3)  # Increased delay for rate limit protection
            
        except Exception as e:
            print(f"✗ ({str(e)[:50]})")
            if evaluation_type == "shortterm":
                result[f"{agent_name}_Short_Term"] = "ERROR"
                result[f"{agent_name}_Long_Term"] = "N/A"
                agent_results[agent_name] = {
                    "short_term": "ERROR",
                    "analysis": str(e)
                }
            else:
                result[f"{agent_name}_Short_Term"] = "N/A"
                result[f"{agent_name}_Long_Term"] = "ERROR"
                agent_results[agent_name] = {
                    "long_term": "ERROR",
                    "analysis": str(e)
                }
            time.sleep(3)
    
    # Run Judge
    print(f"  Running Judge...", end=" ", flush=True)
    time.sleep(5)  # Longer delay before judge
    
    try:
        # IMPROVEMENT #2: Pass all raw metrics to judge for better decision-making
        judge_result = get_judge_analysis(agent_results, evaluation_type, metrics)
        judge_text = judge_result.get("analysis", "")
        
        if not judge_text:
            print(f"✗ (No analysis text)")
            signal = "ERROR"
        else:
            # Extract only the relevant signal based on evaluation type
            if evaluation_type == "shortterm":
                signal = extract_signal(judge_text, "short")
                result["Judge_Short_Term"] = signal
                result["Judge_Long_Term"] = "N/A"  # Not evaluated
            else:  # longterm
                signal = extract_signal(judge_text, "long")
                result["Judge_Short_Term"] = "N/A"  # Not evaluated
                result["Judge_Long_Term"] = signal
            
            result["Judge_Analysis"] = judge_text[:1000]  # First 1000 chars
            
            if signal == "UNCLEAR":
                print(f"⚠ (Signal unclear)")
                # Debug: print first 200 chars to see what we got
                print(f"   Text preview: {judge_text[:200]}")
            else:
                print(f"✓ ({signal})")
        
    except Exception as e:
        print(f"✗ ({str(e)[:50]})")
        if evaluation_type == "shortterm":
            result["Judge_Short_Term"] = "ERROR"
            result["Judge_Long_Term"] = "N/A"
        else:
            result["Judge_Short_Term"] = "N/A"
            result["Judge_Long_Term"] = "ERROR"
        result["Judge_Analysis"] = str(e)
    
    # Add ground truth if available
    if evaluation_type == "longterm":
        result["Ground_Truth_Signal"] = row.get("Correct Signal (Long-Term 3Y)", "")
    else:
        result["Ground_Truth_Signal"] = row.get("Correct Signal (Short-Term 2-6M)", "")
    
    return result


def run_mafas_evaluation(
    csv_file: str,
    evaluation_type: str = "shortterm",
    output_file: str | None = None
) -> pd.DataFrame:
    """Run MAFAS evaluation on all stocks in CSV file.
    
    Args:
        csv_file: Path to input CSV file
        evaluation_type: "shortterm" or "longterm"
        output_file: Optional output CSV path
    
    Returns:
        DataFrame with results
    """
    print(f"\n{'='*70}")
    print(f"MAFAS EVALUATION: {evaluation_type.upper()}")
    print(f"{'='*70}\n")
    
    # Load CSV
    if not os.path.exists(csv_file):
        raise FileNotFoundError(f"CSV file not found: {csv_file}")
    
    print(f"Loading data from: {csv_file}")
    df = pd.read_csv(csv_file)
    print(f"Found {len(df)} stocks to evaluate\n")
    
    # Calculate API calls needed
    api_calls_per_stock = 4  # 3 agents + 1 judge
    total_api_calls = len(df) * api_calls_per_stock
    print(f"⚠️  API CALL ESTIMATE:")
    print(f"   - Stocks: {len(df)}")
    print(f"   - API calls per stock: {api_calls_per_stock}")
    print(f"   - Total API calls needed: {total_api_calls}")
    print(f"   - Free tier limit: 200 requests/day")
    if total_api_calls > 200:
        print(f"   ⚠️  WARNING: This will exceed free tier limit!")
        print(f"   Consider: 1) Using paid API key, 2) Reducing stocks, or 3) Splitting across days")
    print()
    
    # Process each stock
    results = []
    total = len(df)
    
    for idx, row in df.iterrows():
        try:
            result = run_mafas_on_stock(row, evaluation_type)
            results.append(result)
        except Exception as e:
            print(f"\n✗ Error processing row {idx + 1}: {str(e)}")
            continue
        
        # Delay between stocks to avoid rate limiting
        if idx < total - 1:
            print(f"  Waiting 10 seconds before next stock...")
            time.sleep(10)
    
    # Create results DataFrame
    results_df = pd.DataFrame(results)
    
    # Save results
    if output_file is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_dir = os.path.dirname(os.path.abspath(__file__))
        results_dir = os.path.join(base_dir, "results")
        os.makedirs(results_dir, exist_ok=True)
        output_file = os.path.join(
            results_dir,
            f"mafas_results_{evaluation_type}_{timestamp}.csv"
        )
    
    results_df.to_csv(output_file, index=False)
    print(f"\n{'='*70}")
    print(f"✓ Results saved to: {output_file}")
    print(f"{'='*70}\n")
    
    # Print summary
    print_summary(results_df)
    
    return results_df


def print_summary(results_df: pd.DataFrame) -> None:
    """Print summary statistics of the evaluation results."""
    
    print("\n" + "="*70)
    print("SUMMARY STATISTICS")
    print("="*70)
    
    total = len(results_df)
    print(f"\nTotal Stocks Evaluated: {total}")
    
    # Judge signals
    print("\n--- Judge Recommendations ---")
    print("\nShort-Term Signals:")
    if "Judge_Short_Term" in results_df.columns:
        print(results_df["Judge_Short_Term"].value_counts())
    
    print("\nLong-Term Signals:")
    if "Judge_Long_Term" in results_df.columns:
        print(results_df["Judge_Long_Term"].value_counts())
    
    # Individual agent signals
    print("\n--- Individual Agent Signals (Short-Term) ---")
    for agent in ["Cautious Value", "Aggressive Growth", "Technical Trader"]:
        col = f"{agent}_Short_Term"
        if col in results_df.columns:
            print(f"\n{agent}:")
            print(results_df[col].value_counts())
    
    print("\n--- Individual Agent Signals (Long-Term) ---")
    for agent in ["Cautious Value", "Aggressive Growth", "Technical Trader"]:
        col = f"{agent}_Long_Term"
        if col in results_df.columns:
            print(f"\n{agent}:")
            print(results_df[col].value_counts())
    
    # Error count
    error_cols = [col for col in results_df.columns if "ERROR" in str(results_df[col].values)]
    if error_cols:
        print("\n--- Errors ---")
        for col in error_cols:
            error_count = (results_df[col] == "ERROR").sum()
            if error_count > 0:
                print(f"{col}: {error_count} errors")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Run MAFAS evaluation on CSV datasets and save signals"
    )
    parser.add_argument(
        "--type",
        choices=["shortterm", "longterm"],
        default="shortterm",
        help="Evaluation type: shortterm or longterm (default: shortterm)"
    )
    parser.add_argument(
        "--input",
        help="Custom input CSV file path (overrides --type)"
    )
    parser.add_argument(
        "--output",
        help="Output CSV file path (default: auto-generated)"
    )
    
    args = parser.parse_args()
    
    # Determine input file
    if args.input:
        csv_file = args.input
        # Try to infer type from filename
        if "longterm" in csv_file.lower() or "long" in csv_file.lower():
            evaluation_type = "longterm"
        else:
            evaluation_type = "shortterm"
    else:
        base_dir = os.path.dirname(os.path.abspath(__file__))
        if args.type == "longterm":
            # Try comprehensive news version first, then original
            csv_file = os.path.join(base_dir, "news_enhanced", "longterm_comprehensive_news.csv")
            if not os.path.exists(csv_file):
                csv_file = os.path.join(base_dir, "longterm_evaluation_data.csv")
            evaluation_type = "longterm"
        else:
            # Try comprehensive news version first, then original
            csv_file = os.path.join(base_dir, "news_enhanced", "shortterm_comprehensive_news.csv")
            if not os.path.exists(csv_file):
                csv_file = os.path.join(base_dir, "shortterm_evaluation_data.csv")
            evaluation_type = "shortterm"
    
    # Run evaluation
    try:
        results_df = run_mafas_evaluation(
            csv_file=csv_file,
            evaluation_type=evaluation_type,
            output_file=args.output
        )
        print("\n✓ Evaluation complete!")
        
    except Exception as e:
        print(f"\n✗ Error: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()

