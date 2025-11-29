"""
Evaluation Runner: Feed historical data through MAFAS and raw Gemini
Compares multi-agent system vs single Gemini API vs ground truth

Usage:
    python run_evaluation.py --type longterm
    python run_evaluation.py --type shortterm
"""

import os
import sys
import pandas as pd
import json
import argparse
from datetime import datetime
import time

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents.agent_prompts import get_agent_analysis, get_judge_analysis, get_all_analyses
from agents.agent_fetcher import (
    cautious_value_metrics,
    aggressive_growth_metrics,
    technical_trader_metrics,
)

import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)
    MODEL = "gemini-2.0-flash"


def load_evaluation_data(csv_file):
    """Load historical snapshot data from CSV"""
    if not os.path.exists(csv_file):
        print(f"Error: {csv_file} not found!")
        return None
    
    df = pd.read_csv(csv_file)
    print(f"Loaded {len(df)} stocks from {csv_file}")
    return df


def extract_metrics_from_row(row):
    """Extract metrics from CSV row organized by investor type"""
    
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
            "News Sentiment Score": row.get("AG_News Sentiment Score"),
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


def extract_signal(text):
    """Extract BUY/HOLD/SELL signal from LLM response"""
    text = text.upper()
    if "BUY" in text:
        return "BUY"
    elif "SELL" in text:
        return "SELL"
    elif "HOLD" in text:
        return "HOLD"
    else:
        return "UNCLEAR"


def get_raw_gemini_signal(metrics, evaluation_type="longterm"):
    """Get signal from raw Gemini API (without multi-agent orchestration)"""
    
    timeframe = "3 years" if evaluation_type == "longterm" else "2-6 months"
    
    prompt = f"""You are an investment advisor. Analyze these stock metrics and provide a single, decisive recommendation.

Metrics:
{json.dumps(metrics, indent=2)}

Based on these fundamentals and technicals, provide your recommendation for the {timeframe} timeframe:
- BUY if the stock looks attractive
- SELL if the stock looks unattractive
- HOLD if the stock looks neutral

Be decisive. Choose ONE: BUY, HOLD, or SELL."""

    try:
        client = genai.GenerativeModel(MODEL)
        response = client.generate_content(
            prompt,
            generation_config=genai.types.GenerationConfig(
                temperature=0.6,
                max_output_tokens=500
            ),
            stream=False
        )
        if response.text:
            signal = extract_signal(response.text)
            return {
                "signal": signal,
                "full_response": response.text[:200]  # Store first 200 chars
            }
    except Exception as e:
        print(f"  Raw Gemini API Error: {str(e)}")
    
    return {"signal": "ERROR", "full_response": str(e)}


def run_evaluation_on_stock(row, evaluation_type="longterm"):
    """Run MAFAS system + Raw Gemini on a single stock"""
    
    ticker = row.get("Ticker")
    print(f"\nEvaluating {ticker}...")
    
    metrics = extract_metrics_from_row(row)
    
    result = {
        "ticker": ticker,
        "snapshot_date": row.get("Snapshot Date"),
        "snapshot_price": row.get("Snapshot Price"),
        "current_price": row.get("Current Price"),
    }
    
    # Get MAFAS multi-agent analysis
    print(f"  Getting MAFAS analysis (multi-agent)...", end=" ", flush=True)
    try:
        analyses = {
            "Cautious Value": get_agent_analysis(metrics, "Cautious Value"),
            "Aggressive Growth": get_agent_analysis(metrics, "Aggressive Growth"),
            "Technical Trader": get_agent_analysis(metrics, "Technical Trader"),
        }
        time.sleep(1)  # Rate limit protection
        
        judge_result = get_judge_analysis(analyses)
        
        # Extract signals from judge verdict
        judge_text = judge_result.get("analysis", "")
        
        # Parse judge's short-term and long-term signals
        judge_signal = extract_signal(judge_text)
        
        result["mafas_signal"] = judge_signal
        result["mafas_judge_response"] = judge_text[:300]  # First 300 chars
        
        print("✓")
    except Exception as e:
        print(f"✗ ({str(e)})")
        result["mafas_signal"] = "ERROR"
        result["mafas_judge_response"] = str(e)
    
    time.sleep(2)  # Delay before raw Gemini
    
    # Get Raw Gemini signal
    print(f"  Getting Raw Gemini signal...", end=" ", flush=True)
    gemini_result = get_raw_gemini_signal(metrics, evaluation_type)
    result["raw_gemini_signal"] = gemini_result["signal"]
    result["raw_gemini_response"] = gemini_result["full_response"]
    print("✓")
    
    # Add ground truth placeholder
    result["correct_signal"] = row.get(
        "Correct Signal (Long-Term 3Y)" if evaluation_type == "longterm" 
        else "Correct Signal (Short-Term 2-6M)", 
        ""
    )
    
    # Calculate if correct
    if result["correct_signal"]:
        result["mafas_correct"] = 1 if result["mafas_signal"] == result["correct_signal"] else 0
        result["gemini_correct"] = 1 if result["raw_gemini_signal"] == result["correct_signal"] else 0
    else:
        result["mafas_correct"] = None
        result["gemini_correct"] = None
    
    return result


def run_full_evaluation(csv_file, evaluation_type="longterm", output_file=None):
    """Run full evaluation on all stocks in CSV"""
    
    print(f"\n{'='*60}")
    print(f"EVALUATION: {evaluation_type.upper()}")
    print(f"{'='*60}\n")
    
    # Load data
    df = load_evaluation_data(csv_file)
    if df is None:
        return None
    
    # Run evaluation on each stock
    results = []
    for idx, row in df.iterrows():
        try:
            result = run_evaluation_on_stock(row, evaluation_type)
            results.append(result)
        except Exception as e:
            print(f"Error processing {row.get('Ticker')}: {str(e)}")
            continue
        
        # Add delay between stocks to avoid rate limiting
        if idx < len(df) - 1:
            time.sleep(3)
    
    # Create results dataframe
    results_df = pd.DataFrame(results)
    
    # Save results
    if output_file is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = f"{evaluation_type}_evaluation_results_{timestamp}.csv"
    
    results_df.to_csv(output_file, index=False)
    print(f"\n✓ Results saved to {output_file}")
    
    # Print summary statistics
    print_summary_stats(results_df, evaluation_type)
    
    return results_df


def print_summary_stats(results_df, evaluation_type):
    """Print summary statistics of evaluation"""
    
    print(f"\n{'='*60}")
    print(f"SUMMARY STATISTICS")
    print(f"{'='*60}")
    
    total_stocks = len(results_df)
    print(f"\nTotal Stocks Evaluated: {total_stocks}")
    
    # Signal distribution
    print(f"\nMAFAS Signal Distribution:")
    print(results_df['mafas_signal'].value_counts())
    
    print(f"\nRaw Gemini Signal Distribution:")
    print(results_df['raw_gemini_signal'].value_counts())
    
    # Correctness (if ground truth available)
    mafas_correct = results_df['mafas_correct'].sum()
    gemini_correct = results_df['gemini_correct'].sum()
    ground_truth_filled = results_df['correct_signal'].notna().sum()
    
    if ground_truth_filled > 0:
        print(f"\n\nGround Truth Filled: {ground_truth_filled}/{total_stocks}")
        print(f"MAFAS Correct: {mafas_correct}/{ground_truth_filled} ({(mafas_correct/ground_truth_filled)*100:.1f}%)")
        print(f"Raw Gemini Correct: {gemini_correct}/{ground_truth_filled} ({(gemini_correct/ground_truth_filled)*100:.1f}%)")
        print(f"Advantage (MAFAS - Gemini): {((mafas_correct - gemini_correct)/ground_truth_filled)*100:.1f}%")
    else:
        print("\nNote: Ground truth signals not yet filled in CSV")
        print("Fill 'Correct Signal (Long-Term 3Y)' or 'Correct Signal (Short-Term 2-6M)' column to see accuracy")


def main():
    parser = argparse.ArgumentParser(description="Run MAFAS evaluation on historical data")
    parser.add_argument("--type", choices=["longterm", "shortterm"], default="longterm",
                        help="Evaluation type: longterm (3 years) or shortterm (2-6 months)")
    parser.add_argument("--output", help="Output CSV filename")
    
    args = parser.parse_args()
    
    # Determine input CSV file
    if args.type == "longterm":
        csv_file = "longterm_evaluation_data.csv"
    else:
        csv_file = "shortterm_evaluation_data.csv"
    
    # Run evaluation
    results_df = run_full_evaluation(csv_file, args.type, args.output)
    
    if results_df is not None:
        print("\nEvaluation Complete!")
        print("Next steps:")
        print("1. Fill in 'Correct Signal' column with actual outcomes")
        print("2. Run again to see accuracy metrics")


if __name__ == "__main__":
    main()
