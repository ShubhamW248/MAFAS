"""Baseline Test: Single Gemini call vs MAFAS multi-agent system.

This script runs a baseline comparison where a single Gemini API call
analyzes all the data and provides a BUY/HOLD/SELL signal, without
the multi-agent orchestration.

Usage:
    python -m evaluation.run_baseline_test --type shortterm
    python -m evaluation.run_baseline_test --type longterm
"""

from __future__ import annotations

import os
import sys
import time
import argparse
import json
from datetime import datetime
from typing import Dict, Any

import pandas as pd

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
import google.generativeai as genai

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY not found in environment variables")

genai.configure(api_key=GEMINI_API_KEY)
MODEL = "gemini-2.0-flash"


def extract_all_metrics_from_row(row: pd.Series) -> Dict[str, Any]:
    """Extract all available metrics from CSV row for baseline analysis."""
    
    metrics = {
        "Basic Info": {
            "Ticker": row.get("Ticker"),
            "Snapshot Date": row.get("Snapshot Date"),
            "Snapshot Price": row.get("Snapshot Price"),
            "Current Price": row.get("Current Price"),
        },
        "Cautious Value Metrics": {
            "P/E Ratio": row.get("CV_P/E Ratio"),
            "Debt-to-Equity": row.get("CV_Debt-to-Equity"),
            "Profit Margin": row.get("CV_Profit Margin"),
            "Dividend Yield": row.get("CV_Dividend Yield"),
            "52w High": row.get("CV_52w High"),
            "52w Low": row.get("CV_52w Low"),
            "5-Year Return (%)": row.get("CV_5-Year Return (%)"),
        },
        "Aggressive Growth Metrics": {
            "Revenue Growth": row.get("AG_Revenue Growth"),
            "Earnings Growth": row.get("AG_Earnings Growth"),
            "Forward P/E": row.get("AG_Forward P/E"),
            "Trailing P/E": row.get("AG_Trailing P/E"),
            "1-Year Return (%)": row.get("AG_1-Year Return (%)"),
            "News Sentiment Score": row.get("News_Sentiment_Score") or row.get("AG_News Sentiment Score"),
        },
        "Technical Trader Metrics": {
            "RSI(14)": row.get("TT_RSI(14)"),
            "MACD": row.get("TT_MACD"),
            "MACD Signal": row.get("TT_MACD Signal"),
            "SMA50": row.get("TT_SMA50"),
            "SMA200": row.get("TT_SMA200"),
            "Price vs SMA50": row.get("TT_Price vs SMA50"),
            "Recent Volume": row.get("TT_Recent Volume"),
        },
    }
    
    # Add news data if available
    if row.get("Stock_News"):
        metrics["News Context"] = {
            "Stock News": row.get("Stock_News", "")[:500],  # First 500 chars
            "Industry News": row.get("Industry_News", "")[:500],
            "Market News": row.get("Market_News", "")[:500],
            "News Summary": row.get("News_Summary", "")[:300],
            "News Sentiment Score": row.get("News_Sentiment_Score"),
        }
    
    return metrics


def extract_signal(text: str) -> str:
    """Extract BUY/HOLD/SELL signal from LLM response."""
    text_upper = text.upper()
    
    # Look for explicit signal
    if "BUY" in text_upper:
        return "BUY"
    elif "SELL" in text_upper:
        return "SELL"
    elif "HOLD" in text_upper:
        return "HOLD"
    
    return "UNCLEAR"


def get_baseline_signal(metrics: Dict[str, Any], evaluation_type: str) -> Dict[str, Any]:
    """Get BUY/HOLD/SELL signal from single Gemini call with all data.
    
    Args:
        metrics: All metrics for the stock
        evaluation_type: "shortterm" or "longterm"
    """
    
    timeframe = "1-6 months" if evaluation_type == "shortterm" else "1-3 years"
    
    prompt = f"""You are an investment advisor analyzing a stock. You have access to comprehensive financial data including:
- Fundamental metrics (P/E, debt, profit margins, dividends)
- Growth metrics (revenue growth, earnings growth)
- Technical indicators (RSI, MACD, moving averages)
- News sentiment and market context

Analyze this stock and provide a SINGLE, DECISIVE recommendation for the {timeframe} timeframe.

STOCK DATA:
{json.dumps(metrics, indent=2)}

Based on ALL this information, provide:
1. Your recommendation: BUY, HOLD, or SELL (choose ONE)
2. Brief justification (2-3 sentences explaining your decision)

Be DECISIVE. Choose BUY, HOLD, or SELL based on the strongest signals in the data.

OUTPUT FORMAT:
Recommendation: [BUY/HOLD/SELL]
Justification: [2-3 sentences]"""

    max_retries = 5
    retry_delay = 5
    
    for attempt in range(max_retries):
        try:
            client = genai.GenerativeModel(MODEL)
            response = client.generate_content(
                prompt,
                generation_config=genai.types.GenerationConfig(
                    temperature=0.3,
                    max_output_tokens=1000
                ),
                stream=False
            )
            
            if response.text:
                signal = extract_signal(response.text)
                return {
                    "signal": signal,
                    "analysis": response.text[:1000],  # First 1000 chars
                    "status": "success"
                }
            else:
                return {
                    "signal": "ERROR",
                    "analysis": "Empty response",
                    "status": "error"
                }
                
        except Exception as e:
            error_msg = str(e)
            print(f"  API Error (attempt {attempt + 1}): {error_msg[:100]}")
            
            if "429" in error_msg or "Resource exhausted" in error_msg or "quota" in error_msg.lower():
                if attempt < max_retries - 1:
                    wait_time = retry_delay * (attempt + 2)
                    print(f"  Rate limited. Waiting {wait_time} seconds...")
                    time.sleep(wait_time)
                    continue
            
            if attempt < max_retries - 1:
                time.sleep(retry_delay)
                continue
            
            return {
                "signal": "ERROR",
                "analysis": str(e),
                "status": "error"
            }
    
    return {
        "signal": "ERROR",
        "analysis": "Max retries exceeded",
        "status": "error"
    }


def run_baseline_test(
    csv_file: str,
    evaluation_type: str = "shortterm",
    output_file: str | None = None
) -> pd.DataFrame:
    """Run baseline test on all stocks in CSV file.
    
    Args:
        csv_file: Path to input CSV file
        evaluation_type: "shortterm" or "longterm"
        output_file: Optional output CSV path
    
    Returns:
        DataFrame with results
    """
    print(f"\n{'='*70}")
    print(f"BASELINE TEST: {evaluation_type.upper()}")
    print(f"{'='*70}\n")
    
    # Load CSV
    if not os.path.exists(csv_file):
        raise FileNotFoundError(f"CSV file not found: {csv_file}")
    
    print(f"Loading data from: {csv_file}")
    df = pd.read_csv(csv_file)
    print(f"Found {len(df)} stocks to evaluate\n")
    
    # Calculate API calls needed
    total_api_calls = len(df)
    print(f"⚠️  API CALL ESTIMATE:")
    print(f"   - Stocks: {len(df)}")
    print(f"   - API calls needed: {total_api_calls} (1 per stock)")
    print(f"   - Free tier limit: 200 requests/day")
    if total_api_calls > 200:
        print(f"   ⚠️  WARNING: This will exceed free tier limit!")
    print()
    
    # Process each stock
    results = []
    total = len(df)
    
    for idx, row in df.iterrows():
        ticker = str(row.get("Ticker", "")).strip()
        snapshot_date = str(row.get("Snapshot Date", "")).strip()
        
        print(f"[{idx + 1}/{total}] Processing {ticker}...", end=" ", flush=True)
        
        # Extract all metrics
        metrics = extract_all_metrics_from_row(row)
        
        # Get baseline signal
        baseline_result = get_baseline_signal(metrics, evaluation_type)
        
        result = {
            "Ticker": ticker,
            "Snapshot_Date": snapshot_date,
            "Snapshot_Price": row.get("Snapshot Price"),
            "Current_Price": row.get("Current Price"),
            "Evaluation_Type": evaluation_type,
            "Baseline_Signal": baseline_result["signal"],
            "Baseline_Analysis": baseline_result["analysis"],
        }
        
        # Add ground truth
        if evaluation_type == "longterm":
            result["Ground_Truth_Signal"] = row.get("Correct Signal (Long-Term 3Y)", "")
        else:
            result["Ground_Truth_Signal"] = row.get("Correct Signal (Short-Term 2-6M)", "")
        
        results.append(result)
        
        if baseline_result["signal"] == "ERROR":
            print("✗")
        else:
            print(f"✓ ({baseline_result['signal']})")
        
        # Delay between stocks
        if idx < total - 1:
            time.sleep(5)  # 5 second delay to avoid rate limits
    
    # Create results DataFrame
    results_df = pd.DataFrame(results)
    
    # Save results
    if output_file is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_dir = os.path.dirname(os.path.abspath(__file__))
        baseline_dir = os.path.join(base_dir, "baseline_test")
        os.makedirs(baseline_dir, exist_ok=True)
        output_file = os.path.join(
            baseline_dir,
            f"baseline_results_{evaluation_type}_{timestamp}.csv"
        )
    
    results_df.to_csv(output_file, index=False)
    print(f"\n{'='*70}")
    print(f"✓ Results saved to: {output_file}")
    print(f"{'='*70}\n")
    
    # Print summary
    print_summary(results_df, evaluation_type)
    
    return results_df


def print_summary(results_df: pd.DataFrame, evaluation_type: str) -> None:
    """Print summary statistics of baseline test."""
    
    print("\n" + "="*70)
    print("BASELINE TEST SUMMARY")
    print("="*70)
    
    total = len(results_df)
    print(f"\nTotal Stocks Evaluated: {total}")
    
    # Signal distribution
    print("\n--- Baseline Signal Distribution ---")
    if "Baseline_Signal" in results_df.columns:
        print(results_df["Baseline_Signal"].value_counts())
    
    # Accuracy if ground truth available
    if "Ground_Truth_Signal" in results_df.columns:
        valid_df = results_df[
            results_df["Ground_Truth_Signal"].notna() & 
            (results_df["Ground_Truth_Signal"] != "")
        ]
        
        if len(valid_df) > 0:
            correct = (valid_df["Baseline_Signal"] == valid_df["Ground_Truth_Signal"]).sum()
            accuracy = (correct / len(valid_df)) * 100
            
            print(f"\n--- Accuracy ---")
            print(f"Ground Truth Available: {len(valid_df)}/{total}")
            print(f"Correct Predictions: {correct}/{len(valid_df)}")
            print(f"Accuracy: {accuracy:.1f}%")
            
            # Per-class accuracy
            print(f"\n--- Per-Class Accuracy ---")
            for signal in ["BUY", "HOLD", "SELL"]:
                signal_df = valid_df[valid_df["Ground_Truth_Signal"] == signal]
                if len(signal_df) > 0:
                    correct_count = (signal_df["Baseline_Signal"] == signal).sum()
                    print(f"{signal}: {correct_count}/{len(signal_df)} ({correct_count/len(signal_df)*100:.1f}%)")
    
    # Error count
    if "Baseline_Signal" in results_df.columns:
        error_count = (results_df["Baseline_Signal"] == "ERROR").sum()
        if error_count > 0:
            print(f"\n--- Errors ---")
            print(f"Failed API calls: {error_count}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Run baseline test: Single Gemini call vs MAFAS"
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
        evaluation_type = args.type
    else:
        base_dir = os.path.dirname(os.path.abspath(__file__))
        if args.type == "longterm":
            csv_file = os.path.join(base_dir, "news_enhanced", "longterm_comprehensive_news.csv")
            evaluation_type = "longterm"
        else:
            csv_file = os.path.join(base_dir, "news_enhanced", "shortterm_comprehensive_news.csv")
            evaluation_type = "shortterm"
    
    # Run baseline test
    try:
        results_df = run_baseline_test(
            csv_file=csv_file,
            evaluation_type=evaluation_type,
            output_file=args.output
        )
        print("\n✓ Baseline test complete!")
        print("\nNext steps:")
        print("1. Compare baseline results with MAFAS results")
        print("2. Run analysis to see which performs better")
        
    except Exception as e:
        print(f"\n✗ Error: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()

