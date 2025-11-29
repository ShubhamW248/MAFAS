"""
MAFAS Evaluation - Quick Reference & Setup

This file contains helper functions and constants for the evaluation framework.
"""

# Signal Thresholds for Ground Truth
LONGTERM_THRESHOLDS = {
    "BUY": 0.30,      # Return > 30% = BUY
    "HOLD_LOW": -0.10,  # Return between -10% and +30% = HOLD
    "SELL": -0.10,     # Return < -10% = SELL
}

SHORTTERM_THRESHOLDS = {
    "BUY": 0.08,       # Return > 8% = BUY
    "HOLD_LOW": -0.03,   # Return between -3% and +8% = HOLD
    "SELL": -0.03,      # Return < -3% = SELL
}

# Stocks to Evaluate
EVALUATION_STOCKS = [
    "AAPL",   # Tech - Apple
    "MSFT",   # Tech - Microsoft
    "NVDA",   # Tech - NVIDIA
    "TSLA",   # Tech/Growth - Tesla
    "JPM",    # Finance - JPMorgan
    "GS",     # Finance - Goldman Sachs
    "JNJ",    # Healthcare - Johnson & Johnson
    "UNH",    # Healthcare - UnitedHealth
    "AMZN",   # Consumer/Tech - Amazon
    "WMT",    # Consumer - Walmart
    "XOM",    # Energy - ExxonMobil
    "BA",     # Industrials - Boeing
    "NEE",    # Utilities - NextEra Energy
    "PFE",    # Pharma - Pfizer
    "COIN",   # Growth/Crypto - Coinbase
]

# Snapshot Dates
SNAPSHOT_DATES = {
    "longterm": "2022-11-15",   # 3-year evaluation: Nov 2022 → Nov 2025
    "shortterm": "2025-08-15",  # 6-month evaluation: Aug 2025 → Oct/Dec 2025
}

def get_correct_signal(return_pct, eval_type="longterm"):
    """
    Determine correct signal based on actual return
    
    Args:
        return_pct: Return percentage (e.g., 25.5 for 25.5%)
        eval_type: "longterm" or "shortterm"
    
    Returns:
        "BUY", "HOLD", or "SELL"
    """
    return_decimal = return_pct / 100.0
    
    if eval_type == "longterm":
        thresholds = LONGTERM_THRESHOLDS
    else:
        thresholds = SHORTTERM_THRESHOLDS
    
    if return_decimal > thresholds["BUY"]:
        return "BUY"
    elif return_decimal < thresholds["SELL"]:
        return "SELL"
    else:
        return "HOLD"


def print_quickstart():
    """Print quick start guide"""
    guide = """
    MAFAS EVALUATION - QUICK START
    ==============================
    
    1. COLLECT DATA
       python collect_longterm_data.py
       python collect_shortterm_data.py
       
    2. FILL GROUND TRUTH (manually in CSV)
       - Open longterm_evaluation_data.csv
       - For each stock, check actual 3-year return
       - Fill "Correct Signal (Long-Term 3Y)" column using thresholds:
         • BUY if return > 30%
         • HOLD if -10% < return < 30%
         • SELL if return < -10%
       
       - Repeat for shortterm_evaluation_data.csv (2-6 month)
         • BUY if return > 8%
         • HOLD if -3% < return < 8%
         • SELL if return < -3%
       
       - Optional: Fill news sentiment columns
    
    3. RUN EVALUATION
       python run_evaluation.py --type longterm
       python run_evaluation.py --type shortterm
    
    4. ANALYZE RESULTS
       python create_evaluation_report.py --file <results_csv>
    
    OUTPUT FILES
    ============
    • longterm_evaluation_data.csv: Input data with metrics
    • shortterm_evaluation_data.csv: Input data with metrics
    • *_evaluation_results_*.csv: System outputs (MAFAS vs Gemini)
    • evaluation_reports/: Visualizations and analysis
    """
    print(guide)


# Metric Explanations
METRIC_EXPLANATIONS = {
    "P/E Ratio": "Price-to-Earnings - lower is cheaper",
    "Debt-to-Equity": "Financial leverage - lower is less risky",
    "Profit Margin": "Profitability percentage",
    "Dividend Yield": "Annual dividend as % of price",
    "RSI(14)": "Momentum (0-100): <30=oversold, >70=overbought",
    "MACD": "Trend indicator - positive when bullish",
    "SMA50": "50-day moving average - short-term trend",
    "SMA200": "200-day moving average - long-term trend",
    "Revenue Growth": "Year-over-year revenue growth %",
    "Earnings Growth": "Quarterly earnings growth %",
}

if __name__ == "__main__":
    print_quickstart()
    print("\n\nMETRIC GLOSSARY")
    print("=" * 50)
    for metric, explanation in METRIC_EXPLANATIONS.items():
        print(f"{metric:20} - {explanation}")
