"""
Collect historical metrics for Long-Term Evaluation (Nov 2022 snapshot)
This script fetches historical data as it was in November 2022 for 15 diversified stocks.
Ground truth: Compare to actual performance over 3 years (Nov 2022 -> Nov 2025)
"""

import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
import json

# Signal thresholds for long-term (3 years)
def get_correct_signal_longterm(return_pct):
    """Determine correct signal based on actual return for long-term"""
    if return_pct > 30:
        return "BUY"
    elif return_pct < -10:
        return "SELL"
    else:
        return "HOLD"

# Define 15 diversified stocks
STOCKS = [
    "AAPL",   # Tech
    "MSFT",   # Tech
    "NVDA",   # Tech
    "TSLA",   # Tech/Growth
    "JPM",    # Finance
    "GS",     # Finance
    "JNJ",    # Healthcare
    "UNH",    # Healthcare
    "AMZN",   # Consumer/Tech
    "WMT",    # Consumer
    "XOM",    # Energy
    "BA",     # Industrials
    "NEE",    # Utilities
    "PFE",    # Pharma
    "COIN",   # Growth/Crypto
]

def get_historical_metrics(ticker, snapshot_date):
    """
    Fetch metrics as they were on a specific date.
    snapshot_date: datetime object or string "YYYY-MM-DD"
    
    Returns dict with metrics organized by investor type
    """
    try:
        if isinstance(snapshot_date, str):
            snapshot_date = datetime.strptime(snapshot_date, "%Y-%m-%d")
        
        # Fetch historical data
        stock = yf.Ticker(ticker)
        
        # Get info available (note: yfinance historical info is limited)
        info = stock.info
        
        # Get historical price data around snapshot date
        # Get data from a month before to a month after for averaging
        start_date = snapshot_date - timedelta(days=30)
        end_date = snapshot_date + timedelta(days=30)
        
        hist = stock.history(start=start_date, end=end_date)
        
        if hist.empty:
            print(f"No data found for {ticker} on {snapshot_date}")
            return None
        
        # Get price on or closest to snapshot date
        snapshot_price = hist['Close'].iloc[len(hist)//2]  # Approximate middle
        
        # Calculate returns from snapshot date to now
        current_data = stock.history(period="1d")
        if not current_data.empty:
            current_price = current_data['Close'].iloc[-1]
            period_return = ((current_price - snapshot_price) / snapshot_price) * 100
        else:
            current_price = snapshot_price
            period_return = 0
        
        # Fetch 1-year historical for technical indicators at snapshot
        hist_1y = stock.history(start=snapshot_date - timedelta(days=365), end=snapshot_date)
        
        if len(hist_1y) < 50:
            print(f"Insufficient data for {ticker} at {snapshot_date}")
            return None
        
        close_prices = hist_1y['Close']
        
        # Technical indicators
        sma_50 = close_prices.rolling(window=50).mean().iloc[-1] if len(close_prices) >= 50 else None
        sma_200 = close_prices.rolling(window=200).mean().iloc[-1] if len(close_prices) >= 200 else None
        
        # RSI(14)
        delta = close_prices.diff()
        gain = delta.clip(lower=0).fillna(0)
        loss = -delta.clip(upper=0).fillna(0)
        avg_gain = gain.rolling(window=14, min_periods=14).mean()
        avg_loss = loss.rolling(window=14, min_periods=14).mean()
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        rsi_14 = float(rsi.iloc[-1]) if not rsi.empty else None
        
        # MACD
        ema_12 = close_prices.ewm(span=12, adjust=False).mean()
        ema_26 = close_prices.ewm(span=26, adjust=False).mean()
        macd_line = ema_12 - ema_26
        macd_signal = macd_line.ewm(span=9, adjust=False).mean()
        macd_val = float(macd_line.iloc[-1]) if not macd_line.empty else None
        macd_sig = float(macd_signal.iloc[-1]) if not macd_signal.empty else None
        
        # Price vs SMA50
        price_vs_sma50 = float(snapshot_price - sma_50) if sma_50 else None
        
        # Recent volume
        recent_volume = float(hist_1y['Volume'].iloc[-1]) if 'Volume' in hist_1y.columns else None
        
        # Construct metrics by investor type
        metrics = {
            "ticker": ticker,
            "snapshot_date": snapshot_date.strftime("%Y-%m-%d"),
            "snapshot_price": round(float(snapshot_price), 2),
            "current_price": round(float(current_price), 2),
            "period_return_pct": round(period_return, 2),
            
            # Cautious Value Investor Metrics
            "cautious_value": {
                "P/E Ratio": info.get("trailingPE"),
                "Debt-to-Equity": info.get("debtToEquity"),
                "Profit Margin": info.get("profitMargins"),
                "Dividend Yield": info.get("dividendYield"),
                "Current Price": round(float(snapshot_price), 2),
                "52w High": info.get("fiftyTwoWeekHigh"),
                "52w Low": info.get("fiftyTwoWeekLow"),
                "5-Year Return (%)": calculate_historical_return(stock, snapshot_date, years=5),
            },
            
            # Aggressive Growth Investor Metrics
            "aggressive_growth": {
                "Revenue Growth": info.get("revenueGrowth"),
                "Earnings Growth": info.get("earningsQuarterlyGrowth"),
                "Forward P/E": info.get("forwardPE"),
                "Trailing P/E": info.get("trailingPE"),
                "1-Year Return (%)": calculate_historical_return(stock, snapshot_date, years=1),
                "News Article Count": None,  # To be filled manually
                "News Sentiment Score": None,  # To be filled manually
            },
            
            # Technical Trader Metrics
            "technical_trader": {
                "RSI(14)": rsi_14,
                "MACD": macd_val,
                "MACD Signal": macd_sig,
                "SMA50": round(float(sma_50), 2) if sma_50 else None,
                "SMA200": round(float(sma_200), 2) if sma_200 else None,
                "Price vs SMA50": round(price_vs_sma50, 2) if price_vs_sma50 else None,
                "Recent Volume": recent_volume,
                "Current Price": round(float(snapshot_price), 2),
            },
        }
        
        return metrics
    
    except Exception as e:
        print(f"Error fetching data for {ticker}: {str(e)}")
        return None


def calculate_historical_return(stock, snapshot_date, years=1):
    """Calculate return over N years from snapshot date"""
    try:
        start_date = snapshot_date - timedelta(days=years*365)
        hist = stock.history(start=start_date, end=snapshot_date)
        
        if len(hist) < 2:
            return None
        
        start_price = hist['Close'].iloc[0]
        end_price = hist['Close'].iloc[-1]
        
        return_pct = ((end_price - start_price) / start_price) * 100
        return round(return_pct, 2)
    except:
        return None


def collect_longterm_data(output_file="longterm_evaluation_data.csv"):
    """
    Collect Nov 2022 snapshot data for all stocks
    """
    snapshot_date = "2022-11-15"
    
    print(f"Collecting long-term evaluation data (snapshot: {snapshot_date})")
    print(f"Collecting for {len(STOCKS)} stocks...")
    
    data_rows = []
    
    for i, ticker in enumerate(STOCKS, 1):
        print(f"[{i}/{len(STOCKS)}] Fetching {ticker}...", end=" ", flush=True)
        
        metrics = get_historical_metrics(ticker, snapshot_date)
        
        if metrics:
            # Flatten nested structure for CSV
            row = {
                "Ticker": metrics["ticker"],
                "Snapshot Date": metrics["snapshot_date"],
                "Snapshot Price": metrics["snapshot_price"],
                "Current Price": metrics["current_price"],
                "3-Year Return (%)": metrics["period_return_pct"],
                
                # Cautious Value
                "CV_P/E Ratio": metrics["cautious_value"]["P/E Ratio"],
                "CV_Debt-to-Equity": metrics["cautious_value"]["Debt-to-Equity"],
                "CV_Profit Margin": metrics["cautious_value"]["Profit Margin"],
                "CV_Dividend Yield": metrics["cautious_value"]["Dividend Yield"],
                "CV_52w High": metrics["cautious_value"]["52w High"],
                "CV_52w Low": metrics["cautious_value"]["52w Low"],
                "CV_5-Year Return (%)": metrics["cautious_value"]["5-Year Return (%)"],
                
                # Aggressive Growth
                "AG_Revenue Growth": metrics["aggressive_growth"]["Revenue Growth"],
                "AG_Earnings Growth": metrics["aggressive_growth"]["Earnings Growth"],
                "AG_Forward P/E": metrics["aggressive_growth"]["Forward P/E"],
                "AG_Trailing P/E": metrics["aggressive_growth"]["Trailing P/E"],
                "AG_1-Year Return (%)": metrics["aggressive_growth"]["1-Year Return (%)"],
                "AG_News Article Count": metrics["aggressive_growth"]["News Article Count"],
                "AG_News Sentiment Score": metrics["aggressive_growth"]["News Sentiment Score"],
                
                # Technical Trader
                "TT_RSI(14)": metrics["technical_trader"]["RSI(14)"],
                "TT_MACD": metrics["technical_trader"]["MACD"],
                "TT_MACD Signal": metrics["technical_trader"]["MACD Signal"],
                "TT_SMA50": metrics["technical_trader"]["SMA50"],
                "TT_SMA200": metrics["technical_trader"]["SMA200"],
                "TT_Price vs SMA50": metrics["technical_trader"]["Price vs SMA50"],
                "TT_Recent Volume": metrics["technical_trader"]["Recent Volume"],
                
                # Ground Truth (Auto-calculated based on return thresholds)
                "Correct Signal (Long-Term 3Y)": get_correct_signal_longterm(metrics["period_return_pct"]),
            }
            
            data_rows.append(row)
            print("✓")
        else:
            print("✗ (skipped)")
    
    # Create DataFrame and save
    if data_rows:
        df = pd.DataFrame(data_rows)
        df.to_csv(output_file, index=False)
        print(f"\n✓ Data saved to {output_file}")
        print(f"Collected {len(data_rows)}/{len(STOCKS)} stocks")
        return df
    else:
        print("No data collected!")
        return None


if __name__ == "__main__":
    df = collect_longterm_data("longterm_evaluation_data.csv")
    if df is not None:
        print("\nFirst few rows:")
        print(df.head())
