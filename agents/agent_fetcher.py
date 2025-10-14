"""Agent metric fetcher: exposes functions that return the raw metrics each
agent cares about. These are intentionally synchronous and minimal.
"""
from typing import Dict, Any

from tools.financial_data_tools import (
    get_yfinance_basic,
    get_growth_metrics,
    get_news_sentiment,
    compute_technical_indicators,
)


def cautious_value_metrics(ticker: str) -> Dict[str, Any]:
    """Return the 5-6 key metrics for the Cautious Value investor."""
    basic = get_yfinance_basic(ticker)
    out = {
        "P/E Ratio": basic.get("trailingPE"),
        "Debt-to-Equity": basic.get("debtToEquity"),
        "Profit Margin": basic.get("profitMargin"),
        "Dividend Yield": basic.get("dividendYield"),
        "Current Price": basic.get("currentPrice"),
        "52w High": basic.get("fiftyTwoWeekHigh"),
        "52w Low": basic.get("fiftyTwoWeekLow"),
        "5-Year Return (%)": basic.get("fiveYearReturn"),
    }
    return out


def aggressive_growth_metrics(ticker: str) -> Dict[str, Any]:
    """Return the 6 key metrics for the Aggressive Growth investor."""
    growth = get_growth_metrics(ticker)
    news = get_news_sentiment(ticker)

    out = {
        "Revenue Growth": growth.get("revenueGrowth"),
        "Earnings Growth": growth.get("earningsQuarterlyGrowth"),
        "Forward P/E": growth.get("forwardPE"),
        "Trailing P/E": growth.get("trailingPE"),
        "1-Year Return (%)": growth.get("oneYearReturn"),
        "News Article Count": news.get("articleCount"),
        "News Sentiment Score": news.get("sentimentScore"),
        "Recent Headlines": news.get("headlines", [])
    }
    return out


def technical_trader_metrics(ticker: str) -> Dict[str, Any]:
    """Return the 4-5 key metrics for the Technical Trader."""
    tech = compute_technical_indicators(ticker)
    basic = get_yfinance_basic(ticker)

    out = {
        "RSI(14)": tech.get("rsi14"),
        "MACD": tech.get("macd"),
        "MACD Signal": tech.get("macd_signal"),
        "SMA50": tech.get("sma50"),
        "SMA200": tech.get("sma200"),
        "Price vs SMA50": tech.get("price_vs_sma50"),
        "Recent Volume": tech.get("recentVolume"),
        "Current Price": basic.get("currentPrice"),
    }
    return out
