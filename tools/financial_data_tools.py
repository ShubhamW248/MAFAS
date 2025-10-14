"""Minimal financial data helpers using yfinance (and NewsAPI if API key present).

These functions are intentionally lightweight: they try to fetch common metrics
used by the three agent personas and return dictionaries. Missing data is
represented by None and the functions avoid raising for recoverable errors.
"""
from __future__ import annotations
import os
from typing import Dict, Any, Optional
import datetime as _dt

import pandas as pd

# Attempt to load environment variables from a .env file if python-dotenv is available
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    # dotenv isn't required; environment variables may already be set
    pass

try:
    import yfinance as yf
except Exception:  # pragma: no cover - import availability depends on env
    yf = None

try:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
except Exception:
    SentimentIntensityAnalyzer = None


def _safe_ticker(ticker: str):
    if yf is None:
        raise RuntimeError("yfinance is required for data fetching. Install it from requirements.txt")
    return yf.Ticker(ticker)


def get_yfinance_basic(ticker: str) -> Dict[str, Any]:
    """Fetch basic fundamental and price metrics from yfinance.

    Returns keys: trailingPE, forwardPE, debtToEquity, profitMargin, dividendYield,
    fiftyTwoWeekHigh, fiftyTwoWeekLow, fiveYearReturn (float percent) or None.
    """
    tk = _safe_ticker(ticker)
    info = tk.info or {}

    out: Dict[str, Any] = {
        "trailingPE": info.get("trailingPE"),
        "forwardPE": info.get("forwardPE"),
        "debtToEquity": info.get("debtToEquity") or info.get("debtToEquityRatio"),
        "profitMargin": info.get("profitMargins"),
        "dividendYield": info.get("dividendYield"),
        "fiftyTwoWeekHigh": info.get("fiftyTwoWeekHigh"),
        "fiftyTwoWeekLow": info.get("fiftyTwoWeekLow"),
        "currentPrice": info.get("currentPrice") or info.get("regularMarketPrice"),
    }

    # compute 5-year return if history available
    try:
        hist = tk.history(period="5y")["Close"]
        if len(hist) >= 2:
            five_year_return = (hist.iloc[-1] / hist.iloc[0] - 1.0) * 100.0
            out["fiveYearReturn"] = round(float(five_year_return), 2)
        else:
            out["fiveYearReturn"] = None
    except Exception:
        out["fiveYearReturn"] = None

    return out


def get_growth_metrics(ticker: str) -> Dict[str, Any]:
    """Return simple growth-related metrics. Uses yfinance `info` fields
    where possible to avoid heavy parsing of financial statements.

    Keys: revenueGrowth, earningsQuarterlyGrowth, trailingPE, forwardPE, oneYearReturn
    """
    tk = _safe_ticker(ticker)
    info = tk.info or {}

    out: Dict[str, Any] = {
        "revenueGrowth": info.get("revenueGrowth"),
        "earningsQuarterlyGrowth": info.get("earningsQuarterlyGrowth"),
        "trailingPE": info.get("trailingPE"),
        "forwardPE": info.get("forwardPE"),
    }

    # 1-year return
    try:
        hist = tk.history(period="1y")["Close"]
        if len(hist) >= 2:
            one_year_return = (hist.iloc[-1] / hist.iloc[0] - 1.0) * 100.0
            out["oneYearReturn"] = round(float(one_year_return), 2)
        else:
            out["oneYearReturn"] = None
    except Exception:
        out["oneYearReturn"] = None

    return out


def get_news_sentiment(ticker: str, query: Optional[str] = None, api_key: Optional[str] = None) -> Dict[str, Any]:
    """Fetch recent news count and a simple sentiment score using VADER.

    This is a light wrapper that will only attempt a call if NEWSAPI_KEY is present
    (or api_key argument is provided). If NewsAPI isn't available or key is missing,
    the function returns None values.
    """
    key = api_key or os.getenv("NEWSAPI_KEY")
    if not key:
        return {"articleCount": None, "sentimentScore": None}

    q = query or ticker
    # Use the public NewsAPI endpoint via requests to avoid a hard dependency on their client.
    url = "https://newsapi.org/v2/everything"
    params = {"q": q, "language": "en", "pageSize": 20, "apiKey": key}

    try:
        import requests

        resp = requests.get(url, params=params, timeout=10.0)
        data = resp.json()
        articles = data.get("articles", [])
        count = len(articles)
        headlines = [a.get("title", "") for a in articles]

        # sentiment via VADER if available
        if SentimentIntensityAnalyzer is not None and count > 0:
            analyzer = SentimentIntensityAnalyzer()
            scores = [analyzer.polarity_scores(a.get("title", "") + " " + (a.get("description") or "")) for a in articles]
            # compound average
            compound = float(sum(s.get("compound", 0.0) for s in scores) / max(1, len(scores)))
        else:
            compound = None

        return {"articleCount": count, "sentimentScore": compound, "headlines": headlines}
    except Exception:
        return {"articleCount": None, "sentimentScore": None, "headlines": []}


def compute_technical_indicators(ticker: str) -> Dict[str, Any]:
    """Compute RSI(14), MACD signal, SMA50, SMA200, price vs SMA50, recent volume.

    Uses yfinance historical daily data. Returns numeric values or None.
    """
    tk = _safe_ticker(ticker)
    try:
        df = tk.history(period="1y", interval="1d").dropna()
    except Exception:
        return {
            "rsi14": None,
            "macd": None,
            "macd_signal": None,
            "sma50": None,
            "sma200": None,
            "price_vs_sma50": None,
            "recentVolume": None,
        }

    if df.empty or "Close" not in df.columns:
        return {
            "rsi14": None,
            "macd": None,
            "macd_signal": None,
            "sma50": None,
            "sma200": None,
            "price_vs_sma50": None,
            "recentVolume": None,
        }

    close = df["Close"]

    # SMA
    sma50 = close.rolling(window=50).mean().iloc[-1] if len(close) >= 50 else None
    sma200 = close.rolling(window=200).mean().iloc[-1] if len(close) >= 200 else None

    price = close.iloc[-1]
    price_vs_sma50 = None
    if sma50 is not None:
        price_vs_sma50 = float(price - sma50)

    # RSI(14)
    delta = close.diff()
    gain = delta.clip(lower=0).fillna(0)
    loss = -delta.clip(upper=0).fillna(0)
    try:
        avg_gain = gain.rolling(window=14, min_periods=14).mean()
        avg_loss = loss.rolling(window=14, min_periods=14).mean()
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        rsi14 = float(rsi.iloc[-1]) if not rsi.empty and not pd.isna(rsi.iloc[-1]) else None
    except Exception:
        rsi14 = None

    # MACD
    try:
        ema12 = close.ewm(span=12, adjust=False).mean()
        ema26 = close.ewm(span=26, adjust=False).mean()
        macd = ema12 - ema26
        macd_signal = macd.ewm(span=9, adjust=False).mean()
        macd_val = float(macd.iloc[-1]) if not macd.empty else None
        macd_sig = float(macd_signal.iloc[-1]) if not macd_signal.empty else None
    except Exception:
        macd_val = None
        macd_sig = None

    recent_volume = float(df["Volume"].iloc[-1]) if "Volume" in df.columns and not df["Volume"].empty else None

    return {
        "rsi14": rsi14,
        "macd": macd_val,
        "macd_signal": macd_sig,
        "sma50": float(sma50) if sma50 is not None else None,
        "sma200": float(sma200) if sma200 is not None else None,
        "price_vs_sma50": price_vs_sma50,
        "recentVolume": recent_volume,
    }
