"""Create news + sentiment enriched evaluation datasets using Gemini only.

Usage (from repo root):
    python -m evaluation.create_news_sentiment_report

Requirements:
    - Environment variables:
        GEMINI_API_KEY : Google Gemini API key

For each row (ticker + snapshot date), Gemini is asked to describe:
    - Overall market conditions on that date
    - Stock-specific and domain/industry-relevant news
    - A numeric news sentiment score

Outputs:
    evaluation/news_enhanced/shortterm_with_news_gemini.csv
    evaluation/news_enhanced/longterm_with_news_gemini.csv
"""

from __future__ import annotations

import os
import time
import json
from typing import Dict, Any

import pandas as pd

try:
    from dotenv import load_dotenv

    load_dotenv()
except Exception:
    # .env is optional
    pass

import google.generativeai as genai


def _init_gemini_model(model_name: str = "gemini-1.5-flash"):
    """Configure and return a Gemini model instance.

    Requires GEMINI_API_KEY in environment.
    """
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY not set in environment")

    genai.configure(api_key=api_key)
    return genai.GenerativeModel(model_name)


def summarize_news_with_gemini(
    model,
    ticker: str,
    snapshot_date: str | None = None,
    horizon_label: str | None = None,
) -> Dict[str, Any]:
    """Ask Gemini to describe news + sentiment for the given stock and date.

    The prompt asks for:
        - Overall market conditions on that date
        - Stock and domain/industry related relevant news
        - A numeric sentiment score between -1.0 and 1.0

    Returns dict with keys: summary (str), sentiment_score (float from -1 to 1).
    If anything fails, returns empty summary and None score.
    """

    meta_parts = []
    if snapshot_date:
        meta_parts.append(f"on {snapshot_date}")
    if horizon_label:
        meta_parts.append(horizon_label)
    meta_txt = ", ".join(meta_parts) if meta_parts else "for the relevant evaluation period"

    prompt = f"""
You are a financial news and sentiment analyst.

For the stock {ticker}, describe the relevant news and conditions {meta_txt}.

Your tasks:
1. Describe briefly in 3-5 sentences:
   - Overall market conditions (macro / index mood, risk-on vs risk-off) around that time
   - Stock-specific news (earnings, guidance, products, management, major events)
   - Domain/industry-level news that is important for this stock
2. Based on that narrative, provide ONE numeric sentiment score between -1.0 and 1.0 where:
   - -1.0 = extremely negative for the stock
   -  0.0 = neutral or mixed
   -  1.0 = extremely positive for the stock

Return your answer strictly as a JSON object with keys:
- "summary": string
- "sentiment_score": number between -1 and 1

Example JSON:
{{"summary": "...", "sentiment_score": 0.23}}
"""

    try:
        resp = model.generate_content(prompt)
        text = resp.text.strip() if hasattr(resp, "text") else str(resp)

        # Try to isolate JSON if model wrapped it in backticks or text
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            text = text[start : end + 1]

        data = json.loads(text)
        summary = str(data.get("summary", ""))
        score = data.get("sentiment_score")

        try:
            if score is not None:
                score = float(score)
        except Exception:
            score = None

        # Clamp to [-1, 1]
        if isinstance(score, (int, float)):
            if score < -1:
                score = -1.0
            elif score > 1:
                score = 1.0

        return {"summary": summary, "sentiment_score": score}
    except Exception:
        return {"summary": "", "sentiment_score": None}


def enrich_df_with_news_and_sentiment(
    df: pd.DataFrame,
    model,
    horizon_label: str,
    sleep_seconds: float = 1.0,
) -> pd.DataFrame:
    """For each row in df, call Gemini to get narrative + sentiment.

    Expects columns: "Ticker", "Snapshot Date".
    Adds columns:
        gemini_news_summary
        gemini_news_sentiment_score
    """

    rows = []
    for idx, row in df.iterrows():
        ticker = str(row.get("Ticker", "")).strip()
        snapshot_date = str(row.get("Snapshot Date", "")).strip() or None

        if not ticker:
            rows.append(row)
            continue

        # Call Gemini to reason about market + stock + domain news and sentiment
        gemini_result = summarize_news_with_gemini(
            model=model,
            ticker=ticker,
            snapshot_date=snapshot_date,
            horizon_label=horizon_label,
        )

        # Respect rate limit: 1 second delay per stock
        time.sleep(sleep_seconds)

        # Attach new fields
        row = row.copy()
        row["gemini_news_summary"] = gemini_result.get("summary", "")
        row["gemini_news_sentiment_score"] = gemini_result.get("sentiment_score")
        rows.append(row)

    return pd.DataFrame(rows)


def main() -> None:
    base_dir = os.path.dirname(os.path.abspath(__file__))

    shortterm_path = os.path.join(base_dir, "shortterm_evaluation_data.csv")
    longterm_path = os.path.join(base_dir, "longterm_evaluation_data.csv")

    if not os.path.exists(shortterm_path):
        raise FileNotFoundError(f"Missing file: {shortterm_path}")
    if not os.path.exists(longterm_path):
        raise FileNotFoundError(f"Missing file: {longterm_path}")

    print("Initializing Gemini model...")
    model = _init_gemini_model()

    print(f"Loading short-term evaluation data from {shortterm_path}...")
    short_df = pd.read_csv(shortterm_path)

    print(f"Loading long-term evaluation data from {longterm_path}...")
    long_df = pd.read_csv(longterm_path)

    print("Enriching short-term evaluation data with news + Gemini sentiment...")
    short_enriched = enrich_df_with_news_and_sentiment(
        df=short_df,
        model=model,
        horizon_label="short-term (2-6 months)",
        sleep_seconds=1.0,
    )

    print("Enriching long-term evaluation data with news + Gemini sentiment...")
    long_enriched = enrich_df_with_news_and_sentiment(
        df=long_df,
        model=model,
        horizon_label="long-term (3 years)",
        sleep_seconds=1.0,
    )

    out_dir = os.path.join(base_dir, "news_enhanced")
    os.makedirs(out_dir, exist_ok=True)

    short_out = os.path.join(out_dir, "shortterm_with_news_gemini.csv")
    long_out = os.path.join(out_dir, "longterm_with_news_gemini.csv")

    short_enriched.to_csv(short_out, index=False)
    long_enriched.to_csv(long_out, index=False)

    print(f"Saved short-term news + sentiment data to: {short_out}")
    print(f"Saved long-term news + sentiment data to: {long_out}")


if __name__ == "__main__":
    main()
