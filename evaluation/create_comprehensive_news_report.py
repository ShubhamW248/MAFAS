"""Create comprehensive news and sentiment report for evaluation datasets.

This script enriches both short-term and long-term evaluation datasets with:
- Stock-specific news
- Industry/sector news
- Market-wide news
- Overall sentiment score
- Relevant news headlines and summaries

Usage (from repo root):
    python -m evaluation.create_comprehensive_news_report

Requirements:
    - Environment variables:
        GEMINI_API_KEY : Google Gemini API key

Outputs:
    evaluation/news_enhanced/shortterm_comprehensive_news.csv
    evaluation/news_enhanced/longterm_comprehensive_news.csv
"""

from __future__ import annotations

import os
import time
import json
from typing import Dict, Any, List

import pandas as pd

try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

import google.generativeai as genai


def _init_gemini_model(model_name: str = "gemini-2.0-flash"):
    """Configure and return a Gemini model instance.
    
    Requires GEMINI_API_KEY in environment.
    """
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY not set in environment")
    
    genai.configure(api_key=api_key)
    return genai.GenerativeModel(model_name)


def get_comprehensive_news_with_sentiment(
    model,
    ticker: str,
    snapshot_date: str | None = None,
    horizon_label: str | None = None,
) -> Dict[str, Any]:
    """Ask Gemini to provide comprehensive news analysis for stock, industry, and market.
    
    Returns dict with:
        - stock_news: List of stock-specific news items
        - industry_news: List of industry/sector news items
        - market_news: List of market-wide news items
        - overall_sentiment_score: Float from -1.0 to 1.0
        - news_summary: Brief summary of all news
    """
    
    meta_parts = []
    if snapshot_date:
        meta_parts.append(f"on {snapshot_date}")
    if horizon_label:
        meta_parts.append(f"({horizon_label})")
    meta_txt = ", ".join(meta_parts) if meta_parts else "for the relevant evaluation period"
    
    prompt = f"""You are a comprehensive financial news analyst. For the stock {ticker} {meta_txt}, provide detailed news analysis.

Your tasks:
1. **Stock-Specific News**: Provide 3-5 key news items specifically about {ticker} around that time. Include:
   - Earnings announcements, guidance updates
   - Product launches, partnerships, acquisitions
   - Management changes, strategic decisions
   - Major company events

2. **Industry/Sector News**: Provide 3-5 key news items about the industry or sector that {ticker} operates in:
   - Industry trends and developments
   - Regulatory changes affecting the sector
   - Competitive dynamics
   - Sector-specific catalysts

3. **Market-Wide News**: Provide 3-5 key news items about the broader market conditions:
   - Macroeconomic indicators and trends
   - Federal Reserve policy changes
   - Market sentiment and volatility
   - Geopolitical events affecting markets
   - Major index movements

4. **Overall Sentiment Score**: Based on all the news above, provide ONE numeric sentiment score between -1.0 and 1.0 where:
   - -1.0 = extremely negative for {ticker}
   -  0.0 = neutral or mixed
   -  1.0 = extremely positive for {ticker}

5. **Brief Summary**: 2-3 sentences summarizing the overall news environment and its implications for {ticker}

Return your answer strictly as a JSON object with this structure:
{{
    "stock_news": [
        {{"headline": "...", "summary": "..."}},
        ...
    ],
    "industry_news": [
        {{"headline": "...", "summary": "..."}},
        ...
    ],
    "market_news": [
        {{"headline": "...", "summary": "..."}},
        ...
    ],
    "overall_sentiment_score": 0.23,
    "news_summary": "..."
}}

Each news item should have a "headline" (brief title) and "summary" (1-2 sentence description).
The sentiment score must be a number between -1.0 and 1.0.
"""

    max_retries = 3
    retry_delay = 2
    
    for attempt in range(max_retries):
        try:
            resp = model.generate_content(
                prompt,
                generation_config=genai.types.GenerationConfig(
                    temperature=0.5,
                    max_output_tokens=3000
                )
            )
            text = resp.text.strip() if hasattr(resp, "text") else str(resp)
            
            # Try to isolate JSON if model wrapped it in backticks or markdown
            start = text.find("{")
            end = text.rfind("}")
            if start != -1 and end != -1 and end > start:
                text = text[start : end + 1]
            
            data = json.loads(text)
            
            # Extract and validate data
            stock_news = data.get("stock_news", [])
            industry_news = data.get("industry_news", [])
            market_news = data.get("market_news", [])
            sentiment_score = data.get("overall_sentiment_score")
            news_summary = data.get("news_summary", "")
            
            # Validate and clamp sentiment score
            try:
                if sentiment_score is not None:
                    sentiment_score = float(sentiment_score)
                    if sentiment_score < -1:
                        sentiment_score = -1.0
                    elif sentiment_score > 1:
                        sentiment_score = 1.0
            except (ValueError, TypeError):
                sentiment_score = None
            
            # Ensure lists are properly formatted
            if not isinstance(stock_news, list):
                stock_news = []
            if not isinstance(industry_news, list):
                industry_news = []
            if not isinstance(market_news, list):
                market_news = []
            
            return {
                "stock_news": stock_news,
                "industry_news": industry_news,
                "market_news": market_news,
                "overall_sentiment_score": sentiment_score,
                "news_summary": news_summary,
            }
            
        except json.JSONDecodeError as e:
            print(f"  JSON decode error for {ticker} (attempt {attempt + 1}): {e}")
            if attempt < max_retries - 1:
                time.sleep(retry_delay)
                continue
            return _empty_news_result()
            
        except Exception as e:
            error_msg = str(e)
            print(f"  Error for {ticker} (attempt {attempt + 1}): {error_msg}")
            
            if "429" in error_msg or "Resource exhausted" in error_msg:
                if attempt < max_retries - 1:
                    wait_time = retry_delay * (attempt + 2)
                    print(f"  Rate limited. Waiting {wait_time} seconds...")
                    time.sleep(wait_time)
                    continue
            
            if attempt < max_retries - 1:
                time.sleep(retry_delay)
                continue
            
            return _empty_news_result()
    
    return _empty_news_result()


def _empty_news_result() -> Dict[str, Any]:
    """Return empty news result structure."""
    return {
        "stock_news": [],
        "industry_news": [],
        "market_news": [],
        "overall_sentiment_score": None,
        "news_summary": "",
    }


def _format_news_for_csv(news_list: List[Dict[str, str]]) -> str:
    """Format a list of news items into a readable string for CSV."""
    if not news_list:
        return ""
    
    formatted = []
    for i, item in enumerate(news_list[:5], 1):  # Limit to 5 items
        headline = item.get("headline", "")
        summary = item.get("summary", "")
        formatted.append(f"{i}. {headline} | {summary}")
    
    return " | ".join(formatted)


def enrich_df_with_comprehensive_news(
    df: pd.DataFrame,
    model,
    horizon_label: str,
    sleep_seconds: float = 1.0,
) -> pd.DataFrame:
    """For each row in df, call Gemini to get comprehensive news analysis.
    
    Expects columns: "Ticker", "Snapshot Date".
    Adds columns:
        - Stock_News: Formatted stock-specific news
        - Industry_News: Formatted industry news
        - Market_News: Formatted market-wide news
        - News_Sentiment_Score: Overall sentiment score
        - News_Summary: Brief summary
    """
    
    rows = []
    total = len(df)
    
    for idx, row in df.iterrows():
        ticker = str(row.get("Ticker", "")).strip()
        snapshot_date = str(row.get("Snapshot Date", "")).strip() or None
        
        print(f"[{idx + 1}/{total}] Processing {ticker} (date: {snapshot_date})...")
        
        if not ticker:
            print(f"  Skipping row {idx + 1}: No ticker")
            rows.append(row)
            continue
        
        # Call Gemini to get comprehensive news
        news_result = get_comprehensive_news_with_sentiment(
            model=model,
            ticker=ticker,
            snapshot_date=snapshot_date,
            horizon_label=horizon_label,
        )
        
        # Format news for CSV storage
        stock_news_formatted = _format_news_for_csv(news_result.get("stock_news", []))
        industry_news_formatted = _format_news_for_csv(news_result.get("industry_news", []))
        market_news_formatted = _format_news_for_csv(news_result.get("market_news", []))
        
        # Add new columns to row
        row = row.copy()
        row["Stock_News"] = stock_news_formatted
        row["Industry_News"] = industry_news_formatted
        row["Market_News"] = market_news_formatted
        row["News_Sentiment_Score"] = news_result.get("overall_sentiment_score")
        row["News_Summary"] = news_result.get("news_summary", "")
        
        rows.append(row)
        
        # Respect rate limit: delay between calls
        if idx < total - 1:  # Don't sleep after last item
            time.sleep(sleep_seconds)
    
    return pd.DataFrame(rows)


def main() -> None:
    """Main function to process both evaluation datasets."""
    base_dir = os.path.dirname(os.path.abspath(__file__))
    
    shortterm_path = os.path.join(base_dir, "shortterm_evaluation_data.csv")
    longterm_path = os.path.join(base_dir, "longterm_evaluation_data.csv")
    
    if not os.path.exists(shortterm_path):
        raise FileNotFoundError(f"Missing file: {shortterm_path}")
    if not os.path.exists(longterm_path):
        raise FileNotFoundError(f"Missing file: {longterm_path}")
    
    print("=" * 60)
    print("Comprehensive News & Sentiment Report Generator")
    print("=" * 60)
    print("\nInitializing Gemini model...")
    model = _init_gemini_model()
    
    # Process short-term data
    print(f"\n{'=' * 60}")
    print("Processing SHORT-TERM evaluation data")
    print(f"{'=' * 60}")
    print(f"Loading from {shortterm_path}...")
    short_df = pd.read_csv(shortterm_path)
    print(f"Found {len(short_df)} stocks to process")
    
    print("\nEnriching with comprehensive news analysis...")
    print("(This will take approximately 1 second per stock + API call time)")
    short_enriched = enrich_df_with_comprehensive_news(
        df=short_df,
        model=model,
        horizon_label="short-term 2-6 months",
        sleep_seconds=1.0,
    )
    
    # Process long-term data
    print(f"\n{'=' * 60}")
    print("Processing LONG-TERM evaluation data")
    print(f"{'=' * 60}")
    print(f"Loading from {longterm_path}...")
    long_df = pd.read_csv(longterm_path)
    print(f"Found {len(long_df)} stocks to process")
    
    print("\nEnriching with comprehensive news analysis...")
    print("(This will take approximately 1 second per stock + API call time)")
    long_enriched = enrich_df_with_comprehensive_news(
        df=long_df,
        model=model,
        horizon_label="long-term 3 years",
        sleep_seconds=1.0,
    )
    
    # Save results
    out_dir = os.path.join(base_dir, "news_enhanced")
    os.makedirs(out_dir, exist_ok=True)
    
    short_out = os.path.join(out_dir, "shortterm_comprehensive_news.csv")
    long_out = os.path.join(out_dir, "longterm_comprehensive_news.csv")
    
    print(f"\n{'=' * 60}")
    print("Saving results...")
    print(f"{'=' * 60}")
    
    short_enriched.to_csv(short_out, index=False)
    print(f"✓ Saved short-term comprehensive news data to: {short_out}")
    
    long_enriched.to_csv(long_out, index=False)
    print(f"✓ Saved long-term comprehensive news data to: {long_out}")
    
    print(f"\n{'=' * 60}")
    print("COMPLETE!")
    print(f"{'=' * 60}")
    print(f"\nSummary:")
    print(f"  - Short-term: {len(short_enriched)} stocks processed")
    print(f"  - Long-term: {len(long_enriched)} stocks processed")
    print(f"\nNew columns added:")
    print(f"  - Stock_News: Stock-specific news items")
    print(f"  - Industry_News: Industry/sector news items")
    print(f"  - Market_News: Market-wide news items")
    print(f"  - News_Sentiment_Score: Overall sentiment (-1.0 to 1.0)")
    print(f"  - News_Summary: Brief summary of news environment")


if __name__ == "__main__":
    main()

