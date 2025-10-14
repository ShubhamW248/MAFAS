# Multi-Agent Financial Analysis System (MAFAS)

## Concept
MAFAS is a novel financial analysis platform that simulates a "boardroom" of expert AI financial advisors, each with a distinct investment philosophy. When you enter a stock ticker, the system fetches real market and news data, then each agent independently analyzes the company and provides a buy/hold/sell recommendation for both short-term and long-term horizons. The goal is to give retail investors a transparent, multi-perspective view of the market, rather than a single AI opinion.

## Investor Personas
MAFAS currently features three expert agent personas:

- **Cautious Value Investor**
  - Focuses on valuation, financial health, profit margins, dividends, and long-term track record.
  - Inspired by Benjamin Graham and classic value investing principles.

- **Aggressive Growth Investor**
  - Looks for high revenue and earnings growth, market momentum, and positive news sentiment.
  - Inspired by innovation-focused investors like Cathie Wood.

- **Technical Trader**
  - Makes decisions based on price action, technical indicators (RSI, MACD, SMAs), and volume.
  - Ignores fundamentals and focuses on short- to medium-term trading signals.

Each agent provides:
- Short-term and long-term outlooks
- Buy/hold/sell recommendations
- Justification based on their unique philosophy

## Status
- Metrics and agent analyses are displayed via a FastAPI backend and Streamlit frontend.
- Agents are independent; orchestration/debate features are planned for future versions.
