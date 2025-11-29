# MAFAS Evaluation Framework

This folder contains scripts to evaluate the MAFAS multi-agent financial analysis system against historical data and raw Gemini API.

## Overview

The evaluation compares three approaches to stock analysis:
1. **MAFAS** (Multi-Agent System): 3 agents (Cautious Value, Aggressive Growth, Technical Trader) + Judge
2. **Raw Gemini API**: Single LLM call without multi-agent orchestration
3. **Ground Truth**: Actual stock performance (manually filled)

## Setup

### Requirements
```bash
pip install yfinance pandas google-generativeai
```

### Environment Variables
Ensure `GEMINI_API_KEY` is set in your `.env` file

## Workflow

### Step 1: Collect Historical Data

#### Long-Term Evaluation (Nov 2022 → Nov 2025)
```bash
python collect_longterm_data.py
```
- Fetches 15 diversified stocks as they were in November 2022
- Outputs: `longterm_evaluation_data.csv`
- Contains all metrics for 3 investor personas

#### Short-Term Evaluation (Aug 2025 → Oct/Dec 2025)
```bash
python collect_shortterm_data.py
```
- Fetches 15 stocks as they were in August 2025
- Outputs: `shortterm_evaluation_data.csv`
- Same metric structure as long-term

### Step 2: Fill in Ground Truth Signals

After running the data collection scripts:

1. Open the generated CSV file
2. For each stock, look up actual returns from snapshot date to now
3. Fill the "Correct Signal" column:
   - For **Long-Term**: Check 3-year return from Nov 2022 to Nov 2025
     - BUY if return > +30% (beat inflation + gains)
     - HOLD if return between -10% to +30%
     - SELL if return < -10%
   - For **Short-Term**: Check 2-6 month return from Aug 2025
     - BUY if return > +8%
     - HOLD if return between -3% to +8%
     - SELL if return < -3%

4. Also fill `AG_News Sentiment Score` and `AG_News Article Count` manually if desired

### Step 3: Run Evaluation

#### Run Long-Term Evaluation
```bash
python run_evaluation.py --type longterm
```

#### Run Short-Term Evaluation
```bash
python run_evaluation.py --type shortterm
```

#### Custom Output File
```bash
python run_evaluation.py --type longterm --output my_results.csv
```

### Step 4: Analyze Results

Results are saved to: `{evaluation_type}_evaluation_results_{timestamp}.csv`

The results CSV contains:
- **Ticker**: Stock symbol
- **mafas_signal**: BUY/HOLD/SELL from MAFAS Judge
- **raw_gemini_signal**: BUY/HOLD/SELL from Raw Gemini
- **correct_signal**: Ground truth signal (what should have been recommended)
- **mafas_correct**: 1 if MAFAS correct, 0 if wrong
- **gemini_correct**: 1 if Raw Gemini correct, 0 if wrong

## CSV Structure

### Input CSV Columns

**Basic Info:**
- `Ticker`: Stock symbol
- `Snapshot Date`: When metrics were captured
- `Snapshot Price`: Stock price at snapshot
- `Current Price`: Current price
- `Period Return (%)`: Return from snapshot to now

**Cautious Value Metrics (CV_):**
- `CV_P/E Ratio`: Price-to-earnings
- `CV_Debt-to-Equity`: Leverage ratio
- `CV_Profit Margin`: Profitability %
- `CV_Dividend Yield`: Dividend %
- `CV_52w High/Low`: 52-week range
- `CV_5-Year Return (%)`: Historical 5-year return

**Aggressive Growth Metrics (AG_):**
- `AG_Revenue Growth`: YoY revenue growth
- `AG_Earnings Growth`: Earnings growth rate
- `AG_Forward/Trailing P/E`: PE ratios
- `AG_1-Year Return (%)`: 1-year historical return
- `AG_News Article Count`: Number of news articles (to be filled)
- `AG_News Sentiment Score`: Sentiment score -1 to +1 (to be filled)

**Technical Trader Metrics (TT_):**
- `TT_RSI(14)`: Relative strength index
- `TT_MACD`: MACD value
- `TT_MACD Signal`: MACD signal line
- `TT_SMA50/SMA200`: Simple moving averages
- `TT_Price vs SMA50`: Price distance from 50-day MA
- `TT_Recent Volume`: Trading volume

**Ground Truth:**
- `Correct Signal (Long-Term 3Y)` or `Correct Signal (Short-Term 2-6M)`: BUY/HOLD/SELL

## Metrics Collected by Investor Type

### Cautious Value Investor
Uses: P/E, Debt/Equity, Profit Margin, Dividend, 52-week range, historical returns
→ Conservative, long-term focused analysis

### Aggressive Growth Investor
Uses: Revenue Growth, Earnings Growth, Forward P/E, 1-year returns, news sentiment
→ Growth-focused, momentum-aware analysis

### Technical Trader
Uses: RSI, MACD, SMAs, price vs MA, volume
→ Technical signals, price-action focused analysis

## Expected Output

After running evaluation:

```
============================================================
EVALUATION: LONGTERM
============================================================

Evaluating AAPL...
  Getting MAFAS analysis (multi-agent)... ✓
  Getting Raw Gemini signal... ✓

[Results for 15 stocks...]

============================================================
SUMMARY STATISTICS
============================================================

Total Stocks Evaluated: 15

MAFAS Signal Distribution:
BUY     6
HOLD    5
SELL    4

Raw Gemini Signal Distribution:
BUY     5
HOLD    6
SELL    4

Ground Truth Filled: 15/15
MAFAS Correct: 12/15 (80.0%)
Raw Gemini Correct: 10/15 (66.7%)
Advantage (MAFAS - Gemini): 13.3%
```

## Creating Visualization

After evaluation, use `create_evaluation_report.py` to generate:
- Win rate comparison charts
- Signal distribution visualizations
- Accuracy metrics

## Notes

- First run will take time (rate limit delays between API calls)
- Ground truth must be manually filled for accuracy metrics
- News sentiment is optional but recommended for better evaluation
- Subsequent runs are faster (no API calls needed if data cached)

## Troubleshooting

**"No data collected" error:**
- Check internet connection
- Verify yfinance access (some stocks may not have data for old dates)
- Try individual stocks manually first

**API Rate Limit Errors:**
- Script includes retry logic with delays
- If still hitting limits, increase `time.sleep()` values in run_evaluation.py

**CSV Not Generated:**
- Check that snapshot date is valid
- Verify at least some stocks have available data
- Check file permissions in evaluation folder
