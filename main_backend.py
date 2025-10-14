"""FastAPI backend that serves agent metrics for a given ticker."""
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from agents.agent_fetcher import (
    cautious_value_metrics,
    aggressive_growth_metrics,
    technical_trader_metrics,
)
from agents.agent_prompts import get_all_analyses, get_judge_analysis

app = FastAPI(title="Multi-Agent Financial Analysis System")

# Allow CORS for Streamlit frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8501"],  # Streamlit default port
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/analyze/{ticker}")
async def analyze_ticker(ticker: str):
    """Get all agent metrics for a given ticker."""
    try:
        ticker = ticker.upper()
        # First get all metrics
        metrics = {
            "Cautious Value": cautious_value_metrics(ticker),
            "Aggressive Growth": aggressive_growth_metrics(ticker),
            "Technical Trader": technical_trader_metrics(ticker),
        }
        # Then get agent analyses
        analyses = get_all_analyses(metrics)

        # Finally, get the judge's verdict
        judge_analysis = get_judge_analysis(analyses)
        
        return {
            "ticker": ticker,
            "agents": analyses,
            "judge": judge_analysis
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)