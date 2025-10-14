"""Agent personas and analysis functions for the multi-agent system.

Each agent has a distinct investment philosophy and analytical approach:
1. Cautious Value: Benjamin Graham-style conservative investor
2. Aggressive Growth: Cathie Wood-style innovation investor
3. Technical Trader: Pure price action and technical analysis
"""
import os
from typing import Dict, Any
import json
import requests
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
if not OPENROUTER_API_KEY:
    raise ValueError("OPENROUTER_API_KEY not found in environment variables")

# OpenRouter API configuration
OPENROUTER_API_URL = "https://openrouter.ai/api/v1/chat/completions"
HEADERS = {
    "Authorization": f"Bearer {OPENROUTER_API_KEY}",
    "HTTP-Referer": "http://localhost:8501",
    "X-Title": "MAFAS"
}

# Agent System Prompts
CAUTIOUS_VALUE_PROMPT = """You are a conservative value investor in the style of Benjamin Graham. Your analysis focuses on:
- Intrinsic value vs market price
- Strong balance sheets and cash flows
- Margin of safety
- Sustainable competitive advantages
- Long-term perspective (3-5 years minimum)

You examine both quantitative metrics and qualitative factors. You're skeptical of hype and prefer proven business models.

Given metrics for a stock, provide:
1. Short-term outlook (6-12 months)
2. Long-term outlook (3-5 years)
3. Clear buy/hold/sell recommendation for each timeframe
4. Detailed justification based on the metrics and your investment philosophy

Keep responses clear and focused on value-oriented analysis."""

AGGRESSIVE_GROWTH_PROMPT = """You are an aggressive growth investor focused on innovative companies and emerging trends. Your analysis emphasizes:
- Revenue and earnings growth rates
- Market opportunity size
- Competitive positioning in new markets
- Innovation potential
- Network effects and scalability

You look for companies that could be category leaders in 5-10 years. You can tolerate high valuations if growth justifies them.

Given metrics for a stock, provide:
1. Short-term outlook (6-12 months)
2. Long-term outlook (3-5 years)
3. Clear buy/hold/sell recommendation for each timeframe
4. Detailed justification focused on growth potential and market positioning

Keep responses clear and focused on growth-oriented analysis."""

TECHNICAL_TRADER_PROMPT = """You are a technical analysis trader focused purely on price action and technical indicators. Your analysis centers on:
- Trend analysis using moving averages
- Momentum indicators (RSI, MACD)
- Support and resistance levels
- Volume analysis
- Technical patterns and signals

You make decisions based on technical signals regardless of fundamentals. Your timeframes are shorter than fundamental analysts.

Given metrics for a stock, provide:
1. Short-term outlook (1-3 months)
2. Medium-term outlook (6-12 months)
3. Clear buy/hold/sell recommendation for each timeframe
4. Detailed justification based on technical indicators and price action

Keep responses clear and focused on technical analysis."""

def get_agent_analysis(metrics: Dict[str, Any], agent_type: str) -> Dict[str, Any]:
    """Get analysis and recommendations from a specific agent type."""
    
    # Select the appropriate prompt
    if agent_type == "Cautious Value":
        system_prompt = CAUTIOUS_VALUE_PROMPT
        metrics_to_highlight = metrics.get("Cautious Value", {})
    elif agent_type == "Aggressive Growth":
        system_prompt = AGGRESSIVE_GROWTH_PROMPT
        metrics_to_highlight = metrics.get("Aggressive Growth", {})
    else:  # Technical Trader
        system_prompt = TECHNICAL_TRADER_PROMPT
        metrics_to_highlight = metrics.get("Technical Trader", {})

    # Construct the message for the LLM
    user_message = f"""Analyze this stock based on the following metrics:

{json.dumps(metrics_to_highlight, indent=2)}

Additional context (all metrics):
{json.dumps(metrics, indent=2)}

Provide your analysis and recommendations following your investment philosophy."""

    # Call OpenRouter API
    try:
        payload = {
            "model": os.getenv("OPENROUTER_MODEL", "anthropic/claude-3.5-sonnet"),
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message}
            ],
            "temperature": 0.7,
            "max_tokens": 1500
        }
        print(f"Calling OpenRouter API for {agent_type}...")  # Debug log
        response = requests.post(
            url=OPENROUTER_API_URL,
            headers=HEADERS,
            data=json.dumps(payload),
            timeout=30
        )
        if response.status_code == 200:
            result = response.json()
            analysis = result["choices"][0]["message"]["content"]
            return {
                "type": agent_type,
                "analysis": analysis,
                "raw_metrics": metrics_to_highlight
            }
        else:
            error_detail = f"Status: {response.status_code}"
            try:
                error_json = response.json()
                error_detail += f", Message: {error_json.get('error', {}).get('message', 'No error message')}"
            except:
                error_detail += f", Response: {response.text[:200]}"
            print(f"API Error for {agent_type}: {error_detail}")  # Debug log
            return {
                "type": agent_type,
                "error": f"API Error: {error_detail}",
                "raw_metrics": metrics_to_highlight
            }
    except Exception as e:
        return {
            "type": agent_type,
            "error": f"Failed to get analysis: {str(e)}",
            "raw_metrics": metrics_to_highlight
        }

def get_all_analyses(ticker_metrics: Dict[str, Any]) -> Dict[str, Any]:
    """Get analyses from all three agents for a given set of ticker metrics."""
    return {
        "Cautious Value": get_agent_analysis(ticker_metrics, "Cautious Value"),
        "Aggressive Growth": get_agent_analysis(ticker_metrics, "Aggressive Growth"),
        "Technical Trader": get_agent_analysis(ticker_metrics, "Technical Trader")
    }