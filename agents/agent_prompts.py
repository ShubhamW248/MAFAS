"""Agent personas and analysis functions for the multi-agent system.

Each agent has a distinct investment philosophy and analytical approach:
1. Cautious Value: Benjamin Graham-style conservative investor
2. Aggressive Growth: Cathie Wood-style innovation investor
3. Technical Trader: Pure price action and technical analysis
"""
import os
from typing import Dict, Any
import json
import time
import google.generativeai as genai
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY not found in environment variables")

# Configure Google Generative AI
genai.configure(api_key=GEMINI_API_KEY)
MODEL = "gemini-2.0-flash"

# Agent System Prompts
CAUTIOUS_VALUE_PROMPT = """You are a conservative value investor in the style of Benjamin Graham. Your analysis focuses on:
- Intrinsic value vs market price
- Strong balance sheets and cash flows
- Margin of safety
- Sustainable competitive advantages
- Long-term perspective (3-5 years minimum)

You examine both quantitative metrics and qualitative factors. You're skeptical of hype and prefer proven business models.

Given metrics for a stock, provide:
1. Short-term outlook (6-12 months): CLEAR signal - BUY, HOLD, or SELL
2. Long-term outlook (3-5 years): CLEAR signal - BUY, HOLD, or SELL
3. Brief justification (2-3 sentences max per signal)

BE DECISIVE. Do not give grey area answers. Choose one: BUY, HOLD, or SELL."""

AGGRESSIVE_GROWTH_PROMPT = """You are an aggressive growth investor focused on innovative companies and emerging trends. Your analysis emphasizes:
- Revenue and earnings growth rates
- Market opportunity size
- Competitive positioning in new markets
- Innovation potential
- Network effects and scalability

You look for companies that could be category leaders in 5-10 years. You can tolerate high valuations if growth justifies them.

Given metrics for a stock, provide:
1. Short-term outlook (6-12 months): CLEAR signal - BUY, HOLD, or SELL
2. Long-term outlook (3-5 years): CLEAR signal - BUY, HOLD, or SELL
3. Brief justification (2-3 sentences max per signal)

BE DECISIVE. Do not give grey area answers. Choose one: BUY, HOLD, or SELL."""

TECHNICAL_TRADER_PROMPT = """You are a technical analysis trader focused purely on price action and technical indicators. Your analysis centers on:
- Trend analysis using moving averages
- Momentum indicators (RSI, MACD)
- Support and resistance levels
- Volume analysis
- Technical patterns and signals

You make decisions based on technical signals regardless of fundamentals. Your timeframes are shorter than fundamental analysts.

Given metrics for a stock, provide:
1. Short-term outlook (1-3 months): CLEAR signal - BUY, HOLD, or SELL
2. Medium-term outlook (6-12 months): CLEAR signal - BUY, HOLD, or SELL
3. Brief justification (2-3 sentences max per signal)

BE DECISIVE. Do not give grey area answers. Choose one: BUY, HOLD, or SELL."""

JUDGE_PROMPT = """You are the Judge in a financial analysis forum. You have received recommendations from three expert advisors with different philosophies: a Cautious Value Investor, an Aggressive Growth Investor, and a Technical Trader.

Your role is to:
1. **Synthesize their views**: Briefly note the signals and reasoning from each advisor.
2. **Weigh the perspectives**: Consider which perspective is most relevant for the current market.
3. **Make a FINAL DECISION**: Provide ONE clear, decisive recommendation for BOTH short-term (1-6 months) and long-term (1-3 years).
   - Choose ONLY: BUY, HOLD, or SELL
   - Do NOT give grey area answers
4. **Justify your verdict**: 2-3 sentences explaining your choice, referencing the advisors' inputs.

OUTPUT FORMAT:
- Short-Term (1-6 months): [BUY/HOLD/SELL]
- Long-Term (1-3 years): [BUY/HOLD/SELL]
- Justification: [2-3 sentences]

YOU MUST BE DECISIVE. Pick a signal and stick with it."""

NEWS_CONTEXT_PROMPT = """You are a financial news analyst. Given a stock ticker, provide:
1. Recent market trends affecting this stock
2. Industry/sector developments
3. Company-specific news and catalysts
4. Macroeconomic factors
5. Competitive landscape changes

Provide a concise but comprehensive market context that would be useful for investment decisions."""

def get_news_context(ticker: str) -> Dict[str, Any]:
    """Get market context and news analysis from Gemini about a stock."""
    
    user_message = f"""Provide current market context and recent developments for {ticker}:
    
1. What are the key recent news items about {ticker}?
2. What are the major trends in the {ticker} industry/sector?
3. What macroeconomic factors are currently affecting {ticker}?
4. What competitive dynamics should investors know about?
5. Are there any upcoming catalysts or events for {ticker}?

Base your response on your training data knowledge up to April 2024."""

    max_retries = 3
    retry_delay = 1
    
    for attempt in range(max_retries):
        try:
            client = genai.GenerativeModel(MODEL)
            print(f"Getting news context for {ticker}... (attempt {attempt + 1})")
            response = client.generate_content(
                NEWS_CONTEXT_PROMPT + "\n\n" + user_message,
                generation_config=genai.types.GenerationConfig(
                    temperature=0.5,
                    max_output_tokens=1000
                ),
                stream=False
            )
            if response.text:
                return {
                    "ticker": ticker,
                    "news_context": response.text,
                    "status": "success"
                }
            else:
                return {
                    "ticker": ticker,
                    "news_context": "No context available",
                    "status": "empty_response"
                }
        except Exception as e:
            error_msg = str(e)
            print(f"Error getting news context: {error_msg}")
            
            if "429" in error_msg or "Resource exhausted" in error_msg:
                if attempt < max_retries - 1:
                    wait_time = retry_delay * (attempt + 1)
                    print(f"Rate limited. Waiting {wait_time} seconds...")
                    time.sleep(wait_time)
                    continue
            
            return {
                "ticker": ticker,
                "news_context": f"Error: {error_msg}",
                "status": "error"
            }
    
    return {
        "ticker": ticker,
        "news_context": "Max retries exceeded",
        "status": "error"
    }

def get_judge_analysis(analyses: Dict[str, Any]) -> Dict[str, Any]:
    """Get the final verdict from the Judge agent."""
    
    # Construct the message for the LLM
    user_message = f"""Here are the analyses from the three expert advisors:

### Cautious Value Investor Analysis:
{analyses.get("Cautious Value", {}).get("analysis", "No analysis provided.")}

### Aggressive Growth Investor Analysis:
{analyses.get("Aggressive Growth", {}).get("analysis", "No analysis provided.")}

### Technical Trader Analysis:
{analyses.get("Technical Trader", {}).get("analysis", "No analysis provided.")}

Based on these inputs, please provide your final verdict as the Judge."""

    # Call Gemini API with retry logic
    max_retries = 5
    retry_delay = 5  # Longer base delay for judge
    
    for attempt in range(max_retries):
        try:
            client = genai.GenerativeModel(MODEL)
            print(f"Calling Gemini API for Judge... (attempt {attempt + 1}/{max_retries})")  # Debug log
            response = client.generate_content(
                JUDGE_PROMPT + "\n\n" + user_message,
                generation_config=genai.types.GenerationConfig(
                    temperature=0.6,
                    max_output_tokens=1500
                ),
                stream=False
            )
            if response.text:
                return {
                    "type": "Judge",
                    "analysis": response.text,
                }
            else:
                return {
                    "type": "Judge",
                    "error": "Empty response from Gemini API",
                }
        except Exception as e:
            error_msg = str(e)
            print(f"API Error for Judge: {error_msg}")
            
            # Check if it's a rate limit error
            if "429" in error_msg or "Resource exhausted" in error_msg:
                if attempt < max_retries - 1:
                    wait_time = retry_delay * (attempt + 2)
                    print(f"Rate limited. Waiting {wait_time} seconds before retry...")
                    time.sleep(wait_time)
                    continue
            
            return {
                "type": "Judge",
                "error": f"Failed to get analysis: {error_msg}",
            }
    
    return {
        "type": "Judge",
        "error": "Max retries exceeded due to rate limiting",
    }

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

    # Call Gemini API with retry logic
    max_retries = 3
    retry_delay = 2
    
    for attempt in range(max_retries):
        try:
            client = genai.GenerativeModel(MODEL)
            print(f"Calling Gemini API for {agent_type}... (attempt {attempt + 1})")  # Debug log
            response = client.generate_content(
                system_prompt + "\n\n" + user_message,
                generation_config=genai.types.GenerationConfig(
                    temperature=0.7,
                    max_output_tokens=1500
                ),
                stream=False
            )
            if response.text:
                return {
                    "type": agent_type,
                    "analysis": response.text,
                    "raw_metrics": metrics_to_highlight
                }
            else:
                return {
                    "type": agent_type,
                    "error": "Empty response from Gemini API",
                    "raw_metrics": metrics_to_highlight
                }
        except Exception as e:
            error_msg = str(e)
            print(f"API Error for {agent_type}: {error_msg}")
            
            # Check if it's a rate limit error
            if "429" in error_msg or "Resource exhausted" in error_msg:
                if attempt < max_retries - 1:
                    wait_time = retry_delay * (attempt + 1)
                    print(f"Rate limited. Waiting {wait_time} seconds before retry...")
                    time.sleep(wait_time)
                    continue
            
            return {
                "type": agent_type,
                "error": f"Failed to get analysis: {error_msg}",
                "raw_metrics": metrics_to_highlight
            }
    
    return {
        "type": agent_type,
        "error": "Max retries exceeded due to rate limiting",
        "raw_metrics": metrics_to_highlight
    }

def get_all_analyses(ticker_metrics: Dict[str, Any], ticker: str = None) -> Dict[str, Any]:
    """Get analyses from all three agents for a given set of ticker metrics."""
    
    # Get news context first
    news_context = {}
    if ticker:
        print(f"Fetching market context for {ticker}...")
        news_context = get_news_context(ticker)
        time.sleep(2)  # Brief delay before agent calls
    
    analyses = {
        "news_context": news_context,
    }
    
    # Get analysis from Cautious Value agent
    print("Fetching Cautious Value analysis...")
    analyses["Cautious Value"] = get_agent_analysis(ticker_metrics, "Cautious Value")
    time.sleep(1)  # Delay between agents
    
    # Get analysis from Aggressive Growth agent
    print("Fetching Aggressive Growth analysis...")
    analyses["Aggressive Growth"] = get_agent_analysis(ticker_metrics, "Aggressive Growth")
    time.sleep(1)  # Delay between agents
    
    # Get analysis from Technical Trader agent
    print("Fetching Technical Trader analysis...")
    analyses["Technical Trader"] = get_agent_analysis(ticker_metrics, "Technical Trader")
    
    # Add a longer delay before judge call to avoid rate limiting
    print("Waiting 5 seconds before Judge analysis to avoid rate limit...")
    time.sleep(5)
    
    return analyses