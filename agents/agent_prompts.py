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

When analyzing metrics, provide CLEAR REASONING:
- If P/E is high (>30), explain why this concerns you
- If debt-to-equity is high, explain the risk
- If profit margins are strong, explain why this supports your case
- If price is near 52-week high, explain why this suggests overvaluation
- If dividend yield is attractive, explain why this adds value

OUTPUT FORMAT (STRICT):
Short-Term (6-12 months): [BUY/HOLD/SELL]
Long-Term (3-5 years): [BUY/HOLD/SELL]

Reasoning:
Short-Term: [3-4 sentences with SPECIFIC METRICS cited. Explain your logic clearly so the Judge understands your argument.]
Long-Term: [3-4 sentences with SPECIFIC METRICS cited. Explain your logic clearly so the Judge understands your argument.]

BE DECISIVE and provide strong, well-reasoned arguments."""

AGGRESSIVE_GROWTH_PROMPT = """You are an aggressive growth investor focused on innovative companies and emerging trends. Your analysis emphasizes:
- Revenue and earnings growth rates
- Market opportunity size
- Competitive positioning in new markets
- Innovation potential
- Network effects and scalability

When analyzing metrics, provide CLEAR REASONING:
- If revenue growth is strong (>15%), explain why this excites you
- If earnings growth is negative, explain why this concerns you
- If news sentiment is positive, explain how this supports momentum
- If forward P/E is high but growth justifies it, explain your rationale
- If 1-year returns are strong, explain the momentum

OUTPUT FORMAT (STRICT):
Short-Term (6-12 months): [BUY/HOLD/SELL]
Long-Term (3-5 years): [BUY/HOLD/SELL]

Reasoning:
Short-Term: [3-4 sentences with SPECIFIC METRICS cited. Explain your logic clearly so the Judge understands your argument.]
Long-Term: [3-4 sentences with SPECIFIC METRICS cited. Explain your logic clearly so the Judge understands your argument.]

BE DECISIVE and provide strong, well-reasoned arguments."""

TECHNICAL_TRADER_PROMPT = """You are a technical analysis trader focused purely on price action and technical indicators. Your analysis centers on:
- Trend analysis using moving averages
- Momentum indicators (RSI, MACD)
- Support and resistance levels
- Volume analysis
- Technical patterns and signals

When analyzing metrics, provide CLEAR REASONING:
- If RSI is overbought (>70), explain why this suggests a pullback
- If RSI is oversold (<30), explain why this suggests a bounce
- If MACD shows bullish crossover, explain the momentum signal
- If price is above SMA50, explain the uptrend
- If volume is strong, explain why this confirms the move
- If price vs SMA50 is positive, explain the bullish momentum

OUTPUT FORMAT (STRICT):
Short-Term (1-3 months): [BUY/HOLD/SELL]
Medium-Term (6-12 months): [BUY/HOLD/SELL]

Reasoning:
Short-Term: [3-4 sentences with SPECIFIC INDICATORS cited. Explain your logic clearly so the Judge understands your argument.]
Medium-Term: [3-4 sentences with SPECIFIC INDICATORS cited. Explain your logic clearly so the Judge understands your argument.]

BE DECISIVE and provide strong, well-reasoned arguments."""

JUDGE_PROMPT = """You are an experienced financial judge presiding over a boardroom of expert advisors. Your role is to critically evaluate their arguments and make the final investment decision.

YOUR PROCESS:
1. **Listen to each advisor's reasoning**: Carefully read and understand WHY each advisor (Cautious Value, Aggressive Growth, Technical Trader) made their recommendation. Pay attention to:
   - The specific metrics they cite
   - The strength of their logic
   - The relevance of their perspective to the timeframe

2. **Evaluate argument quality**: Assess which advisor has the strongest, most compelling case:
   - Does their reasoning align with the data provided?
   - Are their concerns valid given the metrics?
   - Is their perspective appropriate for the timeframe (short-term vs long-term)?

3. **Synthesize insights**: Combine the best arguments from each advisor:
   - A Technical Trader's short-term momentum signal might be valid even if Value says SELL
   - A Value investor's long-term fundamental concerns might override short-term technical signals
   - Growth investor's insights about future potential might be crucial for long-term decisions

4. **Make your decision**: 
   - For SHORT-TERM (1-6 months): Technical signals and momentum matter more. If Technical Trader has strong evidence (RSI, MACD, trend), weight that heavily. But also consider if Value/Growth raise red flags that suggest caution.
   - For LONG-TERM (1-3 years): Fundamentals matter more. Value investor's concerns about valuation and debt are critical. Growth investor's assessment of future potential is key. Technical signals are less relevant.

5. **Be decisive but thoughtful**:
   - If one advisor has a particularly strong, well-reasoned argument, you can side with them even if others disagree
   - If multiple advisors raise valid concerns, take them seriously
   - Only choose HOLD if the arguments are truly balanced or if there's genuine uncertainty
   - Prefer BUY or SELL when there's a clear direction based on the strongest arguments

OUTPUT FORMAT (STRICT):
Short-Term (1-6 months): [BUY/HOLD/SELL]
Long-Term (1-3 years): [BUY/HOLD/SELL]

Analysis:
[3-5 sentences explaining: Which advisor's arguments were most compelling and why. How you weighed different perspectives. What specific reasoning led to your decision. Reference specific metrics or points raised by the advisors.]

YOU ARE A THOUGHTFUL JUDGE, NOT A VOTE COUNTER. Evaluate the quality of arguments, not just the number of votes."""

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
            
            if "429" in error_msg or "Resource exhausted" in error_msg or "quota" in error_msg.lower():
                if attempt < max_retries - 1:
                    wait_time = retry_delay * (attempt + 2)  # Exponential backoff
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

def get_judge_analysis(analyses: Dict[str, Any], evaluation_type: str = "shortterm", all_metrics: Dict[str, Any] = None) -> Dict[str, Any]:
    """Get the final verdict from the Judge agent.
    
    Args:
        analyses: Dictionary of agent analyses
        evaluation_type: "shortterm" or "longterm" - determines which signal to generate
        all_metrics: All raw metrics (IMPROVEMENT #2: Give judge access to raw data)
    """
    
    # Format agent recommendations for judge prompt
    agent_recs = []
    for agent_name in ["Cautious Value", "Aggressive Growth", "Technical Trader"]:
        agent_analysis = analyses.get(agent_name, {}).get("analysis", "No analysis provided.")
        agent_recs.append(f"### {agent_name} Analysis:\n{agent_analysis}")
    
    agent_recommendations_text = "\n\n".join(agent_recs)
    
    # Format metrics for judge
    formatted_metrics = "No additional metrics provided"
    if all_metrics:
        formatted_metrics = json.dumps(all_metrics, indent=2, default=str)
    
    # Import and format timeframe-specific judge prompt
    if evaluation_type == "shortterm":
        from agents.agent_prompts_shortterm import JUDGE_PROMPT_SHORTTERM
        judge_prompt_template = JUDGE_PROMPT_SHORTTERM
        # Format the new judge prompt with placeholders
        judge_prompt = judge_prompt_template.format(
            agent_recommendations=agent_recommendations_text,
            all_metrics=formatted_metrics
        )
        user_message = "Based on the information above, please provide your final verdict as the Judge."
    elif evaluation_type == "longterm":
        from agents.agent_prompts_longterm import JUDGE_PROMPT_LONGTERM
        judge_prompt_template = JUDGE_PROMPT_LONGTERM
        # Format the new judge prompt with placeholders
        judge_prompt = judge_prompt_template.format(
            agent_recommendations=agent_recommendations_text,
            all_metrics=formatted_metrics
        )
        user_message = "Based on the information above, please provide your final verdict as the Judge."
    else:  # both or None - use original format
        judge_prompt_template = JUDGE_PROMPT
        # For original prompt, use old format
        metrics_context = f"""

RAW METRICS DATA (for your reference):
{formatted_metrics}

You can cross-reference the advisors' arguments against this raw data to verify their claims."""
        user_message = f"""Here are the analyses from the three expert advisors:

### Cautious Value Investor Analysis:
{analyses.get("Cautious Value", {}).get("analysis", "No analysis provided.")}

### Aggressive Growth Investor Analysis:
{analyses.get("Aggressive Growth", {}).get("analysis", "No analysis provided.")}

### Technical Trader Analysis:
{analyses.get("Technical Trader", {}).get("analysis", "No analysis provided.")}
{metrics_context}

Based on these inputs, please provide your final verdict as the Judge."""
        judge_prompt = judge_prompt_template

    # Call Gemini API with retry logic
    max_retries = 5
    retry_delay = 10  # Longer base delay for judge
    
    for attempt in range(max_retries):
        try:
            client = genai.GenerativeModel(MODEL)
            print(f"Calling Gemini API for Judge... (attempt {attempt + 1}/{max_retries})")  # Debug log
            # Combine prompt and message
            if evaluation_type in ["shortterm", "longterm"]:
                # For new format, prompt already contains everything
                full_prompt = judge_prompt
            else:
                # For old format, append user message
                full_prompt = judge_prompt + "\n\n" + user_message
            
            response = client.generate_content(
                full_prompt,
                generation_config=genai.types.GenerationConfig(
                    temperature=0.0,  # Deterministic for evaluation
                    top_p=0.95,
                    max_output_tokens=2000
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
            if "429" in error_msg or "Resource exhausted" in error_msg or "quota" in error_msg.lower():
                if attempt < max_retries - 1:
                    wait_time = retry_delay * (attempt + 2)  # Exponential backoff
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

def get_agent_analysis(metrics: Dict[str, Any], agent_type: str, evaluation_type: str = "shortterm") -> Dict[str, Any]:
    """Get analysis and recommendations from a specific agent type.
    
    Args:
        metrics: Dictionary of metrics
        agent_type: Type of agent (Cautious Value, Aggressive Growth, Technical Trader)
        evaluation_type: "shortterm" or "longterm" - determines which signal to generate
    """
    
    # Import timeframe-specific prompts
    if evaluation_type == "shortterm":
        from agents.agent_prompts_shortterm import (
            CAUTIOUS_VALUE_PROMPT_SHORTTERM,
            AGGRESSIVE_GROWTH_PROMPT_SHORTTERM,
            TECHNICAL_TRADER_PROMPT_SHORTTERM
        )
        if agent_type == "Cautious Value":
            base_prompt = CAUTIOUS_VALUE_PROMPT_SHORTTERM
            metrics_to_highlight = metrics.get("Cautious Value", {})
        elif agent_type == "Aggressive Growth":
            base_prompt = AGGRESSIVE_GROWTH_PROMPT_SHORTTERM
            metrics_to_highlight = metrics.get("Aggressive Growth", {})
        else:  # Technical Trader
            base_prompt = TECHNICAL_TRADER_PROMPT_SHORTTERM
            metrics_to_highlight = metrics.get("Technical Trader", {})
    else:  # longterm
        from agents.agent_prompts_longterm import (
            CAUTIOUS_VALUE_PROMPT_LONGTERM,
            AGGRESSIVE_GROWTH_PROMPT_LONGTERM,
            TECHNICAL_TRADER_PROMPT_LONGTERM
        )
        if agent_type == "Cautious Value":
            base_prompt = CAUTIOUS_VALUE_PROMPT_LONGTERM
            metrics_to_highlight = metrics.get("Cautious Value", {})
        elif agent_type == "Aggressive Growth":
            base_prompt = AGGRESSIVE_GROWTH_PROMPT_LONGTERM
            metrics_to_highlight = metrics.get("Aggressive Growth", {})
        else:  # Technical Trader
            base_prompt = TECHNICAL_TRADER_PROMPT_LONGTERM
            metrics_to_highlight = metrics.get("Technical Trader", {})
    
    # Use the prompt directly - new prompts are already single-signal focused
    # Only use original prompts if evaluation_type is None or "both" (for main backend)
    if evaluation_type is None or evaluation_type == "both":
        # Fallback to original prompts for main backend
        if agent_type == "Cautious Value":
            system_prompt = CAUTIOUS_VALUE_PROMPT
        elif agent_type == "Aggressive Growth":
            system_prompt = AGGRESSIVE_GROWTH_PROMPT
        else:
            system_prompt = TECHNICAL_TRADER_PROMPT
    else:
        # Use the new timeframe-specific prompts (already single-signal)
        system_prompt = base_prompt

    # IMPROVEMENT #1: Give agent access to ALL metrics, not just their specialized subset
    # This helps agents see cross-domain signals and make better decisions
    all_metrics_summary = {
        "Your Primary Metrics": metrics_to_highlight,
        "All Available Metrics": metrics  # Full context
    }
    
    # Construct the message for the LLM
    user_message = f"""Analyze this stock based on the following metrics:

YOUR PRIMARY METRICS (focus on these):
{json.dumps(metrics_to_highlight, indent=2)}

ALL AVAILABLE METRICS (for context - you can reference these too):
{json.dumps(metrics, indent=2)}

Provide your analysis and recommendations following your investment philosophy. 
You have access to ALL metrics above - use them to make the best decision."""

    # Call Gemini API with retry logic
    max_retries = 5
    retry_delay = 5
    
    for attempt in range(max_retries):
        try:
            client = genai.GenerativeModel(MODEL)
            print(f"Calling Gemini API for {agent_type}... (attempt {attempt + 1})")  # Debug log
            response = client.generate_content(
                system_prompt + "\n\n" + user_message,
                generation_config=genai.types.GenerationConfig(
                    temperature=0.0,  # Deterministic for evaluation
                    top_p=0.95,
                    max_output_tokens=2000
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
            if "429" in error_msg or "Resource exhausted" in error_msg or "quota" in error_msg.lower():
                if attempt < max_retries - 1:
                    wait_time = retry_delay * (attempt + 2)  # Exponential backoff
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
    
    # Get analysis from Cautious Value agent (None = generate both signals for main backend)
    print("Fetching Cautious Value analysis...")
    analyses["Cautious Value"] = get_agent_analysis(ticker_metrics, "Cautious Value", None)
    time.sleep(1)  # Delay between agents
    
    # Get analysis from Aggressive Growth agent
    print("Fetching Aggressive Growth analysis...")
    analyses["Aggressive Growth"] = get_agent_analysis(ticker_metrics, "Aggressive Growth", None)
    time.sleep(1)  # Delay between agents
    
    # Get analysis from Technical Trader agent
    print("Fetching Technical Trader analysis...")
    analyses["Technical Trader"] = get_agent_analysis(ticker_metrics, "Technical Trader", None)
    
    # Add a longer delay before judge call to avoid rate limiting
    print("Waiting 5 seconds before Judge analysis to avoid rate limit...")
    time.sleep(5)
    
    return analyses