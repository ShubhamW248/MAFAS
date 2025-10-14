"""Simple runner to fetch and display metrics for each agent.

Usage:
    python -m agents.run_agents AAPL

If no ticker provided, it will raise an error.
"""
import sys
import json

from agents.agent_fetcher import (
    cautious_value_metrics,
    aggressive_growth_metrics,
    technical_trader_metrics,
)


def main(argv=None):
    argv = argv or sys.argv[1:]
    if not argv:
        print("Usage: python -m agents.run_agents <TICKER>")
        sys.exit(1)

    ticker = argv[0].upper()

    print(f"Fetching metrics for: {ticker}\n")

    a1 = cautious_value_metrics(ticker)
    a2 = aggressive_growth_metrics(ticker)
    a3 = technical_trader_metrics(ticker)

    out = {"Cautious Value": a1, "Aggressive Growth": a2, "Technical Trader": a3}
    print(json.dumps(out, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
