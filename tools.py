
''' Function-calling tools for live company data'''
import json
from typing import Any
import yfinance as yf

TOOL_SCHEMA = {
    "type": "function",
    "name": "get_company_fundamentals",
    "description": (
        "Fetch CURRENT market data, valuation ratios, profitability metrics, and a rule-based SWOT for a public company."
        "USE this tool whenever the user asks about present-day stock price, market capitalization, P/E ration,"
        "current valuation, dividend yield, analyst targets, or overall health. DO NOT use it for historical earnings"
        "call content, management commentary or past quarterly results - those come from the transcript content that is"
        "already provided."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "symbol": {
                "type": "string",
                "description": "uppercase Yahoo Finance ticker, e.g.: DELL, MSFT, BRK-B."
            },
            "include_income_statement": {
                "type": "boolean",
                "description": "Include annual revenue/operating income/net income rows",
                "default": False
            },
            "required": ["symbol"]
        },
    },
}
YF_ALIASES = {"BRK-A": "BRK-B", "BF-B": "BF.B"}


def get_company_fundamentals(symbol: str, include_income_statement: bool = False) -> dict:
    sym = YF_ALIASES.get(str(symbol).strip().upper(), str(symbol).strip().upper())
    try:
        info = yf.Ticker(sym).info or {}
    except Exception as e:
        return {"error": f"{type(e).__name__}: {e}", "symbol": sym}
    if not info or info.get("currentPrice") is None:
        return {"error": f"No market data for {sym}", "symbol": sym}

    out = {
        "symbol": sym,
        "company_name": info.get("longName"),
        "sector": info.get("sector"),
        "industry": info.get("industry"),
        "as_of": "live Yahoo Finance quote",
        "current_price": info.get("currentPrice"),
        "market_cap": info.get("marketCap"),
        "trailing_pe": info.get("trailingPE"),
        "forward_pe": info.get("forwardPE"),
        "revenue_ttm": info.get("totalRevenue"),  # total revenue generaetd over last 12 months (ttm -> trailing twleve months)
        "net_income_ttm": info.get("netIncomeToCommon"),
        "profit_margin_pct": round(info["profitMargin"]*100, 2) if info.get("profitMargin") else None,
        "operating_margin_pct": round(info["operatingMargins"] * 100, 2) if info.get("operatingMargins") else None,
        "return_on_equity_pct": round(info["returnOnEquity"] * 100, 2) if info.get("returnOnEquity") else None,
        "debt_to_equity": info.get("debtToEquity"),
        "dividend_yield_pct": round(info["dividendYield"] * 100, 2) if info.get("dividendYield") else None,
        "beta": info.get("beta"),
        "week52_high": info.get("fiftyTwoWeekHigh"),
        "week52_low": info.get("fiftyTwoWeekLow"),
        "target_mean_price": info.get("targetMeanPrice"),
        "revenue_growth_pct": round(info["revenueGrowth"] * 100, 2) if info.get("revenueGrowth") else None,
    }

    if include_income_statement:
        try:
            stmt = yf.Ticker(sym).financials
            if stmt is not None and not stmt.empty:
                keep = [r for r in ("Total Revenue", "Operating Income", "Net Income")
                        if r in stmt.index]
                out["income_statement"] = {
                    str(c.date()): {r: float(stmt.loc[r, c]) for r in keep}
                    for c in stmt.columns
                }
        except Exception:
            out["income_statement"] = None

    return out


def dispatch(name: str, arguments: str | dict) -> Any:
    # Execute a tool call and return a json string for the model
    if name != "get_company_fundamentals":
        return json.dumps({"error": f"Unknown tool: {name}"})
    try:
        args = json.loads(arguments) if isinstance(arguments, str) else arguments
    except json.decoder.JSONDecodeError:
        # gpt-oss sometimes emits partial JSON; salvage the symbol by regex
        import re
        m = re.search(r'"symbol"\s*:\s*"([A-Z.\-]{1-8})"', str(arguments))
        if not m:
            return json.dumps({"error": f"Invalid argument: {arguments[:200]}"})
        args = {"symbol": m.group(1)}

    result = get_company_fundamentals(args.get("symbol", ""), bool(args.get("include_income_statement", False)))
    return json.dumps(result, default=str)



# if __name__ == "__main__":
    # print(get_company_fundamentals(symbol="MSFT"))