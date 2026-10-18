from __future__ import annotations

import json
import os
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest


def _empty_metrics() -> dict[str, Any]:
    return {
        "holdings": pd.DataFrame(), "daily_returns": pd.Series(dtype=float),
        "portfolio_values": pd.Series(dtype=float), "total_value": 0.0,
        "cost_basis": 0.0, "pnl": 0.0, "volatility": 0.0,
        "sharpe": 0.0, "max_drawdown": 0.0, "allocation": pd.DataFrame(),
        "correlation": pd.DataFrame(),
    }


def calculate_metrics(prices: pd.DataFrame, holdings: pd.DataFrame) -> dict[str, Any]:
    """Calculate portfolio valuation and historical risk metrics from close prices."""
    required = {"Ticker", "Shares", "Avg Cost ($)"}
    if prices is None or prices.empty or holdings is None or holdings.empty or not required.issubset(holdings.columns):
        return _empty_metrics()
    clean = holdings.copy()
    clean["Ticker"] = clean["Ticker"].astype(str).str.upper().str.strip()
    clean["Shares"] = pd.to_numeric(clean["Shares"], errors="coerce").fillna(0.0)
    clean["Avg Cost ($)"] = pd.to_numeric(clean["Avg Cost ($)"], errors="coerce").fillna(0.0)
    clean = clean[(clean["Ticker"] != "") & (clean["Shares"] > 0)]
    if clean.empty:
        return _empty_metrics()
    price_history = prices.copy()
    price_history.columns = [str(c).upper().strip() for c in price_history.columns]
    usable = [t for t in clean["Ticker"].unique() if t in price_history.columns]
    if not usable:
        return _empty_metrics()
    clean = clean[clean["Ticker"].isin(usable)].copy()
    price_history = price_history[usable].apply(pd.to_numeric, errors="coerce").sort_index().dropna(how="all")
    if price_history.empty:
        return _empty_metrics()
    latest = price_history.ffill().iloc[-1]
    share_map = clean.groupby("Ticker")["Shares"].sum()
    cost_map = clean.assign(_cost=clean["Shares"] * clean["Avg Cost ($)"]).groupby("Ticker")["_cost"].sum()
    rows = pd.DataFrame({"Ticker": usable})
    rows["Shares"] = rows["Ticker"].map(share_map).astype(float)
    rows["Price ($)"] = rows["Ticker"].map(latest).astype(float)
    rows = rows.dropna(subset=["Price ($)"])
    if rows.empty:
        return _empty_metrics()
    rows["Market Value ($)"] = rows["Shares"] * rows["Price ($)"]
    rows["Cost Basis ($)"] = rows["Ticker"].map(cost_map).fillna(0.0).astype(float)
    rows["P/L ($)"] = rows["Market Value ($)"] - rows["Cost Basis ($)"]
    total = float(rows["Market Value ($)"].sum())
    rows["Allocation (%)"] = rows["Market Value ($)"].div(total).mul(100) if total > 0 else 0.0
    tickers = rows["Ticker"].tolist()
    returns = price_history[tickers].pct_change(fill_method=None).replace([np.inf, -np.inf], np.nan).dropna(how="all")
    weights = rows.set_index("Ticker")["Market Value ($)"].div(total) if total > 0 else rows.set_index("Ticker")["Market Value ($)"] * 0
    portfolio_returns = returns[tickers].fillna(0.0).mul(weights, axis=1).sum(axis=1)
    portfolio_values = (1.0 + portfolio_returns).cumprod() * total
    volatility = float(portfolio_returns.std(ddof=1) * np.sqrt(252)) if len(portfolio_returns) > 1 else 0.0
    sharpe = float((portfolio_returns.mean() * 252) / volatility) if volatility > 0 else 0.0
    max_drawdown = float((portfolio_values / portfolio_values.cummax() - 1.0).min()) if len(portfolio_values) else 0.0
    correlation = returns[tickers].corr() if len(returns) > 1 else pd.DataFrame()
    return {
        "holdings": rows.reset_index(drop=True), "daily_returns": portfolio_returns,
        "portfolio_values": portfolio_values, "total_value": total,
        "cost_basis": float(rows["Cost Basis ($)"].sum()), "pnl": float(rows["P/L ($)"].sum()),
        "volatility": volatility, "sharpe": sharpe, "max_drawdown": max_drawdown,
        "allocation": rows[["Ticker", "Market Value ($)", "Allocation (%)"]].reset_index(drop=True),
        "correlation": correlation,
    }


def generate_insights(metrics: dict[str, Any], returns: pd.Series) -> list[str]:
    """Generate transparent rule-based insights and flag historical statistical outliers."""
    if metrics.get("total_value", 0) <= 0:
        return ["Add valid holdings and load price data to generate portfolio insights."]
    insights = [
        f"Current tracked market value is ${metrics['total_value']:,.2f}; unrealized P/L is ${metrics['pnl']:,.2f}.",
        f"Annualized historical volatility is {metrics['volatility']:.1%}.",
    ]
    allocation = metrics.get("allocation", pd.DataFrame())
    if not allocation.empty:
        top = allocation.sort_values("Allocation (%)", ascending=False).iloc[0]
        if top["Allocation (%)"] >= 35:
            insights.append(f"Concentration alert: {top['Ticker']} represents {top['Allocation (%)']:.1f}% of tracked value.")
        else:
            insights.append(f"Largest holding is {top['Ticker']} at {top['Allocation (%)']:.1f}% of tracked value.")
    values = pd.Series(returns).replace([np.inf, -np.inf], np.nan).dropna()
    if len(values) >= 30 and values.nunique() > 1:
        try:
            labels = IsolationForest(contamination=0.03, random_state=42).fit_predict(values.to_numpy().reshape(-1, 1))
            insights.append(f"ML anomaly scan flagged {int((labels == -1).sum())} unusual daily return(s). Outliers do not predict future losses.")
        except (ValueError, FloatingPointError):
            insights.append("ML anomaly scan could not run on the available return history.")
    else:
        insights.append("ML anomaly scan needs at least 30 varied daily returns; load a longer history.")
    insights.append("Historical analytics are educational and not a prediction of future returns.")
    return insights


def build_ai_context(metrics: dict[str, Any]) -> dict[str, Any]:
    """Build compact JSON-safe context from calculated app metrics."""
    holdings = metrics.get("holdings", pd.DataFrame())
    records = []
    if isinstance(holdings, pd.DataFrame) and not holdings.empty:
        columns = [c for c in ["Ticker", "Shares", "Price ($)", "Market Value ($)", "Cost Basis ($)", "P/L ($)", "Allocation (%)"] if c in holdings]
        records = holdings[columns].replace({np.nan: None}).to_dict(orient="records")

    def finite(value: Any) -> float:
        try:
            number = float(value)
            return number if np.isfinite(number) else 0.0
        except (TypeError, ValueError):
            return 0.0

    return {
        "holdings": records, "total_value_usd": finite(metrics.get("total_value", 0)),
        "cost_basis_usd": finite(metrics.get("cost_basis", 0)), "unrealized_pnl_usd": finite(metrics.get("pnl", 0)),
        "annualized_volatility": finite(metrics.get("volatility", 0)),
        "sharpe_ratio_zero_risk_free_assumption": finite(metrics.get("sharpe", 0)),
        "maximum_drawdown": finite(metrics.get("max_drawdown", 0)),
        "data_note": "Metrics are historical estimates calculated by the app; they are not forecasts.",
    }


def generate_ai_analysis(metrics: dict[str, Any], question: str | None = None) -> str:
    """Use OpenAI Generative AI with computed metrics; never fabricate a fallback answer."""
    from dotenv import load_dotenv
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not api_key:
        return "Generative AI is not configured. Add OPENAI_API_KEY to your local .env file or deployment secrets. The dashboard works without it."
    try:
        from openai import OpenAI
        client = OpenAI(api_key=api_key, timeout=30.0, max_retries=1)
        model = os.getenv("OPENAI_MODEL", "gpt-4.1-mini").strip() or "gpt-4.1-mini"
        context = json.dumps(build_ai_context(metrics), ensure_ascii=False, allow_nan=False)
        prompt = (question or "Provide a concise portfolio risk report with key observations and general risk-management considerations.").strip()[:2000]
        response = client.responses.create(
            model=model,
            instructions=("You are an educational portfolio analytics assistant. Use only supplied JSON for claims about this portfolio. "
                "Do not invent prices, events, statistics, holdings, or predictions. Explain uncertainty. Do not give individualized buy/sell instructions. "
                "If a fact is absent, say it is unavailable. This is educational information, not professional financial advice."),
            input=f"Computed portfolio metrics (JSON):\n{context}\n\nUser request: {prompt}",
        )
        answer = getattr(response, "output_text", "")
        return answer.strip() if answer and answer.strip() else "The AI provider returned an empty response. Please try again later."
    except Exception as exc:
        return f"Generative AI request failed ({type(exc).__name__}). Check API configuration/network and try again. No AI answer was fabricated."
