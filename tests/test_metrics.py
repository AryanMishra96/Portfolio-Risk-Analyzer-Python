import numpy as np
import pandas as pd
from portfolio import build_ai_context, calculate_metrics, generate_ai_analysis, generate_insights


def test_calculate_metrics_value_cost_and_allocation():
    prices = pd.DataFrame({"AAPL": [10.0, 11.0, 12.0], "MSFT": [20.0, 19.0, 22.0]})
    holdings = pd.DataFrame({"Ticker": ["AAPL", "MSFT"], "Shares": [2, 3], "Avg Cost ($)": [9, 21]})
    result = calculate_metrics(prices, holdings)
    assert result["total_value"] == 90.0
    assert result["cost_basis"] == 81.0
    assert result["pnl"] == 9.0
    assert abs(result["allocation"]["Allocation (%)"].sum() - 100) < 1e-9
    assert np.isfinite(result["volatility"])


def test_empty_price_data_returns_safe_defaults():
    result = calculate_metrics(pd.DataFrame(), pd.DataFrame({"Ticker": ["AAPL"], "Shares": [1], "Avg Cost ($)": [10]}))
    assert result["total_value"] == 0.0
    assert result["holdings"].empty


def test_invalid_holdings_are_ignored():
    prices = pd.DataFrame({"AAPL": [10.0, 11.0]})
    holdings = pd.DataFrame({"Ticker": ["AAPL", "BAD"], "Shares": [2, -3], "Avg Cost ($)": [8, 1]})
    result = calculate_metrics(prices, holdings)
    assert result["total_value"] == 22.0
    assert result["holdings"]["Ticker"].tolist() == ["AAPL"]


def test_zero_volatility_is_handled():
    prices = pd.DataFrame({"AAA": [10.0, 10.0, 10.0, 10.0]})
    holdings = pd.DataFrame({"Ticker": ["AAA"], "Shares": [1], "Avg Cost ($)": [10]})
    result = calculate_metrics(prices, holdings)
    assert result["volatility"] == 0.0
    assert result["sharpe"] == 0.0


def test_insights_handles_empty_portfolio():
    result = calculate_metrics(pd.DataFrame(), pd.DataFrame())
    assert generate_insights(result, pd.Series(dtype=float))


def test_ai_context_contains_computed_metrics():
    prices = pd.DataFrame({"AAPL": [10.0, 11.0, 12.0]})
    holdings = pd.DataFrame({"Ticker": ["AAPL"], "Shares": [2], "Avg Cost ($)": [9]})
    result = calculate_metrics(prices, holdings)
    context = build_ai_context(result)
    assert context["total_value_usd"] == 24.0
    assert context["holdings"][0]["Ticker"] == "AAPL"


def test_missing_api_key_has_helpful_message(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    result = generate_ai_analysis({"total_value": 100})
    assert "OPENAI_API_KEY" in result
    assert "not configured" in result


def test_ai_analysis_handles_provider_error_without_crashing(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key-not-real")
    monkeypatch.setenv("OPENAI_BASE_URL", "http://127.0.0.1:1/v1")
    result = generate_ai_analysis({"total_value": 100}, "Summarize this portfolio")
    assert "request failed" in result.lower() or "empty response" in result.lower()
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
