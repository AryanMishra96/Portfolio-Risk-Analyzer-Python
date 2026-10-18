# AI Portfolio Risk Analyzer 📊

A Streamlit dashboard for portfolio valuation, historical risk metrics, and ML-based return-anomaly detection, with an optional **Generative AI Portfolio Analyst** powered by the OpenAI API.

## Features

- Add/edit holdings using ticker, shares, and average purchase cost.
- Fetch historical adjusted prices from Yahoo Finance (data can be delayed or unavailable).
- Portfolio value, cost basis, unrealized P/L, and allocation by holding.
- Interactive Plotly charts and CSV export.
- Annualized historical volatility, simplified Sharpe ratio, maximum drawdown, and return-correlation matrix.
- scikit-learn Isolation Forest to flag unusual historical portfolio returns.
- Optional LLM-powered risk explanations and portfolio Q&A grounded in calculated application metrics.
- Graceful behavior when the LLM key or market data is unavailable.

## Machine Learning vs. Generative AI

- **Machine Learning:** Isolation Forest flags statistical outliers in historical portfolio returns. It does not predict future losses.
- **Generative AI:** When configured with an OpenAI API key, the AI analyst explains calculated portfolio metrics in natural language. It does not invent missing portfolio facts or guarantee outcomes.

## Requirements

Python 3.11 or newer is recommended. Internet access is required for Yahoo Finance data and real OpenAI API calls.

## Run locally

### Windows PowerShell

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
Copy-Item .env.example .env
# Edit .env and set OPENAI_API_KEY if you want Generative AI
streamlit run app.py
```

### macOS/Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
# Edit .env and set OPENAI_API_KEY if you want Generative AI
streamlit run app.py
```

The dashboard works without an OpenAI key. For Generative AI, add your own key to `.env` locally or configure `OPENAI_API_KEY` in hosting secrets. You can optionally set `OPENAI_MODEL`; the default is `gpt-4.1-mini`. API access and billing depend on your OpenAI account.

## Tests

```bash
pytest -q
```

Unit tests use deterministic local price data and do not require live market data. The AI error-path test uses an invalid local endpoint and does not require a real API key or successful external call. A successful real LLM call must be tested separately with valid credentials.

## Deploy on Streamlit Community Cloud

1. Push this repository to GitHub.
2. In Streamlit Community Cloud, create an app from this repository and select `app.py`.
3. Add `OPENAI_API_KEY` and optionally `OPENAI_MODEL` in app secrets if you want the Generative AI analyst.
4. Never commit `.env` or API keys.

## Calculation notes and limitations

- Annualized volatility uses the standard deviation of daily returns multiplied by √252.
- Sharpe ratio uses a simplified zero-risk-free-rate assumption.
- The historical portfolio curve approximates current holdings using current market-value weights; it is not a transaction-level backtest.
- Maximum drawdown is calculated from the reconstructed historical portfolio-value series.
- Yahoo Finance is an unofficial data source; availability, timestamps, and prices are not guaranteed.
- AI output is informational and may be wrong. Verify the underlying data and consult a qualified professional where appropriate. This app does not provide personalized financial advice.
