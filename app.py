from __future__ import annotations

import os
import pandas as pd
import plotly.express as px
import streamlit as st
import yfinance as yf
from dotenv import load_dotenv
from portfolio import calculate_metrics, generate_ai_analysis, generate_insights

load_dotenv()
st.set_page_config(page_title="AI Portfolio Risk Analyzer", page_icon="📊", layout="wide")
DEFAULT_HOLDINGS = pd.DataFrame({
    "Ticker": ["AAPL", "MSFT", "NVDA", "GOOGL"],
    "Shares": [5, 3, 2, 4],
    "Avg Cost ($)": [180, 390, 110, 145],
})


@st.cache_data(ttl=300, show_spinner=False)
def fetch_prices(tickers: tuple[str, ...], period: str = "1y") -> pd.DataFrame:
    """Fetch adjusted historical prices; return empty data on provider failure."""
    if not tickers:
        return pd.DataFrame()
    try:
        raw = yf.download(list(tickers), period=period, auto_adjust=True, progress=False, threads=True)
        if raw.empty:
            return pd.DataFrame()
        if isinstance(raw.columns, pd.MultiIndex):
            if "Close" not in raw.columns.get_level_values(0):
                return pd.DataFrame()
            close = raw["Close"]
        elif "Close" in raw.columns:
            close = raw[["Close"]].rename(columns={"Close": tickers[0]})
        else:
            return pd.DataFrame()
        if isinstance(close, pd.Series):
            close = close.to_frame(name=tickers[0])
        close.columns = [str(c).upper() for c in close.columns]
        return close.dropna(how="all")
    except Exception:
        return pd.DataFrame()


st.title("📊 AI Portfolio Risk Analyzer")
st.caption("Portfolio analytics • Historical risk metrics • Isolation Forest ML • Generative AI analyst")
with st.sidebar:
    st.header("Your portfolio")
    st.write("Edit the sample holdings or enter your own ticker symbols.")
    holdings = st.data_editor(DEFAULT_HOLDINGS, num_rows="dynamic", use_container_width=True, key="holdings_editor")
    period_label = st.selectbox("Historical price range", ["6 months", "1 year", "2 years", "5 years"], index=1)
    period = {"6 months": "6mo", "1 year": "1y", "2 years": "2y", "5 years": "5y"}[period_label]
    analyze = st.button("Analyze portfolio", type="primary", use_container_width=True)
if "analyzed" not in st.session_state:
    st.session_state.analyzed = True

if analyze:
    st.session_state.analyzed = True

if st.session_state.analyzed:
    cleaned = holdings.copy()
    tickers = tuple(sorted(set(cleaned["Ticker"].dropna().astype(str).str.upper().str.strip()) - {""})) if not cleaned.empty and "Ticker" in cleaned.columns else ()
    with st.spinner("Fetching market data from Yahoo Finance..."):
        prices = fetch_prices(tickers, period)
    if prices.empty:
        st.warning("Market data could not be loaded. Check your connection and ticker symbols. No fake live prices are shown.")
        st.info("Example symbols: AAPL, MSFT, NVDA, GOOGL. Data availability and timing are not guaranteed.")
    else:
        metrics = calculate_metrics(prices, holdings)
        if metrics["holdings"].empty:
            st.error("No matching holdings with positive share counts were found in downloaded market data.")
        else:
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Portfolio value", f"${metrics['total_value']:,.2f}")
            pct = metrics["pnl"] / metrics["cost_basis"] if metrics["cost_basis"] else None
            col2.metric("Unrealized P/L", f"${metrics['pnl']:,.2f}", f"{pct:.2%}" if pct is not None else None)
            col3.metric("Annualized volatility", f"{metrics['volatility']:.1%}")
            col4.metric("Maximum drawdown", f"{metrics['max_drawdown']:.1%}")
            overview, risk, ai = st.tabs(["Overview", "Risk & ML", "Generative AI Analyst"])
            with overview:
                left, right = st.columns([1.2, 1])
                with left:
                    st.subheader("Estimated portfolio value history")
                    if not metrics["portfolio_values"].empty:
                        history = metrics["portfolio_values"].rename("Estimated portfolio value").reset_index()
                        history.columns = ["Date", "Estimated portfolio value"]
                        st.plotly_chart(px.line(history, x="Date", y="Estimated portfolio value", template="plotly_dark"), use_container_width=True)
                with right:
                    st.subheader("Allocation by holding")
                    st.plotly_chart(px.pie(metrics["allocation"], names="Ticker", values="Market Value ($)", hole=0.55, template="plotly_dark"), use_container_width=True)
                st.subheader("Holdings breakdown")
                st.dataframe(metrics["holdings"], use_container_width=True, hide_index=True)
                st.download_button("Download holdings CSV", metrics["holdings"].to_csv(index=False).encode("utf-8"), "portfolio_analysis.csv", "text/csv")
            with risk:
                st.subheader("Historical risk metrics")
                st.write(f"**Sharpe ratio:** {metrics['sharpe']:.3f} (simplified zero risk-free-rate assumption).")
                st.write(f"**Maximum drawdown:** {metrics['max_drawdown']:.2%}")
                st.write(f"**Annualized volatility:** {metrics['volatility']:.2%}")
                st.caption("Historical estimates are not forecasts. Portfolio history approximates current weights; it is not a transaction-level backtest.")
                st.subheader("Insights and ML anomaly scan")
                for insight in generate_insights(metrics, metrics["daily_returns"]):
                    st.write("• " + insight)
                if not metrics["correlation"].empty:
                    st.subheader("Historical return correlation")
                    st.plotly_chart(px.imshow(metrics["correlation"], text_auto=".2f", zmin=-1, zmax=1, color_continuous_scale="RdBu", template="plotly_dark"), use_container_width=True)
            with ai:
                st.subheader("Ask your AI Portfolio Analyst")
                st.caption("The LLM receives computed portfolio metrics. AI output is informational, not professional financial advice.")
                question = st.text_area("What would you like to understand?", "Explain the main risks in my portfolio and suggest general diversification considerations.", max_chars=2000)
                if st.button("Generate AI analysis", type="primary"):
                    with st.spinner("Generating analysis..."):
                        st.session_state.ai_answer = generate_ai_analysis(metrics, question)
                if st.session_state.get("ai_answer"):
                    st.markdown(st.session_state.ai_answer)
                if not os.getenv("OPENAI_API_KEY", "").strip():
                    st.info("To enable Generative AI, set OPENAI_API_KEY in .env locally or in deployment secrets.")

st.divider()
st.caption("Educational tool only. Market data can be delayed or unavailable. Verify calculations independently before making financial decisions.")
