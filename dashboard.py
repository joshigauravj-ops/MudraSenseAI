"""Streamlit dashboard for the MudraSense AI market refresh."""

from __future__ import annotations

import asyncio
import os
from pathlib import Path

import streamlit as st

from main import (
    load_positions_csv,
    refresh_positions,
    run_agentic_processes,
    write_agent_results_csv,
)


st.set_page_config(page_title="MudraSense AI", page_icon="INR", layout="wide")
st.title("MudraSense AI")
st.caption("NSE open-position monitor | prices and PnL are calculated by Python")

input_path = Path(
    st.text_input(
        "Positions CSV",
        value=os.getenv("MUDRASENSE_INPUT", "positions.csv"),
        help="Required: ticker, entry_price, quantity, and trade date or transaction_date. See examples/open_positions.template.csv.",
    )
)
configured_provider = os.getenv("MUDRASENSE_PROVIDER", "ollama")
provider = st.selectbox(
    "Inference backend",
    ("ollama", "groq"),
    index=0 if configured_provider == "ollama" else 1,
)

if st.button("Start full workflow", type="primary"):
    try:
        with st.spinner("Fetching NSE market data..."):
            positions = load_positions_csv(input_path)
            refresh_result = refresh_positions(positions)
        with st.spinner(f"Running {provider} analysis agents..."):
            agentic_result = asyncio.run(
                run_agentic_processes(refresh_result, provider=provider)
            )
            write_agent_results_csv(input_path, agentic_result)
            st.session_state["agentic_result"] = agentic_result
    except Exception as exc:
        st.error(f"Workflow did not start: {exc}")

result = st.session_state.get("agentic_result")
if result is None:
    st.info("Enter a CSV path and select Start full workflow.")
else:
    rows = []
    for position in result["positions"]:
        ticker = position.user_input.ticker
        metrics = result["market_metrics"].get(ticker)
        rows.append(
            {
                "Ticker": ticker,
                "Current Price (INR)": position.live_market.current_price,
                "Entry Price (INR)": position.user_input.entry_price,
                "Quantity": position.user_input.quantity,
                "Action": position.user_input.action,
                "Target Price (INR)": position.user_input.target_price,
                "Target Profit (INR)": position.user_input.target_profit_inr,
                "Target Holding": position.user_input.target_holding,
                "Commission + STT (INR)": position.user_input.commission_stt_inr,
                "PnL (INR)": position.current_valuation.absolute_pnl_inr,
                "PnL (%)": position.current_valuation.percentage_pnl,
                "Market Session": metrics["market_session"] if metrics else "Unavailable",
                "Technical Summary": position.agent_evaluation.technical_sentiment_summary,
                "News Analysis": position.agent_evaluation.breaking_news_analysis,
            }
        )

    st.dataframe(rows, use_container_width=True, hide_index=True)
    if result["errors"]:
        st.warning(f"{len(result['errors'])} position refresh(es) failed.")
        with st.expander("View structured errors"):
            for error in result["errors"]:
                st.code(error, language="json")
