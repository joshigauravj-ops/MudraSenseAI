"""Streamlit dashboard for editing and running the MudraSense AI workflow."""

from __future__ import annotations

import asyncio
import os
from pathlib import Path

import streamlit as st

from main import (
    load_positions_csv,
    read_positions_csv_rows,
    refresh_positions,
    run_agentic_processes,
    write_agent_results_csv,
    write_user_csv_rows,
)


_AGENT_OUTPUT_HEADERS = {
    "current price",
    "days high",
    "days low",
    "net change %",
    "market session",
    "pnl",
    "pnl %",
    "technical sentiment",
    "breaking news analysis",
    "risk tier",
}


def _header_key(header: str) -> str:
    """Normalize a displayed header for ownership checks."""
    return " ".join(header.strip().lower().split())


def _load_editor_data(path: Path) -> None:
    """Load raw CSV data into session state when the selected path changes."""
    if st.session_state.get("editor_path") == str(path):
        return
    headers, rows = read_positions_csv_rows(path)
    st.session_state["editor_path"] = str(path)
    st.session_state["editor_headers"] = headers
    st.session_state["editor_rows"] = rows


st.set_page_config(page_title="MudraSense AI", page_icon="INR", layout="wide")
st.title("MudraSense AI")
st.caption("Edit open lots, save the CSV, then run the NSE analysis workflow")

input_path = Path(
    st.text_input(
        "Positions CSV",
        value=os.getenv("MUDRASENSE_INPUT", "positions.csv"),
        help="Required columns: ticker, entry_price, quantity, and trade date or transaction_date.",
    )
)
configured_provider = os.getenv("MUDRASENSE_PROVIDER", "ollama")
provider = st.selectbox(
    "Inference backend",
    ("ollama", "groq"),
    index=0 if configured_provider == "ollama" else 1,
)

try:
    _load_editor_data(input_path)
    headers = st.session_state["editor_headers"]
    rows = st.session_state["editor_rows"]
    disabled_columns = [
        header for header in headers if _header_key(header) in _AGENT_OUTPUT_HEADERS
    ]
    edited_rows = st.data_editor(
        rows,
        key="positions_editor",
        num_rows="dynamic",
        hide_index=True,
        use_container_width=True,
        disabled=disabled_columns,
        column_config={
            "ticker": st.column_config.TextColumn("Ticker (.NS)"),
            "entry_price": st.column_config.NumberColumn("Entry Price (INR)", min_value=0),
            "quantity": st.column_config.NumberColumn("Quantity", min_value=1, step=1),
            "trade date": st.column_config.DateColumn("Trade Date"),
            "transaction_date": st.column_config.DateColumn("Transaction Date"),
        },
    )
    st.caption("Click column headers to sort. Use the table controls to add or remove rows.")

    save_col, run_col = st.columns(2)
    with save_col:
        if st.button("Save CSV edits", type="secondary"):
            try:
                write_user_csv_rows(input_path, headers, list(edited_rows))
                st.session_state["editor_rows"] = list(edited_rows)
                st.success("CSV edits saved after validation.")
            except Exception as exc:
                st.error(f"CSV was not saved: {exc}")

    with run_col:
        if st.button("Save and start full workflow", type="primary"):
            try:
                write_user_csv_rows(input_path, headers, list(edited_rows))
                st.session_state["editor_rows"] = list(edited_rows)
                with st.spinner("Validating and fetching NSE market data..."):
                    positions = load_positions_csv(input_path)
                    refresh_result = refresh_positions(positions)
                with st.spinner(f"Running {provider} analysis agents..."):
                    agentic_result = asyncio.run(
                        run_agentic_processes(refresh_result, provider=provider)
                    )
                write_agent_results_csv(input_path, agentic_result)
                st.session_state["agentic_result"] = agentic_result
                st.success("Workflow completed and agent-owned fields were written to CSV.")
            except Exception as exc:
                st.error(f"Workflow did not start or save: {exc}")
except Exception as exc:
    st.error(f"Could not load CSV: {exc}")


result = st.session_state.get("agentic_result")
if result is not None:
    st.subheader("Latest analysis")
    report_rows = []
    for position in result["positions"]:
        ticker = position.user_input.ticker
        metrics = result["market_metrics"].get(ticker)
        report_rows.append(
            {
                "Ticker": ticker,
                "Current Price (INR)": position.live_market.current_price,
                "Entry Price (INR)": position.user_input.entry_price,
                "Quantity": position.user_input.quantity,
                "Action": position.user_input.action,
                "Target Price (INR)": position.user_input.target_price,
                "PnL (INR)": position.current_valuation.absolute_pnl_inr,
                "PnL (%)": position.current_valuation.percentage_pnl,
                "Market Session": metrics["market_session"] if metrics else "Unavailable",
                "Technical Summary": position.agent_evaluation.technical_sentiment_summary,
                "News Analysis": position.agent_evaluation.breaking_news_analysis,
            }
        )

    st.dataframe(report_rows, use_container_width=True, hide_index=True)
    if result["errors"]:
        st.warning(f"{len(result['errors'])} workflow message(s) returned.")
        with st.expander("View structured messages"):
            for error in result["errors"]:
                st.code(error, language="json")
