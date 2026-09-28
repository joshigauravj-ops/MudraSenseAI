"""Professional trading monitor dashboard for MudraSense AI."""

from __future__ import annotations

import asyncio
import os
from threading import BoundedSemaphore
from decimal import Decimal
from pathlib import Path

import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from market_tools import fetch_price_history
from main import (
    calculate_target_price,
    load_positions_csv,
    read_positions_csv_rows,
    refresh_positions_async,
    run_agentic_processes,
    write_agent_results_csv,
    write_user_csv_rows,
)

_LOGO_PATH = Path(__file__).resolve().parent / "assets" / "branding" / "MudraSenseAI.svg"

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
_CHART_FETCH_LIMIT = BoundedSemaphore(
    max(1, int(os.getenv("MUDRASENSE_CHART_MAX_CONCURRENT_REQUESTS", "4")))
)


def _header_key(header: str) -> str:
    return " ".join(header.strip().lower().split())


def _currency(value: object) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "₹0.00"
    return f"₹{number:,.2f}"


def _pct(value: object) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "0.00%"
    return f"{number:.2f}%"


def _tone_color(value: object) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "#7dd3fc"
    if number > 0:
        return "#34d399"
    if number < 0:
        return "#f87171"
    return "#7dd3fc"


def _swing_trendline(bars: list[dict], field: str, *, is_high: bool, label: str, color: str):
    """Draw a valid support/resistance ray through confirmed swing pivots."""
    pivot_radius = 2
    pivot_points = []
    for index in range(pivot_radius, len(bars) - pivot_radius):
        value = bars[index][field]
        neighbors = [
            bars[neighbor][field]
            for neighbor in range(index - pivot_radius, index + pivot_radius + 1)
            if neighbor != index
        ]
        is_pivot = value > max(neighbors) if is_high else value < min(neighbors)
        if is_pivot:
            pivot_points.append((index, value))

    valid_lines = []
    for first_point_index, (first_index, first_value) in enumerate(pivot_points[:-1]):
        for second_index, second_value in pivot_points[first_point_index + 1 :]:
            slope = (second_value - first_value) / (second_index - first_index)
            if (is_high and slope >= 0) or (not is_high and slope <= 0):
                continue

            tolerance = max(abs(first_value), abs(second_value)) * 0.002
            is_respected = all(
                bars[index][field]
                <= first_value + slope * (index - first_index) + tolerance
                if is_high
                else bars[index][field]
                >= first_value + slope * (index - first_index) - tolerance
                for index in range(first_index + 1, second_index)
            )
            if is_respected:
                valid_lines.append((second_index, second_index - first_index, first_index, first_value, slope))

    if not valid_lines:
        return None

    second_index, _, first_index, first_value, _ = max(valid_lines)
    second_value = bars[second_index][field]
    return go.Scatter(
        x=[bars[first_index]["timestamp"], bars[second_index]["timestamp"]],
        y=[first_value, second_value],
        mode="lines",
        line={"color": color, "width": 2},
        name=label,
        hovertemplate=f"{label} · ₹%{{y:,.2f}}<extra></extra>",
    )


@st.cache_data(ttl=60, max_entries=128, show_spinner=False)
def _cached_price_history(
    ticker: str,
    period: str,
    interval: str,
    timeout_seconds: float,
) -> list[dict] | str:
    if not _CHART_FETCH_LIMIT.acquire(timeout=timeout_seconds):
        return "Chart history provider is busy. Retry shortly."
    try:
        return fetch_price_history(
            ticker,
            period=period,
            interval=interval,
            timeout_seconds=timeout_seconds,
        )
    finally:
        _CHART_FETCH_LIMIT.release()


def _close_chart_dialog() -> None:
    st.session_state.pop("open_chart_ticker", None)


@st.dialog(
    "Market chart",
    width="medium",
    dismissible=True,
    on_dismiss=_close_chart_dialog,
)
def _render_chart_dialog(ticker: str, positions: list) -> None:
    ticker_positions = [
        position for position in positions if position.user_input.ticker == ticker
    ]
    if not ticker_positions:
        st.info("No open positions are available for this ticker.")
        return

    selected_position = ticker_positions[0]
    chart_head, timeframe_col = st.columns([3, 1])
    with chart_head:
        st.markdown(f"#### {ticker} · NSE")
        st.caption("Candlestick · Volume · INR")
    with timeframe_col:
        timeframe = st.selectbox(
            "Range",
            ("1D", "5D", "1M", "3M", "1Y"),
            index=2,
            key=f"chart_timeframe_{ticker}",
            label_visibility="collapsed",
        )

    history_options = {
        "1D": ("1d", "5m"),
        "5D": ("5d", "15m"),
        "1M": ("1mo", "1h"),
        "3M": ("3mo", "1d"),
        "1Y": ("1y", "1d"),
    }
    period, interval = history_options[timeframe]
    bars = _cached_price_history(
        ticker,
        period,
        interval,
        float(os.getenv("MUDRASENSE_MARKET_TIMEOUT_SECONDS", "10")),
    )
    if isinstance(bars, str):
        st.warning("Historical chart data is unavailable for this symbol right now.")
        with st.expander("Chart data details"):
            st.code(bars, language="json")
        return

    bars = sorted(bars, key=lambda bar: bar["timestamp"])
    timestamps = [bar["timestamp"] for bar in bars]
    latest_bar = bars[-1]
    previous_close = bars[-2]["close"] if len(bars) > 1 else latest_bar["close"]
    chart_change_percent = (
        (Decimal(str(latest_bar["close"])) - Decimal(str(previous_close)))
        / Decimal(str(previous_close))
        * Decimal("100")
        if previous_close
        else Decimal("0")
    )
    price_low = min(bar["low"] for bar in bars)
    price_high = max(bar["high"] for bar in bars)
    price_padding = max((price_high - price_low) * 0.08, price_high * 0.005)
    visible_price_range = (max(0, price_low - price_padding), price_high + price_padding)
    figure = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.025,
        row_heights=[0.78, 0.22],
    )
    figure.add_trace(
        go.Candlestick(
            x=timestamps,
            open=[bar["open"] for bar in bars],
            high=[bar["high"] for bar in bars],
            low=[bar["low"] for bar in bars],
            close=[bar["close"] for bar in bars],
            increasing_line_color="#26a69a",
            decreasing_line_color="#ef5350",
            name=ticker,
        ),
        row=1,
        col=1,
    )
    for trendline in (
        _swing_trendline(
            bars,
            "low",
            is_high=False,
            label="Support trendline",
            color="#26a69a",
        ),
        _swing_trendline(
            bars,
            "high",
            is_high=True,
            label="Resistance trendline",
            color="#ef5350",
        ),
    ):
        if trendline is not None:
            figure.add_trace(trendline, row=1, col=1)
    figure.add_trace(
        go.Bar(
            x=timestamps,
            y=[bar["volume"] for bar in bars],
            marker_color=[
                "rgba(38,166,154,0.55)" if bar["close"] >= bar["open"] else "rgba(239,83,80,0.55)"
                for bar in bars
            ],
            name="Volume",
        ),
        row=2,
        col=1,
    )
    off_chart_levels = []
    for label, value, color in (
        ("Target", calculate_target_price(selected_position), "#26a69a"),
        ("Stop", selected_position.user_input.stop_loss_price, "#ef5350"),
    ):
        if value is not None:
            level = float(value)
            if visible_price_range[0] <= level <= visible_price_range[1]:
                figure.add_hline(
                    y=level,
                    line_dash="dot",
                    line_color=color,
                    annotation_text=f"{label} · ₹{value:.2f}",
                    annotation_position="top left",
                    row=1,
                    col=1,
                )
            else:
                off_chart_levels.append(f"{label} ₹{value:.2f}")
    figure.update_layout(
        template="plotly_dark",
        height=260,
        margin={"l": 8, "r": 8, "t": 12, "b": 8},
        paper_bgcolor="#0b0e11",
        plot_bgcolor="#0b0e11",
        font={"color": "#d1d4dc", "family": "Arial, sans-serif", "size": 11},
        showlegend=False,
        xaxis_rangeslider_visible=False,
        hovermode="x unified",
        dragmode="pan",
    )
    figure.update_xaxes(
        showgrid=True,
        gridcolor="rgba(120, 123, 134, 0.16)",
        zeroline=False,
        rangeslider_visible=False,
        rangebreaks=(
            [
                {"bounds": ["sat", "mon"]},
                {"pattern": "hour", "bounds": [15.5, 9.25]},
            ]
            if interval in {"5m", "15m", "1h"}
            else [{"bounds": ["sat", "mon"]}]
        ),
    )
    figure.update_yaxes(
        side="right",
        showgrid=True,
        gridcolor="rgba(120, 123, 134, 0.16)",
        zeroline=False,
        range=visible_price_range,
        row=1,
        col=1,
    )
    figure.update_yaxes(
        side="right",
        showgrid=False,
        zeroline=False,
        row=2,
        col=1,
    )
    st.metric(
        "Latest chart close",
        _currency(latest_bar["close"]),
        _pct(chart_change_percent),
    )
    st.caption(
        f"O {_currency(latest_bar['open'])} · H {_currency(latest_bar['high'])} · "
        f"L {_currency(latest_bar['low'])} · C {_currency(latest_bar['close'])} · "
        f"{latest_bar['timestamp']}"
    )
    st.plotly_chart(
        figure,
        use_container_width=True,
        config={
            "displaylogo": False,
            "scrollZoom": True,
            "modeBarButtonsToRemove": ["lasso2d", "select2d"],
        },
    )
    if off_chart_levels:
        st.caption(f"Outside visible price range: {', '.join(off_chart_levels)}")
    st.caption(f"{len(ticker_positions)} open lot(s)")
    for lot_number, position in enumerate(ticker_positions, start=1):
        st.markdown(f"**Lot {lot_number} · {position.user_input.action}**")
        st.caption(f"Entry {_currency(position.user_input.entry_price)} · Qty {position.user_input.quantity}")
        trigger = position.trigger.status
        risk = position.agent_evaluation.risk_tier
        if trigger != "No Trigger" and trigger:
            st.error(trigger)
        elif risk == "High":
            st.warning("High risk")
        else:
            st.caption(f"Risk: {risk}")


def _render_market_workspace(result: dict | None, fallback_positions: list) -> None:
    positions = result.get("positions", []) if result else fallback_positions
    if not positions:
        st.info("Add positions to the watchlist to begin monitoring.")
        return

    tickers = list(dict.fromkeys(position.user_input.ticker for position in positions))
    st.markdown("#### Watchlist")
    headers = st.columns([1.5, 1.1, 1, 1.2, 1.4])
    for column, label in zip(headers, ("Ticker", "Last price", "Move", "Open PnL", "Status")):
        column.caption(label)

    clicked_ticker = None
    for ticker in tickers:
        ticker_positions = [
            position for position in positions if position.user_input.ticker == ticker
        ]
        representative = ticker_positions[0]
        total_pnl = sum(
            (position.current_valuation.absolute_pnl_inr for position in ticker_positions),
            Decimal("0"),
        )
        active_trigger = next(
            (
                position.trigger.status
                for position in ticker_positions
                if position.trigger.status and position.trigger.status != "No Trigger"
            ),
            None,
        )
        row = st.columns([1.5, 1.1, 1, 1.2, 1.4])
        if row[0].button(ticker, key=f"open_chart_{ticker}", help="Open price chart"):
            clicked_ticker = ticker
        row[1].write(_currency(representative.live_market.current_price))
        row[2].write(_pct(representative.live_market.net_change_percent))
        row[3].write(_currency(total_pnl))
        if active_trigger:
            row[4].error(active_trigger)
        elif any(position.agent_evaluation.risk_tier == "High" for position in ticker_positions):
            row[4].warning("High risk")
        else:
            row[4].write("Monitoring")

    if clicked_ticker:
        st.session_state["open_chart_ticker"] = clicked_ticker
    open_ticker = st.session_state.get("open_chart_ticker")
    if open_ticker in tickers:
        _render_chart_dialog(open_ticker, positions)


def _load_editor_data(path: Path) -> None:
    if st.session_state.get("editor_path") == str(path):
        return
    headers, rows = read_positions_csv_rows(path)
    st.session_state["editor_path"] = str(path)
    st.session_state["editor_headers"] = headers
    st.session_state["editor_rows"] = rows


def _render_kpis(result: dict | None) -> None:
    if result is None:
        return

    positions = result.get("positions", [])
    if not positions:
        st.info("No positions loaded for this watchlist.")
        return

    total_investment = sum((p.user_input.entry_price * p.user_input.quantity) for p in positions)
    total_pnl = sum((p.current_valuation.absolute_pnl_inr for p in positions), Decimal("0"))
    avg_change = sum((p.live_market.net_change_percent for p in positions), Decimal("0")) / Decimal(len(positions))
    risk_count = sum(1 for p in positions if p.agent_evaluation.risk_tier == "High")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(
            f"""
            <div class='metric-card'><div class='metric-label'>Portfolio PnL</div><div class='metric-value' style='color:{_tone_color(total_pnl)};'>{_currency(total_pnl)}</div></div>
            """,
            unsafe_allow_html=True,
        )
    with col2:
        st.markdown(
            f"""
            <div class='metric-card'><div class='metric-label'>Investment</div><div class='metric-value'>{_currency(total_investment)}</div></div>
            """,
            unsafe_allow_html=True,
        )
    with col3:
        st.markdown(
            f"""
            <div class='metric-card'><div class='metric-label'>Avg. Move</div><div class='metric-value' style='color:{_tone_color(avg_change)};'>{_pct(avg_change)}</div></div>
            """,
            unsafe_allow_html=True,
        )
    with col4:
        st.markdown(
            f"""
            <div class='metric-card'><div class='metric-label'>High Risk</div><div class='metric-value'>{risk_count}</div></div>
            """,
            unsafe_allow_html=True,
        )


def _render_report_table(result: dict | None) -> None:
    if result is None:
        return

    report_rows = []
    for position in result["positions"]:
        ticker = position.user_input.ticker
        metrics = result["market_metrics"].get(ticker)
        report_rows.append(
            {
                "Ticker": ticker,
                "Price": position.live_market.current_price,
                "Day High": position.live_market.days_high,
                "Day Low": position.live_market.days_low,
                "Net Change %": position.live_market.net_change_percent,
                "PnL": position.current_valuation.absolute_pnl_inr,
                "PnL %": position.current_valuation.percentage_pnl,
                "Action": position.user_input.action,
                "Target": calculate_target_price(position),
                "Stop Loss": position.user_input.stop_loss_price,
                "Risk": position.agent_evaluation.risk_tier,
                "Trigger": position.trigger.status,
                "Technical": position.agent_evaluation.technical_sentiment_summary,
                "News": position.agent_evaluation.breaking_news_analysis,
                "Market Session": metrics["market_session"] if metrics else "Unavailable",
            }
        )

    st.dataframe(
        report_rows,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Price": st.column_config.NumberColumn("Price", format="₹%.2f"),
            "Day High": st.column_config.NumberColumn("Day High", format="₹%.2f"),
            "Day Low": st.column_config.NumberColumn("Day Low", format="₹%.2f"),
            "PnL": st.column_config.NumberColumn("PnL", format="₹%.2f"),
            "PnL %": st.column_config.NumberColumn("PnL %", format="%.2f%%"),
            "Net Change %": st.column_config.NumberColumn("Net Change %", format="%.2f%%"),
            "Target": st.column_config.NumberColumn("Target", format="₹%.2f"),
            "Stop Loss": st.column_config.NumberColumn("Stop Loss", format="₹%.2f"),
        },
    )


st.set_page_config(page_title="MudraSense AI", page_icon="INR", layout="wide")

st.markdown(
    """
    <style>
        :root {
            --bg: #131722;
            --panel: #1e222d;
            --panel-soft: #191d27;
            --line: rgba(120, 123, 134, 0.24);
            --text: #d1d4dc;
            --muted: #787b86;
            --primary: #2962ff;
            --success: #26a69a;
            --warning: #f0b90b;
            --danger: #ef5350;
            --shadow: 0 8px 24px rgba(0, 0, 0, 0.22);
        }

        .stApp {
            background: var(--bg);
            color: var(--text);
        }

        .block-container {
            padding-top: 2rem;
            padding-bottom: 3rem;
            padding-left: 1rem;
            padding-right: 1rem;
        }

        .muted {
            color: var(--muted);
            font-size: 0.84rem;
            margin-top: 0.2rem;
        }

        .metric-card {
            border: 1px solid var(--line);
            border-radius: 18px;
            background: var(--panel);
            padding: 1rem 1.1rem;
            min-height: 110px;
            box-shadow: var(--shadow);
        }

        .metric-label {
            color: var(--muted);
            font-size: 0.75rem;
            letter-spacing: 0.12em;
            text-transform: uppercase;
            margin-bottom: 0.7rem;
        }

        .metric-value {
            font-size: clamp(1.5rem, 2vw, 2.1rem);
            font-weight: 800;
            color: var(--text);
            line-height: 1.2;
        }

        [data-testid="stSidebar"] {
            background: #1e222d;
            border-right: 1px solid var(--line);
        }

        div[data-testid="stDataFrame"] > div {
            border-radius: 14px;
            overflow: hidden;
            border: 1px solid var(--line);
        }

    </style>
    """,
    unsafe_allow_html=True,
)


with st.container(border=True):
    logo_col, menu_col, status_col = st.columns(
        [1.0, 1.6, 0.8],
        vertical_alignment="center",
    )
    with logo_col:
        st.image(str(_LOGO_PATH), width=250)
    with menu_col:
        active_view = st.segmented_control(
            "Main navigation",
            ["Market monitor", "Trade analysis", "My Listing"],
            default="Market monitor",
            key="active_dashboard_view",
            label_visibility="collapsed",
            width="stretch",
        )
    with status_col:
        st.markdown(
            "<div style='text-align:right;padding-top:0.8rem;'>"
            "<span style='display:inline-block;padding:0.45rem 0.9rem;border-radius:999px;"
            "background:rgba(41,98,255,0.14);color:#8ab4f8;"
            "border:1px solid rgba(41,98,255,0.25);font-size:0.74rem;font-weight:700;"
            "letter-spacing:0.08em;text-transform:uppercase;'>Market data · on demand</span></div>",
            unsafe_allow_html=True,
        )


with st.sidebar:
    with st.expander("Portfolio controls", expanded=True):
        input_path = Path(
            st.text_input(
                "Positions CSV",
                value=os.getenv("MUDRASENSE_INPUT", "positions.csv"),
                help="Required columns: ticker, entry_price, quantity, and transaction_date.",
            )
        )

        configured_provider = os.getenv("MUDRASENSE_PROVIDER", "ollama")
        provider = st.selectbox(
            "Inference backend",
            ("ollama", "groq"),
            index=0 if configured_provider == "ollama" else 1,
        )

        st.markdown("### Refresh parameters")
        max_parallel = st.slider(
            "Max concurrent tickers",
            min_value=1,
            max_value=50,
            value=int(os.getenv("MUDRASENSE_MAX_CONCURRENT_TICKERS", "10")),
        )

        if st.button("Refresh watchlist", use_container_width=True, type="primary"):
            try:
                positions = load_positions_csv(input_path)
                with st.spinner("Refreshing market data across watchlist using bounded parallel fetches..."):
                    refresh_result = asyncio.run(
                        refresh_positions_async(
                            positions,
                            timeout_seconds=float(os.getenv("MUDRASENSE_MARKET_TIMEOUT_SECONDS", "10")),
                            max_attempts=int(os.getenv("MUDRASENSE_MARKET_RETRY_ATTEMPTS", "3")),
                            retry_delay_seconds=float(os.getenv("MUDRASENSE_RETRY_DELAY_SECONDS", "1.0")),
                            max_concurrent_tickers=max_parallel,
                        )
                    )
                st.session_state["agentic_result"] = refresh_result
                st.success("Market data refreshed successfully.")
            except Exception as exc:
                st.error(f"Refresh failed: {exc}")

        if st.button("Run full analysis", use_container_width=True):
            try:
                positions = load_positions_csv(input_path)
                with st.spinner("Running market + analysis pipeline..."):
                    refresh_result = asyncio.run(
                        refresh_positions_async(
                            positions,
                            timeout_seconds=float(os.getenv("MUDRASENSE_MARKET_TIMEOUT_SECONDS", "10")),
                            max_attempts=int(os.getenv("MUDRASENSE_MARKET_RETRY_ATTEMPTS", "3")),
                            retry_delay_seconds=float(os.getenv("MUDRASENSE_RETRY_DELAY_SECONDS", "1.0")),
                            max_concurrent_tickers=max_parallel,
                        )
                    )
                    agentic_result = asyncio.run(
                        run_agentic_processes(refresh_result, provider=provider)
                    )
                write_agent_results_csv(input_path, agentic_result)
                st.session_state["agentic_result"] = agentic_result
                st.success("Portfolio analysis complete.")
            except Exception as exc:
                st.error(f"Workflow failed: {exc}")


monitor_result = st.session_state.get("agentic_result")
if active_view == "Market monitor":
    try:
        monitor_positions = (
            monitor_result["positions"]
            if monitor_result is not None
            else load_positions_csv(input_path)
        )
        _render_market_workspace(monitor_result, monitor_positions)
    except Exception as exc:
        st.info(f"Load a valid positions CSV to open the market monitor: {exc}")

elif active_view == "Trade analysis":
    if monitor_result is None:
        st.info("Run a watchlist refresh or full analysis to view trade analysis.")
    else:
        st.markdown("### Position and agent analysis")
        _render_report_table(monitor_result)
        with st.expander("System messages"):
            if monitor_result["errors"]:
                for error in monitor_result["errors"]:
                    st.code(error, language="json")
            else:
                st.success("No operational warnings or fetch issues detected.")

elif active_view == "My Listing":
    st.markdown("### My Listing")
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
                "trade date": st.column_config.TextColumn("Trade Date"),
                "transaction_date": st.column_config.TextColumn("Transaction Date"),
            },
        )

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
            if st.button("Save and run workflow", type="primary"):
                try:
                    write_user_csv_rows(input_path, headers, list(edited_rows))
                    st.session_state["editor_rows"] = list(edited_rows)
                    with st.spinner("Validating and fetching market data..."):
                        positions = load_positions_csv(input_path)
                        refresh_result = asyncio.run(
                            refresh_positions_async(
                                positions,
                                timeout_seconds=float(os.getenv("MUDRASENSE_MARKET_TIMEOUT_SECONDS", "10")),
                                max_attempts=int(os.getenv("MUDRASENSE_MARKET_RETRY_ATTEMPTS", "3")),
                                retry_delay_seconds=float(os.getenv("MUDRASENSE_RETRY_DELAY_SECONDS", "1.0")),
                                max_concurrent_tickers=max_parallel,
                            )
                        )
                    with st.spinner(f"Running {provider} analysis agents..."):
                        agentic_result = asyncio.run(
                            run_agentic_processes(refresh_result, provider=provider)
                        )
                    write_agent_results_csv(input_path, agentic_result)
                    st.session_state["agentic_result"] = agentic_result
                    st.success("Workflow completed successfully.")
                except Exception as exc:
                    st.error(f"Workflow did not start or save: {exc}")
    except Exception as exc:
        st.error(f"Could not load CSV: {exc}")


