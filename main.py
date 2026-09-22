"""Runnable MudraSense AI wrapper and dashboard launcher."""

from __future__ import annotations

import argparse
import asyncio
import csv
import os
import shutil
import subprocess
import sys
from decimal import Decimal
from pathlib import Path
from typing import TypedDict, cast

from dotenv import load_dotenv

from market_tools import (
    MarketMovementMetrics,
    fetch_breaking_news,
    fetch_market_movement,
    normalize_nse_ticker,
    update_position_pnl,
)
from schemas.positions import OpenTradePosition
from analysis_graph import (
    GroqBackend,
    OllamaBackend,
    apply_agent_summaries,
    build_analysis_graph,
    build_supervised_analysis_graph,
)


load_dotenv()


class PositionCsvRow(TypedDict):
    """Normalized columns in the local positions CSV file."""

    ticker: str
    entry_price: str
    quantity: str
    transaction_date: str
    current_price: str
    target_price: str
    target_profit: str
    target_holding: str
    strike_price: str
    action: str
    commission_stt: str
    pnl: str


class RefreshResult(TypedDict):
    """Output of one market refresh run."""

    positions: list[OpenTradePosition]
    market_metrics: dict[str, MarketMovementMetrics]
    errors: list[str]


class AgenticResult(TypedDict):
    """Output after qualitative agents and supervisor processing."""

    positions: list[OpenTradePosition]
    market_metrics: dict[str, MarketMovementMetrics]
    errors: list[str]


_OUTPUT_COLUMNS = [
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
]


_REQUIRED_COLUMNS = {"ticker", "entry_price", "quantity", "transaction_date"}
_OPTIONAL_COLUMN_ALIASES = {
    "current price": "current_price",
    "target price": "target_price",
    "target profit": "target_profit",
    "target holding": "target_holding",
    "trade date": "transaction_date",
    "strike price": "strike_price",
    "action (b/s)": "action",
    "commission+stt": "commission_stt",
    "commission + stt": "commission_stt",
    "pnl": "pnl",
}


def _normalize_csv_headers(fieldnames: list[str] | None) -> dict[str, str]:
    """Map human-readable template headers to canonical loader names."""
    normalized: dict[str, str] = {}
    for fieldname in fieldnames or []:
        key = " ".join(fieldname.strip().lower().split())
        canonical = _OPTIONAL_COLUMN_ALIASES.get(key, key.replace(" ", "_"))
        normalized[fieldname] = canonical
    return normalized


def _optional_decimal(row: dict[str, str], column: str) -> Decimal | None:
    """Parse an optional money field, preserving blank cells as None."""
    value = row.get(column, "").strip()
    return Decimal(value) if value else None


def load_positions_csv(path: Path) -> list[OpenTradePosition]:
    """Load user position inputs from a CSV file into typed Pydantic models."""
    if not path.is_file():
        raise FileNotFoundError(
            f"Positions CSV not found: {path}. Copy examples/open_positions.template.csv "
            "to a local file and pass it with --input."
        )

    with path.open(newline="", encoding="utf-8-sig") as csv_file:
        reader = csv.DictReader(csv_file)
        header_map = _normalize_csv_headers(reader.fieldnames)
        columns = set(header_map.values())
        missing_columns = _REQUIRED_COLUMNS - columns
        if missing_columns:
            missing = ", ".join(sorted(missing_columns))
            raise ValueError(f"CSV is missing required columns: {missing}")

        positions: list[OpenTradePosition] = []
        validation_errors: list[str] = []
        for row_number, raw_row in enumerate(reader, start=2):
            try:
                row = {
                    header_map[key]: value or ""
                    for key, value in raw_row.items()
                    if key in header_map
                }
                typed_row = cast(PositionCsvRow, row)
                ticker = normalize_nse_ticker(typed_row["ticker"])
                entry_price = Decimal(typed_row["entry_price"])
                quantity = int(typed_row["quantity"])
                reference_current_price = _optional_decimal(row, "current_price") or entry_price
                position = OpenTradePosition.model_validate(
                    {
                        "user_input": {
                            "ticker": ticker,
                            "entry_price": entry_price,
                            "quantity": quantity,
                            "transaction_date": typed_row["transaction_date"].strip(),
                            "target_price": _optional_decimal(row, "target_price"),
                            "target_profit_inr": _optional_decimal(row, "target_profit"),
                            "target_holding": row.get("target_holding", "").strip() or None,
                            "strike_price": _optional_decimal(row, "strike_price"),
                            "action": row.get("action", "B").strip().upper() or "B",
                            "commission_stt_inr": _optional_decimal(row, "commission_stt") or Decimal("0"),
                            "reported_pnl_inr": None,
                        },
                        "live_market": {
                            "current_price": reference_current_price,
                            "days_high": reference_current_price,
                            "days_low": reference_current_price,
                            "volatility_index": None,
                            "net_change_percent": Decimal("0"),
                            "total_invested_capital": entry_price * quantity,
                        },
                        "current_valuation": {
                            "absolute_pnl_inr": Decimal("0"),
                            "percentage_pnl": Decimal("0"),
                        },
                        "agent_evaluation": {
                            "technical_sentiment_summary": "Pending market refresh.",
                            "breaking_news_analysis": "Pending news refresh.",
                            "risk_tier": "Medium",
                        },
                    },
                )
                positions.append(position)
            except Exception as exc:
                validation_errors.append(f"row {row_number}: {exc}")

    if validation_errors:
        details = "\n".join(f"- {error}" for error in validation_errors)
        raise ValueError(f"CSV validation failed; no market or agent work started:\n{details}")

    if not positions:
        raise ValueError(f"CSV contains no positions: {path}")
    return positions


def read_positions_csv_rows(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    """Read raw CSV rows for the dashboard editor without changing values."""
    if not path.is_file():
        raise FileNotFoundError(f"Positions CSV not found: {path}")
    with path.open(newline="", encoding="utf-8-sig") as csv_file:
        reader = csv.DictReader(csv_file)
        headers = list(reader.fieldnames or [])
        if not headers:
            raise ValueError("CSV must contain a header row")
        return headers, [dict(row) for row in reader]


def write_user_csv_rows(
    path: Path,
    headers: list[str],
    rows: list[dict[str, object]],
) -> None:
    """Validate and atomically save dashboard-edited CSV rows."""
    temporary_path = path.with_suffix(f"{path.suffix}.edit.tmp")
    with temporary_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=headers, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(
            {
                header: _format_csv_value(row.get(header, ""))
                for header in headers
            }
            for row in rows
        )

    try:
        load_positions_csv(temporary_path)
        _backup_csv(path)
        temporary_path.replace(path)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()


def _format_csv_value(value: object) -> str:
    """Serialize Decimal and scalar model values without locale formatting."""
    return str(value) if value is not None else ""


def _backup_csv(path: Path) -> Path:
    """Copy the current CSV to a sibling backup before replacing it."""
    backup_path = path.with_name(f"{path.name}.backup")
    shutil.copy2(path, backup_path)
    return backup_path


def write_agent_results_csv(path: Path, result: AgenticResult) -> None:
    """Persist agent-owned outputs without changing any target-prefixed column.

    Rows are matched by input order because repeated tickers represent
    independent lots and cannot safely be matched by ticker alone.
    """
    if not path.is_file():
        raise FileNotFoundError(f"Positions CSV not found: {path}")

    with path.open(newline="", encoding="utf-8-sig") as csv_file:
        reader = csv.DictReader(csv_file)
        original_headers = list(reader.fieldnames or [])
        rows = list(reader)

    if len(rows) != len(result["positions"]):
        raise ValueError("CSV row count changed during workflow; refusing to write results")

    headers = list(original_headers)
    existing_canonical = {_normalize_csv_headers(headers).get(header, header) for header in headers}
    for output_column in _OUTPUT_COLUMNS:
        output_canonical = _normalize_csv_headers([output_column])[output_column]
        if output_column not in headers and output_canonical not in existing_canonical:
            headers.append(output_column)

    header_map = _normalize_csv_headers(headers)
    output_by_canonical = {
        "current_price": "current_price",
        "days_high": "days_high",
        "days_low": "days_low",
        "net_change_percent": "net_change_%",
        "market_session": "market_session",
        "pnl": "pnl",
        "percentage_pnl": "pnl_%",
        "technical_sentiment_summary": "technical_sentiment",
        "breaking_news_analysis": "breaking_news_analysis",
        "risk_tier": "risk_tier",
    }

    for row, position in zip(rows, result["positions"]):
        ticker = position.user_input.ticker
        metrics = result["market_metrics"].get(ticker)
        if metrics is not None:
            market_values = {
                "current_price": metrics["current_price"],
                "days_high": metrics["days_high"],
                "days_low": metrics["days_low"],
                "net_change_percent": metrics["net_change_percent"],
                "market_session": metrics["market_session"],
            }
            for key, value in market_values.items():
                canonical_column = output_by_canonical[key]
                actual_column = next(
                    (header for header, canonical in header_map.items() if canonical == canonical_column),
                    canonical_column,
                )
                row[actual_column] = _format_csv_value(value)

        valuation_values = {
            "pnl": position.current_valuation.absolute_pnl_inr,
            "percentage_pnl": position.current_valuation.percentage_pnl,
            "technical_sentiment_summary": position.agent_evaluation.technical_sentiment_summary,
            "breaking_news_analysis": position.agent_evaluation.breaking_news_analysis,
            "risk_tier": position.agent_evaluation.risk_tier,
        }
        for key, value in valuation_values.items():
            canonical_column = output_by_canonical[key]
            actual_column = next(
                (header for header, canonical in header_map.items() if canonical == canonical_column),
                canonical_column,
            )
            row[actual_column] = _format_csv_value(value)

    temporary_path = path.with_suffix(f"{path.suffix}.tmp")
    with temporary_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=headers, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    _backup_csv(path)
    temporary_path.replace(path)


def _apply_market_metrics(
    position: OpenTradePosition,
    metrics: MarketMovementMetrics,
) -> OpenTradePosition | str:
    """Copy provider values into position state before deterministic PnL math."""
    updated = position.model_copy(deep=True)
    updated.live_market.current_price = metrics["current_price"]
    updated.live_market.days_high = metrics["days_high"]
    updated.live_market.days_low = metrics["days_low"]
    updated.live_market.net_change_percent = metrics["net_change_percent"]
    return update_position_pnl(updated, metrics["current_price"])


def refresh_positions(positions: list[OpenTradePosition]) -> RefreshResult:
    """Fetch current market values and update every position deterministically."""
    refreshed_positions: list[OpenTradePosition] = []
    market_metrics: dict[str, MarketMovementMetrics] = {}
    errors: list[str] = []

    for position in positions:
        ticker = position.user_input.ticker
        result = market_metrics.get(ticker)
        if result is None:
            fetched = fetch_market_movement(
                ticker,
                timeout_seconds=float(os.getenv("MUDRASENSE_MARKET_TIMEOUT_SECONDS", "10")),
            )
            if isinstance(fetched, str):
                errors.append(fetched)
                refreshed_positions.append(position)
                continue
            result = fetched
            market_metrics[ticker] = result

        updated = _apply_market_metrics(position, result)
        if isinstance(updated, str):
            errors.append(updated)
            refreshed_positions.append(position)
        else:
            refreshed_positions.append(updated)

    return {
        "positions": refreshed_positions,
        "market_metrics": market_metrics,
        "errors": errors,
    }


def _print_report(result: RefreshResult) -> None:
    """Print a compact terminal report suitable for scheduled runs."""
    print("MudraSense AI market refresh")
    for position in result["positions"]:
        ticker = position.user_input.ticker
        valuation = position.current_valuation
        metrics = result["market_metrics"].get(ticker)
        session = metrics["market_session"] if metrics else "Unavailable"
        print(
            f"{ticker}: INR {position.live_market.current_price:.2f} | "
            f"PnL INR {valuation.absolute_pnl_inr:.2f} "
            f"({valuation.percentage_pnl:.2f}%) | {session}"
        )

    if result["errors"]:
        print(f"\nErrors: {len(result['errors'])}")
        for error in result["errors"]:
            print(error)


def _create_backend(provider: str):
    """Create the configured free-inference backend lazily."""
    if provider == "groq":
        return GroqBackend()
    if provider == "ollama":
        return OllamaBackend()
    raise ValueError(f"unsupported provider: {provider}")


async def run_agentic_processes(
    result: RefreshResult,
    *,
    provider: str,
) -> AgenticResult:
    """Run news, technical, and supervisor agents after market validation."""
    errors = list(result["errors"])
    positions = list(result["positions"])
    try:
        backend = _create_backend(provider)
    except Exception as exc:
        errors.append(f"agent provider initialization failed: {exc}")
        return {
            "positions": positions,
            "market_metrics": result["market_metrics"],
            "errors": errors,
        }

    news_results = await asyncio.gather(
        *(
            fetch_breaking_news(
                position.user_input.ticker,
                timeout_seconds=float(os.getenv("MUDRASENSE_NEWS_TIMEOUT_SECONDS", "10")),
            )
            for position in positions
        ),
        return_exceptions=True,
    )
    updated_positions: list[OpenTradePosition] = []
    market_data = result["market_metrics"]
    for position, news_result in zip(positions, news_results):
        if isinstance(news_result, Exception):
            news_headlines = []
            errors.append(f"{position.user_input.ticker} news process failed: {news_result}")
        elif isinstance(news_result, str):
            news_headlines = []
            errors.append(news_result)
        else:
            news_headlines = news_result

        metrics = market_data.get(position.user_input.ticker)
        if metrics is None:
            updated_positions.append(position)
            continue

        graph = build_analysis_graph(
            technical_backend=backend,
            news_backend=backend,
        )
        agent_state = graph.invoke(
            {
                "position": position,
                "live_market": metrics,
                "technical_indicators": {},
                "news_headlines": news_headlines,
            }
        )
        errors.extend(agent_state.get("errors", []))
        if "technical_summary" in agent_state and "news_analysis" in agent_state:
            position = apply_agent_summaries(
                position,
                technical_summary=agent_state["technical_summary"],
                news_analysis=agent_state["news_analysis"],
            )
        updated_positions.append(position)

    supervisor = build_supervised_analysis_graph(
        risk_backend=backend,
        loss_threshold_percent=Decimal(
            os.getenv("MUDRASENSE_LOSS_THRESHOLD_PERCENT", "-5.00")
        ),
        intraday_drop_threshold_percent=Decimal(
            os.getenv("MUDRASENSE_INTRADAY_DROP_THRESHOLD_PERCENT", "-3.00")
        ),
    )
    supervisor_result = supervisor.invoke(
        {
            "positions": updated_positions,
            "market_data": market_data,
            "technical_indicators_by_ticker": {},
        }
    )
    errors.extend(supervisor_result.get("errors", []))
    return {
        "positions": updated_positions,
        "market_metrics": market_data,
        "errors": errors,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run MudraSense AI market refresh")
    parser.add_argument(
        "--input",
        type=Path,
        default=Path(os.getenv("MUDRASENSE_INPUT", "positions.csv")),
        help="CSV file containing the required position columns; see the template",
    )
    parser.add_argument(
        "--ui",
        action="store_true",
        help="Launch the Streamlit dashboard instead of the terminal report",
    )
    parser.add_argument(
        "--provider",
        choices=("ollama", "groq"),
        default=os.getenv("MUDRASENSE_PROVIDER", "ollama"),
        help="Free inference backend used after market refresh (default: ollama)",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.ui:
        subprocess.run(
            [sys.executable, "-m", "streamlit", "run", "dashboard.py"],
            check=True,
        )
        return

    positions = load_positions_csv(args.input)
    refreshed = refresh_positions(positions)
    _print_report(refreshed)
    agentic_result = asyncio.run(run_agentic_processes(refreshed, provider=args.provider))
    write_agent_results_csv(args.input, agentic_result)
    print(f"Agent-owned fields written to {args.input}.")
    print(f"\nAgentic workflow completed using {args.provider}.")
    if agentic_result["errors"]:
        print(f"Agentic workflow messages: {len(agentic_result['errors'])}")
        for error in agentic_result["errors"]:
            print(error)


if __name__ == "__main__":
    main()
