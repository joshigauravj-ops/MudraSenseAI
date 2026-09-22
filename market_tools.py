"""Market-data and deterministic valuation helpers for NSE positions.

This module contains no agent or LLM calls. Numeric calculations are performed
with :class:`decimal.Decimal` before any qualitative layer consumes the state.
"""

from __future__ import annotations

import asyncio
import json
import re
import traceback
from datetime import datetime, time
from decimal import Decimal, InvalidOperation, ROUND_HALF_UP
from email.utils import parsedate_to_datetime
from typing import Literal, TypedDict
from urllib.parse import quote_plus
from zoneinfo import ZoneInfo

import certifi
import requests
import yfinance as yf
from bs4 import BeautifulSoup

from schemas.positions import OpenTradePosition


_NSE_SYMBOL_PATTERN = re.compile(r"^[A-Z0-9&-]+$")
_MONEY_QUANTUM = Decimal("0.01")
_PERCENT_QUANTUM = Decimal("0.01")
_IST = ZoneInfo("Asia/Kolkata")
_NSE_OPEN = time(9, 15)
_NSE_CLOSE = time(15, 30)

MarketSessionStatus = Literal["Live Intraday", "Post-Market Close"]


class MarketMovementMetrics(TypedDict):
    """Market movement values returned for one NSE ticker."""

    ticker: str
    current_price: Decimal
    days_high: Decimal
    days_low: Decimal
    net_change_percent: Decimal
    captured_at: str
    market_session: MarketSessionStatus


class NewsHeadline(TypedDict):
    """A headline and publication timestamp extracted from Google News RSS."""

    title: str
    publication_timestamp: str


def get_market_session_status(
    captured_at: datetime | None = None,
) -> MarketSessionStatus:
    """Classify a timestamp against regular NSE trading hours in IST."""
    timestamp = captured_at or datetime.now(_IST)
    if timestamp.tzinfo is None:
        timestamp = timestamp.replace(tzinfo=_IST)
    timestamp = timestamp.astimezone(_IST)
    is_weekday = timestamp.weekday() < 5
    is_open = _NSE_OPEN <= timestamp.time() < _NSE_CLOSE
    return "Live Intraday" if is_weekday and is_open else "Post-Market Close"


def normalize_nse_ticker(ticker: str) -> str:
    """Return an uppercase NSE ticker with exactly one ``.NS`` suffix."""
    if not isinstance(ticker, str) or not ticker.strip():
        raise ValueError("ticker must be a non-empty string")

    symbol = ticker.strip().upper().split(".", maxsplit=1)[0]
    if not _NSE_SYMBOL_PATTERN.fullmatch(symbol):
        raise ValueError(f"invalid NSE ticker symbol: {ticker!r}")
    return f"{symbol}.NS"


def _decimal(value: object, field_name: str) -> Decimal:
    """Convert a provider value to a finite Decimal."""
    try:
        converted = Decimal(str(value))
    except (InvalidOperation, ValueError, TypeError) as exc:
        raise ValueError(f"{field_name} is not numeric: {value!r}") from exc

    if not converted.is_finite():
        raise ValueError(f"{field_name} must be finite")
    return converted


def _error_traceback(operation: str, ticker: str, exc: Exception) -> str:
    """Serialize an operational failure without raising it to the caller."""
    return json.dumps(
        {
            "error": operation,
            "ticker": ticker,
            "exception_type": type(exc).__name__,
            "message": str(exc),
            "traceback": traceback.format_exc(),
        },
        ensure_ascii=True,
    )


def _last_numeric(series: object, field_name: str) -> Decimal:
    """Read the latest non-null value from a yfinance Series-like object."""
    if series is None or not hasattr(series, "dropna"):
        raise ValueError(f"yfinance did not return {field_name} data")

    values = series.dropna()
    if values.empty:
        raise ValueError(f"yfinance returned no {field_name} data")
    return _decimal(values.iloc[-1], field_name)


def fetch_market_movement(
    ticker: str,
    *,
    timeout_seconds: float = 10.0,
) -> MarketMovementMetrics | str:
    """Fetch the latest daily movement metrics for an NSE ticker.

    The return value is either a typed metrics mapping or a JSON-formatted
    structured error string. ``timeout_seconds`` is passed to yfinance so a
    stalled provider request is contained at this boundary.
    """
    normalized_ticker = ticker
    try:
        normalized_ticker = normalize_nse_ticker(ticker)
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be greater than zero")

        history = yf.Ticker(normalized_ticker).history(
            period="2d",
            interval="1d",
            auto_adjust=False,
            timeout=timeout_seconds,
        )
        if history is None or history.empty:
            raise TimeoutError(f"no market data returned for {normalized_ticker}")

        current_price = _last_numeric(history["Close"], "current price")
        days_high = _last_numeric(history["High"], "day high")
        days_low = _last_numeric(history["Low"], "day low")
        previous_close = (
            _decimal(history["Close"].dropna().iloc[-2], "previous close")
            if len(history["Close"].dropna()) >= 2
            else current_price
        )
        if previous_close <= 0:
            raise ValueError("previous close must be greater than zero")

        net_change_percent = ((current_price - previous_close) / previous_close * 100).quantize(
            _PERCENT_QUANTUM,
            rounding=ROUND_HALF_UP,
        )
        captured_at = datetime.now(_IST)
        return {
            "ticker": normalized_ticker,
            "current_price": current_price.quantize(_MONEY_QUANTUM, rounding=ROUND_HALF_UP),
            "days_high": days_high.quantize(_MONEY_QUANTUM, rounding=ROUND_HALF_UP),
            "days_low": days_low.quantize(_MONEY_QUANTUM, rounding=ROUND_HALF_UP),
            "net_change_percent": net_change_percent,
            "captured_at": captured_at.isoformat(),
            "market_session": get_market_session_status(captured_at),
        }
    except Exception as exc:
        return _error_traceback("market_data_fetch_failed", normalized_ticker, exc)


def update_position_pnl(
    position: OpenTradePosition,
    current_price: Decimal | int | float | str,
) -> OpenTradePosition | str:
    """Update one position's price and calculate its open PnL deterministically.

    For buys, gross PnL is ``(current - entry) * quantity``; for sells the
    price delta is reversed. Commission and STT are then deducted. The
    percentage return is net PnL divided by invested capital. Both values are
    rounded to paise-scale precision with Decimal arithmetic; no model or
    external service is used.
    """
    try:
        price = _decimal(current_price, "current price")
        if price <= 0:
            raise ValueError("current price must be greater than zero")

        entry_price = position.user_input.entry_price
        quantity = Decimal(position.user_input.quantity)
        invested_capital = (entry_price * quantity).quantize(
            _MONEY_QUANTUM,
            rounding=ROUND_HALF_UP,
        )
        if invested_capital <= 0:
            raise ValueError("invested capital must be greater than zero")

        price_delta = price - entry_price
        if position.user_input.action == "S":
            price_delta = -price_delta
        absolute_pnl = (
            price_delta * quantity - position.user_input.commission_stt_inr
        ).quantize(
            _MONEY_QUANTUM,
            rounding=ROUND_HALF_UP,
        )
        percentage_pnl = (absolute_pnl / invested_capital * 100).quantize(
            _PERCENT_QUANTUM,
            rounding=ROUND_HALF_UP,
        )

        updated_position = position.model_copy(deep=True)
        updated_position.live_market.current_price = price.quantize(
            _MONEY_QUANTUM,
            rounding=ROUND_HALF_UP,
        )
        updated_position.live_market.total_invested_capital = invested_capital
        updated_position.current_valuation.absolute_pnl_inr = absolute_pnl
        updated_position.current_valuation.percentage_pnl = percentage_pnl
        return updated_position
    except Exception as exc:
        return _error_traceback("position_pnl_update_failed", position.user_input.ticker, exc)


def _fetch_news_sync(ticker_name: str, timeout_seconds: float) -> list[NewsHeadline]:
    """Fetch and parse RSS on a worker thread."""
    normalized_ticker = normalize_nse_ticker(ticker_name)
    query = quote_plus(f"{normalized_ticker} stock NSE India")
    url = f"https://news.google.com/rss/search?q={query}"

    response = requests.get(
        url,
        timeout=timeout_seconds,
        verify=certifi.where(),
        headers={"User-Agent": "MudraSenseAI/1.0"},
    )
    response.raise_for_status()
    soup = BeautifulSoup(response.content, "xml")

    headlines: list[NewsHeadline] = []
    for item in soup.find_all("item")[:5]:
        title_node = item.find("title")
        publication_node = item.find("pubDate")
        title = title_node.get_text(" ", strip=True) if title_node else ""
        publication_timestamp = (
            publication_node.get_text(" ", strip=True) if publication_node else ""
        )
        if not title:
            continue
        try:
            publication_timestamp = parsedate_to_datetime(publication_timestamp).isoformat()
        except (TypeError, ValueError, OverflowError):
            pass
        headlines.append(
            {
                "title": title,
                "publication_timestamp": publication_timestamp,
            }
        )

    return headlines


async def fetch_breaking_news(
    ticker_name: str,
    *,
    timeout_seconds: float = 10.0,
) -> list[NewsHeadline] | str:
    """Asynchronously return up to five Google News RSS headlines.

    ``requests`` is run in a worker thread so this async API does not block the
    caller's event loop. Failures are returned as structured JSON strings.
    """
    normalized_ticker = ticker_name
    try:
        normalized_ticker = normalize_nse_ticker(ticker_name)
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be greater than zero")
        return await asyncio.to_thread(_fetch_news_sync, normalized_ticker, timeout_seconds)
    except Exception as exc:
        return _error_traceback("news_fetch_failed", normalized_ticker, exc)
