"""Market-data and deterministic valuation helpers for NSE positions.

This module contains no agent or LLM calls. Numeric calculations are performed
with :class:`decimal.Decimal` before any qualitative layer consumes the state.
"""

from __future__ import annotations

import asyncio
import json
import math
import re
import time
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

from schemas.positions import OpenTradePosition, PositionTrigger


_NSE_SYMBOL_PATTERN = re.compile(r"^[A-Z0-9&-]+$")
_MONEY_QUANTUM = Decimal("0.01")
_PERCENT_QUANTUM = Decimal("0.01")
_IST = ZoneInfo("Asia/Kolkata")
_NSE_OPEN = time(9, 15)
_NSE_CLOSE = time(15, 30)

MarketSessionStatus = Literal["Live Intraday", "Post-Market Close"]


class PriceBar(TypedDict):
    """Validated OHLCV candle for chart rendering."""

    timestamp: str
    open: float
    high: float
    low: float
    close: float
    volume: int


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


def _build_market_metric_payload(
    ticker: str,
    history: object,
) -> MarketMovementMetrics:
    """Convert a yfinance history payload into the standardized market metrics format."""
    if history is None or getattr(history, "empty", True):
        raise TimeoutError(f"no market data returned for {ticker}")

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
        "ticker": ticker,
        "current_price": current_price.quantize(_MONEY_QUANTUM, rounding=ROUND_HALF_UP),
        "days_high": days_high.quantize(_MONEY_QUANTUM, rounding=ROUND_HALF_UP),
        "days_low": days_low.quantize(_MONEY_QUANTUM, rounding=ROUND_HALF_UP),
        "net_change_percent": net_change_percent,
        "captured_at": captured_at.isoformat(),
        "market_session": get_market_session_status(captured_at),
    }


def fetch_market_movement(
    ticker: str,
    *,
    timeout_seconds: float = 10.0,
    max_attempts: int = 3,
    retry_delay_seconds: float = 1.0,
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
        if max_attempts < 1:
            raise ValueError("max_attempts must be at least one")
        if retry_delay_seconds < 0:
            raise ValueError("retry_delay_seconds cannot be negative")
    except Exception as exc:
        return _error_traceback("market_data_fetch_failed", normalized_ticker, exc)

    last_error: Exception | None = None
    for attempt in range(max_attempts):
        try:
            history = yf.Ticker(normalized_ticker).history(
                period="2d",
                interval="1d",
                auto_adjust=False,
                timeout=timeout_seconds,
            )
            return _build_market_metric_payload(normalized_ticker, history)
        except Exception as exc:
            last_error = exc
            if attempt + 1 < max_attempts:
                time.sleep(retry_delay_seconds)

    return _error_traceback(
        "market_data_fetch_failed",
        normalized_ticker,
        last_error or RuntimeError("unknown market data failure"),
    )


def fetch_market_movements_batch(
    tickers: list[str],
    *,
    timeout_seconds: float = 10.0,
    max_attempts: int = 3,
    retry_delay_seconds: float = 1.0,
) -> dict[str, MarketMovementMetrics | str]:
    """Fetch multiple ticker metrics in one bulk yfinance request when possible.

    This lowers network chatter and reduces the number of active threads for large
    portfolios by reusing a single provider batch call instead of one call per ticker.
    """
    normalized_tickers: list[str] = []
    errors: dict[str, str] = {}
    for raw_ticker in tickers:
        try:
            normalized = normalize_nse_ticker(raw_ticker)
        except Exception as exc:
            errors[raw_ticker] = _error_traceback("market_data_fetch_failed", raw_ticker, exc)
            continue
        if normalized not in normalized_tickers:
            normalized_tickers.append(normalized)

    if not normalized_tickers:
        return errors

    batch: dict[str, MarketMovementMetrics | str] = {}
    for attempt in range(max_attempts):
        try:
            bundle = yf.Tickers(" ".join(normalized_tickers))
            for ticker in normalized_tickers:
                try:
                    history = bundle.tickers[ticker].history(
                        period="2d",
                        interval="1d",
                        auto_adjust=False,
                        timeout=timeout_seconds,
                    )
                    batch[ticker] = _build_market_metric_payload(ticker, history)
                except Exception as exc:
                    batch[ticker] = _error_traceback("market_data_fetch_failed", ticker, exc)
            return batch
        except Exception as exc:
            if attempt + 1 < max_attempts:
                time.sleep(retry_delay_seconds)
                continue
            for ticker in normalized_tickers:
                batch[ticker] = _error_traceback("market_data_fetch_failed", ticker, exc)
            return batch

    return batch


def fetch_price_history(
    ticker: str,
    *,
    period: str = "1mo",
    interval: str = "1d",
    timeout_seconds: float = 10.0,
) -> list[PriceBar] | str:
    """Fetch validated OHLCV bars for a ticker chart."""
    normalized_ticker = ticker
    try:
        normalized_ticker = normalize_nse_ticker(ticker)
        if period not in {"1d", "5d", "1mo", "3mo", "6mo", "1y", "2y", "5y"}:
            raise ValueError("unsupported chart period")
        if interval not in {"1m", "2m", "5m", "15m", "30m", "60m", "90m", "1h", "1d", "1wk"}:
            raise ValueError("unsupported chart interval")
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be greater than zero")

        history = yf.Ticker(normalized_ticker).history(
            period=period,
            interval=interval,
            auto_adjust=False,
            timeout=timeout_seconds,
        )
        if history is None or history.empty:
            raise TimeoutError(f"no price history returned for {normalized_ticker}")

        bars: list[PriceBar] = []
        for timestamp, row in history.iterrows():
            try:
                open_price, high_price, low_price, close_price = (
                    float(row[column]) for column in ("Open", "High", "Low", "Close")
                )
            except (KeyError, TypeError, ValueError):
                continue
            if not all(math.isfinite(value) for value in (open_price, high_price, low_price, close_price)):
                continue
            volume_value = row.get("Volume", 0)
            volume = int(volume_value) if math.isfinite(float(volume_value or 0)) else 0
            bars.append(
                {
                    "timestamp": timestamp.isoformat(),
                    "open": open_price,
                    "high": high_price,
                    "low": low_price,
                    "close": close_price,
                    "volume": volume,
                }
            )
        if not bars:
            raise ValueError(f"no complete OHLC bars returned for {normalized_ticker}")
        return bars
    except Exception as exc:
        return _error_traceback("price_history_fetch_failed", normalized_ticker, exc)


def evaluate_position_trigger(position: OpenTradePosition) -> PositionTrigger:
    """Evaluate target and stop-loss conditions using authoritative runtime values."""
    price = position.live_market.current_price
    user_input = position.user_input
    stop_loss_hit = (
        user_input.stop_loss_price is not None
        and (price <= user_input.stop_loss_price if user_input.action == "B" else price >= user_input.stop_loss_price)
    )
    target_price_hit = (
        user_input.target_price is not None
        and (price >= user_input.target_price if user_input.action == "B" else price <= user_input.target_price)
    )
    target_profit_hit = (
        user_input.target_profit_percent is not None
        and position.current_valuation.percentage_pnl >= user_input.target_profit_percent
    )
    if stop_loss_hit:
        return PositionTrigger(
            status="Stop Loss Hit",
            reason=f"Price INR {price:.2f} reached the stop-loss level.",
        )
    if target_price_hit or target_profit_hit:
        reason = "Target price reached." if target_price_hit else "Target profit reached."
        return PositionTrigger(status="Target Hit", reason=reason)
    return PositionTrigger()


def calculate_target_price(position: OpenTradePosition) -> Decimal | None:
    """Derive a target price from target PnL percentage when no price is supplied."""
    target_percent = position.user_input.target_profit_percent
    if target_percent is None or position.user_input.target_price is not None:
        return position.user_input.target_price

    invested_capital = (
        position.user_input.entry_price * Decimal(position.user_input.quantity)
    ).quantize(_MONEY_QUANTUM, rounding=ROUND_HALF_UP)
    target_profit = (invested_capital * target_percent / Decimal("100")).quantize(
        _MONEY_QUANTUM,
        rounding=ROUND_HALF_UP,
    )
    price_delta = (
        (target_profit + position.user_input.commission_stt_inr)
        / Decimal(position.user_input.quantity)
    ).quantize(_MONEY_QUANTUM, rounding=ROUND_HALF_UP)
    target_price = (
        position.user_input.entry_price + price_delta
        if position.user_input.action == "B"
        else position.user_input.entry_price - price_delta
    ).quantize(_MONEY_QUANTUM, rounding=ROUND_HALF_UP)
    if target_price <= 0:
        raise ValueError("target profit percentage produces a non-positive target price")
    return target_price


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


async def fetch_breaking_news_batch(
    tickers: list[str],
    *,
    timeout_seconds: float = 10.0,
    max_concurrent: int = 5,
) -> dict[str, list[NewsHeadline] | str]:
    """Fetch RSS headlines for multiple tickers with bounded concurrency.

    This keeps the news pipeline efficient without spawning unbounded threads for
    every position in a large watchlist.
    """
    if not tickers:
        return {}

    semaphore = asyncio.Semaphore(max(1, max_concurrent))

    async def fetch_one(ticker: str):
        async with semaphore:
            return await fetch_breaking_news(ticker, timeout_seconds=timeout_seconds)

    results = await asyncio.gather(
        *(fetch_one(ticker) for ticker in tickers),
        return_exceptions=True,
    )

    payload: dict[str, list[NewsHeadline] | str] = {}
    for ticker, result in zip(tickers, results):
        if isinstance(result, Exception):
            payload[ticker] = _error_traceback("news_fetch_failed", ticker, result)
        else:
            payload[ticker] = result
    return payload
