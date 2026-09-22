"""LangGraph-compatible qualitative analysis nodes for open NSE positions.

The agents only interpret already-calculated values and text. They never
calculate PnL, infer missing prices, or mutate numeric portfolio fields.
"""

from __future__ import annotations

import json
import os
import traceback
from collections.abc import Mapping, Sequence
from datetime import datetime
from decimal import Decimal
from typing import Protocol, TypedDict

from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, ConfigDict, Field
from dotenv import load_dotenv

from market_tools import MarketMovementMetrics, NewsHeadline
from schemas.positions import AgentEvaluation, OpenTradePosition


load_dotenv()


class TechnicalIndicators(TypedDict, total=False):
    """Optional indicators calculated by the deterministic market layer."""

    volatility_index: Decimal | None
    rsi_14: Decimal | None
    moving_average_20: Decimal | None


class AnalysisState(TypedDict, total=False):
    """Shared state passed between the two analysis nodes."""

    position: OpenTradePosition
    live_market: MarketMovementMetrics
    technical_indicators: TechnicalIndicators
    news_headlines: list[NewsHeadline]
    technical_summary: str
    news_analysis: str
    errors: list[str]


class SupervisorState(TypedDict, total=False):
    """Portfolio state consumed by the autonomous guardrail branch."""

    positions: list[OpenTradePosition]
    market_data: dict[str, MarketMovementMetrics]
    technical_indicators_by_ticker: dict[str, TechnicalIndicators]
    loss_threshold_percent: Decimal
    intraday_drop_threshold_percent: Decimal
    guardrail_triggered: bool
    risk_tickers: list[str]
    risk_reasons: list[str]
    risk_alert: "RiskAlertPayload"
    webhook_result: str
    errors: list[str]


class RiskAlertPayload(BaseModel):
    """Emergency alert drafted by the risk-control branch."""

    model_config = ConfigDict(extra="forbid")

    alert_type: str = Field(default="OPEN_POSITION_RISK")
    tickers: list[str] = Field(min_length=1)
    total_current_losses_inr: Decimal = Field(ge=0)
    reasoning: str = Field(min_length=1)
    created_at: str


class ChatBackend(Protocol):
    """Minimal provider contract implemented by Groq and Ollama adapters."""

    def complete(self, system_prompt: str, user_prompt: str) -> str:
        """Return one qualitative completion."""
        ...


class GroqBackend:
    """Groq chat adapter using the configured API key from the environment."""

    def __init__(self, model: str | None = None) -> None:
        from groq import Groq

        self._client = Groq()
        self._model = model or os.getenv("GROQ_MODEL", "llama3-70b-8192")

    def complete(self, system_prompt: str, user_prompt: str) -> str:
        response = self._client.chat.completions.create(
            model=self._model,
            temperature=0,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )
        content = response.choices[0].message.content
        if not content or not content.strip():
            raise ValueError("Groq returned an empty completion")
        return content.strip()


class OllamaBackend:
    """Local Ollama chat adapter; no cloud API key is required."""

    def __init__(self, model: str | None = None) -> None:
        from ollama import Client

        self._client = Client(host=os.getenv("OLLAMA_HOST", "http://127.0.0.1:11434"))
        self._model = model or os.getenv("OLLAMA_MODEL", "llama3:70b")

    def complete(self, system_prompt: str, user_prompt: str) -> str:
        response = self._client.chat(
            model=self._model,
            options={"temperature": 0},
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )
        if not isinstance(response, Mapping):
            raise TypeError("Ollama returned an unexpected response")
        message = response.get("message")
        if not isinstance(message, Mapping):
            raise TypeError("Ollama response did not contain a message")
        content = message.get("content")
        if not isinstance(content, str) or not content.strip():
            raise ValueError("Ollama returned an empty completion")
        return content.strip()


_TECHNICAL_SYSTEM_PROMPT = """You are the Technical Analyst for an Indian NSE equity monitor.
Write exactly two concise sentences about immediate momentum and whether the
stock appears overbought, oversold, or neither in the Indian market context.
Use only the supplied observations and indicators. INR means Indian rupees.
You are forbidden to calculate, guess, forecast, or alter any stock price,
percentage, PnL, quantity, or other numeric financial value. Do not invent
indicators or facts. If evidence is insufficient, say so explicitly. Return
plain text only, with no bullets, JSON, or headings."""

_NEWS_SYSTEM_PROMPT = """You are the Corporate News and Sentiment Analyst for an Indian NSE equity monitor.
Review the supplied Google News RSS headlines and write a concise qualitative
summary of material corporate developments. Filter generic market noise and
prioritize SEBI or other regulatory filings, earnings reports, material
corporate actions, and governance updates affecting the named stock.
You are forbidden to calculate, guess, forecast, or alter any stock price,
percentage, PnL, quantity, or other numeric financial value. Use only supplied
headlines, distinguish reported facts from uncertainty, and do not invent
news. Return plain text only, with no bullets, JSON, or headings."""

_RISK_CONTROL_SYSTEM_PROMPT = """You are the Risk Control Agent for an Indian NSE portfolio monitor.
Draft a concise emergency risk-warning explanation from the supplied,
authoritative trigger reasons. Mention the affected tickers and explain why
human review is urgent. INR means Indian rupees.
You are forbidden to calculate, guess, forecast, or alter any stock price,
percentage, PnL, quantity, loss total, threshold, or other numeric financial
value. Do not invent a stop-loss or trading action. Return plain text only,
with no bullets, JSON, or headings."""


def _format_context(value: object) -> str:
    """Serialize typed context without converting Decimal values to floats."""
    return json.dumps(value, default=str, ensure_ascii=True, indent=2)


def _node_error(node_name: str, ticker: str, exc: Exception) -> str:
    """Represent a failed agent call as a structured state error."""
    return json.dumps(
        {
            "error": node_name,
            "ticker": ticker,
            "exception_type": type(exc).__name__,
            "message": str(exc),
            "traceback": traceback.format_exc(),
        },
        ensure_ascii=True,
    )


def technical_analyst_node(
    state: AnalysisState,
    *,
    backend: ChatBackend,
) -> AnalysisState:
    """Analyze supplied market movement and indicators without numeric edits."""
    ticker = state["position"].user_input.ticker
    try:
        user_prompt = (
            f"Stock: {ticker}\n"
            "The following values are authoritative read-only observations. "
            "Do not recompute or change them.\n"
            f"Live calculated price data: {_format_context(state['live_market'])}\n"
            f"Basic indicators: {_format_context(state.get('technical_indicators', {}))}"
        )
        return {"technical_summary": backend.complete(_TECHNICAL_SYSTEM_PROMPT, user_prompt)}
    except Exception as exc:
        return {"errors": [_node_error("technical_agent_failed", ticker, exc)]}


def corporate_news_node(
    state: AnalysisState,
    *,
    backend: ChatBackend,
) -> AnalysisState:
    """Filter supplied RSS headlines for material corporate developments."""
    ticker = state["position"].user_input.ticker
    try:
        user_prompt = (
            f"Stock: {ticker}\n"
            "These are the only headlines you may use:\n"
            f"{_format_context(state.get('news_headlines', []))}"
        )
        return {"news_analysis": backend.complete(_NEWS_SYSTEM_PROMPT, user_prompt)}
    except Exception as exc:
        return {"errors": [_node_error("news_agent_failed", ticker, exc)]}


def supervisor_guardrail_node(state: SupervisorState) -> SupervisorState:
    """Inspect every open position and deterministically decide risk routing."""
    try:
        loss_threshold = state.get("loss_threshold_percent", Decimal("-5.00"))
        intraday_drop_threshold = state.get(
            "intraday_drop_threshold_percent",
            Decimal("-3.00"),
        )
        if loss_threshold >= 0 or intraday_drop_threshold >= 0:
            raise ValueError("risk thresholds must be negative percentages")

        market_data = state.get("market_data", {})
        indicators_by_ticker = state.get("technical_indicators_by_ticker", {})
        risk_tickers: list[str] = []
        risk_reasons: list[str] = []
        for position in state.get("positions", []):
            ticker = position.user_input.ticker
            pnl_percent = position.current_valuation.percentage_pnl
            if pnl_percent <= loss_threshold:
                if ticker not in risk_tickers:
                    risk_tickers.append(ticker)
                risk_reasons.append(
                    f"{ticker} position PnL is {pnl_percent}% which is at or below "
                    f"the {loss_threshold}% loss threshold."
                )

            snapshot = market_data.get(ticker)
            if snapshot is None:
                continue
            intraday_change = snapshot["net_change_percent"]
            vix_value = indicators_by_ticker.get(ticker, {}).get("volatility_index")
            high_vix_drop = (
                intraday_change < 0
                and vix_value is not None
                and vix_value >= Decimal("20.00")
            )
            if intraday_change <= intraday_drop_threshold or high_vix_drop:
                if ticker not in risk_tickers:
                    risk_tickers.append(ticker)
                volatility_detail = (
                    f" with India VIX at {vix_value}"
                    if high_vix_drop
                    else ""
                )
                risk_reasons.append(
                    f"{ticker} shows a {intraday_change}% intraday decline"
                    f"{volatility_detail}, breaching the market-volatility guardrail."
                )

        return {
            "guardrail_triggered": bool(risk_reasons),
            "risk_tickers": risk_tickers,
            "risk_reasons": risk_reasons,
        }
    except Exception as exc:
        return {
            "guardrail_triggered": False,
            "errors": [_node_error("supervisor_guardrail_failed", "portfolio", exc)],
        }


def risk_control_agent_node(
    state: SupervisorState,
    *,
    backend: ChatBackend,
) -> SupervisorState:
    """Draft an emergency payload from supervisor-owned numeric facts."""
    try:
        if not state.get("guardrail_triggered"):
            return {}

        positions = state.get("positions", [])
        losses = [
            position.current_valuation.absolute_pnl_inr
            for position in positions
            if position.current_valuation.absolute_pnl_inr < 0
        ]
        total_current_losses = -sum(losses, Decimal("0.00"))
        tickers = state.get("risk_tickers", [])
        if not tickers:
            raise ValueError("guardrail triggered without affected tickers")

        reasoning = backend.complete(
            _RISK_CONTROL_SYSTEM_PROMPT,
            "Authoritative supervisor facts; do not recalculate or modify them.\n"
            f"Affected tickers: {_format_context(tickers)}\n"
            f"Total current losses in INR: {total_current_losses}\n"
            f"Trigger reasons: {_format_context(state.get('risk_reasons', []))}",
        )
        alert = RiskAlertPayload(
            tickers=tickers,
            total_current_losses_inr=total_current_losses,
            reasoning=reasoning,
            created_at=datetime.now().astimezone().isoformat(),
        )
        return {"risk_alert": alert}
    except Exception as exc:
        return {"errors": [_node_error("risk_control_agent_failed", "portfolio", exc)]}


def mock_webhook_broadcast(payload: RiskAlertPayload) -> str:
    """Simulate broadcasting an urgent alert to Discord or Telegram."""
    banner = (
        "\n!!! URGENT RISK ALERT !!!\n"
        f"Ticker(s): {', '.join(payload.tickers)}\n"
        f"Total current losses (INR): {payload.total_current_losses_inr}\n"
        f"Reason: {payload.reasoning}\n"
        "!!! END RISK ALERT !!!"
    )
    print(banner)
    return "mock_webhook_broadcasted"


def _risk_route(state: SupervisorState) -> str:
    """Choose the risk branch only when the supervisor found a trigger."""
    return "risk_control_agent" if state.get("guardrail_triggered") else END


def build_supervised_analysis_graph(
    *,
    risk_backend: ChatBackend,
    loss_threshold_percent: Decimal = Decimal("-5.00"),
    intraday_drop_threshold_percent: Decimal = Decimal("-3.00"),
):
    """Build a portfolio graph with conditional emergency risk routing."""
    if loss_threshold_percent >= 0 or intraday_drop_threshold_percent >= 0:
        raise ValueError("risk thresholds must be negative percentages")

    def configured_supervisor(state: SupervisorState) -> SupervisorState:
        configured_state = {
            **state,
            "loss_threshold_percent": loss_threshold_percent,
            "intraday_drop_threshold_percent": intraday_drop_threshold_percent,
        }
        return supervisor_guardrail_node(configured_state)

    graph = StateGraph(SupervisorState)
    graph.add_node("supervisor_guardrail", configured_supervisor)
    graph.add_node(
        "risk_control_agent",
        lambda state: risk_control_agent_node(state, backend=risk_backend),
    )
    graph.add_node(
        "mock_webhook",
        lambda state: {
            "webhook_result": mock_webhook_broadcast(state["risk_alert"])
        }
        if "risk_alert" in state
        else {},
    )
    graph.add_edge(START, "supervisor_guardrail")
    graph.add_conditional_edges(
        "supervisor_guardrail",
        _risk_route,
        {"risk_control_agent": "risk_control_agent", END: END},
    )
    graph.add_edge("risk_control_agent", "mock_webhook")
    graph.add_edge("mock_webhook", END)
    return graph.compile()


def build_analysis_graph(
    *,
    technical_backend: ChatBackend,
    news_backend: ChatBackend,
):
    """Build a sequential LangGraph: technical analysis then news analysis."""
    graph = StateGraph(AnalysisState)
    graph.add_node(
        "technical_analyst",
        lambda state: technical_analyst_node(state, backend=technical_backend),
    )
    graph.add_node(
        "corporate_news",
        lambda state: corporate_news_node(state, backend=news_backend),
    )
    graph.add_edge(START, "technical_analyst")
    graph.add_edge("technical_analyst", "corporate_news")
    graph.add_edge("corporate_news", END)
    return graph.compile()


def apply_agent_summaries(
    position: OpenTradePosition,
    *,
    technical_summary: str,
    news_analysis: str,
) -> OpenTradePosition:
    """Copy qualitative outputs into the position while preserving all numbers."""
    updated = position.model_copy(deep=True)
    updated.agent_evaluation = AgentEvaluation(
        technical_sentiment_summary=technical_summary,
        breaking_news_analysis=news_analysis,
        risk_tier=position.agent_evaluation.risk_tier,
    )
    return updated
