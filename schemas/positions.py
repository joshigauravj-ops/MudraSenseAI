"""Schemas for open NSE trade positions.

The models in this module describe state received from market-data and agent
layers. They do not calculate PnL or run any agent workflow.
"""

from datetime import date
from decimal import Decimal
from typing import Literal, TypedDict

from pydantic import BaseModel, ConfigDict, Field, field_validator


RiskTier = Literal["Low", "Medium", "High"]
TradeAction = Literal["B", "S"]


class UserInputPositionData(TypedDict):
    """JSON-compatible shape for the user's original trade input."""

    ticker: str
    entry_price: Decimal
    quantity: int
    transaction_date: date
    target_price: Decimal | None
    target_profit_inr: Decimal | None
    target_holding: str | None
    strike_price: Decimal | None
    action: TradeAction
    commission_stt_inr: Decimal
    reported_pnl_inr: Decimal | None


class LiveMarketLayerData(TypedDict):
    """JSON-compatible shape for the current market snapshot."""

    current_price: Decimal
    days_high: Decimal
    days_low: Decimal
    volatility_index: Decimal | None
    net_change_percent: Decimal
    total_invested_capital: Decimal


class CurrentValuationData(TypedDict):
    """JSON-compatible shape for the deterministic valuation output."""

    absolute_pnl_inr: Decimal
    percentage_pnl: Decimal


class AgentEvaluationData(TypedDict):
    """JSON-compatible shape for qualitative agent output."""

    technical_sentiment_summary: str
    breaking_news_analysis: str
    risk_tier: RiskTier


class UserInputPosition(BaseModel):
    """User-supplied details for one open position."""

    model_config = ConfigDict(extra="forbid")

    ticker: str = Field(description="Canonical NSE ticker, for example RELIANCE.NS")
    entry_price: Decimal = Field(gt=0, description="Entry price in INR")
    quantity: int = Field(gt=0, description="Number of shares")
    transaction_date: date
    target_price: Decimal | None = Field(default=None, gt=0, description="User target price in INR")
    target_profit_inr: Decimal | None = Field(
        default=None,
        description="User target profit in INR",
    )
    target_holding: str | None = Field(
        default=None,
        description="User holding horizon, for example 30D or long-term",
    )
    strike_price: Decimal | None = Field(default=None, gt=0, description="Optional strike price in INR")
    action: TradeAction = Field(default="B", description="B for buy or S for sell")
    commission_stt_inr: Decimal = Field(
        default=Decimal("0.00"),
        ge=0,
        description="Brokerage, commission, and STT in INR",
    )
    reported_pnl_inr: Decimal | None = Field(
        default=None,
        description="User-reported PnL; never used instead of calculated PnL",
    )

    @field_validator("ticker", mode="before")
    @classmethod
    def normalize_nse_ticker(cls, value: object) -> str:
        """Normalize a bare symbol to the NSE Yahoo Finance suffix format."""
        if not isinstance(value, str) or not value.strip():
            raise ValueError("ticker must be a non-empty string")

        ticker = value.strip().upper()
        symbol = ticker.split(".", maxsplit=1)[0]
        return f"{symbol}.NS"


class LiveMarketLayer(BaseModel):
    """Latest market values associated with the open position."""

    model_config = ConfigDict(extra="forbid")

    current_price: Decimal = Field(gt=0, description="Current price in INR")
    days_high: Decimal = Field(gt=0, description="Current trading day's high in INR")
    days_low: Decimal = Field(gt=0, description="Current trading day's low in INR")
    volatility_index: Decimal | None = Field(
        default=None,
        ge=0,
        description="NSE India VIX value when available",
    )
    net_change_percent: Decimal
    total_invested_capital: Decimal = Field(
        gt=0,
        description="Entry price multiplied by quantity, in INR",
    )


class CurrentValuation(BaseModel):
    """Deterministic valuation values calculated by the Python layer."""

    model_config = ConfigDict(extra="forbid")

    absolute_pnl_inr: Decimal = Field(description="Absolute PnL in INR")
    percentage_pnl: Decimal = Field(description="PnL as a percentage")


class AgentEvaluation(BaseModel):
    """Qualitative analysis produced after the numeric layers are available."""

    model_config = ConfigDict(extra="forbid")

    technical_sentiment_summary: str = Field(min_length=1)
    breaking_news_analysis: str = Field(min_length=1)
    risk_tier: RiskTier


class OpenTradePosition(BaseModel):
    """Complete typed state for one open Indian equity position."""

    model_config = ConfigDict(extra="forbid")

    user_input: UserInputPosition
    live_market: LiveMarketLayer
    current_valuation: CurrentValuation
    agent_evaluation: AgentEvaluation


class OpenTradePositions(BaseModel):
    """Validated collection matching the sample positions JSON document."""

    model_config = ConfigDict(extra="forbid")

    positions: list[OpenTradePosition] = Field(min_length=1)


class OpenTradePositionData(TypedDict):
    """Top-level raw payload shape used before Pydantic validation."""

    user_input: UserInputPositionData
    live_market: LiveMarketLayerData
    current_valuation: CurrentValuationData
    agent_evaluation: AgentEvaluationData
