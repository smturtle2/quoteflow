from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Mapping


ModelSide = Literal["bid", "ask"]
AggressorSide = Literal["buy", "sell"]
SessionPhase = Literal["open", "mid", "close"]
ParticipantType = Literal["passive_lp", "inventory_mm", "noise_taker", "informed_meta"]
ShockName = Literal["fair_jump", "vol_burst", "liquidity_drought", "one_sided_taker_surge"]


@dataclass(frozen=True)
class MetaOrderState:
    id: int
    side: AggressorSide
    initial_qty: float
    remaining_qty: float
    urgency: float
    decay_half_life: int
    age: int = 0


@dataclass(frozen=True)
class ShockState:
    name: ShockName
    intensity: float
    remaining_steps: int
    side: str | None = None


@dataclass(frozen=True)
class EngineContext:
    session_phase: SessionPhase
    session_progress: float
    seasonality: Mapping[str, float]
    hidden_vol: float
    excitation: Mapping[str, float]
    burst_state: str
    shock: ShockState | None
    meta_orders: Mapping[AggressorSide, MetaOrderState | None]
    spread_excess: float
    best_depth_deficit_bid: float
    best_depth_deficit_ask: float
    imbalance_displacement: float
    directional_anchor: float


PARTICIPANT_TYPES: tuple[ParticipantType, ...] = (
    "passive_lp",
    "inventory_mm",
    "noise_taker",
    "informed_meta",
)

EXCITATION_KEYS: tuple[str, ...] = (
    "market_buy",
    "market_sell",
    "cancel_bid_near",
    "cancel_ask_near",
    "limit_bid_near",
    "limit_ask_near",
)

SHOCK_NAMES: tuple[ShockName, ...] = (
    "fair_jump",
    "vol_burst",
    "liquidity_drought",
    "one_sided_taker_surge",
)
