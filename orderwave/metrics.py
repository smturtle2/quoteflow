from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from orderwave.book import OrderBook
from orderwave.utils import EPSILON, clamp, tick_to_price


@dataclass(frozen=True)
class MarketFeatures:
    mid_tick: float
    mid_price: float
    microprice: float
    spread_ticks: int
    spread_price: float
    depth_imbalance: float
    recent_flow_imbalance: float
    recent_return: float
    trade_strength: float
    near_best_depth_ratio: float
    rolling_volatility: float
    buy_aggr_volume: float
    sell_aggr_volume: float
    best_bid_qty: int
    best_ask_qty: int
    thin_bid_best: float
    thin_ask_best: float
    top_bid_depth: int
    top_ask_depth: int
    signed_flow: float
    realized_vol: float


def compute_features(
    book: OrderBook,
    *,
    tick_size: float,
    depth_levels: int,
    buy_flow: Sequence[float],
    sell_flow: Sequence[float],
    mid_returns: Sequence[float],
    buy_exec_ema: float | None = None,
    sell_exec_ema: float | None = None,
) -> MarketFeatures:
    if book.best_bid_tick is None or book.best_ask_tick is None:
        raise ValueError("book must contain both bid and ask liquidity")

    best_bid_price = tick_to_price(book.best_bid_tick, tick_size)
    best_ask_price = tick_to_price(book.best_ask_tick, tick_size)
    mid_tick = (book.best_bid_tick + book.best_ask_tick) / 2.0
    mid_price = tick_to_price(mid_tick, tick_size)

    best_bid_qty = book.best_qty("bid")
    best_ask_qty = book.best_qty("ask")
    microprice = _compute_microprice(best_bid_price, best_ask_price, best_bid_qty, best_ask_qty, tick_size)

    bids = book.top_levels("bid", depth_levels)
    asks = book.top_levels("ask", depth_levels)
    top_bid_depth = sum(qty for _, qty in bids)
    top_ask_depth = sum(qty for _, qty in asks)
    depth_imbalance = _safe_balance(top_bid_depth, top_ask_depth)

    buy_aggr_volume = float(sum(buy_flow))
    sell_aggr_volume = float(sum(sell_flow))
    signed_flow = _safe_balance(buy_aggr_volume, sell_aggr_volume)
    trade_strength = _safe_balance(
        buy_exec_ema if buy_exec_ema is not None else buy_aggr_volume,
        sell_exec_ema if sell_exec_ema is not None else sell_aggr_volume,
    )

    near_best_depth_ratio = float(
        clamp(
            (best_bid_qty + best_ask_qty) / max(top_bid_depth + top_ask_depth, EPSILON),
            0.0,
            1.0,
        )
    )

    avg_bid_depth = top_bid_depth / max(len(bids), 1)
    avg_ask_depth = top_ask_depth / max(len(asks), 1)
    thin_bid_best = float(clamp(1.0 - (best_bid_qty / max(avg_bid_depth, 1.0)), 0.0, 1.0))
    thin_ask_best = float(clamp(1.0 - (best_ask_qty / max(avg_ask_depth, 1.0)), 0.0, 1.0))

    recent_return = float(mid_returns[-1]) if mid_returns else 0.0
    realized_vol = float(np.std(mid_returns)) if len(mid_returns) > 1 else 0.0

    return MarketFeatures(
        mid_tick=mid_tick,
        mid_price=mid_price,
        microprice=microprice,
        spread_ticks=book.spread_ticks,
        spread_price=tick_to_price(book.spread_ticks, tick_size),
        depth_imbalance=depth_imbalance,
        recent_flow_imbalance=signed_flow,
        recent_return=recent_return,
        trade_strength=trade_strength,
        near_best_depth_ratio=near_best_depth_ratio,
        rolling_volatility=realized_vol,
        buy_aggr_volume=buy_aggr_volume,
        sell_aggr_volume=sell_aggr_volume,
        best_bid_qty=best_bid_qty,
        best_ask_qty=best_ask_qty,
        thin_bid_best=thin_bid_best,
        thin_ask_best=thin_ask_best,
        top_bid_depth=top_bid_depth,
        top_ask_depth=top_ask_depth,
        signed_flow=signed_flow,
        realized_vol=realized_vol,
    )


def _compute_microprice(
    best_bid_price: float,
    best_ask_price: float,
    best_bid_qty: int,
    best_ask_qty: int,
    tick_size: float,
) -> float:
    total_qty = best_bid_qty + best_ask_qty
    if total_qty <= 0:
        return (best_bid_price + best_ask_price) / 2.0
    microprice = ((best_ask_price * best_bid_qty) + (best_bid_price * best_ask_qty)) / total_qty
    return tick_to_price(microprice / tick_size, tick_size)


def _safe_balance(left: float, right: float) -> float:
    return float((left - right) / max(left + right, EPSILON))
