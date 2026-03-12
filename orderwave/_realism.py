from __future__ import annotations

"""Internal realism metrics for aggregate order-book runs."""

from dataclasses import asdict, dataclass, fields
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Sequence

    from orderwave.market import Market


_RANK_COUNT = 5
_IMPACT_HORIZONS = (1, 2, 5, 10)
_REGIME_TIGHT = 0
_REGIME_FRAGILE = 2


@dataclass(frozen=True)
class RealismProfile:
    seed: int | None
    steps: int
    one_tick_spread_share: float
    wide_spread_share: float
    spread_acf1: float
    trade_sign_acf1: float
    same_step_impact_corr: float
    next_step_impact_corr: float
    bid_gap_gt1_share_top5: float
    ask_gap_gt1_share_top5: float
    spread_recovery_median: float
    bid_rank_mean_depth: tuple[float, ...]
    ask_rank_mean_depth: tuple[float, ...]
    bid_rank_std_depth: tuple[float, ...]
    ask_rank_std_depth: tuple[float, ...]
    bid_gap_run_median: float
    ask_gap_run_median: float
    impact_decay_abs: tuple[float, ...]
    buy_shock_cancel_skew: float
    sell_shock_cancel_skew: float
    buy_shock_refill_skew: float
    sell_shock_refill_skew: float
    tight_regime_share: float
    fragile_regime_share: float
    connected_depth_share: float
    isolated_depth_share: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def profile_market_realism(market: Market, *, steps: int) -> RealismProfile:
    bid_gap_flags: list[float] = []
    ask_gap_flags: list[float] = []
    bid_gap_runs: list[int] = []
    ask_gap_runs: list[int] = []
    bid_gap_active = 0
    ask_gap_active = 0
    bid_rank_depths: list[tuple[float, ...]] = []
    ask_rank_depths: list[tuple[float, ...]] = []
    buy_cancel_skews: list[float] = []
    sell_cancel_skews: list[float] = []
    buy_refill_skews: list[float] = []
    sell_refill_skews: list[float] = []
    connected_shares: list[float] = []
    isolated_shares: list[float] = []
    tight_steps = 0
    fragile_steps = 0

    for _ in range(int(steps)):
        market.step()

        bid_levels = market._book.levels("bid", _RANK_COUNT + 1)  # type: ignore[attr-defined]
        ask_levels = market._book.levels("ask", _RANK_COUNT + 1)  # type: ignore[attr-defined]
        bid_rank_depths.append(_rank_depths(bid_levels))
        ask_rank_depths.append(_rank_depths(ask_levels))

        bid_flags = _gap_flags(bid_levels)
        ask_flags = _gap_flags(ask_levels)
        bid_gap_flags.extend(bid_flags)
        ask_gap_flags.extend(ask_flags)
        bid_gap_active = _update_gap_run(bid_gap_runs, bid_gap_active, any(flag > 0.0 for flag in bid_flags))
        ask_gap_active = _update_gap_run(ask_gap_runs, ask_gap_active, any(flag > 0.0 for flag in ask_flags))

        connected, isolated = _connectivity_shares(bid_levels, ask_levels)
        connected_shares.append(connected)
        isolated_shares.append(isolated)

        kernel = market._kernel  # type: ignore[attr-defined]
        if kernel.liquidity_regime == _REGIME_TIGHT:
            tight_steps += 1
        if kernel.liquidity_regime == _REGIME_FRAGILE:
            fragile_steps += 1

        signed_flow_step = float(market._buy_aggr_volume - market._sell_aggr_volume)  # type: ignore[attr-defined]
        if signed_flow_step > 0.0:
            buy_cancel_skews.append(kernel.ask_cancel_pressure - kernel.bid_cancel_pressure)
            buy_refill_skews.append(kernel.bid_refill_lag - kernel.ask_refill_lag)
        elif signed_flow_step < 0.0:
            sell_cancel_skews.append(kernel.bid_cancel_pressure - kernel.ask_cancel_pressure)
            sell_refill_skews.append(kernel.ask_refill_lag - kernel.bid_refill_lag)

    if bid_gap_active > 0:
        bid_gap_runs.append(bid_gap_active)
    if ask_gap_active > 0:
        ask_gap_runs.append(ask_gap_active)

    history = market.get_history()
    spread_ticks = (history["spread"] / market.tick_size).round().to_numpy(dtype=float)
    signed_flow = (history["buy_aggr_volume"] - history["sell_aggr_volume"]).to_numpy(dtype=float)
    trade_sign = np.sign(signed_flow)
    mid_prices = history["mid_price"].to_numpy(dtype=float)
    mid_returns = history["mid_price"].diff().fillna(0.0).to_numpy(dtype=float)

    valid_sign = (trade_sign[1:] != 0.0) & (trade_sign[:-1] != 0.0)
    trade_sign_acf = _corr(trade_sign[1:][valid_sign], trade_sign[:-1][valid_sign]) if valid_sign.any() else 0.0

    bid_rank_array = np.asarray(bid_rank_depths, dtype=float) if bid_rank_depths else np.zeros((0, _RANK_COUNT), dtype=float)
    ask_rank_array = np.asarray(ask_rank_depths, dtype=float) if ask_rank_depths else np.zeros((0, _RANK_COUNT), dtype=float)

    return RealismProfile(
        seed=market.seed,
        steps=int(steps),
        one_tick_spread_share=float(np.mean(spread_ticks == 1.0)),
        wide_spread_share=float(np.mean(spread_ticks >= 4.0)),
        spread_acf1=_autocorr1(spread_ticks),
        trade_sign_acf1=trade_sign_acf,
        same_step_impact_corr=_corr(signed_flow[1:], mid_returns[1:]),
        next_step_impact_corr=_corr(signed_flow[:-1], mid_returns[1:]),
        bid_gap_gt1_share_top5=float(np.mean(bid_gap_flags)) if bid_gap_flags else 0.0,
        ask_gap_gt1_share_top5=float(np.mean(ask_gap_flags)) if ask_gap_flags else 0.0,
        spread_recovery_median=_median_spread_recovery(spread_ticks),
        bid_rank_mean_depth=_tuple_mean(bid_rank_array),
        ask_rank_mean_depth=_tuple_mean(ask_rank_array),
        bid_rank_std_depth=_tuple_std(bid_rank_array),
        ask_rank_std_depth=_tuple_std(ask_rank_array),
        bid_gap_run_median=_median_runs(bid_gap_runs),
        ask_gap_run_median=_median_runs(ask_gap_runs),
        impact_decay_abs=tuple(_impact_corr(mid_prices, signed_flow, horizon) for horizon in _IMPACT_HORIZONS),
        buy_shock_cancel_skew=_safe_mean(buy_cancel_skews),
        sell_shock_cancel_skew=_safe_mean(sell_cancel_skews),
        buy_shock_refill_skew=_safe_mean(buy_refill_skews),
        sell_shock_refill_skew=_safe_mean(sell_refill_skews),
        tight_regime_share=float(tight_steps / max(int(steps), 1)),
        fragile_regime_share=float(fragile_steps / max(int(steps), 1)),
        connected_depth_share=_safe_mean(connected_shares),
        isolated_depth_share=_safe_mean(isolated_shares),
    )


def aggregate_realism_profiles(profiles: Sequence[RealismProfile]) -> RealismProfile:
    if not profiles:
        raise ValueError("profiles must not be empty")

    aggregated: dict[str, Any] = {"seed": None, "steps": int(np.mean([profile.steps for profile in profiles]))}
    for field in fields(RealismProfile):
        if field.name in {"seed", "steps"}:
            continue
        values = [getattr(profile, field.name) for profile in profiles]
        first = values[0]
        if isinstance(first, tuple):
            aggregated[field.name] = tuple(
                float(np.nanmean([float(value[index]) for value in values])) for index in range(len(first))
            )
        else:
            aggregated[field.name] = float(np.nanmean(np.asarray(values, dtype=float)))
    return RealismProfile(**aggregated)


def _rank_depths(levels: tuple[tuple[int, int], ...]) -> tuple[float, ...]:
    depths = [0.0] * _RANK_COUNT
    for index, (_, qty) in enumerate(levels[:_RANK_COUNT]):
        depths[index] = float(qty)
    return tuple(depths)


def _gap_flags(levels: tuple[tuple[int, int], ...]) -> list[float]:
    flags: list[float] = []
    for index in range(min(_RANK_COUNT, max(0, len(levels) - 1))):
        left_tick = levels[index][0]
        right_tick = levels[index + 1][0]
        flags.append(1.0 if abs(left_tick - right_tick) > 1 else 0.0)
    return flags


def _update_gap_run(runs: list[int], active_run: int, has_gap: bool) -> int:
    if has_gap:
        return active_run + 1
    if active_run > 0:
        runs.append(active_run)
    return 0


def _connectivity_shares(
    bid_levels: tuple[tuple[int, int], ...],
    ask_levels: tuple[tuple[int, int], ...],
) -> tuple[float, float]:
    connected = 0
    isolated = 0
    total = 0
    for levels in (bid_levels, ask_levels):
        ticks = [tick for tick, _ in levels[: max(_RANK_COUNT + 1, 6)]]
        tick_set = set(ticks)
        for tick in ticks[2:]:
            total += 1
            if (tick - 1) in tick_set or (tick + 1) in tick_set:
                connected += 1
            else:
                isolated += 1
    if total == 0:
        return 0.0, 0.0
    return connected / float(total), isolated / float(total)


def _tuple_mean(values: np.ndarray) -> tuple[float, ...]:
    if values.size == 0:
        return tuple(0.0 for _ in range(_RANK_COUNT))
    return tuple(float(np.mean(values[:, index])) for index in range(values.shape[1]))


def _tuple_std(values: np.ndarray) -> tuple[float, ...]:
    if values.size == 0:
        return tuple(0.0 for _ in range(_RANK_COUNT))
    return tuple(float(np.std(values[:, index])) for index in range(values.shape[1]))


def _impact_corr(mid_prices: np.ndarray, signed_flow: np.ndarray, horizon: int) -> float:
    if mid_prices.size <= horizon or signed_flow.size <= horizon:
        return 0.0
    future_move = mid_prices[horizon:] - mid_prices[:-horizon]
    response = _corr(signed_flow[:-horizon], future_move)
    return abs(response)


def _median_runs(runs: list[int]) -> float:
    if not runs:
        return 0.0
    return float(np.median(np.asarray(runs, dtype=float)))


def _safe_mean(values: list[float]) -> float:
    if not values:
        return 0.0
    return float(np.mean(np.asarray(values, dtype=float)))


def _autocorr1(values: np.ndarray) -> float:
    if values.size < 2:
        return 0.0
    return _corr(values[1:], values[:-1])


def _corr(lhs: np.ndarray, rhs: np.ndarray) -> float:
    if lhs.size == 0 or rhs.size == 0:
        return 0.0
    if np.std(lhs) <= 1e-12 or np.std(rhs) <= 1e-12:
        return 0.0
    return float(np.corrcoef(lhs, rhs)[0, 1])


def _median_spread_recovery(spread_ticks: np.ndarray) -> float:
    lags: list[int] = []
    in_event = False
    event_start = 0
    for index, spread in enumerate(spread_ticks):
        if not in_event and spread >= 3.0:
            in_event = True
            event_start = index
            continue
        if in_event and spread <= 2.0:
            lags.append(index - event_start)
            in_event = False
    if not lags:
        return float("nan")
    return float(np.median(np.asarray(lags, dtype=float)))


__all__ = ["RealismProfile", "aggregate_realism_profiles", "profile_market_realism"]
