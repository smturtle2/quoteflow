from __future__ import annotations

"""Internal realism metrics for aggregate order-book runs."""

from collections import Counter
from dataclasses import asdict, dataclass, fields
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Sequence

    from orderwave.market import Market


_RANK_COUNT = 5
_IMPACT_HORIZONS = (1, 2, 5, 10)


@dataclass(frozen=True)
class RealismProfile:
    seed: int | None
    steps: int
    normalized_net_drift: float
    up_step_share: float
    down_step_share: float
    variance_ratio_5: float
    variance_ratio_20: float
    one_tick_spread_share: float
    wide_spread_share: float
    spread_acf1: float
    trade_sign_acf1: float
    trade_sign_acf5: float
    same_step_impact_corr: float
    next_step_impact_corr: float
    flow_return_sign_agreement: float
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
    visible_one_side_thin_share: float
    visible_one_vs_many_share: float
    full_book_extreme_share: float
    avg_visible_levels_bid: float
    avg_visible_levels_ask: float
    avg_full_levels_bid: float
    avg_full_levels_ask: float
    near_touch_connectivity_bid: float
    near_touch_connectivity_ask: float
    visible_pair_entropy: float
    full_pair_entropy: float

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
    visible_pairs: Counter[tuple[int, int]] = Counter()
    full_pairs: Counter[tuple[int, int]] = Counter()
    visible_one_side_thin = 0
    visible_one_vs_many = 0
    full_book_extreme = 0
    bid_visible_levels_sum = 0.0
    ask_visible_levels_sum = 0.0
    bid_full_levels_sum = 0.0
    ask_full_levels_sum = 0.0
    bid_connectivity: list[float] = []
    ask_connectivity: list[float] = []

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

        bid_visible = len(market._book.levels("bid", market.levels))  # type: ignore[attr-defined]
        ask_visible = len(market._book.levels("ask", market.levels))  # type: ignore[attr-defined]
        bid_full = market._book.level_count("bid")  # type: ignore[attr-defined]
        ask_full = market._book.level_count("ask")  # type: ignore[attr-defined]
        visible_pairs[(bid_visible, ask_visible)] += 1
        full_pairs[(bid_full, ask_full)] += 1
        bid_visible_levels_sum += bid_visible
        ask_visible_levels_sum += ask_visible
        bid_full_levels_sum += bid_full
        ask_full_levels_sum += ask_full

        if bid_visible <= 1 or ask_visible <= 1:
            visible_one_side_thin += 1
        if (bid_visible <= 1 and ask_visible >= 4) or (ask_visible <= 1 and bid_visible >= 4):
            visible_one_vs_many += 1
        if (bid_full <= 2 and ask_full >= 7) or (ask_full <= 2 and bid_full >= 7):
            full_book_extreme += 1

        bid_connectivity.append(_near_touch_connectivity(bid_levels))
        ask_connectivity.append(_near_touch_connectivity(ask_levels))

    if bid_gap_active > 0:
        bid_gap_runs.append(bid_gap_active)
    if ask_gap_active > 0:
        ask_gap_runs.append(ask_gap_active)

    history = market.get_history()
    spread_ticks = (history["spread"] / market.tick_size).round().to_numpy(dtype=float)
    signed_flow = (history["buy_aggr_volume"] - history["sell_aggr_volume"]).to_numpy(dtype=float)
    trade_sign = np.sign(signed_flow)
    mid_prices = history["mid_price"].to_numpy(dtype=float)
    mid_ticks = mid_prices / market.tick_size
    mid_returns = history["mid_price"].diff().fillna(0.0).to_numpy(dtype=float)

    bid_rank_array = np.asarray(bid_rank_depths, dtype=float) if bid_rank_depths else np.zeros((0, _RANK_COUNT), dtype=float)
    ask_rank_array = np.asarray(ask_rank_depths, dtype=float) if ask_rank_depths else np.zeros((0, _RANK_COUNT), dtype=float)

    return RealismProfile(
        seed=market.seed,
        steps=int(steps),
        normalized_net_drift=_normalized_net_drift(mid_ticks, int(steps)),
        up_step_share=float(np.mean(mid_returns > 0.0)),
        down_step_share=float(np.mean(mid_returns < 0.0)),
        variance_ratio_5=_variance_ratio(mid_ticks, 5),
        variance_ratio_20=_variance_ratio(mid_ticks, 20),
        one_tick_spread_share=float(np.mean(spread_ticks == 1.0)),
        wide_spread_share=float(np.mean(spread_ticks >= 4.0)),
        spread_acf1=_autocorr(spread_ticks, 1),
        trade_sign_acf1=_sign_autocorr(trade_sign, 1),
        trade_sign_acf5=_sign_autocorr(trade_sign, 5),
        same_step_impact_corr=_corr(signed_flow[1:], mid_returns[1:]),
        next_step_impact_corr=_corr(signed_flow[:-1], mid_returns[1:]),
        flow_return_sign_agreement=_sign_agreement(signed_flow[1:], mid_returns[1:]),
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
        visible_one_side_thin_share=float(visible_one_side_thin / max(int(steps), 1)),
        visible_one_vs_many_share=float(visible_one_vs_many / max(int(steps), 1)),
        full_book_extreme_share=float(full_book_extreme / max(int(steps), 1)),
        avg_visible_levels_bid=float(bid_visible_levels_sum / max(int(steps), 1)),
        avg_visible_levels_ask=float(ask_visible_levels_sum / max(int(steps), 1)),
        avg_full_levels_bid=float(bid_full_levels_sum / max(int(steps), 1)),
        avg_full_levels_ask=float(ask_full_levels_sum / max(int(steps), 1)),
        near_touch_connectivity_bid=_safe_mean(bid_connectivity),
        near_touch_connectivity_ask=_safe_mean(ask_connectivity),
        visible_pair_entropy=_normalized_entropy(visible_pairs),
        full_pair_entropy=_normalized_entropy(full_pairs),
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


def _near_touch_connectivity(levels: tuple[tuple[int, int], ...]) -> float:
    if len(levels) <= 1:
        return 0.0
    adjacent = 0.0
    pairs = min(3, len(levels) - 1)
    for index in range(pairs):
        adjacent += 1.0 if abs(levels[index][0] - levels[index + 1][0]) == 1 else 0.0
    return adjacent / float(pairs)


def _update_gap_run(runs: list[int], active_run: int, has_gap: bool) -> int:
    if has_gap:
        return active_run + 1
    if active_run > 0:
        runs.append(active_run)
    return 0


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
    future_move = mid_prices[horizon:] - mid_prices[horizon - 1 : -1]
    response = _corr(signed_flow[:-horizon], future_move)
    return abs(response)


def _sign_agreement(lhs: np.ndarray, rhs: np.ndarray) -> float:
    if lhs.size == 0 or rhs.size == 0:
        return 0.0
    lhs_sign = np.sign(lhs)
    rhs_sign = np.sign(rhs)
    mask = (lhs_sign != 0.0) & (rhs_sign != 0.0)
    if not np.any(mask):
        return 0.0
    return float(np.mean(lhs_sign[mask] == rhs_sign[mask]))


def _median_runs(runs: list[int]) -> float:
    if not runs:
        return 0.0
    return float(np.median(np.asarray(runs, dtype=float)))


def _normalized_entropy(counter: Counter[tuple[int, int]]) -> float:
    total = sum(counter.values())
    if total <= 0 or len(counter) <= 1:
        return 0.0
    probabilities = np.asarray([count / total for count in counter.values()], dtype=float)
    entropy = -float(np.sum(probabilities * np.log(probabilities)))
    return entropy / float(np.log(len(counter)))


def _safe_mean(values: list[float]) -> float:
    if not values:
        return 0.0
    return float(np.mean(np.asarray(values, dtype=float)))


def _normalized_net_drift(mid_ticks: np.ndarray, steps: int) -> float:
    if mid_ticks.size <= 1:
        return 0.0
    return abs(float(mid_ticks[-1] - mid_ticks[0])) / float(max(int(steps), 1) ** 0.5)


def _variance_ratio(mid_ticks: np.ndarray, horizon: int) -> float:
    if mid_ticks.size <= horizon + 1:
        return 0.0
    step_move = np.diff(mid_ticks)
    if step_move.size == 0:
        return 0.0
    step_var = float(np.var(step_move))
    if step_var <= 1e-12:
        return 0.0
    horizon_move = mid_ticks[horizon:] - mid_ticks[:-horizon]
    return float(np.var(horizon_move) / (float(horizon) * step_var))


def _sign_autocorr(signs: np.ndarray, lag: int) -> float:
    if signs.size <= lag:
        return 0.0
    current = signs[lag:]
    previous = signs[:-lag]
    valid = (current != 0.0) & (previous != 0.0)
    if not valid.any():
        return 0.0
    return _corr(current[valid], previous[valid])


def _autocorr(values: np.ndarray, lag: int) -> float:
    if values.size <= lag:
        return 0.0
    return _corr(values[lag:], values[:-lag])


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
