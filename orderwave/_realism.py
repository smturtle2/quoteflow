from __future__ import annotations

"""Internal realism metrics for aggregate order-book runs."""

from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Sequence

    from orderwave.market import Market


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

    def to_dict(self) -> dict[str, float | int | None]:
        return asdict(self)


def profile_market_realism(market: Market, *, steps: int) -> RealismProfile:
    bid_gap_flags: list[float] = []
    ask_gap_flags: list[float] = []
    for _ in range(int(steps)):
        market.step()
        bid_gap_flags.extend(_gap_flags(market._book.levels("bid", 6)))  # type: ignore[attr-defined]
        ask_gap_flags.extend(_gap_flags(market._book.levels("ask", 6)))  # type: ignore[attr-defined]

    history = market.get_history()
    spread_ticks = (history["spread"] / market.tick_size).round().to_numpy(dtype=float)
    signed_flow = (history["buy_aggr_volume"] - history["sell_aggr_volume"]).to_numpy(dtype=float)
    trade_sign = np.sign(signed_flow)
    mid_returns = history["mid_price"].diff().fillna(0.0).to_numpy(dtype=float)

    valid_sign = (trade_sign[1:] != 0.0) & (trade_sign[:-1] != 0.0)
    trade_sign_acf = _corr(trade_sign[1:][valid_sign], trade_sign[:-1][valid_sign]) if valid_sign.any() else 0.0

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
    )


def aggregate_realism_profiles(profiles: Sequence[RealismProfile]) -> RealismProfile:
    if not profiles:
        raise ValueError("profiles must not be empty")
    arrays = {field: np.array([getattr(profile, field) for profile in profiles], dtype=float) for field in profiles[0].to_dict() if field not in {"seed", "steps"}}
    return RealismProfile(
        seed=None,
        steps=int(np.mean([profile.steps for profile in profiles])),
        one_tick_spread_share=float(np.mean(arrays["one_tick_spread_share"])),
        wide_spread_share=float(np.mean(arrays["wide_spread_share"])),
        spread_acf1=float(np.mean(arrays["spread_acf1"])),
        trade_sign_acf1=float(np.mean(arrays["trade_sign_acf1"])),
        same_step_impact_corr=float(np.mean(arrays["same_step_impact_corr"])),
        next_step_impact_corr=float(np.mean(arrays["next_step_impact_corr"])),
        bid_gap_gt1_share_top5=float(np.mean(arrays["bid_gap_gt1_share_top5"])),
        ask_gap_gt1_share_top5=float(np.mean(arrays["ask_gap_gt1_share_top5"])),
        spread_recovery_median=float(np.nanmean(arrays["spread_recovery_median"])),
    )


def _gap_flags(levels: tuple[tuple[int, int], ...]) -> list[float]:
    flags: list[float] = []
    for index in range(min(5, max(0, len(levels) - 1))):
        left_tick = levels[index][0]
        right_tick = levels[index + 1][0]
        flags.append(1.0 if abs(left_tick - right_tick) > 1 else 0.0)
    return flags


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
