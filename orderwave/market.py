from __future__ import annotations

"""Public market simulator API."""

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd

from orderwave.book import AggressorSide, BookSide, OrderBook
from orderwave.config import MarketConfig, coerce_config
from orderwave.history import HistoryBuffer
from orderwave.utils import bounded_int, clamp, compute_depth_imbalance, price_to_tick, tick_to_price
from orderwave.visualization import (
    VisualHistoryStore,
    plot_market_overview,
    plot_order_book,
)
from orderwave.visualization import (
    plot_heatmap as render_heatmap,
)

CaptureMode = Literal["summary", "visual"]
_LimitFamily = Literal["join", "refill", "gap", "shelf", "wall"]

_REGIME_TIGHT = 0
_REGIME_NORMAL = 1
_REGIME_FRAGILE = 2


@dataclass(frozen=True)
class SimulationResult:
    """Collected outputs returned by ``Market.run``."""

    snapshot: dict[str, object]
    history: pd.DataFrame


@dataclass(frozen=True)
class _SummaryState:
    best_bid_tick: int
    best_ask_tick: int
    mid_tick: float
    spread_ticks: int
    bid_depth: int
    ask_depth: int
    depth_imbalance: float
    center_tick: int


@dataclass
class _KernelState:
    buy_flow_impulse: float = 0.0
    sell_flow_impulse: float = 0.0
    bid_cancel_pressure: float = 0.0
    ask_cancel_pressure: float = 0.0
    bid_refill_lag: float = 0.0
    ask_refill_lag: float = 0.0
    gap_pressure: float = 0.0
    flow_predictability: float = 0.0
    book_elasticity: float = 1.0
    latent_fair_offset: float = 0.0
    liquidity_regime: int = _REGIME_NORMAL
    regime_steps_left: int = 18
    execution_side: int = 0
    execution_strength: float = 0.0
    execution_steps_left: int = 0


@dataclass(frozen=True)
class _KernelParams:
    flow_decay: float = 0.95
    cancel_decay: float = 0.91
    refill_decay: float = 0.96
    gap_decay: float = 0.90
    predictability_decay: float = 0.97
    elasticity_decay: float = 0.95
    latent_offset_decay: float = 0.98
    flow_to_fair: float = 0.16
    fair_to_pressure: float = 0.48
    imbalance_to_pressure: float = 0.18
    flow_to_pressure: float = 0.34
    gap_to_pressure: float = 0.10
    predictability_to_pressure: float = 0.12
    market_pressure_scale: float = 0.88
    market_flow_scale: float = 1.08
    market_cancel_scale: float = 0.34
    market_predictability_scale: float = 0.42
    market_fragility_scale: float = 0.18
    limit_pressure_scale: float = 0.28
    limit_gap_scale: float = 0.18
    limit_cancel_scale: float = 0.14
    limit_lag_scale: float = 0.48
    limit_elasticity_scale: float = 0.14
    cancel_pressure_scale: float = 1.00
    cancel_gap_scale: float = 0.26
    cancel_signal_scale: float = 0.20
    cancel_elasticity_scale: float = 0.15
    taker_to_flow: float = 0.72
    taker_to_cancel: float = 0.84
    taker_to_gap: float = 0.44
    taker_to_refill_lag: float = 0.82
    cancel_to_gap: float = 0.30
    cancel_to_refill_lag: float = 0.30
    refill_relief: float = 0.90
    gap_fill_relief: float = 0.66
    deep_relief: float = 0.10
    join_relief: float = 0.65
    side_cross_relief: float = 0.10
    join_size_scale: float = 0.52
    refill_size_scale: float = 0.82
    gap_size_scale: float = 0.92
    shelf_size_scale: float = 1.00
    wall_size_scale: float = 1.24
    cancel_size_scale: float = 0.62
    market_size_scale: float = 1.06
    market_size_predictability_dampen: float = 0.22
    emergency_compress_scale: float = 0.84
    touch_cancel_wide_penalty: float = 1.10
    touch_cancel_tight_penalty: float = 0.12
    shock_to_predictability: float = 0.16
    gap_to_predictability: float = 0.10
    predictability_to_market_count: float = 0.34
    predictability_to_wall: float = 0.16
    spread_wide_refill_boost: float = 0.50
    spread_wide_join_boost: float = 0.46
    regime_stress_scale: float = 0.18
    regime_calm_scale: float = 0.15
    execution_decay: float = 0.96
    execution_start_scale: float = 0.34
    execution_reinforce_scale: float = 0.52
    execution_market_bias: float = 1.08
    execution_market_extra: float = 0.90
    execution_limit_drag: float = 0.32


@dataclass(frozen=True)
class _EventPlan:
    bid_cancel: int
    ask_cancel: int
    buy_market: int
    sell_market: int
    bid_join: int
    ask_join: int
    bid_refill: int
    ask_refill: int
    bid_gap_fill: int
    ask_gap_fill: int
    bid_shelf: int
    ask_shelf: int
    bid_wall: int
    ask_wall: int


class Market:
    """Aggregate order-book simulator with a queue-reactive, regime-aware kernel."""

    def __init__(
        self,
        init_price: float = 100.0,
        tick_size: float = 0.01,
        levels: int = 5,
        seed: int | None = None,
        config: MarketConfig | Mapping[str, object] | None = None,
        *,
        capture: CaptureMode = "summary",
    ) -> None:
        if tick_size <= 0.0:
            raise ValueError("tick_size must be positive")
        if levels <= 0:
            raise ValueError("levels must be positive")
        if capture not in {"summary", "visual"}:
            raise ValueError("capture must be 'summary' or 'visual'")

        self.tick_size = float(tick_size)
        self.levels = int(levels)
        self.seed = seed
        self.capture = capture
        self.config = coerce_config(config)
        self._kernel_params = _KernelParams()
        self._rng = np.random.default_rng(seed)

        self._book = OrderBook(self.tick_size)
        self._history = HistoryBuffer()
        self._step = 0
        self._book_capacity = self.levels + self.config.max_spread_ticks + 4
        self._visual_window_ticks = max(
            12,
            self.levels + self.config.max_spread_ticks + self.config.max_fair_move_ticks + 2,
        )
        self._visual_history = (
            VisualHistoryStore(depth_window_ticks=self._visual_window_ticks) if self.capture == "visual" else None
        )
        self._summary_state_cache: _SummaryState | None = None

        self._init_tick = price_to_tick(init_price, self.tick_size)
        self._last_trade_tick = self._init_tick
        self._fair_tick = float(self._init_tick)
        self._buy_aggr_volume = 0.0
        self._sell_aggr_volume = 0.0
        self._kernel = _KernelState()

        self._seed_book(self._init_tick)
        self._record_state()

    def get(self) -> dict[str, object]:
        """Return the current public market snapshot."""

        state = self._summary_state()
        return {
            "step": self._step,
            "last_price": tick_to_price(self._last_trade_tick, self.tick_size),
            "mid_price": tick_to_price(state.mid_tick, self.tick_size),
            "best_bid": tick_to_price(state.best_bid_tick, self.tick_size),
            "best_ask": tick_to_price(state.best_ask_tick, self.tick_size),
            "spread": tick_to_price(state.spread_ticks, self.tick_size),
            "bids": self._public_levels("bid"),
            "asks": self._public_levels("ask"),
            "bid_depth": state.bid_depth,
            "ask_depth": state.ask_depth,
            "depth_imbalance": state.depth_imbalance,
            "buy_aggr_volume": float(self._buy_aggr_volume),
            "sell_aggr_volume": float(self._sell_aggr_volume),
            "fair_price": tick_to_price(self._fair_tick, self.tick_size),
        }

    def get_history(self) -> pd.DataFrame:
        """Return the summary history as a ``pandas.DataFrame``."""

        return self._history.dataframe()

    def step(self) -> dict[str, object]:
        """Advance the simulator by one step and return the new snapshot."""

        self._step += 1
        self._buy_aggr_volume = 0.0
        self._sell_aggr_volume = 0.0

        self._decay_kernel_state()
        pre_state = self._event_state()
        self._advance_slow_state(pre_state)
        self._advance_fair_price()
        state = self._event_state()
        plan = self._sample_event_plan(state)

        for _ in range(plan.bid_cancel):
            self._apply_cancel_event("bid")
        for _ in range(plan.ask_cancel):
            self._apply_cancel_event("ask")

        for aggressor in self._market_sequence(plan.buy_market, plan.sell_market):
            self._apply_market_slice(aggressor)

        for side, family, count in self._limit_sequence(plan):
            for _ in range(count):
                self._apply_limit_event(side, family)

        self._repair_book()
        self._record_state()
        return self.get()

    def gen(self, steps: int) -> dict[str, object]:
        """Run ``steps`` iterations and return the final snapshot."""

        for _ in range(self._coerce_steps(steps)):
            self.step()
        return self.get()

    def run(self, steps: int) -> SimulationResult:
        """Run ``steps`` iterations and return both snapshot and history."""

        snapshot = self.gen(steps)
        return SimulationResult(snapshot=snapshot, history=self.get_history())

    def plot(
        self,
        *,
        max_steps: int = 1200,
        price_window_ticks: int | None = None,
        title: str | None = None,
        figsize: tuple[float, float] | None = None,
    ):
        """Render the overview plot with a level-ranked heatmap."""

        store = self._require_visual_capture()
        return plot_market_overview(
            self.get_history(),
            store,
            tick_size=self.tick_size,
            max_steps=max_steps,
            price_window_ticks=price_window_ticks,
            title=title,
            figsize=figsize,
        )

    def plot_heatmap(
        self,
        *,
        anchor: Literal["mid", "price"] = "mid",
        max_steps: int = 1200,
        price_window_ticks: int | None = None,
        title: str | None = None,
        figsize: tuple[float, float] | None = None,
    ):
        """Render a standalone heatmap from captured visual history."""

        store = self._require_visual_capture()
        return render_heatmap(
            store,
            tick_size=self.tick_size,
            anchor=anchor,
            max_steps=max_steps,
            price_window_ticks=price_window_ticks,
            title=title,
            figsize=figsize,
        )

    def plot_book(
        self,
        *,
        levels: int | None = None,
        title: str | None = None,
        figsize: tuple[float, float] | None = None,
    ):
        """Render the current order book."""

        return plot_order_book(
            self._book,
            tick_size=self.tick_size,
            levels=levels or self.levels,
            title=title,
            figsize=figsize,
        )

    def _summary_state(self) -> _SummaryState:
        if self._summary_state_cache is not None:
            return self._summary_state_cache
        best_bid_tick = self._book.best_bid_tick
        best_ask_tick = self._book.best_ask_tick
        if best_bid_tick is None or best_ask_tick is None:
            raise RuntimeError("order book must remain two-sided")
        mid_tick = (best_bid_tick + best_ask_tick) / 2.0
        spread_ticks = best_ask_tick - best_bid_tick
        bid_depth = self._book.total_depth("bid", self.levels)
        ask_depth = self._book.total_depth("ask", self.levels)
        depth_imbalance = compute_depth_imbalance(bid_depth, ask_depth)
        self._summary_state_cache = _SummaryState(
            best_bid_tick=best_bid_tick,
            best_ask_tick=best_ask_tick,
            mid_tick=mid_tick,
            spread_ticks=spread_ticks,
            bid_depth=bid_depth,
            ask_depth=ask_depth,
            depth_imbalance=depth_imbalance,
            center_tick=int(round(mid_tick)),
        )
        return self._summary_state_cache

    def _event_state(self) -> _SummaryState:
        best_bid_tick = self._book.best_bid_tick
        best_ask_tick = self._book.best_ask_tick
        reference_tick = int(round(self._reference_tick()))
        if best_bid_tick is None:
            synthetic_bid = reference_tick - 1
            if best_ask_tick is not None:
                synthetic_bid = min(synthetic_bid, best_ask_tick - 1)
            best_bid_tick = synthetic_bid
        if best_ask_tick is None:
            synthetic_ask = reference_tick + 1
            synthetic_ask = max(synthetic_ask, best_bid_tick + 1)
            best_ask_tick = synthetic_ask
        if best_bid_tick >= best_ask_tick:
            best_bid_tick = min(best_bid_tick, best_ask_tick - 1)
            best_ask_tick = max(best_ask_tick, best_bid_tick + 1)
        mid_tick = (best_bid_tick + best_ask_tick) / 2.0
        spread_ticks = best_ask_tick - best_bid_tick
        bid_depth = self._book.total_depth("bid", self.levels)
        ask_depth = self._book.total_depth("ask", self.levels)
        return _SummaryState(
            best_bid_tick=best_bid_tick,
            best_ask_tick=best_ask_tick,
            mid_tick=mid_tick,
            spread_ticks=spread_ticks,
            bid_depth=bid_depth,
            ask_depth=ask_depth,
            depth_imbalance=compute_depth_imbalance(bid_depth, ask_depth),
            center_tick=int(round(mid_tick)),
        )

    def _invalidate_summary_state(self) -> None:
        self._summary_state_cache = None

    def _seed_book(self, center_tick: int) -> None:
        for offset in range(self._book_capacity):
            size_scale = 0.90 + (0.12 * min(offset, 5))
            self._book.add_limit("bid", center_tick - 1 - offset, self._scaled_quantity(size_scale))
            self._book.add_limit("ask", center_tick + 1 + offset, self._scaled_quantity(size_scale))
        self._invalidate_summary_state()

    def _record_state(self) -> None:
        state = self._summary_state()
        self._history.append(
            step=self._step,
            last_price=tick_to_price(self._last_trade_tick, self.tick_size),
            mid_price=tick_to_price(state.mid_tick, self.tick_size),
            best_bid=tick_to_price(state.best_bid_tick, self.tick_size),
            best_ask=tick_to_price(state.best_ask_tick, self.tick_size),
            spread=tick_to_price(state.spread_ticks, self.tick_size),
            bid_depth=state.bid_depth,
            ask_depth=state.ask_depth,
            depth_imbalance=state.depth_imbalance,
            buy_aggr_volume=self._buy_aggr_volume,
            sell_aggr_volume=self._sell_aggr_volume,
            fair_price=tick_to_price(self._fair_tick, self.tick_size),
        )
        if self._visual_history is not None:
            self._visual_history.append(
                step=self._step,
                center_tick=state.center_tick,
                best_bid_tick=state.best_bid_tick,
                best_ask_tick=state.best_ask_tick,
                signed_depth=self._book.signed_window(state.center_tick, self._visual_window_ticks),
            )

    def _coerce_steps(self, steps: int) -> int:
        if steps < 0:
            raise ValueError("steps must be non-negative")
        return int(steps)

    def _decay_kernel_state(self) -> None:
        params = self._kernel_params
        self._kernel.buy_flow_impulse *= params.flow_decay
        self._kernel.sell_flow_impulse *= params.flow_decay
        self._kernel.bid_cancel_pressure *= params.cancel_decay
        self._kernel.ask_cancel_pressure *= params.cancel_decay
        self._kernel.bid_refill_lag *= params.refill_decay
        self._kernel.ask_refill_lag *= params.refill_decay
        self._kernel.gap_pressure *= params.gap_decay
        self._kernel.flow_predictability *= params.predictability_decay
        self._kernel.book_elasticity = clamp(
            (self._kernel.book_elasticity * params.elasticity_decay)
            + (0.04 * self._regime_value(tight=1.08, normal=1.00, fragile=0.90)),
            0.25,
            1.60,
        )
        self._kernel.latent_fair_offset *= params.latent_offset_decay
        self._kernel.regime_steps_left = max(0, self._kernel.regime_steps_left - 1)
        self._kernel.execution_strength *= params.execution_decay
        self._kernel.execution_steps_left = max(0, self._kernel.execution_steps_left - 1)
        if self._kernel.execution_steps_left <= 0 or self._kernel.execution_strength < 0.05:
            self._kernel.execution_side = 0
            self._kernel.execution_strength = 0.0
            self._kernel.execution_steps_left = 0

    def _advance_slow_state(self, state: _SummaryState) -> None:
        params = self._kernel_params
        spread_component = self._spread_component(state)
        stress = (
            0.22 * abs(self._net_flow())
            + 0.24 * self._kernel.gap_pressure
            + 0.20 * spread_component
            + 0.17 * (self._kernel.bid_cancel_pressure + self._kernel.ask_cancel_pressure)
            - 0.16 * self._kernel.book_elasticity
        )
        calm = clamp(
            self._kernel.book_elasticity
            - (0.45 * spread_component)
            - (0.20 * self._kernel.gap_pressure)
            - (0.08 * abs(self._net_flow())),
            0.0,
            2.0,
        )

        self._kernel.flow_predictability = clamp(
            (0.88 * self._kernel.flow_predictability)
            + (params.shock_to_predictability * abs(self._net_flow()))
            + (params.gap_to_predictability * self._kernel.gap_pressure)
            + (0.08 * abs(state.depth_imbalance)),
            0.0,
            2.5,
        )

        target_elasticity = clamp(
            1.02
            - (0.32 * spread_component)
            - (0.18 * self._kernel.gap_pressure)
            - (0.12 * (self._kernel.bid_cancel_pressure + self._kernel.ask_cancel_pressure))
            + self._regime_value(tight=0.12, normal=0.0, fragile=-0.12),
            0.25,
            1.50,
        )
        self._kernel.book_elasticity = clamp(
            (0.90 * self._kernel.book_elasticity) + (0.10 * target_elasticity),
            0.25,
            1.60,
        )

        offset_target = clamp(
            (0.55 * self._net_flow()) + (0.12 * state.depth_imbalance),
            -float(self.config.max_fair_move_ticks) * 1.5,
            float(self.config.max_fair_move_ticks) * 1.5,
        )
        self._kernel.latent_fair_offset = clamp(
            (0.94 * self._kernel.latent_fair_offset) + (0.06 * offset_target),
            -float(self.config.max_fair_move_ticks) * 2.0,
            float(self.config.max_fair_move_ticks) * 2.0,
        )
        self._advance_execution_episode(state)
        self._transition_regime(stress=stress, calm=calm)

    def _advance_execution_episode(self, state: _SummaryState) -> None:
        params = self._kernel_params
        net_flow = self._net_flow()
        if abs(net_flow) < 0.08 and self._kernel.flow_predictability < 0.14:
            return
        side = 1 if net_flow >= 0.0 else -1
        same_side_pressure = self._side_signal("bid" if side > 0 else "ask", state)
        trigger_strength = (
            params.execution_start_scale * abs(net_flow)
            + 0.12 * self._kernel.flow_predictability
            + 0.08 * max(0.0, same_side_pressure)
        )
        if self._kernel.execution_side == side:
            self._kernel.execution_strength = clamp(
                self._kernel.execution_strength + (params.execution_reinforce_scale * abs(net_flow)),
                0.0,
                2.0,
            )
            self._kernel.execution_steps_left = min(
                28,
                max(self._kernel.execution_steps_left, int(6 + round(8.0 * trigger_strength))),
            )
            return
        if trigger_strength < 0.14:
            return
        if self._rng.random() < min(0.70, trigger_strength):
            self._kernel.execution_side = side
            self._kernel.execution_strength = clamp(trigger_strength, 0.0, 1.6)
            self._kernel.execution_steps_left = int(6 + self._rng.integers(0, 8))

    def _transition_regime(self, *, stress: float, calm: float) -> None:
        params = self._kernel_params
        regime = self._kernel.liquidity_regime
        can_move = self._kernel.regime_steps_left <= 0
        if regime == _REGIME_TIGHT:
            hazard = clamp(0.01 + (params.regime_stress_scale * stress), 0.0, 0.35)
            if can_move and self._rng.random() < hazard:
                if stress > 0.90 and self._rng.random() < 0.30:
                    self._set_regime(_REGIME_FRAGILE)
                else:
                    self._set_regime(_REGIME_NORMAL)
        elif regime == _REGIME_NORMAL:
            tighten = clamp(0.01 + (params.regime_calm_scale * max(0.0, calm - 0.55)), 0.0, 0.24)
            weaken = clamp(0.01 + (params.regime_stress_scale * max(0.0, stress - 0.40)), 0.0, 0.30)
            if stress > 1.00 and self._rng.random() < min(0.40, weaken * 1.25):
                self._set_regime(_REGIME_FRAGILE)
            elif can_move:
                draw = self._rng.random()
                if draw < tighten:
                    self._set_regime(_REGIME_TIGHT)
                elif draw < tighten + weaken:
                    self._set_regime(_REGIME_FRAGILE)
        else:
            stabilize = clamp(0.03 + (params.regime_calm_scale * calm), 0.0, 0.35)
            if stress > 1.10:
                self._kernel.regime_steps_left = max(self._kernel.regime_steps_left, 8)
                return
            if can_move and self._rng.random() < stabilize:
                if calm > 1.05 and self._rng.random() < 0.25:
                    self._set_regime(_REGIME_TIGHT)
                else:
                    self._set_regime(_REGIME_NORMAL)

    def _set_regime(self, regime: int) -> None:
        self._kernel.liquidity_regime = regime
        if regime == _REGIME_TIGHT:
            self._kernel.regime_steps_left = int(self._rng.integers(16, 34))
        elif regime == _REGIME_FRAGILE:
            self._kernel.regime_steps_left = int(self._rng.integers(10, 24))
        else:
            self._kernel.regime_steps_left = int(self._rng.integers(12, 28))

    def _advance_fair_price(self) -> None:
        params = self._kernel_params
        reference_tick = self._reference_tick()
        net_flow = self._net_flow()
        fair_anchor = reference_tick + self._kernel.latent_fair_offset
        reversion = self.config.mean_reversion * (fair_anchor - self._fair_tick)
        shock = float(
            self._rng.normal(
                0.0,
                self.config.fair_price_vol * self._regime_value(tight=0.82, normal=1.00, fragile=1.12),
            )
        )
        directional_memory = 0.0
        if abs(net_flow) > 1e-9:
            directional_memory = params.predictability_to_pressure * self._kernel.flow_predictability * np.sign(net_flow)
        move = clamp(
            reversion + shock + (params.flow_to_fair * net_flow) + directional_memory,
            -float(self.config.max_fair_move_ticks),
            float(self.config.max_fair_move_ticks),
        )
        self._fair_tick += move

    def _sample_event_plan(self, state: _SummaryState) -> _EventPlan:
        bid_limit = self._poisson_count(self._limit_rate("bid", state))
        ask_limit = self._poisson_count(self._limit_rate("ask", state))
        bid_cancel = self._poisson_count(self._cancel_rate("bid", state))
        ask_cancel = self._poisson_count(self._cancel_rate("ask", state))
        buy_market = self._poisson_count(self._market_rate("buy", state))
        sell_market = self._poisson_count(self._market_rate("sell", state))
        buy_market, sell_market = self._rebalance_market_counts(buy_market, sell_market)
        bid_join, bid_refill, bid_gap, bid_shelf, bid_wall = self._split_limit_counts("bid", bid_limit, state)
        ask_join, ask_refill, ask_gap, ask_shelf, ask_wall = self._split_limit_counts("ask", ask_limit, state)
        return _EventPlan(
            bid_cancel=bid_cancel,
            ask_cancel=ask_cancel,
            buy_market=buy_market,
            sell_market=sell_market,
            bid_join=bid_join,
            ask_join=ask_join,
            bid_refill=bid_refill,
            ask_refill=ask_refill,
            bid_gap_fill=bid_gap,
            ask_gap_fill=ask_gap,
            bid_shelf=bid_shelf,
            ask_shelf=ask_shelf,
            bid_wall=bid_wall,
            ask_wall=ask_wall,
        )

    def _poisson_count(self, rate: float) -> int:
        return int(self._rng.poisson(max(rate, 0.0)))

    def _rebalance_market_counts(self, buy_count: int, sell_count: int) -> tuple[int, int]:
        if self._kernel.execution_side == 0 or self._kernel.execution_strength <= 0.0:
            return buy_count, sell_count
        extra_rate = self._kernel.execution_strength * (
            self._kernel_params.execution_market_extra
            + (0.10 * self._kernel.flow_predictability)
        )
        extra = self._poisson_count(extra_rate)
        if self._kernel.execution_side > 0:
            buy_count += extra
            if sell_count > 0 and extra > 0 and self._rng.random() < 0.35:
                sell_count -= 1
        else:
            sell_count += extra
            if buy_count > 0 and extra > 0 and self._rng.random() < 0.35:
                buy_count -= 1
        return buy_count, sell_count

    def _split_limit_counts(self, side: BookSide, count: int, state: _SummaryState) -> tuple[int, int, int, int, int]:
        if count <= 0:
            return (0, 0, 0, 0, 0)
        signal = self._side_signal(side, state)
        own_cancel = self._side_cancel_pressure(side)
        own_refill_lag = self._side_refill_lag(side)
        execution_penalty = self._execution_penalty(side)
        gap_flag = 1.0 if self._open_gap_ticks(side, max_pairs=6) else 0.0
        spread_component = self._spread_component(state)
        elasticity = clamp(self._kernel.book_elasticity, 0.0, 1.5)
        fragility = 1.0 - clamp(self._kernel.book_elasticity, 0.25, 1.25)
        regime_join = self._regime_value(tight=0.20, normal=0.0, fragile=-0.16)
        regime_refill = self._regime_value(tight=0.10, normal=0.0, fragile=-0.08)
        regime_gap = self._regime_value(tight=-0.04, normal=0.0, fragile=0.18)
        regime_wall = self._regime_value(tight=-0.08, normal=0.0, fragile=0.16)

        join_weight = max(
            0.04,
            0.14
            + (0.56 * max(0.0, signal))
            + (0.26 * spread_component)
            + regime_join
            - (0.30 * own_refill_lag)
            - (0.08 * self._kernel.gap_pressure)
            - (0.30 * execution_penalty),
        )
        refill_weight = max(
            0.04,
            0.26
            + (0.34 * own_cancel)
            + (0.28 * spread_component)
            + (0.12 * elasticity)
            + regime_refill
            - (0.20 * own_refill_lag)
            - (0.22 * execution_penalty),
        )
        gap_weight = max(
            0.04,
            0.12
            + (0.62 * gap_flag)
            + (0.34 * self._kernel.gap_pressure)
            + (0.10 * own_cancel)
            + (0.10 * execution_penalty)
            + regime_gap,
        )
        shelf_weight = max(
            0.04,
            0.18
            + (0.20 * elasticity)
            + (0.10 * self._kernel.gap_pressure)
            + (0.14 * max(0.0, -signal))
            + self._regime_value(tight=0.08, normal=0.0, fragile=-0.02),
        )
        wall_weight = max(
            0.04,
            0.08
            + (0.14 * own_refill_lag)
            + (0.10 * fragility)
            + (0.08 * self._kernel.flow_predictability)
            + (0.06 * execution_penalty)
            + regime_wall,
        )
        weights = np.array([join_weight, refill_weight, gap_weight, shelf_weight, wall_weight], dtype=float)
        split = self._rng.multinomial(count, weights / weights.sum())
        return int(split[0]), int(split[1]), int(split[2]), int(split[3]), int(split[4])

    def _market_rate(self, aggressor: AggressorSide, state: _SummaryState) -> float:
        params = self._kernel_params
        side: BookSide = "bid" if aggressor == "buy" else "ask"
        signal = self._side_signal(side, state)
        flow_impulse = self._kernel.buy_flow_impulse if aggressor == "buy" else self._kernel.sell_flow_impulse
        opposite_cancel = self._kernel.ask_cancel_pressure if aggressor == "buy" else self._kernel.bid_cancel_pressure
        spread_penalty = self._spread_component(state)
        fragility = clamp(1.0 - self._kernel.book_elasticity, 0.0, 1.0)
        execution_bias = self._execution_bias(aggressor)
        regime_scale = self._regime_value(
            tight=0.96,
            normal=1.00,
            fragile=1.10,
        )
        base_rate = self.config.market_rate * 0.52
        scale = (
            0.60
            + (params.market_pressure_scale * max(0.0, signal))
            + (params.market_flow_scale * flow_impulse)
            + (params.market_cancel_scale * opposite_cancel)
            + (params.market_predictability_scale * self._kernel.flow_predictability)
            + (params.market_fragility_scale * fragility)
            + (0.10 * self._kernel.gap_pressure)
            + (params.execution_market_bias * execution_bias)
            - (0.18 * spread_penalty)
        )
        return base_rate * regime_scale * max(0.06, scale)

    def _limit_rate(self, side: BookSide, state: _SummaryState) -> float:
        params = self._kernel_params
        signal = self._side_signal(side, state)
        own_cancel = self._side_cancel_pressure(side)
        own_refill_lag = self._side_refill_lag(side)
        execution_drag = 0.0
        if self._kernel.execution_side != 0:
            if (self._kernel.execution_side > 0 and side == "ask") or (self._kernel.execution_side < 0 and side == "bid"):
                execution_drag = self._kernel.execution_strength
        base_rate = self.config.limit_rate * 0.42
        regime_scale = self._regime_value(tight=1.08, normal=1.00, fragile=0.95)
        scale = (
            0.68
            + (params.limit_pressure_scale * max(0.0, signal))
            + (params.limit_gap_scale * self._kernel.gap_pressure)
            - (params.limit_lag_scale * own_refill_lag)
            + (params.limit_elasticity_scale * self._kernel.book_elasticity)
            + (params.limit_cancel_scale * min(0.6, own_cancel))
            + (0.22 * self._spread_component(state))
            - (params.execution_limit_drag * execution_drag)
        )
        return base_rate * regime_scale * max(0.08, scale)

    def _cancel_rate(self, side: BookSide, state: _SummaryState) -> float:
        params = self._kernel_params
        signal = abs(self._side_signal(side, state))
        base_rate = self.config.cancel_rate * 0.54
        regime_scale = self._regime_value(tight=0.86, normal=1.00, fragile=1.18)
        scale = (
            0.60
            + (params.cancel_pressure_scale * self._side_cancel_pressure(side))
            + (params.cancel_gap_scale * self._kernel.gap_pressure)
            + (params.cancel_signal_scale * signal)
            - (params.cancel_elasticity_scale * self._kernel.book_elasticity)
        )
        return base_rate * regime_scale * max(0.08, scale)

    def _pressure_signal(self, state: _SummaryState) -> float:
        params = self._kernel_params
        fair_signal = clamp(
            ((self._fair_tick + self._kernel.latent_fair_offset) - state.mid_tick) / float(self.config.max_fair_move_ticks),
            -1.0,
            1.0,
        )
        flow_direction = 0.0
        net_flow = self._net_flow()
        if abs(net_flow) > 1e-9:
            flow_direction = self._kernel.flow_predictability * float(np.sign(net_flow))
        return clamp(
            (params.fair_to_pressure * fair_signal)
            + (params.imbalance_to_pressure * state.depth_imbalance)
            + (params.flow_to_pressure * net_flow)
            + (params.predictability_to_pressure * flow_direction)
            - (params.gap_to_pressure * self._kernel.gap_pressure)
            + self._regime_value(tight=0.04, normal=0.0, fragile=-0.04),
            -1.0,
            1.0,
        )

    def _side_signal(self, side: BookSide, state: _SummaryState) -> float:
        pressure = self._pressure_signal(state)
        return pressure if side == "bid" else -pressure

    def _net_flow(self) -> float:
        return self._kernel.buy_flow_impulse - self._kernel.sell_flow_impulse

    def _spread_component(self, state: _SummaryState) -> float:
        if self.config.max_spread_ticks <= 1:
            return 0.0
        return clamp(
            (state.spread_ticks - 1) / float(self.config.max_spread_ticks - 1),
            0.0,
            1.0,
        )

    def _regime_value(self, *, tight: float, normal: float, fragile: float) -> float:
        regime = self._kernel.liquidity_regime
        if regime == _REGIME_TIGHT:
            return tight
        if regime == _REGIME_FRAGILE:
            return fragile
        return normal

    def _side_cancel_pressure(self, side: BookSide) -> float:
        return self._kernel.bid_cancel_pressure if side == "bid" else self._kernel.ask_cancel_pressure

    def _side_refill_lag(self, side: BookSide) -> float:
        return self._kernel.bid_refill_lag if side == "bid" else self._kernel.ask_refill_lag

    def _execution_bias(self, aggressor: AggressorSide) -> float:
        if self._kernel.execution_side == 0 or self._kernel.execution_strength <= 0.0:
            return 0.0
        is_buy = aggressor == "buy"
        same_side = (self._kernel.execution_side > 0 and is_buy) or (self._kernel.execution_side < 0 and not is_buy)
        return self._kernel.execution_strength if same_side else -0.30 * self._kernel.execution_strength

    def _execution_penalty(self, side: BookSide) -> float:
        if self._kernel.execution_side == 0 or self._kernel.execution_strength <= 0.0:
            return 0.0
        if self._kernel.execution_side > 0 and side == "ask":
            return self._kernel.execution_strength
        if self._kernel.execution_side < 0 and side == "bid":
            return self._kernel.execution_strength
        return 0.0

    def _market_sequence(self, buy_count: int, sell_count: int) -> list[AggressorSide]:
        if buy_count <= 0 and sell_count <= 0:
            return []
        predictability = self._kernel.flow_predictability + max(0.0, self._kernel.execution_strength - 0.20)
        dominant_buy = self._kernel.buy_flow_impulse >= self._kernel.sell_flow_impulse
        if self._kernel.execution_side != 0:
            dominant_buy = self._kernel.execution_side > 0
        dominant_count = buy_count if dominant_buy else sell_count
        other_count = sell_count if dominant_buy else buy_count
        dominant_side: AggressorSide = "buy" if dominant_buy else "sell"
        other_side: AggressorSide = "sell" if dominant_buy else "buy"

        if predictability < 0.35 or dominant_count <= 0 or other_count <= 0:
            sequence: list[AggressorSide] = []
            sequence.extend(["buy"] * buy_count)
            sequence.extend(["sell"] * sell_count)
            self._rng.shuffle(sequence)
            return sequence

        sequence = []
        remaining_dominant = dominant_count
        remaining_other = other_count
        block_size = max(1, min(4, int(round(1.0 + predictability))))
        while remaining_dominant > 0 or remaining_other > 0:
            dominant_run = min(remaining_dominant, block_size)
            sequence.extend([dominant_side] * dominant_run)
            remaining_dominant -= dominant_run
            if remaining_other > 0:
                sequence.append(other_side)
                remaining_other -= 1
        return sequence

    def _limit_sequence(self, plan: _EventPlan) -> tuple[tuple[BookSide, _LimitFamily, int], ...]:
        return (
            ("bid", "join", plan.bid_join),
            ("ask", "join", plan.ask_join),
            ("bid", "refill", plan.bid_refill),
            ("ask", "refill", plan.ask_refill),
            ("bid", "gap", plan.bid_gap_fill),
            ("ask", "gap", plan.ask_gap_fill),
            ("bid", "shelf", plan.bid_shelf),
            ("ask", "shelf", plan.ask_shelf),
            ("bid", "wall", plan.bid_wall),
            ("ask", "wall", plan.ask_wall),
        )

    def _sample_quantity(self) -> int:
        raw_value = float(self._rng.lognormal(self.config.size_mean, self.config.size_dispersion))
        return bounded_int(raw_value, self.config.min_order_qty, self.config.max_order_qty)

    def _scaled_quantity(self, scale: float) -> int:
        return bounded_int(
            round(self._sample_quantity() * max(scale, 0.20)),
            self.config.min_order_qty,
            self.config.max_order_qty,
        )

    def _apply_market_slice(self, aggressor: AggressorSide) -> None:
        state = self._event_state()
        depth_before = max(1, state.ask_depth if aggressor == "buy" else state.bid_depth)
        side: BookSide = "bid" if aggressor == "buy" else "ask"
        flow_impulse = self._kernel.buy_flow_impulse if aggressor == "buy" else self._kernel.sell_flow_impulse
        predictability_dampen = 1.0 / (1.0 + (self._kernel_params.market_size_predictability_dampen * self._kernel.flow_predictability))
        qty = self._scaled_quantity(
            self._kernel_params.market_size_scale
            * predictability_dampen
            * self._regime_value(tight=0.95, normal=1.0, fragile=0.92)
            + (0.14 * flow_impulse)
            + (0.12 * max(0.0, self._side_signal(side, state)))
        )
        result = self._book.execute_market(aggressor, qty)
        if result.filled_qty <= 0:
            return

        if result.last_fill_tick is not None:
            self._last_trade_tick = result.last_fill_tick
        filled_qty = float(result.filled_qty)
        impact = clamp(filled_qty / float(depth_before), 0.0, 1.8)

        if aggressor == "buy":
            self._buy_aggr_volume += filled_qty
            self._on_market_pressure("buy", impact)
        else:
            self._sell_aggr_volume += filled_qty
            self._on_market_pressure("sell", impact)
        self._invalidate_summary_state()

    def _on_market_pressure(self, aggressor: AggressorSide, impact: float) -> None:
        params = self._kernel_params
        if aggressor == "buy":
            self._kernel.buy_flow_impulse = clamp(
                (params.flow_decay * self._kernel.buy_flow_impulse) + (params.taker_to_flow * impact),
                0.0,
                3.0,
            )
            self._kernel.sell_flow_impulse *= 0.96
            self._kernel.ask_cancel_pressure = clamp(
                (self._kernel.ask_cancel_pressure * 0.82) + (params.taker_to_cancel * impact),
                0.0,
                3.0,
            )
            self._kernel.ask_refill_lag = clamp(
                max(self._kernel.ask_refill_lag * 0.84, params.taker_to_refill_lag * impact),
                0.0,
                2.8,
            )
            self._kernel.bid_cancel_pressure *= 0.96
            self._kernel.bid_refill_lag *= 0.90
        else:
            self._kernel.sell_flow_impulse = clamp(
                (params.flow_decay * self._kernel.sell_flow_impulse) + (params.taker_to_flow * impact),
                0.0,
                3.0,
            )
            self._kernel.buy_flow_impulse *= 0.96
            self._kernel.bid_cancel_pressure = clamp(
                (self._kernel.bid_cancel_pressure * 0.82) + (params.taker_to_cancel * impact),
                0.0,
                3.0,
            )
            self._kernel.bid_refill_lag = clamp(
                max(self._kernel.bid_refill_lag * 0.84, params.taker_to_refill_lag * impact),
                0.0,
                2.8,
            )
            self._kernel.ask_cancel_pressure *= 0.96
            self._kernel.ask_refill_lag *= 0.90
        self._kernel.gap_pressure = clamp(
            (self._kernel.gap_pressure * 0.86) + (params.taker_to_gap * impact),
            0.0,
            3.0,
        )
        self._kernel.flow_predictability = clamp(
            (0.90 * self._kernel.flow_predictability) + (0.10 * impact),
            0.0,
            2.5,
        )
        self._kernel.book_elasticity = clamp(self._kernel.book_elasticity - (0.05 * impact), 0.25, 1.60)
        self._reinforce_execution(aggressor, impact)

    def _reinforce_execution(self, aggressor: AggressorSide, impact: float) -> None:
        side = 1 if aggressor == "buy" else -1
        if self._kernel.execution_side == side:
            self._kernel.execution_strength = clamp(
                self._kernel.execution_strength + (self._kernel_params.execution_reinforce_scale * impact),
                0.0,
                2.0,
            )
            self._kernel.execution_steps_left = min(32, self._kernel.execution_steps_left + 3)
            return
        if impact < 0.10:
            return
        if self._rng.random() < min(0.75, 0.18 + impact):
            self._kernel.execution_side = side
            self._kernel.execution_strength = clamp(
                0.16 + (self._kernel_params.execution_reinforce_scale * impact),
                0.0,
                1.6,
            )
            self._kernel.execution_steps_left = int(6 + self._rng.integers(0, 8))

    def _apply_cancel_event(self, side: BookSide) -> None:
        state = self._event_state()
        tick = self._cancel_tick(side, state)
        if tick is None:
            return
        level_qty = self._book.level_qty(side, tick)
        if level_qty <= 0:
            return
        qty = min(level_qty, self._cancel_quantity(side))
        if qty <= 0:
            return
        best_tick_before = self._book.best_bid_tick if side == "bid" else self._book.best_ask_tick
        was_touch = best_tick_before == tick
        canceled = self._book.cancel_level(side, tick, qty)
        if canceled <= 0:
            return
        self._on_cancel_pressure(side, canceled=float(canceled), was_touch=was_touch)
        self._invalidate_summary_state()

    def _cancel_tick(self, side: BookSide, state: _SummaryState) -> int | None:
        levels = self._book.levels(side, min(self._book_capacity, 10))
        if not levels:
            return None
        max_qty = max(qty for _, qty in levels)
        spread_component = self._spread_component(state)
        weights: list[float] = []
        for index, (tick, qty) in enumerate(levels):
            thinness = 1.0 - (float(qty) / float(max(max_qty, 1)))
            gap_adjacent = 1.0 if self._tick_has_gap_neighbor(side, tick, levels) else 0.0
            mid_rank_bonus = 0.40 if 1 <= index <= 4 else 0.0
            touch_penalty = 0.0
            if index == 0:
                touch_penalty = (
                    self._kernel_params.touch_cancel_wide_penalty * spread_component
                    + self._regime_value(
                        tight=self._kernel_params.touch_cancel_tight_penalty,
                        normal=0.0,
                        fragile=0.10,
                    )
                )
            weights.append(
                max(
                    0.03,
                    (1.65 / float(index + 1))
                    + (0.68 * thinness)
                    + (0.72 * gap_adjacent)
                    + mid_rank_bonus
                    + (0.34 * self._side_cancel_pressure(side))
                    - touch_penalty,
                )
            )
        return levels[self._weighted_index(weights)][0]

    def _cancel_quantity(self, side: BookSide) -> int:
        scale = (
            self._kernel_params.cancel_size_scale
            + (0.16 * self._side_cancel_pressure(side))
            + (0.10 * self._kernel.gap_pressure)
            - (0.06 * self._kernel.book_elasticity)
        )
        return self._scaled_quantity(scale)

    def _tick_has_gap_neighbor(self, side: BookSide, tick: int, levels: tuple[tuple[int, int], ...]) -> bool:
        level_ticks = [level_tick for level_tick, _ in levels]
        index = level_ticks.index(tick)
        if index > 0:
            upper = level_ticks[index - 1]
            if abs(upper - tick) > 1:
                return True
        if index + 1 < len(level_ticks):
            lower = level_ticks[index + 1]
            if abs(tick - lower) > 1:
                return True
        return False

    def _on_cancel_pressure(self, side: BookSide, *, canceled: float, was_touch: bool) -> None:
        params = self._kernel_params
        impact = clamp(canceled / float(max(self.config.max_order_qty, 1)), 0.0, 1.0)
        if side == "bid":
            self._kernel.bid_cancel_pressure = clamp(
                (self._kernel.bid_cancel_pressure * 0.90) + impact,
                0.0,
                3.0,
            )
            self._kernel.bid_refill_lag = clamp(
                max(self._kernel.bid_refill_lag * 0.92, params.cancel_to_refill_lag * impact),
                0.0,
                2.8,
            )
        else:
            self._kernel.ask_cancel_pressure = clamp(
                (self._kernel.ask_cancel_pressure * 0.90) + impact,
                0.0,
                3.0,
            )
            self._kernel.ask_refill_lag = clamp(
                max(self._kernel.ask_refill_lag * 0.92, params.cancel_to_refill_lag * impact),
                0.0,
                2.8,
            )
        if was_touch:
            self._kernel.gap_pressure = clamp(
                (self._kernel.gap_pressure * 0.90) + (params.cancel_to_gap * impact),
                0.0,
                3.0,
            )
        self._kernel.book_elasticity = clamp(self._kernel.book_elasticity - (0.03 * impact), 0.25, 1.60)

    def _apply_limit_event(self, side: BookSide, family: _LimitFamily) -> None:
        state = self._event_state()
        signal = self._side_signal(side, state)
        if family == "join":
            tick = self._join_tick(side, state)
            qty = self._scaled_quantity(self._kernel_params.join_size_scale + (0.10 * max(0.0, signal)))
        elif family == "refill":
            tick = self._refill_tick(side, state)
            qty = self._scaled_quantity(
                self._kernel_params.refill_size_scale
                + (0.20 * self._side_cancel_pressure(side))
                - (0.08 * self._side_refill_lag(side))
            )
        elif family == "gap":
            tick = self._gap_fill_tick(side, state)
            qty = self._scaled_quantity(
                self._kernel_params.gap_size_scale
                + (0.22 * self._kernel.gap_pressure)
                + (0.08 * self._side_cancel_pressure(side))
            )
        elif family == "shelf":
            tick = self._shelf_tick(side, state)
            qty = self._scaled_quantity(
                self._kernel_params.shelf_size_scale
                + (0.14 * self._kernel.book_elasticity)
                + (0.08 * max(0.0, -signal))
            )
        else:
            tick = self._wall_tick(side, state)
            qty = self._scaled_quantity(
                self._kernel_params.wall_size_scale
                + (0.12 * self._side_refill_lag(side))
                + (0.10 * self._kernel.flow_predictability)
            )
        self._book.add_limit(side, tick, qty)
        self._on_limit_relief(side, family)
        self._invalidate_summary_state()

    def _join_tick(self, side: BookSide, state: _SummaryState) -> int:
        if side == "bid":
            if state.spread_ticks > 1:
                return state.best_bid_tick + 1
            return state.best_bid_tick
        if state.spread_ticks > 1:
            return state.best_ask_tick - 1
        return state.best_ask_tick

    def _refill_tick(self, side: BookSide, state: _SummaryState) -> int:
        levels = self._book.levels(side, min(max(self.levels, 5), self._book_capacity))
        if not levels:
            return self._join_tick(side, state)
        top = levels[: min(5, len(levels))]
        max_qty = max(qty for _, qty in top)
        spread_component = self._spread_component(state)
        weights: list[float] = []
        for index, (tick, qty) in enumerate(top):
            thinness = 1.0 - (float(qty) / float(max(max_qty, 1)))
            gap_adjacent = 1.0 if self._tick_has_gap_neighbor(side, tick, top) else 0.0
            touch_bonus = 0.55 * spread_component if index == 0 else 0.0
            weights.append(
                max(
                    0.04,
                    (1.25 / float(index + 1))
                    + (0.95 * thinness)
                    + (0.70 * gap_adjacent)
                    + touch_bonus,
                )
            )
        return top[self._weighted_index(weights)][0]

    def _gap_fill_tick(self, side: BookSide, state: _SummaryState) -> int:
        candidates = self._open_gap_ticks(side, max_pairs=10)
        if not candidates:
            return self._shelf_tick(side, state)
        weights = np.linspace(float(len(candidates)), 1.0, num=len(candidates), dtype=float)
        weights = weights + (0.24 * self._kernel.gap_pressure) + (0.12 * self._spread_component(state))
        return candidates[self._weighted_index(weights)]

    def _shelf_tick(self, side: BookSide, state: _SummaryState) -> int:
        candidates, weights = self._deep_candidates(side, state, mode="connected")
        if candidates:
            return candidates[self._weighted_index(weights)]
        return self._join_tick(side, state)

    def _wall_tick(self, side: BookSide, state: _SummaryState) -> int:
        candidates, weights = self._deep_candidates(side, state, mode="isolated")
        if candidates:
            return candidates[self._weighted_index(weights)]
        fallback, fallback_weights = self._deep_candidates(side, state, mode="connected")
        if fallback:
            return fallback[self._weighted_index(fallback_weights)]
        return self._emergency_quote_tick(side, state)

    def _open_gap_ticks(self, side: BookSide, *, max_pairs: int) -> list[int]:
        levels = self._book.levels(side, min(self._book_capacity, max_pairs + 1))
        candidates: list[int] = []
        for index in range(len(levels) - 1):
            current_tick = levels[index][0]
            next_tick = levels[index + 1][0]
            gap = abs(current_tick - next_tick)
            if gap <= 1:
                continue
            near_touch_bonus = 2 if index < 2 else 1
            for offset in range(1, gap):
                candidate = current_tick - offset if side == "bid" else current_tick + offset
                weight = max(1, near_touch_bonus + 1 - offset)
                candidates.extend([candidate] * weight)
        return candidates

    def _deep_candidates(self, side: BookSide, state: _SummaryState, *, mode: Literal["connected", "isolated"]) -> tuple[list[int], list[float]]:
        occupied = {tick for tick, _ in self._book.levels(side)}
        candidates: list[int] = []
        weights: list[float] = []
        max_distance = self._book_capacity + self.config.max_spread_ticks + 4
        best_tick = state.best_bid_tick if side == "bid" else state.best_ask_tick
        start_distance = 1 if mode == "connected" else 2
        for distance in range(start_distance, max_distance + 1):
            tick = best_tick - distance if side == "bid" else best_tick + distance
            if side == "bid" and tick < 0:
                break
            if tick in occupied:
                continue
            if side == "bid" and tick >= state.best_ask_tick:
                continue
            if side == "ask" and tick <= state.best_bid_tick:
                continue

            adjacent_count = 0
            if (tick - 1) in occupied:
                adjacent_count += 1
            if (tick + 1) in occupied:
                adjacent_count += 1

            if mode == "connected":
                if adjacent_count == 0:
                    continue
                score = (1.30 + (0.40 * adjacent_count)) * float(self.config.level_decay ** max(distance - 1, 0))
            else:
                min_isolated_distance = max(8, self.levels + 3)
                if adjacent_count > 0 or distance < min_isolated_distance:
                    continue
                score = (0.52 + (0.04 * distance)) * float(self.config.level_decay ** max(distance - min_isolated_distance, 0))

            candidates.append(tick)
            weights.append(max(0.02, score))
        return candidates, weights

    def _weighted_index(self, weights: list[float] | np.ndarray) -> int:
        normalized = np.asarray(weights, dtype=float)
        normalized = normalized / normalized.sum()
        return int(self._rng.choice(normalized.size, p=normalized))

    def _on_limit_relief(self, side: BookSide, family: _LimitFamily) -> None:
        params = self._kernel_params
        if side == "bid":
            self._kernel.bid_cancel_pressure *= 1.0 - params.side_cross_relief
            if family == "join":
                self._kernel.bid_refill_lag *= params.join_relief
            else:
                self._kernel.bid_refill_lag *= params.refill_relief
        else:
            self._kernel.ask_cancel_pressure *= 1.0 - params.side_cross_relief
            if family == "join":
                self._kernel.ask_refill_lag *= params.join_relief
            else:
                self._kernel.ask_refill_lag *= params.refill_relief

        if family == "gap":
            self._kernel.gap_pressure *= params.gap_fill_relief
        elif family in {"shelf", "wall"}:
            self._kernel.gap_pressure *= 1.0 - params.deep_relief

        if family in {"join", "refill", "shelf"}:
            self._kernel.book_elasticity = clamp(self._kernel.book_elasticity + 0.03, 0.25, 1.60)
        elif family == "wall":
            self._kernel.book_elasticity = clamp(self._kernel.book_elasticity - 0.01, 0.25, 1.60)

    def _repair_book(self) -> None:
        self._ensure_two_sided()
        self._book.clear_crossed_quotes()
        self._ensure_two_sided()
        self._compress_spread()
        self._book.trim("bid", self._book_capacity)
        self._book.trim("ask", self._book_capacity)
        self._ensure_two_sided()
        self._invalidate_summary_state()

    def _ensure_two_sided(self) -> None:
        state = self._event_state()
        if self._book.best_bid_tick is None:
            self._book.add_limit("bid", self._emergency_quote_tick("bid", state), self._scaled_quantity(0.80))
        if self._book.best_ask_tick is None:
            state = self._event_state()
            self._book.add_limit("ask", self._emergency_quote_tick("ask", state), self._scaled_quantity(0.80))

    def _compress_spread(self) -> None:
        for _ in range(self.config.max_spread_ticks + 2):
            state = self._event_state()
            if state.spread_ticks <= self.config.max_spread_ticks:
                return
            favored_bid = self._pressure_signal(state) >= 0.0
            if favored_bid:
                tick = max(0, state.best_ask_tick - self.config.max_spread_ticks)
                tick = min(tick, state.best_ask_tick - 1)
                self._book.add_limit("bid", tick, self._scaled_quantity(self._kernel_params.emergency_compress_scale))
                continue

            tick = state.best_bid_tick + self.config.max_spread_ticks
            tick = max(tick, state.best_bid_tick + 1)
            self._book.add_limit("ask", tick, self._scaled_quantity(self._kernel_params.emergency_compress_scale))

    def _emergency_quote_tick(self, side: BookSide, state: _SummaryState) -> int:
        reference_tick = int(round(self._reference_tick()))
        if side == "bid":
            return max(0, min(reference_tick - 1, state.best_ask_tick - 1))
        return max(reference_tick + 1, state.best_bid_tick + 1)

    def _public_levels(self, side: BookSide) -> list[dict[str, float]]:
        return [
            {"price": tick_to_price(tick, self.tick_size), "qty": float(qty)}
            for tick, qty in self._book.levels(side, self.levels)
        ]

    def _reference_tick(self) -> float:
        mid_tick = self._book.mid_tick()
        if mid_tick is not None:
            return mid_tick
        return float(self._last_trade_tick)

    def _require_visual_capture(self) -> VisualHistoryStore:
        if self._visual_history is None:
            raise RuntimeError("plot() and plot_heatmap() require Market(..., capture='visual')")
        return self._visual_history


__all__ = ["Market", "SimulationResult"]
