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
from orderwave.visualization import VisualHistoryStore, plot_market_overview, plot_order_book
from orderwave.visualization import plot_heatmap as render_heatmap

CaptureMode = Literal["summary", "visual"]


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


@dataclass(frozen=True)
class _SideObservables:
    depth: np.ndarray
    log_depth: np.ndarray
    occupancy: np.ndarray
    shortage: np.ndarray
    neighbor_log: np.ndarray
    support: float
    front_depth: float
    connectedness: float
    touch_shortage: float


@dataclass
class _LatentState:
    total_liquidity_log: float
    side_budget: float
    execution_state: float
    resilience: float
    realized_flow_memory: float
    bid_mix: np.ndarray
    ask_mix: np.ndarray
    bid_mass: np.ndarray
    ask_mass: np.ndarray


@dataclass(frozen=True)
class _LatentParams:
    state_persistence: float = 0.92
    state_noise: float = 0.10
    mix_persistence: float = 0.86
    mass_persistence: float = 0.82
    mass_diffusion: float = 0.24
    cox_shape: float = 3.0


class Market:
    """Aggregate order-book simulator driven by dynamic latent distribution synthesis."""

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
        self._params = _LatentParams()
        self._rng = np.random.default_rng(seed)

        self._step = 0
        self._center_tick = price_to_tick(init_price, self.tick_size)
        self._last_trade_tick = self._center_tick
        self._fair_tick = float(self._center_tick)

        self._depth_cells = max(
            18,
            self.levels + self.config.max_spread_ticks + self.config.max_fair_move_ticks + 18,
        )
        self._book_capacity = self._depth_cells
        self._grid: np.ndarray = np.arange(self._depth_cells, dtype=float)
        self._grid_norm = self._grid / max(float(self._depth_cells - 1), 1.0)
        self._base_profile = np.power(self.config.level_decay, self._grid)
        self._near_kernel = _normalize_positive(self._base_profile)
        mid_center = float(self.levels)
        mid_width = max(float(self.levels) / 2.0, 1.0)
        far_center = min(
            float(self._depth_cells - 1),
            float(self.levels + self.config.max_spread_ticks + self.config.max_fair_move_ticks),
        )
        far_width = max(float(self.config.max_spread_ticks + self.config.max_fair_move_ticks), 1.0)
        self._front_basis = self._near_kernel
        self._mid_basis = _normalize_positive(np.exp(-0.5 * np.square((self._grid - mid_center) / mid_width)))
        self._far_basis = _normalize_positive(np.exp(-0.5 * np.square((self._grid - far_center) / far_width)))
        self._touch_horizon = min(
            self._depth_cells,
            self.levels + self.config.max_spread_ticks + self.config.max_fair_move_ticks,
        )
        self._touch_weights = _normalize_positive(self._front_basis[: self._touch_horizon])
        self._profile_target = self._base_profile / (1.0 + self._base_profile)

        self._visible_bid: np.ndarray = np.zeros(self._depth_cells, dtype=np.int64)
        self._visible_ask: np.ndarray = np.zeros(self._depth_cells, dtype=np.int64)
        self._latent = self._initial_latent_state()

        self._book = OrderBook(self.tick_size)
        self._history = HistoryBuffer()
        self._summary_state_cache: _SummaryState | None = None
        self._buy_aggr_volume = 0.0
        self._sell_aggr_volume = 0.0

        self._visual_window_ticks = max(
            12,
            self.levels + self.config.max_spread_ticks + self.config.max_fair_move_ticks + 2,
        )
        self._visual_history = (
            VisualHistoryStore(depth_window_ticks=self._visual_window_ticks) if self.capture == "visual" else None
        )

        self._seed_visible_book()
        self._sync_book_from_visible()
        self._fair_tick = self._summary_state().mid_tick
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
        return self._history.dataframe()

    def step(self) -> dict[str, object]:
        self._step += 1
        self._buy_aggr_volume = 0.0
        self._sell_aggr_volume = 0.0

        state = self._summary_state()
        fair_gap = self._advance_latent_state(state)
        bid_mean, ask_mean = self._latent_reveal_means(fair_gap, state.depth_imbalance)
        self._apply_cancel_thinning(fair_gap, bid_mean, ask_mean)
        pre_refill_scale = float(
            _sigmoid(
                _mean_signal(
                    self._spread_component(state.spread_ticks),
                    self._side_observables(self._visible_bid).touch_shortage,
                    self._side_observables(self._visible_ask).touch_shortage,
                )
            )
        )
        self._apply_limit_reveals(bid_mean, ask_mean, scale=pre_refill_scale)
        self._apply_market_flow(fair_gap, state.depth_imbalance)
        self._repair_visible_arrays()
        self._recenter_visible_arrays()

        post_state = self._visible_state()
        post_fair_gap = self._fair_gap(post_state.mid_tick)
        post_bid_mean, post_ask_mean = self._latent_reveal_means(post_fair_gap, post_state.depth_imbalance)
        refill_scale = float(
            _sigmoid(
                _mean_signal(
                    self._latent.resilience,
                    self._spread_component(post_state.spread_ticks),
                    self._side_observables(self._visible_bid).touch_shortage,
                    self._side_observables(self._visible_ask).touch_shortage,
                )
            )
        )
        self._apply_limit_reveals(post_bid_mean, post_ask_mean, scale=refill_scale)
        self._repair_visible_arrays()
        self._recenter_visible_arrays()
        self._sync_book_from_visible()
        self._record_state()
        return self.get()

    def gen(self, steps: int) -> dict[str, object]:
        for _ in range(self._coerce_steps(steps)):
            self.step()
        return self.get()

    def run(self, steps: int) -> SimulationResult:
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
        return plot_order_book(
            self._book,
            tick_size=self.tick_size,
            levels=levels or self.levels,
            title=title,
            figsize=figsize,
        )

    def _initial_latent_state(self) -> _LatentState:
        common_mix = 0.05 * self._rng.normal(size=3)
        bid_mix = common_mix + (0.02 * self._rng.normal(size=3))
        ask_mix = common_mix + (0.02 * self._rng.normal(size=3))
        bid_mass = _normalize_positive(self._front_basis * np.exp(0.05 * self._rng.normal(size=self._depth_cells)))
        ask_mass = _normalize_positive(self._front_basis * np.exp(0.05 * self._rng.normal(size=self._depth_cells)))
        return _LatentState(
            total_liquidity_log=0.0,
            side_budget=0.0,
            execution_state=0.0,
            resilience=0.0,
            realized_flow_memory=0.0,
            bid_mix=bid_mix.astype(float),
            ask_mix=ask_mix.astype(float),
            bid_mass=bid_mass.astype(float),
            ask_mass=ask_mass.astype(float),
        )

    def _seed_visible_book(self) -> None:
        bid_mean, ask_mean = self._latent_reveal_means(0.0, 0.0)
        self._visible_bid = np.asarray(self._rng.poisson(bid_mean * 1.40), dtype=np.int64)
        self._visible_ask = np.asarray(self._rng.poisson(ask_mean * 1.40), dtype=np.int64)
        self._repair_visible_arrays()
        self._recenter_visible_arrays()

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
            center_tick=_center_from_quotes(best_bid_tick, best_ask_tick),
        )
        return self._summary_state_cache

    def _invalidate_summary_state(self) -> None:
        self._summary_state_cache = None

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

    def _visible_state(self) -> _SummaryState:
        bid_index = _first_positive_index(self._visible_bid)
        ask_index = _first_positive_index(self._visible_ask)
        if bid_index is None:
            bid_index = 0
        if ask_index is None:
            ask_index = 0
        best_bid_tick = self._tick_at_distance("bid", bid_index)
        best_ask_tick = self._tick_at_distance("ask", ask_index)
        mid_tick = (best_bid_tick + best_ask_tick) / 2.0
        bid_depth = int(np.sum(self._visible_bid[: self.levels]))
        ask_depth = int(np.sum(self._visible_ask[: self.levels]))
        return _SummaryState(
            best_bid_tick=best_bid_tick,
            best_ask_tick=best_ask_tick,
            mid_tick=mid_tick,
            spread_ticks=best_ask_tick - best_bid_tick,
            bid_depth=bid_depth,
            ask_depth=ask_depth,
            depth_imbalance=compute_depth_imbalance(bid_depth, ask_depth),
            center_tick=_center_from_quotes(best_bid_tick, best_ask_tick),
        )

    def _coerce_steps(self, steps: int) -> int:
        if steps < 0:
            raise ValueError("steps must be non-negative")
        return int(steps)

    def _advance_latent_state(self, state: _SummaryState) -> float:
        params = self._params
        bid_obs = self._side_observables(self._visible_bid)
        ask_obs = self._side_observables(self._visible_ask)
        spread_component = self._spread_component(state.spread_ticks)
        depth_imbalance = _continuous_imbalance(bid_obs.front_depth, ask_obs.front_depth)
        support_imbalance = _continuous_imbalance(bid_obs.support, ask_obs.support)
        shortage_imbalance = bid_obs.touch_shortage - ask_obs.touch_shortage
        avg_shortage = 0.5 * (bid_obs.touch_shortage + ask_obs.touch_shortage)
        avg_front = 0.5 * (bid_obs.front_depth + ask_obs.front_depth)
        front_target = max(float(self.levels) + self.config.limit_rate, 1.0)
        front_gap = clamp((front_target - avg_front) / front_target, -1.0, 1.0)

        self._latent.total_liquidity_log = clamp(
            (params.state_persistence * self._latent.total_liquidity_log)
            + _mean_signal(front_gap, avg_shortage, -spread_component)
            + float(self._rng.normal(0.0, params.state_noise)),
            -1.25,
            1.25,
        )
        self._latent.side_budget = clamp(
            (params.state_persistence * self._latent.side_budget)
            + _mean_signal(shortage_imbalance, -depth_imbalance, -support_imbalance)
            + float(self._rng.normal(0.0, params.state_noise)),
            -1.5,
            1.5,
        )

        fair_gap = self._fair_gap(state.mid_tick)
        self._latent.execution_state = clamp(
            (params.state_persistence * self._latent.execution_state)
            + _mean_signal(fair_gap, -depth_imbalance, -self._latent.realized_flow_memory)
            + float(self._rng.normal(0.0, params.state_noise)),
            -2.0,
            2.0,
        )
        self._latent.resilience = clamp(
            (params.state_persistence * self._latent.resilience)
            + _mean_signal(avg_shortage, spread_component, abs(self._latent.realized_flow_memory))
            - 1.0
            + float(self._rng.normal(0.0, params.state_noise)),
            -1.25,
            2.25,
        )

        move = clamp(
            (self.config.mean_reversion * (state.mid_tick - self._fair_tick))
            + float(self._rng.normal(0.0, self.config.fair_price_vol)),
            -float(self.config.max_fair_move_ticks),
            float(self.config.max_fair_move_ticks),
        )
        self._fair_tick += move

        self._latent.bid_mix = self._advance_mix_state(self._latent.bid_mix, bid_obs)
        self._latent.ask_mix = self._advance_mix_state(self._latent.ask_mix, ask_obs)
        self._latent.bid_mass = self._advance_mass_distribution(self._latent.bid_mass, self._latent.bid_mix, bid_obs)
        self._latent.ask_mass = self._advance_mass_distribution(self._latent.ask_mass, self._latent.ask_mix, ask_obs)
        return self._fair_gap(state.mid_tick)

    def _advance_mix_state(self, mix_state: np.ndarray, obs: _SideObservables) -> np.ndarray:
        params = self._params
        drive = np.asarray(
            [
                float(np.dot(self._front_basis, obs.shortage)),
                float(np.dot(self._mid_basis, obs.shortage)),
                float(np.dot(self._far_basis, obs.shortage)),
            ],
            dtype=float,
        )
        regime = np.asarray(
            [
                max(self._latent.resilience, 0.0),
                0.0,
                max(-self._latent.resilience, 0.0),
            ],
            dtype=float,
        )
        centered = (drive + regime) - np.mean(drive + regime)
        noise = self._rng.normal(0.0, params.state_noise, size=3)
        return np.clip((params.mix_persistence * mix_state) + centered + noise, -3.0, 3.0)

    def _advance_mass_distribution(
        self,
        mass: np.ndarray,
        mix_state: np.ndarray,
        obs: _SideObservables,
    ) -> np.ndarray:
        params = self._params
        target = self._mass_target(obs, mix_state)
        noisy_target = target * np.exp(self._rng.normal(0.0, params.state_noise, size=self._depth_cells))
        neighbor = _neighbor_mean(mass)
        advanced = (
            (params.mass_persistence * mass)
            + (params.mass_diffusion * neighbor)
            + ((1.0 - params.mass_persistence) * noisy_target)
        )
        return _normalize_positive(advanced)

    def _side_observables(self, visible: np.ndarray) -> _SideObservables:
        depth: np.ndarray = visible.astype(float)
        log_depth = np.log1p(depth)
        occupancy = depth / (1.0 + depth)
        shortage = np.clip(
            (self._profile_target - occupancy) / np.maximum(self._profile_target, 1e-9),
            0.0,
            1.5,
        )
        neighbor_log = self._neighbor_log_support(log_depth)
        support = float(np.sum(self._touch_weights * log_depth[: self._touch_horizon]))
        front_depth = float(np.sum(depth[: self.levels]))
        if self._touch_horizon <= 1:
            connectedness = 0.0
        else:
            connected_pairs = np.sqrt(
                np.clip(
                    occupancy[: self._touch_horizon - 1] * occupancy[1 : self._touch_horizon],
                    0.0,
                    1.0,
                )
            )
            connectedness = float(np.mean(connected_pairs))
        touch_shortage = float(np.sum(self._touch_weights * shortage[: self._touch_horizon]))
        return _SideObservables(
            depth=depth,
            log_depth=log_depth,
            occupancy=occupancy,
            shortage=shortage,
            neighbor_log=neighbor_log,
            support=support,
            front_depth=front_depth,
            connectedness=connectedness,
            touch_shortage=touch_shortage,
        )

    def _neighbor_log_support(self, log_depth: np.ndarray) -> np.ndarray:
        neighbor = np.empty_like(log_depth)
        neighbor[0] = log_depth[1]
        neighbor[-1] = log_depth[-2]
        neighbor[1:-1] = 0.5 * (log_depth[:-2] + log_depth[2:])
        return neighbor

    def _mixture_profile(self, mix_state: np.ndarray) -> np.ndarray:
        weights = _normalize_positive(np.exp(np.clip(mix_state - np.max(mix_state), -20.0, 20.0)))
        profile = (
            (weights[0] * self._front_basis)
            + (weights[1] * self._mid_basis)
            + (weights[2] * self._far_basis)
        )
        return _normalize_positive(profile)

    def _mass_target(self, obs: _SideObservables, mix_state: np.ndarray) -> np.ndarray:
        basis = self._mixture_profile(mix_state)
        shortage = _normalize_positive(obs.shortage + 1e-12)
        front_shortage = _normalize_positive((1.0 + obs.shortage) * self._front_basis)
        return _normalize_positive(basis + shortage + front_shortage)

    def _latent_reveal_means(self, fair_gap: float, depth_imbalance: float) -> tuple[np.ndarray, np.ndarray]:
        bid_obs = self._side_observables(self._visible_bid)
        ask_obs = self._side_observables(self._visible_ask)
        liquidity_state = clamp(
            self._latent.total_liquidity_log + _mean_signal(bid_obs.touch_shortage, ask_obs.touch_shortage),
            -1.0,
            1.0,
        )
        base_budget = self.config.limit_rate * float(np.exp(liquidity_state))
        split = float(_sigmoid(_mean_signal(self._latent.side_budget, -depth_imbalance, fair_gap)))

        bid_budget = base_budget * split * float(np.exp(clamp(bid_obs.touch_shortage, 0.0, 1.0)))
        ask_budget = base_budget * (1.0 - split) * float(np.exp(clamp(ask_obs.touch_shortage, 0.0, 1.0)))

        bid_score = self._cell_scores(bid_obs, self._latent.bid_mass)
        ask_score = self._cell_scores(ask_obs, self._latent.ask_mass)
        bid_mean = bid_budget * (bid_score / np.sum(bid_score))
        ask_mean = ask_budget * (ask_score / np.sum(ask_score))
        bid_mean = bid_mean * self._rng.gamma(self._params.cox_shape, 1.0 / self._params.cox_shape, size=self._depth_cells)
        ask_mean = ask_mean * self._rng.gamma(self._params.cox_shape, 1.0 / self._params.cox_shape, size=self._depth_cells)
        return bid_mean.astype(float), ask_mean.astype(float)

    def _cell_scores(self, obs: _SideObservables, mass: np.ndarray) -> np.ndarray:
        score = (
            mass
            * np.exp(np.clip(obs.shortage, 0.0, 1.0))
            * (1.0 + np.maximum(obs.neighbor_log, 0.0))
            / (1.0 + obs.log_depth)
        )
        return np.maximum(score, 1e-9)

    def _apply_cancel_thinning(self, fair_gap: float, bid_mean: np.ndarray, ask_mean: np.ndarray) -> None:
        bid_obs = self._side_observables(self._visible_bid)
        ask_obs = self._side_observables(self._visible_ask)
        depth_term = self._grid_norm
        latent_imbalance = _continuous_imbalance(
            float(np.sum(bid_mean[: self.levels])),
            float(np.sum(ask_mean[: self.levels])),
        )
        adverse_signal = float(np.tanh(self._latent.execution_state + fair_gap + latent_imbalance))
        baseline = float(np.log(self.config.cancel_rate / self.config.limit_rate))
        resilience_buffer = max(self._latent.resilience, 0.0)

        bid_signal = (
            baseline
            - adverse_signal
            + (depth_term - np.mean(depth_term))
            + bid_obs.log_depth
            - np.log1p(bid_mean)
            - bid_obs.shortage
            - resilience_buffer
        )
        ask_signal = (
            baseline
            + adverse_signal
            + (depth_term - np.mean(depth_term))
            + ask_obs.log_depth
            - np.log1p(ask_mean)
            - ask_obs.shortage
            - resilience_buffer
        )

        bid_cancel = self._rng.binomial(self._visible_bid, _sigmoid(bid_signal))
        ask_cancel = self._rng.binomial(self._visible_ask, _sigmoid(ask_signal))
        self._visible_bid = np.maximum(0, self._visible_bid - bid_cancel)
        self._visible_ask = np.maximum(0, self._visible_ask - ask_cancel)

    def _apply_limit_reveals(self, bid_mean: np.ndarray, ask_mean: np.ndarray, *, scale: float = 1.0) -> None:
        reveal_scale = max(scale, 0.0)
        self._visible_bid = self._visible_bid + np.asarray(
            self._rng.poisson(bid_mean * reveal_scale),
            dtype=np.int64,
        )
        self._visible_ask = self._visible_ask + np.asarray(
            self._rng.poisson(ask_mean * reveal_scale),
            dtype=np.int64,
        )

    def _apply_market_flow(self, fair_gap: float, depth_imbalance: float) -> None:
        spread_component = self._spread_component(self._visible_state().spread_ticks)
        signal = float(
            np.tanh(
                _mean_signal(
                    self._latent.execution_state,
                    fair_gap,
                    -depth_imbalance,
                    self._latent.realized_flow_memory,
                )
            )
        )
        activity_state = float(
            _mean_signal(abs(signal), -spread_component, max(-self._latent.resilience, 0.0))
        )
        activity = self.config.market_rate * float(np.exp(activity_state - 0.5))
        activity *= max(1.0 - spread_component, 1.0 / max(float(self.levels + 1), 1.0))
        activity *= float(self._rng.gamma(self._params.cox_shape, 1.0 / self._params.cox_shape))

        total_count = int(self._rng.poisson(max(activity, 0.0)))
        buy_prob = float(_sigmoid(signal))
        depth_before = max(
            1.0,
            float(np.sum(self._visible_bid[: self.levels])) + float(np.sum(self._visible_ask[: self.levels])),
        )

        for _ in range(total_count):
            aggressor: AggressorSide = "buy" if self._rng.random() < buy_prob else "sell"
            opposing_visible = self._visible_ask if aggressor == "buy" else self._visible_bid
            front_depth = float(np.sum(opposing_visible[: self.levels]))
            qty_scale = _slice_liquidity_scale(front_depth, self.levels)
            qty = self._scaled_quantity(qty_scale)
            filled = self._execute_market_order(aggressor, qty)
            if aggressor == "buy":
                self._buy_aggr_volume += float(filled)
            else:
                self._sell_aggr_volume += float(filled)

        net_flow = (self._buy_aggr_volume - self._sell_aggr_volume) / depth_before
        self._latent.realized_flow_memory = clamp(
            (self._params.state_persistence * self._latent.realized_flow_memory) + net_flow,
            -2.0,
            2.0,
        )

    def _execute_market_order(self, aggressor: AggressorSide, qty: int) -> int:
        remaining = int(qty)
        filled = 0
        if remaining <= 0:
            return 0

        visible = self._visible_ask if aggressor == "buy" else self._visible_bid
        side: BookSide = "ask" if aggressor == "buy" else "bid"
        for distance, resting in enumerate(visible):
            if remaining <= 0:
                break
            if resting <= 0:
                continue
            take = min(int(resting), remaining)
            visible[distance] -= take
            remaining -= take
            filled += take
            self._last_trade_tick = self._tick_at_distance(side, distance)
        return filled

    def _repair_visible_arrays(self) -> None:
        self._ensure_side_presence("bid")
        self._ensure_side_presence("ask")
        self._enforce_spread_cap()

    def _ensure_side_presence(self, side: BookSide) -> None:
        visible = self._visible_bid if side == "bid" else self._visible_ask
        if np.any(visible > 0):
            return
        visible[0] = max(visible[0], self._sample_quantity())

    def _enforce_spread_cap(self) -> None:
        for _ in range(self.config.max_spread_ticks + 2):
            bid_index = _first_positive_index(self._visible_bid)
            ask_index = _first_positive_index(self._visible_ask)
            if bid_index is None or ask_index is None:
                return
            spread_ticks = bid_index + ask_index + 1
            if spread_ticks <= self.config.max_spread_ticks:
                return

            target_bid = max(0, self.config.max_spread_ticks - ask_index - 1)
            target_ask = max(0, self.config.max_spread_ticks - bid_index - 1)
            if ask_index > bid_index:
                self._visible_ask[target_ask] += self._sample_quantity()
            elif bid_index > ask_index:
                self._visible_bid[target_bid] += self._sample_quantity()
            elif self._fair_tick >= float(self._center_tick):
                self._visible_bid[target_bid] += self._sample_quantity()
            else:
                self._visible_ask[target_ask] += self._sample_quantity()

    def _recenter_visible_arrays(self) -> None:
        bid_index = _first_positive_index(self._visible_bid)
        ask_index = _first_positive_index(self._visible_ask)
        if bid_index is None or ask_index is None:
            return
        best_bid_tick = self._tick_at_distance("bid", bid_index)
        best_ask_tick = self._tick_at_distance("ask", ask_index)
        new_center = _center_from_quotes(best_bid_tick, best_ask_tick)
        delta = new_center - self._center_tick
        if delta == 0:
            return
        self._visible_bid = _shift_bid(self._visible_bid, delta)
        self._visible_ask = _shift_ask(self._visible_ask, delta)
        self._center_tick = new_center

    def _sync_book_from_visible(self) -> None:
        book = OrderBook(self.tick_size)
        for distance, qty in enumerate(self._visible_bid):
            if qty <= 0:
                continue
            book.add_limit("bid", self._tick_at_distance("bid", distance), int(qty))
        for distance, qty in enumerate(self._visible_ask):
            if qty <= 0:
                continue
            book.add_limit("ask", self._tick_at_distance("ask", distance), int(qty))
        self._book = book
        self._invalidate_summary_state()

    def _tick_at_distance(self, side: BookSide, distance: int) -> int:
        if side == "bid":
            return max(0, self._center_tick - int(distance))
        return self._center_tick + 1 + int(distance)

    def _spread_component(self, spread_ticks: int) -> float:
        if self.config.max_spread_ticks <= 1:
            return 0.0
        return clamp((spread_ticks - 1) / float(self.config.max_spread_ticks - 1), 0.0, 1.0)

    def _fair_gap(self, mid_tick: float) -> float:
        return clamp((self._fair_tick - mid_tick) / float(self.config.max_fair_move_ticks), -1.0, 1.0)

    def _sample_quantity(self) -> int:
        raw_value = float(self._rng.lognormal(self.config.size_mean, self.config.size_dispersion))
        return bounded_int(raw_value, self.config.min_order_qty, self.config.max_order_qty)

    def _scaled_quantity(self, scale: float) -> int:
        scale_floor = 1.0 / max(float(self.config.max_order_qty), 1.0)
        return bounded_int(
            round(self._sample_quantity() * max(scale, scale_floor)),
            self.config.min_order_qty,
            self.config.max_order_qty,
        )

    def _public_levels(self, side: BookSide) -> list[dict[str, float]]:
        return [
            {"price": tick_to_price(tick, self.tick_size), "qty": float(qty)}
            for tick, qty in self._book.levels(side, self.levels)
        ]

    def _require_visual_capture(self) -> VisualHistoryStore:
        if self._visual_history is None:
            raise RuntimeError("plot() and plot_heatmap() require Market(..., capture='visual')")
        return self._visual_history


def _sigmoid(values: np.ndarray | float) -> np.ndarray | float:
    clipped = np.clip(values, -60.0, 60.0)
    return 1.0 / (1.0 + np.exp(-clipped))


def _normalize_positive(values: np.ndarray) -> np.ndarray:
    clipped = np.maximum(values.astype(float), 1e-12)
    total = float(np.sum(clipped))
    if total <= 1e-12:
        return np.full_like(clipped, 1.0 / max(clipped.size, 1), dtype=float)
    return clipped / total


def _mean_signal(*values: float) -> float:
    return float(np.mean(np.asarray(values, dtype=float)))


def _first_positive_index(values: np.ndarray) -> int | None:
    positions = np.flatnonzero(values > 0)
    if positions.size == 0:
        return None
    return int(positions[0])


def _center_from_quotes(best_bid_tick: int, best_ask_tick: int) -> int:
    spread_ticks = max(best_ask_tick - best_bid_tick, 1)
    return best_bid_tick + ((spread_ticks - 1) // 2)


def _shift_bid(values: np.ndarray, delta: int) -> np.ndarray:
    shifted = np.zeros_like(values)
    if delta >= 0:
        if delta < values.size:
            shifted[delta:] = values[: values.size - delta]
        return shifted
    offset = -delta
    if offset < values.size:
        shifted[: values.size - offset] = values[offset:]
    return shifted


def _shift_ask(values: np.ndarray, delta: int) -> np.ndarray:
    shifted = np.zeros_like(values)
    if delta >= 0:
        if delta < values.size:
            shifted[: values.size - delta] = values[delta:]
        return shifted
    offset = -delta
    if offset < values.size:
        shifted[offset:] = values[: values.size - offset]
    return shifted


def _neighbor_mean(values: np.ndarray) -> np.ndarray:
    neighbor = np.empty_like(values, dtype=float)
    neighbor[0] = values[1]
    neighbor[-1] = values[-2]
    neighbor[1:-1] = 0.5 * (values[:-2] + values[2:])
    return neighbor


def _slice_liquidity_scale(front_depth: float, levels: int) -> float:
    reference = max(float((2 * levels) + 2), 1.0)
    ratio = max(front_depth, 0.0) / (reference + max(front_depth, 0.0))
    return clamp(ratio, 1.0 / reference, 1.0)


def _continuous_imbalance(lhs: float, rhs: float) -> float:
    total = lhs + rhs
    if total <= 1e-12:
        return 0.0
    return (lhs - rhs) / total


__all__ = ["Market", "SimulationResult"]
