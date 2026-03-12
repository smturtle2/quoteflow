# API Reference

[Docs index](https://github.com/smturtle2/quoteflow/blob/main/docs/en/README.md) | [한국어](https://github.com/smturtle2/quoteflow/blob/main/docs/ko/api.md)

## Public Import

```python
from orderwave import Market
```

`Market` is the supported public entry point.
Internal modules such as `orderwave.model` and `orderwave._model` are implementation details rather than stable library API.

For typed helpers:

```python
from orderwave.market import BookLevel, MarketSnapshot, SimulationResult
```

## `Market`

```python
Market(
    init_price=100.0,
    tick_size=0.01,
    levels=5,
    seed=None,
    config=None,
    *,
    preset=None,
    logging_mode=None,
    liquidity_backstop=None,
)
```

`Market` seeds an initial aggregate order book at `step == 0`, records compact history immediately, and keeps private visual history for built-in plotting.
`orderwave` should still be read as an aggregate order-book market-state simulator: it focuses on path, book, and regime dynamics rather than order-level fill precision.

### `step() -> dict`

Advance the simulator by one micro-batch and return the latest snapshot.

### `gen(steps: int) -> dict`

Advance the simulator by `steps` micro-batches and return the latest snapshot.

### `run(steps: int) -> SimulationResult`

Advance the simulator by `steps` micro-batches and return a bundled result object.

`SimulationResult` contains:

- `snapshot`
- `history`
- `event_history`
- `debug_history`
- `labeled_event_history`

In `history_only` mode, the history table remains available and the event/debug fields are `None`.

### `get() -> dict`

Return the current snapshot.

### `get_snapshot() -> MarketSnapshot`

Return the current snapshot as a typed dataclass.

Snapshot fields include:

- `step`
- `day`
- `session_step`
- `session_phase`
- `last_price`
- `mid_price`
- `microprice`
- `best_bid`
- `best_ask`
- `spread`
- `bids`
- `asks`
- `last_trade_side`
- `last_trade_qty`
- `buy_aggr_volume`
- `sell_aggr_volume`
- `trade_strength`
- `depth_imbalance`
- `regime`
- `visible_levels_bid`
- `visible_levels_ask`
- `drought_age`
- `recovery_pressure`
- `impact_residue`
- `regime_dwell`
- `inventory_pressure`

`trade_strength` is a realized-trade signed imbalance from an EWMA of aggressor buy and sell volume.
Quote-only book changes do not move it.

### `get_history() -> pandas.DataFrame`

Return compact history from the initial seeded book through the current step.

Important columns:

- `step`
- `day`
- `session_step`
- `session_phase`
- `last_price`
- `mid_price`
- `microprice`
- `best_bid`
- `best_ask`
- `spread`
- `buy_aggr_volume`
- `sell_aggr_volume`
- `trade_strength`
- `depth_imbalance`
- `regime`
- `top_n_bid_qty`
- `top_n_ask_qty`
- `realized_vol`
- `signed_flow`
- `visible_levels_bid`
- `visible_levels_ask`
- `drought_age`
- `recovery_pressure`
- `impact_residue`
- `regime_dwell`
- `inventory_pressure`

### `get_event_history() -> pandas.DataFrame`

Return the applied event log from `step == 1` onward.

Columns:

- `step`
- `event_idx`
- `day`
- `session_step`
- `session_phase`
- `event_type`
- `side`
- `level`
- `price`
- `requested_qty`
- `applied_qty`
- `fill_qty`
- `fills`
- `best_bid_after`
- `best_ask_after`
- `mid_price_after`
- `last_trade_price_after`
- `regime`

The log records applied events only.
`market` rows include a `fills` list of `(price, qty)` tuples covering the sweep path.

### `get_debug_history() -> pandas.DataFrame`

Return the event-aligned latent debug stream.

Columns:

- `step`
- `event_idx`
- `day`
- `session_step`
- `session_phase`
- `microphase`
- `source`
- `participant_type`
- `meta_order_id`
- `meta_order_side`
- `meta_order_progress`
- `burst_state`
- `shock_state`
- `drought_age`
- `recovery_pressure`
- `impact_residue`
- `regime_dwell`
- `inventory_pressure`
- `flow_toxicity`
- `maker_stress`
- `quote_revision_wave`
- `refill_pressure`
- `visible_levels_bid`
- `visible_levels_ask`

Interpretation notes:

- `microphase` exposes the internal time-structure bucket used by the engine
- `flow_toxicity` tracks how adverse recent aggressive flow looks to passive liquidity
- `maker_stress` tracks how unstable or defensive the passive side has become
- `quote_revision_wave` flags structural pre-withdrawal revision events
- `refill_pressure` tracks post-depletion passive replenishment pressure

### `get_labeled_event_history() -> pandas.DataFrame`

Return event history joined with aligned debug labels on `step` and `event_idx`.

### `plot(*, levels: int | None = None, title: str | None = None, figsize: tuple[float, float] | None = None) -> matplotlib.figure.Figure`

Render the main overview figure with:

- `mid_price`
- `last_price`
- bid/ask spread band
- `trade_strength`
- signed visible-depth heatmap

### `plot_book(*, levels: int | None = None, title: str | None = None, figsize: tuple[float, float] | None = None) -> matplotlib.figure.Figure`

Render the current order book on a real price axis.
Bid and ask depth are mirrored around zero and the figure highlights best bid, best ask, and microprice.

### `plot_diagnostics(*, imbalance_bins: int = 8, max_lag: int = 12, title: str | None = None, figsize: tuple[float, float] | None = None) -> matplotlib.figure.Figure`

Render a diagnostics figure that includes the classical market-state panels and, when full debug data is available, additional microphase and revision/refill panels.

### Logging Modes

- `logging_mode="full"` keeps summary, event, debug, and plotting history
- `logging_mode="history_only"` keeps summary history plus overview/book plotting state only

In `history_only` mode:

- `get_history()` still works
- `plot()` and `plot_book()` still work
- `get_event_history()`, `get_debug_history()`, `get_labeled_event_history()`, and `plot_diagnostics()` raise `RuntimeError`
