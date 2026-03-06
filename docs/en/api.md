# API Reference

[Docs index](https://github.com/smturtle2/quoteflow/blob/main/docs/en/README.md) | [한국어](https://github.com/smturtle2/quoteflow/blob/main/docs/ko/api.md)

## Public Import

```python
from orderwave import Market
```

## `Market`

```python
Market(
    init_price=100.0,
    tick_size=0.01,
    levels=5,
    seed=None,
    config=None,
)
```

`Market` is the main public entry point. It seeds an initial aggregate order book at `step == 0`, records compact history immediately, and keeps a private visual history for built-in plotting.

### `step() -> dict`

Advance the simulator by one micro-batch and return the latest snapshot.

### `gen(steps: int) -> dict`

Advance the simulator by `steps` micro-batches and return the latest snapshot.

### `get() -> dict`

Return the current snapshot.

Snapshot fields:

- `step`
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

### `get_history() -> pandas.DataFrame`

Return compact history from the initial seeded book through the current step.

Minimum columns:

- `step`
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

Additional convenience columns may include summary depth and volatility fields.

### `plot(*, levels: int | None = None, title: str | None = None, figsize: tuple[float, float] | None = None) -> matplotlib.figure.Figure`

Render the built-in overview figure with:

- `mid_price`
- `last_price`
- bid/ask spread band
- `trade_strength`
- signed visible-depth heatmap

`levels` defaults to the market's visible depth and is clamped to the internal book buffer.

### `plot_book(*, levels: int | None = None, title: str | None = None, figsize: tuple[float, float] | None = None) -> matplotlib.figure.Figure`

Render the current order book on a real price axis. Bid and ask depth are mirrored around zero and the figure highlights best bid, best ask, and microprice.

### `plot_diagnostics(*, imbalance_bins: int = 8, max_lag: int = 12, title: str | None = None, figsize: tuple[float, float] | None = None) -> matplotlib.figure.Figure`

Render a 2x2 diagnostics figure with:

- spread distribution
- depth imbalance to next mid-return relationship
- absolute-return autocorrelation
- regime occupancy

This method requires at least two recorded history rows.

## `orderwave.config.MarketConfig`

```python
from orderwave.config import MarketConfig
```

`MarketConfig` is the advanced configuration type. It keeps the external surface small by exposing:

- `preset`
- `book_buffer_levels`
- `flow_window`
- `vol_window`
- `limit_rate_scale`
- `market_rate_scale`
- `cancel_rate_scale`
- `fair_price_vol_scale`
- `regime_transition_scale`

`config` passed to `Market` can be either a `MarketConfig` instance or a plain mapping with the same keys.
