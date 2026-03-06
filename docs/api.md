# API Reference

[Docs index](https://github.com/smturtle2/quoteflow/blob/main/docs/README.md) | [한국어](https://github.com/smturtle2/quoteflow/blob/main/docs/ko/api.md)

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

`Market` is the main public entry point. It seeds an initial aggregate order book at `step == 0` and records history immediately.

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
