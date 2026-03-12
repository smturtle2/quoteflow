# orderwave Documentation

`orderwave` is a compact event-based aggregate book simulator. It is meant for synthetic order-book paths, lightweight experiments, and deterministic smoke-scale research runs.

## Runtime Surface

- `Market`
- `MarketConfig`
- `SimulationResult`

Everything else is internal. The package no longer exposes plotting helpers, validation helpers, presets, or latent market-state labels.

## MarketConfig

`MarketConfig` exposes exactly these controls:

- `limit_rate`
- `market_rate`
- `cancel_rate`
- `fair_price_vol`
- `mean_reversion`
- `level_decay`
- `size_mean`
- `size_dispersion`
- `min_order_qty`
- `max_order_qty`
- `max_spread_ticks`
- `max_fair_move_ticks`

Validation rules:

- rates must be greater than `0`
- `mean_reversion` must be in `[0, 1]`
- `level_decay` must be in `(0, 1)`
- `size_dispersion` must be greater than `0`
- `min_order_qty` must be at least `1`
- `max_order_qty` must be greater than or equal to `min_order_qty`
- `max_spread_ticks` must be at least `1`
- `max_fair_move_ticks` must be at least `1`

## Snapshot and History

`get()` returns the latest snapshot as a plain dictionary. `get_history()` returns the same core fields over time:

- `step`
- `last_price`
- `mid_price`
- `best_bid`
- `best_ask`
- `spread`
- `bid_depth`
- `ask_depth`
- `depth_imbalance`
- `buy_aggr_volume`
- `sell_aggr_volume`
- `fair_price`

The snapshot also includes visible `bids` and `asks` for the current book view.

## Documentation Images

Regenerate all documentation assets with:

```bash
python -m scripts.render_doc_images
```

The script writes:

- `docs/assets/orderwave-built-in-overview.png`
- `docs/assets/orderwave-built-in-current-book.png`
- `docs/assets/orderwave-built-in-diagnostics.png`
- `docs/assets/orderwave-built-in-presets.png`
