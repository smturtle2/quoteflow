# orderwave

Compact aggregate order-book simulation for Python, with readable built-in heatmaps.

`orderwave` keeps the runtime model small: a sparse bid/ask book, Poisson limit/market/cancel flow, bounded mean-reverting fair value, and a light liquidity-state kernel that creates sweep and refill structure without the older heuristic stack.

![Overview](docs/assets/orderwave-built-in-overview.png)

## Install

```bash
pip install orderwave
```

## Quick Start

```python
from orderwave import Market

market = Market(seed=42, capture="visual")
result = market.run(steps=1_000)

snapshot = result.snapshot
history = result.history
overview = market.plot()
heatmap = market.plot_heatmap(anchor="price")
book = market.plot_book()
```

## Public API

- `Market(...)`: create a simulator with an initial price, tick size, visible depth, seed, optional `MarketConfig`, and `capture="summary" | "visual"`.
- `step()`: advance one step and return the latest snapshot.
- `gen(steps)`: run multiple steps and return the latest snapshot.
- `run(steps)`: return `SimulationResult(snapshot=..., history=...)`.
- `get()`: return the current snapshot as a `dict`.
- `get_history()`: return the summary history as a `pandas.DataFrame`.
- `plot()`: render the price path with a mid-anchored signed-depth heatmap. Requires `capture="visual"`.
- `plot_heatmap(anchor="mid" | "price")`: render a standalone heatmap. Requires `capture="visual"`.
- `plot_book()`: render the current order book.

`capture="summary"` keeps the fast path lean. `capture="visual"` stores a fixed signed-depth window around the moving market center so the heatmap can show sweep, void, and refill structure clearly.

## Snapshot and History

Snapshot fields:

- `step`
- `last_price`
- `mid_price`
- `best_bid`
- `best_ask`
- `spread`
- `bids`
- `asks`
- `bid_depth`
- `ask_depth`
- `depth_imbalance`
- `buy_aggr_volume`
- `sell_aggr_volume`
- `fair_price`

History columns:

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

## Model

- Fair price follows a bounded mean-reverting Gaussian process.
- Limit, market, and cancel counts are sampled from Poisson distributions.
- Event side is driven by fair-value gap, depth imbalance, and recent signed flow.
- Limit placement mixes inside join/improve, best-level refill, and deeper wall placement.
- Aggressive flow raises side-specific stress and refill pressure so the heatmap shows asymmetric withdrawal and recovery.

## Documentation Assets

![Book](docs/assets/orderwave-built-in-current-book.png)

![Diagnostics](docs/assets/orderwave-built-in-diagnostics.png)

![Variants](docs/assets/orderwave-built-in-presets.png)

Regenerate the documentation images with:

```bash
python -m scripts.render_doc_images
```

Render the standalone heatmap example with:

```bash
python -m examples.plot_market_heatmap --output artifacts/orderwave_heatmap.png
```

More docs:

- English: [docs/en/README.md](docs/en/README.md)
- Korean: [README.ko.md](README.ko.md)
- Release process: [docs/en/releasing.md](docs/en/releasing.md)
