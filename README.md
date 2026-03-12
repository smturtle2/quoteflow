# orderwave

Compact aggregate order-book simulation for Python, with readable built-in heatmaps.

`orderwave` keeps the runtime model small: a sparse bid/ask book, bounded mean-reverting fair value, and a latent-liquidity Cox kernel that reveals, cancels, and sweeps aggregate depth from a stochastic hidden state instead of a large hand-written heuristic tree.

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
- `plot()`: render the price path with a stable level-ranked signed-depth heatmap. Requires `capture="visual"`.
- `plot_heatmap(anchor="mid" | "price")`: render a standalone heatmap on stable level coordinates. Requires `capture="visual"`.
- `plot_book()`: render the current order book.

`capture="summary"` keeps the fast path lean. `capture="visual"` stores a fixed signed-depth window around the moving market center so the heatmap can show sweep, void, and refill structure clearly. Heatmap rows are always fixed visible ranks, laid out as `ask N ... ask 1 | bid 1 ... bid N`, so they do not drift vertically with price.

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

- Fair price follows a bounded mean-reverting Gaussian process with weak flow coupling.
- Hidden liquidity evolves as a stochastic latent state: total liquidity, side skew, flow bias, and depth-cell fields move first, then visible limit/cancel/market flow is sampled from those states through Cox-Poisson style intensities.
- Visible depth is not rebuilt with symptom-specific rules. Thin-side recovery comes from shortage-aware reveal budgets, connected queue scoring, and smooth cancel thinning rather than hard visible-level floors.
- Repair is safety-only: it prevents one-sided or crossed books and enforces the spread cap, but it does not cosmetically repad every visible rank.

## Realism Profiling

Profile generic microstructure behavior with:

```bash
python -m scripts.profile_realism --steps 5000
```

The profiler reports spread/impact persistence, trade-sign autocorrelation, top-rank gap frequency, per-rank depth shape, visible/full-book one-sidedness, near-touch connectivity, and pair-distribution entropy.

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
