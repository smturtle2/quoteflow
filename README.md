# orderwave

Compact aggregate order-book simulation for Python.

`orderwave` generates a sparse bid/ask book, samples limit/market/cancel flow from explicit distributions, and records a small summary history. The public API is intentionally narrow: build a `Market`, step or run it, inspect the latest snapshot, and read the summary frame.

![Overview](docs/assets/orderwave-built-in-overview.png)

## Install

```bash
pip install orderwave
```

## Quick Start

```python
from orderwave import Market

market = Market(seed=42)
result = market.run(steps=1_000)

snapshot = result.snapshot
history = result.history
```

Common overrides stay in `config`:

```python
from orderwave import Market, MarketConfig

config = MarketConfig(
    market_rate=3.0,
    fair_price_vol=0.45,
    max_spread_ticks=4,
)

market = Market(seed=7, config=config)
market.gen(steps=500)
snapshot = market.get()
```

## Public API

- `Market(...)`: create a simulator with an initial price, tick size, visible depth, seed, and optional `MarketConfig`.
- `step()`: advance one step and return the latest snapshot.
- `gen(steps)`: run multiple steps and return the latest snapshot.
- `run(steps)`: return `SimulationResult(snapshot=..., history=...)`.
- `get()`: return the current snapshot as a `dict`.
- `get_history()`: return the summary history as a `pandas.DataFrame`.

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

The simulator keeps only one internal model family:

- Fair price follows a bounded mean-reverting Gaussian process.
- Limit, market, and cancel counts are sampled from Poisson distributions.
- Event side is driven by fair-value gap and current depth imbalance.
- Event level is sampled from a truncated decay distribution.
- Event size is sampled from a bounded lognormal distribution.

There are no presets, participant taxonomies, latent regimes, validation pipelines, or plotting APIs in the runtime package.

## Documentation Assets

![Book](docs/assets/orderwave-built-in-current-book.png)

![Diagnostics](docs/assets/orderwave-built-in-diagnostics.png)

![Variants](docs/assets/orderwave-built-in-presets.png)

Regenerate the documentation images with:

```bash
python -m scripts.render_doc_images
```

More docs:

- English: [docs/en/README.md](docs/en/README.md)
- Korean: [README.ko.md](README.ko.md)
- Release process: [docs/en/releasing.md](docs/en/releasing.md)
