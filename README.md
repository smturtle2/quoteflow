# orderwave

`orderwave` is a microstructure-first synthetic market simulator.

It does not random walk price. It generates stochastic order flow and limit-order-book updates, then lets price emerge from executions, cancellations, and inside-spread quote improvement.

## Why `orderwave`

- Public API stays extremely small: `from orderwave import Market`
- Price is an outcome of book dynamics, not a directly sampled process
- Internal engine models limit arrivals, marketable flow, cancellations, hidden fair value, and regime shifts
- Same seed produces the same path

## Installation

Install from source:

```bash
pip install -e .
```

Install with test dependencies:

```bash
pip install -e .[dev]
```

Once published:

```bash
pip install orderwave
```

## Quick Start

```python
from orderwave import Market

market = Market(seed=42)

market.step()
market.gen(steps=1_000)

snapshot = market.get()
history = market.get_history()

print(snapshot["mid_price"], snapshot["best_bid"], snapshot["best_ask"])
print(history.tail())
```

## Public API

```python
from orderwave import Market

market = Market(
    init_price=100.0,
    tick_size=0.01,
    levels=5,
    seed=42,
    config={"preset": "balanced"},
)
```

Available methods:

- `step()` -> advance one micro-batch and return the latest snapshot
- `gen(steps=n)` -> advance `n` steps and return the latest snapshot
- `get()` -> return the current snapshot
- `get_history()` -> return compact history as a `pandas.DataFrame`

Supported presets:

- `balanced`
- `trend`
- `volatile`

`config` accepts either a plain `dict` or `orderwave.config.MarketConfig`.

## Snapshot Shape

`Market.get()` returns a dictionary with:

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

`last_price` is the last trade price. If the book moves without a trade, `mid_price` can change while `last_price` stays fixed.

## Model Outline

Each `step()` is treated as a micro-batch:

1. Compute state features from the current book
2. Update the market regime: `calm`, `directional`, or `stressed`
3. Update hidden fair value
4. Sample new limit orders, marketable flow, and cancellations
5. Shuffle events and apply them to the book
6. Record a snapshot and a compact history row

Price moves only through book mechanics:

- market buy removes ask liquidity
- market sell removes bid liquidity
- cancellation removes the best quote
- a new order improves the quote inside the spread

## Diagnostics Example

```python
from orderwave import Market

market = Market(seed=7, config={"preset": "trend"})
market.gen(steps=5_000)
history = market.get_history()

mid_ret = history["mid_price"].diff().fillna(0.0)
abs_ret = mid_ret.abs()
spread_mean = history["spread"].mean()
imbalance_lead_corr = history["depth_imbalance"].corr(mid_ret.shift(-1).fillna(0.0))
vol_cluster = abs_ret.autocorr(lag=1)

print("spread mean:", spread_mean)
print("imbalance -> next return corr:", imbalance_lead_corr)
print("|return| lag-1 autocorr:", vol_cluster)
```

## Benchmark

Run the benchmark helper:

```bash
python benchmarks/benchmark.py --steps 100000 --preset balanced
```

## PyPI Release Workflow

This repository includes `.github/workflows/workflow.yml` for CI and PyPI publishing.

The workflow:

- runs tests on pushes and pull requests
- builds sdist and wheel artifacts
- publishes to PyPI only when a GitHub Release is published

To enable trusted publishing on PyPI, configure a GitHub trusted publisher for:

- repository owner: `smturtle2`
- repository name: `quoteflow`
- workflow filename: `.github/workflows/workflow.yml`
- environment name: `pypi`

If the PyPI project does not exist yet, create a pending publisher for the project name `orderwave` before the first release.
