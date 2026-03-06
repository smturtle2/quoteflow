# orderwave

[![PyPI version](https://img.shields.io/pypi/v/orderwave.svg)](https://pypi.org/project/orderwave/)
[![Python versions](https://img.shields.io/pypi/pyversions/orderwave.svg)](https://pypi.org/project/orderwave/)
[![Release workflow](https://github.com/smturtle2/quoteflow/actions/workflows/workflow.yml/badge.svg)](https://github.com/smturtle2/quoteflow/actions/workflows/workflow.yml)

Order-flow-driven synthetic market simulation for Python.

`orderwave` does not sample price first and explain it later. It simulates a sparse limit order book, stochastic limit arrivals, marketable flow, cancellations, and inside-spread quote improvement, then lets price emerge from those book changes.

[English Docs](https://github.com/smturtle2/quoteflow/tree/main/docs) | [한국어 README](https://github.com/smturtle2/quoteflow/blob/main/README.ko.md) | [한국어 문서](https://github.com/smturtle2/quoteflow/tree/main/docs/ko)

![orderwave overview](https://raw.githubusercontent.com/smturtle2/quoteflow/main/docs/assets/orderwave-overview.png)

## Why orderwave

- Minimal public API: `from orderwave import Market`
- Price changes only as a consequence of book mechanics
- Hidden fair value biases flow without directly overwriting price
- Regime switching creates calmer, directional, and stressed periods
- Same seed, same path

## Installation

```bash
pip install orderwave
```

For local development:

```bash
pip install -e .[dev]
```

## Quick Start

```python
from orderwave import Market

market = Market(seed=42, config={"preset": "balanced"})

market.step()
market.gen(steps=1_000)

snapshot = market.get()
history = market.get_history()

print(snapshot["mid_price"], snapshot["best_bid"], snapshot["best_ask"])
print(history.tail())
```

## What You Get

- `mid_price` as the primary quote-driven path
- `last_price` as the most recent executed trade price
- Visible bid/ask ladders with aggregate depth
- Compact history as a `pandas.DataFrame`
- Presets for balanced, trend, and volatile behavior

## Public API

```python
from orderwave import Market

market = Market(
    init_price=100.0,
    tick_size=0.01,
    levels=5,
    seed=42,
    config={"preset": "trend"},
)
```

| API | Purpose |
| --- | --- |
| `step()` | Run one micro-batch and return the latest snapshot |
| `gen(steps=n)` | Run `n` micro-batches and return the latest snapshot |
| `get()` | Return the current snapshot |
| `get_history()` | Return a compact `pandas.DataFrame` history |

Advanced configuration is available through `orderwave.config.MarketConfig`.

## Snapshot Semantics

`Market.get()` returns a dictionary with prices, spread, visible depth, recent aggressive volume, trade strength, depth imbalance, and regime.

Important distinction:

- `mid_price` can move when quotes improve, cancel, or get depleted
- `last_price` only changes when a trade actually executes

## Visualization Example

The repository includes a matplotlib example that plots price, trade strength, and a visible-book heatmap:

```bash
pip install matplotlib
python examples/plot_market_heatmap.py --steps 2000 --preset trend
```

Save directly to a file:

```bash
python examples/plot_market_heatmap.py --steps 2000 --preset trend --output artifacts/orderwave_heatmap.png
```

## Presets At A Glance

![orderwave presets](https://raw.githubusercontent.com/smturtle2/quoteflow/main/docs/assets/orderwave-presets.png)

`balanced`, `trend`, and `volatile` reuse the same public API while shifting spread behavior, flow pressure, and hidden fair-price dynamics.

## Diagnostics

![orderwave diagnostics](https://raw.githubusercontent.com/smturtle2/quoteflow/main/docs/assets/orderwave-diagnostics.png)

The simulator is designed to expose useful microstructure diagnostics such as spread variation, imbalance lead, volatility clustering, and regime occupancy.

## Documentation

- [Documentation index](https://github.com/smturtle2/quoteflow/blob/main/docs/README.md)
- [Getting started](https://github.com/smturtle2/quoteflow/blob/main/docs/getting-started.md)
- [API reference](https://github.com/smturtle2/quoteflow/blob/main/docs/api.md)
- [Examples](https://github.com/smturtle2/quoteflow/blob/main/docs/examples.md)
- [Release guide](https://github.com/smturtle2/quoteflow/blob/main/docs/releasing.md)
- [한국어 문서 인덱스](https://github.com/smturtle2/quoteflow/blob/main/docs/ko/README.md)

## Design Guarantees

- Price is never random-walked directly
- Quote improvement, best-quote depletion, and market execution are the only price-moving mechanisms
- Visible history starts at `step == 0` with the seeded initial book
- Aggregate depth is modeled without exposing per-order FIFO complexity in v1
