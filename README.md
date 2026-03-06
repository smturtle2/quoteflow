# orderwave

[![PyPI version](https://img.shields.io/pypi/v/orderwave.svg)](https://pypi.org/project/orderwave/)
[![Python versions](https://img.shields.io/pypi/pyversions/orderwave.svg)](https://pypi.org/project/orderwave/)
[![Release workflow](https://github.com/smturtle2/quoteflow/actions/workflows/workflow.yml/badge.svg)](https://github.com/smturtle2/quoteflow/actions/workflows/workflow.yml)

Order-flow-driven synthetic market simulation for Python, with built-in visualization.

`orderwave` does not random-walk price directly. It simulates a sparse limit order book, stochastic limit arrivals, marketable flow, cancellations, and inside-spread quote improvement, then lets price emerge from those book changes. The same `Market` object can also render the path, the current book snapshot, and microstructure diagnostics without extra plotting glue.

[English Docs](https://github.com/smturtle2/quoteflow/tree/main/docs/en) | [한국어 README](https://github.com/smturtle2/quoteflow/blob/main/README.ko.md) | [한국어 문서](https://github.com/smturtle2/quoteflow/tree/main/docs/ko)

![orderwave overview](https://raw.githubusercontent.com/smturtle2/quoteflow/main/docs/assets/orderwave-built-in-overview.png)

## Why orderwave

- Minimal public entry point: `from orderwave import Market`
- Price changes only as a consequence of book mechanics
- Hidden fair value biases order flow without directly overwriting price
- Same seed, same path
- Built-in figures for overview, current book, and diagnostics

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

market = Market(seed=42, config={"preset": "trend"})
market.gen(steps=1_000)

snapshot = market.get()
history = market.get_history()
overview = market.plot()
book = market.plot_book()
diagnostics = market.plot_diagnostics()

print(snapshot["mid_price"], snapshot["best_bid"], snapshot["best_ask"])
print(history.tail())

overview.savefig("orderwave-overview.png")
```

## API Surface

| API | Purpose |
| --- | --- |
| `step()` | Run one micro-batch and return the latest snapshot |
| `gen(steps=n)` | Run `n` micro-batches and return the latest snapshot |
| `get()` | Return the current snapshot |
| `get_history()` | Return compact `pandas.DataFrame` history |
| `plot()` | Render price, spread, trade strength, and visible-book heatmap |
| `plot_book()` | Render the current order book on a real price axis |
| `plot_diagnostics()` | Render spread, imbalance, volatility, and regime diagnostics |

Advanced configuration is available through `orderwave.config.MarketConfig`.

## Built-in Visualization

All plotting methods return `matplotlib.figure.Figure` and leave save/show control to the caller.

- `plot()` renders the main overview: price, spread, trade strength, and signed visible-depth heatmap
- `plot_book()` renders the current order book on a real price axis
- `plot_diagnostics()` renders spread, imbalance, volatility, and regime diagnostics

![orderwave current book](https://raw.githubusercontent.com/smturtle2/quoteflow/main/docs/assets/orderwave-built-in-current-book.png)

![orderwave diagnostics](https://raw.githubusercontent.com/smturtle2/quoteflow/main/docs/assets/orderwave-built-in-diagnostics.png)

The overview heatmap keeps signed depth. Ask liquidity is red, bid liquidity is blue, `0` maps to a light gray midpoint, and missing levels render as blank background instead of black cells.

## Presets At A Glance

![orderwave presets](https://raw.githubusercontent.com/smturtle2/quoteflow/main/docs/assets/orderwave-built-in-presets.png)

`balanced`, `trend`, and `volatile` reuse the same public API while shifting spread behavior, flow pressure, cancellation pressure, and hidden fair-price dynamics.

## Core Semantics

`Market.get()` returns a compact dictionary with prices, spread, visible depth, aggressive volume, trade strength, depth imbalance, and regime.

Important distinction:

- `mid_price` can move when quotes improve, cancel, or get depleted
- `last_price` only changes when a trade actually executes

Core guarantees:

- Price is never random-walked directly
- Quote improvement, best-quote depletion, and market execution are the only price-moving mechanisms
- Visible history starts at `step == 0` with the seeded initial book
- Aggregate depth is modeled without exposing per-order FIFO complexity in v1

## Docs

- [Documentation index](https://github.com/smturtle2/quoteflow/blob/main/docs/en/README.md)
- [Getting started](https://github.com/smturtle2/quoteflow/blob/main/docs/en/getting-started.md)
- [API reference](https://github.com/smturtle2/quoteflow/blob/main/docs/en/api.md)
- [Examples](https://github.com/smturtle2/quoteflow/blob/main/docs/en/examples.md)
- [Release guide](https://github.com/smturtle2/quoteflow/blob/main/docs/en/releasing.md)
- [한국어 문서 인덱스](https://github.com/smturtle2/quoteflow/blob/main/docs/ko/README.md)
