# orderwave

[![PyPI version](https://img.shields.io/pypi/v/orderwave.svg)](https://pypi.org/project/orderwave/)
[![Python versions](https://img.shields.io/pypi/pyversions/orderwave.svg)](https://pypi.org/project/orderwave/)
[![Release workflow](https://github.com/smturtle2/quoteflow/actions/workflows/workflow.yml/badge.svg)](https://github.com/smturtle2/quoteflow/actions/workflows/workflow.yml)

Order-flow-driven synthetic market simulation for Python, with built-in visualization.

`orderwave` does not random-walk price directly. It simulates a sparse aggregate limit order book, participant-conditioned limit flow, marketable flow, cancellations, latent meta-orders, exogenous shocks, and session-aware state changes, then lets price emerge from those book mechanics. The same `Market` object can render the path, the current book snapshot, and realism-oriented diagnostics without extra plotting glue.

[English Docs](https://github.com/smturtle2/quoteflow/tree/main/docs/en) | [한국어 README](https://github.com/smturtle2/quoteflow/blob/main/README.ko.md) | [한국어 문서](https://github.com/smturtle2/quoteflow/tree/main/docs/ko)

![orderwave overview](https://raw.githubusercontent.com/smturtle2/quoteflow/main/docs/assets/orderwave-built-in-overview.png)

## Why orderwave

- Minimal public entry point: `from orderwave import Market`
- Price changes only as a consequence of book mechanics
- Hidden fair value, session clock, shocks, and meta-orders bias flow without directly overwriting price
- Same seed, same path
- Built-in figures for overview, current book, and diagnostics
- Thin public event history plus optional latent debug history

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
events = market.get_event_history()
debug = market.get_debug_history()
overview = market.plot()
book = market.plot_book()
diagnostics = market.plot_diagnostics()

print(snapshot["session_phase"], snapshot["mid_price"], snapshot["best_bid"], snapshot["best_ask"])
print(history.tail())
print(events.tail())
print(debug.tail())

overview.savefig("orderwave-overview.png")
```

## Benchmark

The repository benchmark reports both throughput and micro-batch realism metrics.

```bash
python benchmarks/benchmark.py --steps 5000 --preset balanced
```

Expected output includes:

- `steps_per_second`
- `events_per_step`
- `market_flow=buy=... sell=... buy_share=...`

## Validation Sweep

The repository also includes a longer-form validation runner that summarizes preset separation, seed stability, invariants, and knob sensitivity into CSV, PNG, and markdown artifacts.

```bash
python scripts/validate_orderwave.py --steps 10000 --seeds 20 --outdir artifacts/validation
```

## API Surface

| API | Purpose |
| --- | --- |
| `step()` | Run one micro-batch and return the latest snapshot |
| `gen(steps=n)` | Run `n` micro-batches and return the latest snapshot |
| `get()` | Return the current snapshot |
| `get_history()` | Return compact `pandas.DataFrame` history |
| `get_event_history()` | Return the applied event log as a `pandas.DataFrame` |
| `get_debug_history()` | Return event-aligned latent debug history for advanced inspection |
| `plot()` | Render price, spread, trade strength, and visible-book heatmap |
| `plot_book()` | Render the current order book on a real price axis |
| `plot_diagnostics()` | Render session, excitation, imbalance, spread/volatility, resiliency, and occupancy diagnostics |

Advanced configuration is available through `orderwave.config.MarketConfig`.

## Built-in Visualization

All plotting methods return `matplotlib.figure.Figure` and leave save/show control to the caller.

- `plot()` renders the main overview: price, spread, execution-only trade strength, and signed visible-depth heatmap
- `plot_book()` renders the current order book on a real price axis
- `plot_diagnostics()` renders session phase profile, imbalance lead, market-flow excitation, spread-volatility coupling, depletion resiliency, and regime or shock occupancy

![orderwave current book](https://raw.githubusercontent.com/smturtle2/quoteflow/main/docs/assets/orderwave-built-in-current-book.png)

![orderwave diagnostics](https://raw.githubusercontent.com/smturtle2/quoteflow/main/docs/assets/orderwave-built-in-diagnostics.png)

The overview heatmap keeps signed depth. Ask liquidity is red, bid liquidity is blue, `0` maps to a light gray midpoint, and missing levels render as blank background instead of black cells.

## Presets At A Glance

![orderwave presets](https://raw.githubusercontent.com/smturtle2/quoteflow/main/docs/assets/orderwave-built-in-presets.png)

`balanced`, `trend`, and `volatile` reuse the same public API while shifting spread behavior, flow pressure, cancellation pressure, and hidden fair-price dynamics.

## Core Semantics

`Market.get()` returns a compact dictionary with session clock fields, prices, spread, visible depth, aggressive volume, trade strength, depth imbalance, and regime.

`trade_strength` is an execution-only signed imbalance. It is computed from an EWMA of realized aggressor buy and sell volume, so quote-only book changes do not move it.

Important distinction:

- `mid_price` can move when quotes improve, cancel, or get depleted
- `last_price` only changes when a trade actually executes
- `day`, `session_step`, and `session_phase` expose the synthetic intraday clock

Core guarantees:

- Price is never random-walked directly
- Quote improvement, best-quote depletion, and market execution are the only price-moving mechanisms
- Visible history starts at `step == 0` with the seeded initial book
- Applied limit, market, and cancel events are available through `get_event_history()`
- Participant type, meta-order progress, burst state, and shock state are available through `get_debug_history()`
- Aggregate depth is modeled without exposing per-order FIFO complexity in v1

## Docs

- [Documentation index](https://github.com/smturtle2/quoteflow/blob/main/docs/en/README.md)
- [Getting started](https://github.com/smturtle2/quoteflow/blob/main/docs/en/getting-started.md)
- [API reference](https://github.com/smturtle2/quoteflow/blob/main/docs/en/api.md)
- [Examples](https://github.com/smturtle2/quoteflow/blob/main/docs/en/examples.md)
- [Release guide](https://github.com/smturtle2/quoteflow/blob/main/docs/en/releasing.md)
- [한국어 문서 인덱스](https://github.com/smturtle2/quoteflow/blob/main/docs/ko/README.md)
