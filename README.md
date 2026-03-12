# orderwave

[![PyPI version](https://img.shields.io/pypi/v/orderwave.svg)](https://pypi.org/project/orderwave/)
[![Python versions](https://img.shields.io/pypi/pyversions/orderwave.svg)](https://pypi.org/project/orderwave/)
[![Release workflow](https://github.com/smturtle2/quoteflow/actions/workflows/workflow.yml/badge.svg)](https://github.com/smturtle2/quoteflow/actions/workflows/workflow.yml)

Order-flow-driven aggregate order-book market-state simulation for Python, with built-in visualization.

`orderwave` models a sparse aggregate order book, participant-conditioned limit flow, marketable flow, adverse quote revision, passive refill, latent meta-orders, exogenous shocks, and session-aware state changes. Price is still an outcome of book mechanics rather than a directly random-walked process.

`orderwave` is an aggregate order-book market-state simulator.
It is built for believable intraday path generation, book-state diagnostics, preset comparison, and sandbox experimentation.
It is not an order-level matching or fill-precision simulator.

[English Docs](https://github.com/smturtle2/quoteflow/tree/main/docs/en) | [한국어 README](https://github.com/smturtle2/quoteflow/blob/main/README.ko.md) | [한국어 문서](https://github.com/smturtle2/quoteflow/tree/main/docs/ko)

![orderwave overview](https://raw.githubusercontent.com/smturtle2/quoteflow/main/docs/assets/orderwave-built-in-overview.png)

## Why orderwave

- Minimal public entry point: `from orderwave import Market`
- `Market` is the supported public API; engine and model internals stay private
- Price changes only through quote improvement, depletion, marketable flow, and refill dynamics
- Internal microphases shape open release, midday lull, power-hour activity, and closing imbalance behavior
- Same seed, same path
- Built-in overview, current-book, and diagnostics figures
- Thin public history plus richer event-aligned debug history when you want the latent state

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

market = Market(seed=42, preset="trend")
result = market.run(steps=1_000)

snapshot = market.get_snapshot()
history = market.get_history()
events = market.get_event_history()
debug = market.get_debug_history()
overview = market.plot()
book = market.plot_book()
diagnostics = market.plot_diagnostics()

print(snapshot.session_phase, snapshot.mid_price, snapshot.best_bid, snapshot.best_ask)
print(history.tail())
print(events.tail())
print(debug[["microphase", "maker_stress", "quote_revision_wave", "refill_pressure"]].tail())

overview.savefig("orderwave-overview.png")
```

For long runs where compact history and plotting are enough:

```python
fast_market = Market(seed=7, preset="balanced", logging_mode="history_only")
summary = fast_market.run(steps=10_000).history
figure = fast_market.plot()
```

## Core Realism Model

- Aggregate visible book levels rather than per-order FIFO
- Participant mix: `passive_lp`, `inventory_mm`, `noise_taker`, `informed_meta`
- Market-first step cycle with structural pre-withdrawal and post-refill behavior
- Regime persistence plus hazard-like switching between `calm`, `directional`, and `stressed`
- Session phase and internal microphase structure: `open_release`, `morning_trend`, `midday_lull`, `power_hour`, `closing_imbalance`
- Event-aligned latent diagnostics including `flow_toxicity`, `maker_stress`, `quote_revision_wave`, and `refill_pressure`

## API Surface

| API | Purpose |
| --- | --- |
| `step()` | Run one micro-batch and return the latest snapshot |
| `gen(steps=n)` | Run `n` micro-batches and return the latest snapshot |
| `run(steps=n)` | Run `n` micro-batches and return a bundled typed result |
| `get()` | Return the current snapshot |
| `get_snapshot()` | Return the current snapshot as a typed dataclass |
| `get_history()` | Return compact `pandas.DataFrame` history |
| `get_event_history()` | Return the applied event log as a `pandas.DataFrame` |
| `get_debug_history()` | Return event-aligned latent debug history |
| `get_labeled_event_history()` | Return event history joined with latent debug labels |
| `plot()` | Render price, spread, trade strength, and signed visible-depth heatmap |
| `plot_book()` | Render the current order book on a real price axis |
| `plot_diagnostics()` | Render session, excitation, imbalance, spread/volatility, resiliency, occupancy, microphase, and revision/refill diagnostics |

Advanced configuration is available through `orderwave.config.MarketConfig`.
Common settings can also be passed directly as `Market(..., preset="trend", logging_mode="history_only", liquidity_backstop="off")`.

`logging_mode="history_only"` keeps summary history plus overview/book plotting data, but disables `get_event_history()`, `get_debug_history()`, and `plot_diagnostics()`.
Default `liquidity_backstop="on_empty"` restores a missing side without forcing minimum visible depth after every step.
Use `"always"` for more aggressively stabilized books or `"off"` to allow thinner post-step liquidity.

## Built-in Visualization

All plotting methods return `matplotlib.figure.Figure` and leave save/show control to the caller.

- `plot()` renders the main overview: mid/last price, spread band, trade strength, and signed visible-depth heatmap
- `plot_book()` renders the current order book on a real price axis
- `plot_diagnostics()` renders:
  session phase profile, imbalance lead, market-flow excitation, spread-volatility coupling, depletion resiliency, regime/shock occupancy, microphase stress profile, and revision/refill pressure

![orderwave current book](https://raw.githubusercontent.com/smturtle2/quoteflow/main/docs/assets/orderwave-built-in-current-book.png)

![orderwave diagnostics](https://raw.githubusercontent.com/smturtle2/quoteflow/main/docs/assets/orderwave-built-in-diagnostics.png)

The overview heatmap keeps signed depth. Ask liquidity is red, bid liquidity is blue, `0` maps to a light gray midpoint, and missing levels render as blank background.

## Performance Measurement

Use the single performance runner for throughput, memory, and `full` vs `history_only` logging comparison.

```bash
python -m scripts.measure_performance --preset balanced --seeds 20 --steps 20000 --outdir artifacts/performance
```

The runner writes:

- `performance_metrics.csv`
- `performance_summary.csv`
- `performance_logging_modes.csv`
- `performance_summary.md`

## Validation Sweep

The repository also ships a validation runner for preset sweeps, sensitivity checks, reproducibility, and soak tests.

```bash
python -m scripts.validate_orderwave --profile quality_regression --jobs 4 --outdir artifacts/validation
```

The runner writes:

- `validation_summary.md`
- `run_metrics.csv`
- `preset_summary.csv`
- `sensitivity_summary.csv`
- `invariant_failures.csv`
- `acceptance_decision.md`
- `diagnostics_<preset>_<seed>.png` when diagnostics rendering is enabled

Release builds use a shorter smoke workload:

```bash
python -m scripts.validate_orderwave --profile release_smoke --outdir artifacts/validation-release --baseline-json tests/golden/validation_release_baseline.json --fail-on-baseline-drift
```

## Core Semantics

- `mid_price` moves when quotes improve, cancel, or get depleted
- `last_price` only changes on realized trades
- `day`, `session_step`, and `session_phase` expose the synthetic session clock
- `trade_strength` is a realized-trade signed imbalance from aggressor buy/sell EWMA
- `get_debug_history()` adds latent fields such as `microphase`, `flow_toxicity`, `maker_stress`, `quote_revision_wave`, and `refill_pressure`

Core guarantees:

- Price is never random-walked directly
- Quote improvement, best-quote depletion, and market trades are the only price-moving mechanisms
- Visible history starts at `step == 0` with the seeded initial book
- Applied limit, market, and cancel events are available through `get_event_history()`
- Aggregate depth is modeled without exposing per-order FIFO complexity in v1

## Docs

- [Documentation index](https://github.com/smturtle2/quoteflow/blob/main/docs/en/README.md)
- [Getting started](https://github.com/smturtle2/quoteflow/blob/main/docs/en/getting-started.md)
- [API reference](https://github.com/smturtle2/quoteflow/blob/main/docs/en/api.md)
- [Examples](https://github.com/smturtle2/quoteflow/blob/main/docs/en/examples.md)
- [Release guide](https://github.com/smturtle2/quoteflow/blob/main/docs/en/releasing.md)
- [한국어 문서 인덱스](https://github.com/smturtle2/quoteflow/blob/main/docs/ko/README.md)

`orderwave.validation` is also a supported advanced API for reproducibility checks, sensitivity sweeps, and validation pipelines.
