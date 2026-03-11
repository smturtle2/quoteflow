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
- `Market` is the supported public API; engine and model internals are implementation details
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

market = Market(seed=42, preset="trend")
result = market.run(steps=1_000)

snapshot = market.get_snapshot()
history = market.get_history()
events = market.get_labeled_event_history()
overview = market.plot()
book = market.plot_book()
diagnostics = market.plot_diagnostics()

print(snapshot.session_phase, snapshot.mid_price, snapshot.best_bid, snapshot.best_ask)
print(history.tail())
print(events.tail())
print(result.debug_history.tail())

overview.savefig("orderwave-overview.png")
```

For lighter long runs where you only need summary history, visible book snapshots, and trade strength:

```python
fast_market = Market(seed=7, preset="balanced", logging_mode="history_only")
summary = fast_market.run(steps=10_000).history
figure = fast_market.plot()
```

## Performance Measurement

Use the single performance runner when you want a quick throughput check plus a `full` vs `history_only` logging comparison.

```bash
python -m scripts.measure_performance --preset balanced --seeds 20 --steps 20000 --outdir artifacts/performance
```

The runner writes:

- `performance_metrics.csv`
- `performance_summary.csv`
- `performance_logging_modes.csv`
- `performance_summary.md`

## Validation Sweep

The repository also includes a validation runner for preset sweeps, knob sensitivity, reproducibility checks, and soak tests.

```bash
python -m scripts.validate_orderwave --profile full --jobs 4 --outdir artifacts/validation
```

The runner writes:

- `validation_summary.md`
- `run_metrics.csv`
- `preset_summary.csv`
- `sensitivity_summary.csv`
- `invariant_failures.csv`
- `acceptance_decision.md`
- `diagnostics_<preset>_<seed>.png` when diagnostics rendering is enabled

Release builds run a separate `Release Validation` job that executes the shorter `--profile release` regression and compares it against `tests/golden/validation_release_baseline.json` before PyPI publish.
That release profile is intentionally kept small so the CI validation gate stays fast.

The next engine improvement target is intentionally narrow: finer intra-step event feedback only.

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
| `get_debug_history()` | Return event-aligned latent debug history for advanced inspection |
| `get_labeled_event_history()` | Return event history joined with latent debug labels |
| `plot()` | Render price, spread, trade strength, and visible-book heatmap |
| `plot_book()` | Render the current order book on a real price axis |
| `plot_diagnostics()` | Render session, excitation, imbalance, spread/volatility, resiliency, and occupancy diagnostics |

Advanced configuration is available through `orderwave.config.MarketConfig`.
Common settings can also be passed directly as `Market(..., preset="trend", logging_mode="history_only", liquidity_backstop="off")`.

`logging_mode="history_only"` keeps summary history plus overview/book plotting data, but disables `get_event_history()`, `get_debug_history()`, and `plot_diagnostics()`.
Default `liquidity_backstop="always"` keeps the synthetic market two-sided and observable by default.
It also restores minimum visible depth after each step so the baseline path stays readable.
`"on_empty"` and `"off"` are available when you want to allow thinner or missing post-step liquidity.

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

`Market.run()` returns a `SimulationResult` bundle with the typed snapshot plus whichever tables are available for the current logging mode.

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

`orderwave.validation` is also a supported advanced API for reproducibility checks, sensitivity sweeps, and validation pipelines.
