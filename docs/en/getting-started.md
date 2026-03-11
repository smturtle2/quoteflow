# Getting Started

[Docs index](https://github.com/smturtle2/quoteflow/blob/main/docs/en/README.md) | [한국어](https://github.com/smturtle2/quoteflow/blob/main/docs/ko/getting-started.md)

## Install

```bash
pip install orderwave
```

For development:

```bash
pip install -e .[dev]
```

## Minimal Example

```python
from orderwave import Market

market = Market(seed=42, preset="trend")
result = market.run(steps=1_000)

snapshot = market.get_snapshot()
history = market.get_history()
events = market.get_labeled_event_history()
figure = market.plot()
```

For lighter runs where you only need compact history plus overview/book plots:

```python
fast_market = Market(seed=7, preset="balanced", logging_mode="history_only")
summary = fast_market.run(steps=10_000).history
overview = fast_market.plot()
```

## Constructor

```python
Market(
    init_price=100.0,
    tick_size=0.01,
    levels=5,
    seed=None,
    config=None,
    *,
    preset=None,
    logging_mode=None,
    liquidity_backstop=None,
)
```

- `init_price`: initial reference price, snapped to the nearest tick
- `tick_size`: order book price increment
- `levels`: visible depth returned by `get()` and default plot depth
- `seed`: deterministic random seed
- `config`: `dict` or `orderwave.config.MarketConfig`
- `preset`: shortcut for the most common config selection
- `logging_mode`: shortcut for `config["logging_mode"]`
- `liquidity_backstop`: shortcut for `config["liquidity_backstop"]`
- `config["logging_mode"]`: `"full"` or `"history_only"`
- `config["liquidity_backstop"]`: `"always"` (default), `"on_empty"`, or `"off"`

## Built-in Plots

```python
overview = market.plot()
book = market.plot_book()
diagnostics = market.plot_diagnostics()
```

- `plot()` renders price, spread, trade strength, and a signed visible-book heatmap
- `plot_book()` renders the current order book on a real price axis
- `plot_diagnostics()` renders session profile, market-flow excitation, imbalance lead, spread-volatility coupling, resiliency, and regime or shock occupancy
- `plot_diagnostics()` requires `logging_mode="full"`

Every plotting method returns a `matplotlib.figure.Figure`. Saving or displaying the figure stays under user control.

## Presets

- `balanced`: default setting with moderate flow and spread behavior
- `trend`: stronger directional persistence and fair-value pressure
- `volatile`: wider spread tendency and higher cancellation or market pressure

## Snapshot Behavior

The current state returned by `get()` is intentionally compact.

- `mid_price` follows the best bid and ask
- `last_price` only updates on realized trades
- `day`, `session_step`, and `session_phase` expose the synthetic session clock
- `trade_strength` is a symmetric `[-1, 1]` signed flow indicator
- `bids` and `asks` contain up to `levels` visible price levels

## Advanced Inspection

- `get_event_history()` returns the applied event stream only
- `get_debug_history()` returns participant type, meta-order progress, burst state, and shock state aligned to the same `step` and `event_idx` keys
- `get_labeled_event_history()` returns the joined event/debug table without manual `merge(...)`
- `run()` returns a `SimulationResult` bundle with the typed snapshot plus available tables
- `history_only` mode keeps `get_history()`, `plot()`, and `plot_book()`, but disables `get_event_history()`, `get_debug_history()`, and `plot_diagnostics()`
- default `liquidity_backstop="on_empty"` restores a missing side without forcing minimum visible depth after every step

## Reproducibility

The simulator uses a NumPy random generator seeded at construction time. Two markets created with the same arguments and seed should generate the same path.
