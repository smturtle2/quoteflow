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

For long runs where you only need compact history plus overview/book plots:

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

- `init_price`: initial reference price snapped to the nearest tick
- `tick_size`: order book price increment
- `levels`: visible depth returned by `get()` and used by default in plots
- `seed`: deterministic random seed
- `config`: `dict` or `orderwave.config.MarketConfig`
- `preset`: shortcut for preset selection
- `logging_mode`: shortcut for `config["logging_mode"]`
- `liquidity_backstop`: shortcut for `config["liquidity_backstop"]`

## What The Engine Simulates

- Aggregate visible depth rather than per-order FIFO
- Participant-conditioned `limit`, `market`, and `cancel` flow
- Regimes: `calm`, `directional`, `stressed`
- Session phase plus internal microphase structure
- Structural pre-withdrawal before aggressive bursts and passive refill after depletion
- Latent stress diagnostics: `flow_toxicity`, `maker_stress`, `quote_revision_wave`, `refill_pressure`

## Built-in Plots

```python
overview = market.plot()
book = market.plot_book()
diagnostics = market.plot_diagnostics()
```

- `plot()` renders price, spread, trade strength, and a signed visible-book heatmap
- `plot_book()` renders the current order book on a real price axis
- `plot_diagnostics()` renders session profile, imbalance lead, market-flow excitation, spread-volatility coupling, depletion resiliency, regime/shock occupancy, microphase stress profile, and revision/refill pressure
- `plot_diagnostics()` requires `logging_mode="full"`

## Presets

- `balanced`: smoother refill, moderate directional pressure, lower stress persistence
- `trend`: stronger directional dwell and meta-order persistence
- `volatile`: wider spread tails, heavier cancel/revision pressure, slower refill recovery

## Advanced Inspection

- `get_event_history()` returns applied `limit`, `market`, and `cancel` events
- `get_debug_history()` returns event-aligned latent labels and stress fields
- `get_labeled_event_history()` returns the joined event/debug table
- `run()` returns a `SimulationResult` bundle with typed snapshot plus available tables
- `history_only` mode keeps `get_history()`, `plot()`, and `plot_book()`, but disables event/debug APIs and diagnostics

## Reproducibility

The simulator uses a NumPy random generator seeded at construction time.
Two markets created with the same arguments and seed should produce the same path, event log, and debug history.
