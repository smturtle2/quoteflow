# Getting Started

[Docs index](https://github.com/smturtle2/quoteflow/blob/main/docs/README.md) | [한국어](https://github.com/smturtle2/quoteflow/blob/main/docs/ko/getting-started.md)

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

market = Market(seed=42, config={"preset": "trend"})
market.gen(steps=1_000)

snapshot = market.get()
history = market.get_history()
figure = market.plot()
```

## Constructor

```python
Market(
    init_price=100.0,
    tick_size=0.01,
    levels=5,
    seed=None,
    config=None,
)
```

- `init_price`: initial reference price, snapped to the nearest tick
- `tick_size`: order book price increment
- `levels`: visible depth returned by `get()` and default plot depth
- `seed`: deterministic random seed
- `config`: `dict` or `orderwave.config.MarketConfig`

## Built-in Plots

```python
overview = market.plot()
book = market.plot_book()
diagnostics = market.plot_diagnostics()
```

- `plot()` renders price, spread, trade strength, and a signed visible-book heatmap
- `plot_book()` renders the current order book on a real price axis
- `plot_diagnostics()` renders spread, imbalance, volatility, and regime diagnostics

Every plotting method returns a `matplotlib.figure.Figure`. Saving or displaying the figure stays under user control.

## Presets

- `balanced`: default setting with moderate flow and spread behavior
- `trend`: stronger directional persistence and fair-value pressure
- `volatile`: wider spread tendency and higher cancellation or market pressure

## Snapshot Behavior

The current state returned by `get()` is intentionally compact.

- `mid_price` follows the best bid and ask
- `last_price` only updates on real executions
- `trade_strength` is a symmetric `[-1, 1]` signed flow indicator
- `bids` and `asks` contain up to `levels` visible price levels

## Reproducibility

The simulator uses a NumPy random generator seeded at construction time. Two markets created with the same arguments and seed should generate the same path.
