# orderwave

`orderwave` is an order-flow-driven market simulator.

It does not random-walk price directly. Instead, it simulates a limit order book with stochastic limit arrivals, marketable flow, cancellations, and inside-spread quote improvement, then lets price emerge from those book changes.

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

market = Market(seed=42)

market.step()
market.gen(steps=1_000)

snapshot = market.get()
history = market.get_history()

print(snapshot["mid_price"], snapshot["best_bid"], snapshot["best_ask"])
print(history.tail())
```

## Why orderwave

- Minimal public API: `from orderwave import Market`
- Price is an outcome of book dynamics, not a separately sampled process
- Hidden fair value and regime shifts bias order flow without directly overwriting price
- Deterministic paths under the same seed

## API

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

Methods:

- `step()` returns the latest snapshot after one micro-batch
- `gen(steps=n)` advances `n` steps and returns the latest snapshot
- `get()` returns the current snapshot
- `get_history()` returns a compact `pandas.DataFrame`

Supported presets:

- `balanced`
- `trend`
- `volatile`

`config` accepts either a plain `dict` or `orderwave.config.MarketConfig`.

## Snapshot

`Market.get()` returns:

```python
{
    "step": int,
    "last_price": float,
    "mid_price": float,
    "microprice": float,
    "best_bid": float,
    "best_ask": float,
    "spread": float,
    "bids": [{"price": float, "qty": float}, ...],
    "asks": [{"price": float, "qty": float}, ...],
    "last_trade_side": "buy" | "sell" | None,
    "last_trade_qty": float,
    "buy_aggr_volume": float,
    "sell_aggr_volume": float,
    "trade_strength": float,
    "depth_imbalance": float,
    "regime": str,
}
```

`last_price` is the last executed trade price. If the book changes without a trade, `mid_price` can move while `last_price` stays fixed.

## Model

Each `step()` is a micro-batch:

1. Compute state features from the current book
2. Update regime: `calm`, `directional`, or `stressed`
3. Update hidden fair value
4. Sample limit orders, marketable flow, and cancellations
5. Shuffle events and apply them to the book
6. Record the snapshot and compact history row

Price moves only through book mechanics:

- market buy removes ask liquidity
- market sell removes bid liquidity
- cancellation depletes the best quote
- a new limit order improves the quote inside the spread

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

## Maintainer Release

PyPI publishing is wired through [`workflow.yml`](https://github.com/smturtle2/quoteflow/blob/main/.github/workflows/workflow.yml).

On 2026-03-06, [GitHub Actions release event docs](https://docs.github.com/en/actions/using-workflows/events-that-trigger-workflows#release) document that `release.types: [published]` triggers when a release is published, while drafts themselves do not trigger workflows. [PyPI trusted publishing docs](https://docs.pypi.org/trusted-publishers/using-a-publisher/) document the `id-token: write` flow used by the publish job.

Release flow:

1. Update `version` in `pyproject.toml`
2. Commit and push to `main`
3. In GitHub, open `Releases`
4. Click `Draft a new release`
5. Create a tag like `v0.1.0`
6. Set the release title, then click `Publish release`
7. GitHub Actions runs tests, builds the distributions, and publishes to PyPI

Trusted Publisher settings for PyPI:

- PyPI project name: `orderwave`
- Repository owner: `smturtle2`
- Repository name: `quoteflow`
- Workflow filename: `.github/workflows/workflow.yml`
- Environment name: `pypi`

If `orderwave` does not exist on PyPI yet, create the project through a pending publisher first. PyPI notes that a pending publisher does not reserve the name until the first successful publish.
