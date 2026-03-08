# Examples

[Docs index](https://github.com/smturtle2/quoteflow/blob/main/docs/en/README.md) | [한국어](https://github.com/smturtle2/quoteflow/blob/main/docs/ko/examples.md)

## Built-in Overview Plot

```python
from orderwave import Market

market = Market(seed=7, config={"preset": "trend"})
market.gen(steps=2_000)

event_history = market.get_event_history()
debug_history = market.get_debug_history()
figure = market.plot(levels=8, title="orderwave overview")
figure.savefig("orderwave-overview.png")

print(event_history.tail())
print(debug_history.tail())
```

![Overview image](../assets/orderwave-built-in-overview.png)

## Current Book Snapshot

```python
book_figure = market.plot_book(levels=8, title="Current order book")
book_figure.savefig("orderwave-current-book.png")
```

![Current book](../assets/orderwave-built-in-current-book.png)

## Diagnostics

```python
diagnostics = market.plot_diagnostics(max_lag=12, title="Diagnostics")
diagnostics.savefig("orderwave-diagnostics.png")
```

![Diagnostics snapshot](../assets/orderwave-built-in-diagnostics.png)

These built-in figures are meant to answer three different questions quickly:

- what path did the simulator generate?
- what does the current book look like?
- does the path have useful microstructure signals?

## Event Flow Inspection

```python
events = market.get_event_history()
market_fills = events.loc[events["event_type"] == "market", ["step", "side", "fill_qty", "fills"]]

print(market_fills.tail())
```

`get_event_history()` exposes the applied event stream, not just the sampled intents. That makes it easier to inspect sweep paths, cancellation pressure, and quote replenishment in the exact order they hit the book.

## Latent Debug Inspection

```python
debug = market.get_debug_history()
joined = market.get_event_history().merge(debug, on=["step", "event_idx"], how="inner")

print(joined[["step", "event_idx", "event_type", "participant_type", "meta_order_id", "shock_state"]].tail())
```

`get_debug_history()` is the advanced inspection view. It keeps latent driver labels out of the thin public event log while still making participant mix, burst state, meta-order progress, and shock state auditable.

## Compact History-Only Runs

```python
fast_market = Market(seed=11, config={"preset": "balanced", "logging_mode": "history_only"})
fast_market.gen(steps=20_000)

summary = fast_market.get_history()
figure = fast_market.plot(title="Compact overview")
figure.savefig("orderwave-history-only.png")
```

`history_only` mode is the lighter option for long sweeps when you only need compact history, visible-book plotting, and trade strength. In this mode, `get_event_history()`, `get_debug_history()`, and `plot_diagnostics()` intentionally raise `RuntimeError`.

## CLI Example

The repository includes [`examples/plot_market_heatmap.py`](https://github.com/smturtle2/quoteflow/blob/main/examples/plot_market_heatmap.py), which calls `Market.plot()` directly.

```bash
python examples/plot_market_heatmap.py --steps 2000 --preset trend --output artifacts/orderwave_heatmap.png
```

## Performance Measurement

Use the single performance script when you want a quick throughput check, a floor check, and a `full` vs `history_only` comparison after engine changes.

```bash
python scripts/measure_performance.py --preset balanced --seeds 20 --steps 20000 --outdir artifacts/performance
```

The script writes:

- `performance_metrics.csv`
- `performance_summary.csv`
- `performance_logging_modes.csv`
- `performance_summary.md`

## Validation Sweep

Use the validation runner when you want the full synthetic market-state validation pipeline rather than a single throughput snapshot.

```bash
python scripts/validate_orderwave.py --baseline-steps 20000 --baseline-seeds 20 --jobs 4 --outdir artifacts/validation
```

The runner writes:

- `validation_summary.md`
- `run_metrics.csv`
- `preset_summary.csv`
- `sensitivity_summary.csv`
- `invariant_failures.csv`
- `acceptance_decision.md`
- `diagnostics_<preset>_<seed>.png`

Release builds compare this output against `tests/golden/validation_baseline.json` and stop before publish if the baseline drifts.

The next engine improvement target is intentionally narrow: finer intra-step event feedback only.

## Preset Comparison

![Preset comparison](../assets/orderwave-built-in-presets.png)

Preset comparison remains a docs-only figure generated from the same public simulation API with different presets.

## Regenerating Docs Images

```bash
python scripts/render_doc_images.py
```
