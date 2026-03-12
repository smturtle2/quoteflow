# orderwave Documentation

`orderwave` is a compact event-based aggregate book simulator with two runtime modes:

- `capture="summary"` for the fast path
- `capture="visual"` for overview and heatmap plots

## Runtime Surface

- `Market`
- `MarketConfig`
- `SimulationResult`

## MarketConfig

`MarketConfig` still exposes only the statistical controls for flow intensity, fair-value movement, price-level decay, order-size bounds, and spread/fair-move limits.

## Runtime Model

- The engine stays aggregate-book only. It does not simulate per-order FIFO queues.
- Realism comes from latent distribution synthesis. A hidden stochastic state combines total-liquidity, side-skew, and side-specific depth distributions before visible limit/cancel/market events are sampled.
- Thin-side recovery is shortage-driven and smooth: shortage distributions and near-touch distributions are re-synthesized continuously instead of using hard visible-level floors.
- Repair is safety-only, so visible holes and delayed refill can survive instead of being erased every step.

## Plotting

The plotting surface is:

- `plot()` for the overview figure
- `plot_heatmap(anchor="mid" | "price")` for a standalone signed-depth heatmap
- `plot_book()` for the current ladder snapshot

Heatmap semantics:

- Heatmap rows are fixed visible ranks, ordered as `ask N ... ask 1 | bid 1 ... bid N`.
- The y-axis never switches to price and does not drift vertically when the market moves.
- `anchor="mid"` and `anchor="price"` are kept for API compatibility, but both render the same stable level-ranked rows.
- Colors are signed depth with robust asinh scaling, so one large wall does not flatten the rest of the book.

## Documentation Images

Regenerate all documentation assets with:

```bash
python -m scripts.render_doc_images
```

Standalone example:

```bash
python -m examples.plot_market_heatmap --output artifacts/orderwave_heatmap.png
```

Realism profile:

```bash
python -m scripts.profile_realism --steps 5000
```

The profiler emits spread/impact persistence, rank-level depth shape, visible/full-book one-sidedness, near-touch connectivity, and pair-distribution entropy.
