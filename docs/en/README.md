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

## Plotting

The plotting surface is:

- `plot()` for the overview figure
- `plot_heatmap(anchor="mid" | "price")` for a standalone signed-depth heatmap
- `plot_book()` for the current ladder snapshot

Heatmap semantics:

- `anchor="mid"` centers rows on the moving market center and is best for reading sweeps and refills.
- `anchor="price"` uses the real price axis and is best for reading drift and persistent walls.
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
