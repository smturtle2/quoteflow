# Examples

[Docs index](https://github.com/smturtle2/quoteflow/blob/main/docs/en/README.md) | [한국어](https://github.com/smturtle2/quoteflow/blob/main/docs/ko/examples.md)

## Built-in Overview Plot

```python
from orderwave import Market

market = Market(seed=7, preset="trend")
result = market.run(steps=2_000)

labeled_events = market.get_labeled_event_history()
figure = market.plot(levels=8, title="orderwave overview")
figure.savefig("orderwave-overview.png")

print(result.snapshot)
print(labeled_events.tail())
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
diagnostics = market.plot_diagnostics(max_lag=12, title="Microstructure diagnostics")
diagnostics.savefig("orderwave-diagnostics.png")
```

![Diagnostics snapshot](../assets/orderwave-built-in-diagnostics.png)

The diagnostics figure now covers:

- session phase profile
- depth imbalance lead
- market-flow excitation
- spread-volatility coupling
- depletion resiliency
- regime and shock occupancy
- microphase stress profile
- revision and refill pressure

## Event Flow Inspection

```python
labeled = market.get_labeled_event_history()
market_fills = labeled.loc[
    labeled["event_type"] == "market",
    ["step", "side", "fill_qty", "participant_type", "source"],
]

print(market_fills.tail())
```

## Latent Stress Inspection

```python
debug = market.get_debug_history()

print(
    debug[
        [
            "step",
            "event_idx",
            "microphase",
            "maker_stress",
            "flow_toxicity",
            "quote_revision_wave",
            "refill_pressure",
        ]
    ].tail()
)
```

This view is useful when you want to inspect structural pre-withdrawal and passive refill behavior directly rather than inferring it from price/spread alone.

## Compact History-Only Runs

```python
fast_market = Market(seed=11, preset="balanced", logging_mode="history_only")
result = fast_market.run(steps=20_000)

summary = result.history
figure = fast_market.plot(title="Compact overview")
figure.savefig("orderwave-history-only.png")
```

`history_only` mode is the lighter option for long sweeps when you only need compact history, visible-book plotting, and realized-trade imbalance.

## CLI Example

```bash
python -m examples.plot_market_heatmap --steps 2000 --preset trend --output artifacts/orderwave_heatmap.png
```

## Performance Measurement

```bash
python -m scripts.measure_performance --preset balanced --seeds 20 --steps 20000 --outdir artifacts/performance
```

## Validation Sweep

```bash
python -m scripts.validate_orderwave --profile quality_regression --jobs 4 --outdir artifacts/validation
```

The validation pipeline checks reproducibility, preset separation, stylized structure, sensitivity direction, and soak behavior for the current aggregate market-state model.

## Regenerating Docs Images

```bash
python -m scripts.render_doc_images
```
