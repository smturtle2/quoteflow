# Examples

[Docs index](https://github.com/smturtle2/quoteflow/blob/main/docs/README.md) | [한국어](https://github.com/smturtle2/quoteflow/blob/main/docs/ko/examples.md)

## Quick Diagnostics

```python
from orderwave import Market

market = Market(seed=7, config={"preset": "trend"})
market.gen(steps=5_000)
history = market.get_history()

mid_ret = history["mid_price"].diff().fillna(0.0)
abs_ret = mid_ret.abs()

print("spread mean:", history["spread"].mean())
print("imbalance -> next return corr:", history["depth_imbalance"].corr(mid_ret.shift(-1).fillna(0.0)))
print("|return| lag-1 autocorr:", abs_ret.autocorr(lag=1))
```

## Price + Trade Strength + Heatmap

The repository includes [`examples/plot_market_heatmap.py`](https://github.com/smturtle2/quoteflow/blob/main/examples/plot_market_heatmap.py).

![Overview image](assets/orderwave-overview.png)

Run it locally:

```bash
pip install matplotlib
python examples/plot_market_heatmap.py --steps 2000 --preset trend
```

Save output:

```bash
python examples/plot_market_heatmap.py --steps 2000 --preset trend --output artifacts/orderwave_heatmap.png
```

The example renders:

- `mid_price` and `last_price`
- `trade_strength`
- a signed visible-book heatmap with `ask n ... ask 1, bid 1 ... bid n`

The heatmap keeps signed depth in the color scale while rendering zero exactly as black.

## Preset Comparison

![Preset comparison](assets/orderwave-presets.png)

This figure is generated from the same simulator with different presets and the same seed so the behavioral differences stay easy to compare.

## Diagnostics Snapshot

![Diagnostics snapshot](assets/orderwave-diagnostics.png)

This image summarizes three useful checks for a synthetic market path:

- spread is not locked to a single value
- depth imbalance has directional information about the next move
- absolute returns show persistence through positive autocorrelation

To regenerate the documentation images:

```bash
python scripts/render_doc_images.py
```
