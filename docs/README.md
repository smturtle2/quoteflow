# orderwave Docs

[README](https://github.com/smturtle2/quoteflow/blob/main/README.md) | [한국어 문서](https://github.com/smturtle2/quoteflow/tree/main/docs/ko)

`orderwave` is a compact Python library for simulating a state-conditioned aggregate limit order book and visualizing the result directly from the same `Market` object.

![Overview](assets/orderwave-overview.png)

## Pages

- [Getting started](https://github.com/smturtle2/quoteflow/blob/main/docs/getting-started.md)
- [API reference](https://github.com/smturtle2/quoteflow/blob/main/docs/api.md)
- [Examples](https://github.com/smturtle2/quoteflow/blob/main/docs/examples.md)
- [Releasing](https://github.com/smturtle2/quoteflow/blob/main/docs/releasing.md)

## Built-in Visualization

- `Market.plot()` for the main overview figure
- `Market.plot_book()` for the current order book on a real price axis
- `Market.plot_diagnostics()` for spread, imbalance, volatility, and regime checks

![Current book](assets/orderwave-current-book.png)

![Diagnostics](assets/orderwave-diagnostics.png)
