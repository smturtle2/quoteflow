# orderwave Docs

[README](https://github.com/smturtle2/quoteflow/blob/main/README.md) | [한국어 문서](https://github.com/smturtle2/quoteflow/tree/main/docs/ko)

`orderwave` is a compact Python library for simulating a session-aware, state-conditioned aggregate limit order book and visualizing the result directly from the same `Market` object.
`Market` is the supported public API; internal engine and model modules are intentionally not documented as stable imports.

![Overview](../assets/orderwave-built-in-overview.png)

## Pages

- [Getting started](https://github.com/smturtle2/quoteflow/blob/main/docs/en/getting-started.md)
- [API reference](https://github.com/smturtle2/quoteflow/blob/main/docs/en/api.md)
- [Examples](https://github.com/smturtle2/quoteflow/blob/main/docs/en/examples.md)
- [Releasing](https://github.com/smturtle2/quoteflow/blob/main/docs/en/releasing.md)

## Built-in Visualization

- `Market.plot()` for the main overview figure
- `Market.plot_book()` for the current order book on a real price axis
- `Market.plot_diagnostics()` for session, excitation, imbalance, resiliency, and regime or shock checks

![Current book](../assets/orderwave-built-in-current-book.png)

![Diagnostics](../assets/orderwave-built-in-diagnostics.png)
