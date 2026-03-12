# orderwave Docs

[README](https://github.com/smturtle2/quoteflow/blob/main/README.md) | [한국어 문서](https://github.com/smturtle2/quoteflow/tree/main/docs/ko)

`orderwave` is a compact Python library for simulating a session-aware, state-conditioned aggregate limit order book and visualizing the result from the same `Market` object.
`Market` is the supported public API; internal engine and model modules are intentionally not documented as stable imports.

It is an aggregate order-book market-state simulator.
It is not an order-level matching or fill-precision simulator.

![Overview](../assets/orderwave-built-in-overview.png)

## Pages

- [Getting started](https://github.com/smturtle2/quoteflow/blob/main/docs/en/getting-started.md)
- [API reference](https://github.com/smturtle2/quoteflow/blob/main/docs/en/api.md)
- [Examples](https://github.com/smturtle2/quoteflow/blob/main/docs/en/examples.md)
- [Releasing](https://github.com/smturtle2/quoteflow/blob/main/docs/en/releasing.md)

## What Changed In The Current Engine

- Internal microphases shape open release, midday lull, power hour, and closing imbalance behavior
- The event cycle now emphasizes marketable flow first, then adverse quote revision, then passive refill
- Debug history includes `microphase`, `flow_toxicity`, `maker_stress`, `quote_revision_wave`, and `refill_pressure`
- Diagnostics add microphase and revision/refill panels on top of the original market-state checks

## Built-in Visualization

- `Market.plot()` for the main overview figure
- `Market.plot_book()` for the current order book on a real price axis
- `Market.plot_diagnostics()` for session, excitation, imbalance, resiliency, regime/shock, microphase, and revision/refill checks

![Current book](../assets/orderwave-built-in-current-book.png)

![Diagnostics](../assets/orderwave-built-in-diagnostics.png)
