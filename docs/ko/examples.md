# 예제

[문서 인덱스](https://github.com/smturtle2/quoteflow/blob/main/docs/ko/README.md) | [English](https://github.com/smturtle2/quoteflow/blob/main/docs/en/examples.md)

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

현재 diagnostics figure는 다음을 다룹니다.

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

이 view는 structural pre-withdrawal과 passive refill을 가격/스프레드만 보고 추정하는 대신 직접 확인할 때 유용합니다.

## Compact History-Only Runs

```python
fast_market = Market(seed=11, preset="balanced", logging_mode="history_only")
result = fast_market.run(steps=20_000)

summary = result.history
figure = fast_market.plot(title="Compact overview")
figure.savefig("orderwave-history-only.png")
```

`history_only` 모드는 compact history, visible-book plotting, realized-trade imbalance만 필요할 때 가벼운 옵션입니다.

## CLI Example

```bash
python -m examples.plot_market_heatmap --steps 2000 --preset trend --output artifacts/orderwave_heatmap.png
```

## 성능 측정

```bash
python -m scripts.measure_performance --preset balanced --seeds 20 --steps 20000 --outdir artifacts/performance
```

## 검증 스윕

```bash
python -m scripts.validate_orderwave --profile quality_regression --jobs 4 --outdir artifacts/validation
```

validation pipeline은 현재 aggregate market-state model에 대해 reproducibility, preset separation, stylized structure, sensitivity direction, soak behavior를 점검합니다.

## 문서 이미지 재생성

```bash
python -m scripts.render_doc_images
```
