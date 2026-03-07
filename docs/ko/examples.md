# 예제

[문서 인덱스](https://github.com/smturtle2/quoteflow/blob/main/docs/ko/README.md) | [English](https://github.com/smturtle2/quoteflow/blob/main/docs/en/examples.md)

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

## 현재 호가 스냅샷

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

이 built-in figure들은 서로 다른 질문에 답하도록 설계했습니다.

- 시뮬레이터가 어떤 경로를 만들었는가?
- 현재 호가장은 어떤 모양인가?
- 생성된 경로가 유용한 미시구조 신호를 갖는가?

## Event Flow 확인

```python
events = market.get_event_history()
market_fills = events.loc[events["event_type"] == "market", ["step", "side", "fill_qty", "fills"]]

print(market_fills.tail())
```

`get_event_history()`는 샘플링된 의도 이벤트가 아니라 실제 적용 순서 그대로의 event stream을 노출합니다. 덕분에 sweep 경로, 취소 압력, quote replenishment를 그대로 검증할 수 있습니다.

## Latent Debug 확인

```python
debug = market.get_debug_history()
joined = market.get_event_history().merge(debug, on=["step", "event_idx"], how="inner")

print(joined[["step", "event_idx", "event_type", "participant_type", "meta_order_id", "shock_state"]].tail())
```

`get_debug_history()`는 고급 검증용 뷰입니다. 얇은 public event log에는 숨은 원인 라벨을 넣지 않되, participant mix, burst state, meta-order progress, shock state는 별도 테이블에서 감사 가능하게 유지합니다.

## CLI 예제

저장소에는 [`examples/plot_market_heatmap.py`](https://github.com/smturtle2/quoteflow/blob/main/examples/plot_market_heatmap.py) 예제가 있고, 이제 내부적으로 `Market.plot()`을 직접 호출합니다.

```bash
python examples/plot_market_heatmap.py --steps 2000 --preset trend --output artifacts/orderwave_heatmap.png
```

## 벤치마크

엔진 변경 후 처리량과 event density를 함께 확인하려면 benchmark 스크립트를 사용하면 됩니다.

```bash
python benchmarks/benchmark.py --steps 5000 --preset balanced
```

스크립트는 다음 값을 출력합니다.

- `steps_per_second`
- `events_per_step`
- `market_flow=buy=... sell=... buy_share=...`

## 검증 스윕

단일 throughput 측정이 아니라 multi-seed 검증 계획 전체를 실행하려면 validation runner를 사용하면 됩니다.

```bash
python scripts/validate_orderwave.py --steps 10000 --seeds 20 --outdir artifacts/validation
```

runner는 다음 산출물을 생성합니다.

- `validation-runs.csv`
- `validation-summary.csv`
- `validation-reproducibility.csv`
- `report.md`
- preset 요약 PNG와 representative diagnostics PNG

## Preset 비교

![Preset comparison](../assets/orderwave-built-in-presets.png)

preset comparison 그림은 문서 전용이지만, 동일한 public simulation API를 다른 preset으로 실행해 생성합니다.

## 문서 이미지 다시 생성

```bash
python scripts/render_doc_images.py
```
