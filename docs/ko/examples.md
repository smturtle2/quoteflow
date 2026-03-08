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

## Compact history-only 실행

```python
fast_market = Market(seed=11, config={"preset": "balanced", "logging_mode": "history_only"})
fast_market.gen(steps=20_000)

summary = fast_market.get_history()
figure = fast_market.plot(title="Compact overview")
figure.savefig("orderwave-history-only.png")
```

`history_only`는 compact history, visible-book plotting, trade strength만 필요할 때 쓰는 경량 모드입니다. 이 모드에서는 `get_event_history()`, `get_debug_history()`, `plot_diagnostics()`가 의도적으로 `RuntimeError`를 발생시킵니다.

## CLI 예제

저장소에는 [`examples/plot_market_heatmap.py`](https://github.com/smturtle2/quoteflow/blob/main/examples/plot_market_heatmap.py) 예제가 있고, 이제 내부적으로 `Market.plot()`을 직접 호출합니다.

```bash
python examples/plot_market_heatmap.py --steps 2000 --preset trend --output artifacts/orderwave_heatmap.png
```

## 성능 측정

엔진 변경 후 처리량, floor check, `full` vs `history_only` 비교를 한 번에 확인하려면 성능 측정 스크립트를 사용하면 됩니다.

```bash
python scripts/measure_performance.py --preset balanced --seeds 20 --steps 20000 --outdir artifacts/performance
```

생성 산출물:

- `performance_metrics.csv`
- `performance_summary.csv`
- `performance_logging_modes.csv`
- `performance_summary.md`

## 검증 스윕

단일 throughput 측정이 아니라 synthetic market-state 검증 파이프라인을 돌리려면 validation runner를 사용하면 됩니다.

```bash
python scripts/validate_orderwave.py --profile full --jobs 4 --outdir artifacts/validation
```

runner는 다음 산출물을 생성합니다.

- `validation_summary.md`
- `run_metrics.csv`
- `preset_summary.csv`
- `sensitivity_summary.csv`
- `invariant_failures.csv`
- `acceptance_decision.md`
- diagnostics 렌더링이 켜진 경우 `diagnostics_<preset>_<seed>.png`

release 빌드는 별도의 `Release Validation` job에서 더 짧은 `--profile release` 회귀를 돌리고 `tests/golden/validation_release_baseline.json`과 비교한 뒤 publish를 진행합니다.
이 release profile은 CI 릴리스 게이트를 빠르게 유지하도록 아주 작게 유지합니다.

다음 엔진 개선 범위는 의도적으로 좁게 유지합니다. 다음 단계는 finer intra-step event feedback만 다룹니다.

## Preset 비교

![Preset comparison](../assets/orderwave-built-in-presets.png)

preset comparison 그림은 문서 전용이지만, 동일한 public simulation API를 다른 preset으로 실행해 생성합니다.

## 문서 이미지 다시 생성

```bash
python scripts/render_doc_images.py
```
