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
- 생성된 경로가 유용한 시장상태 신호를 갖는가?

## Event Flow 확인

```python
labeled_events = market.get_labeled_event_history()
market_fills = labeled_events.loc[labeled_events["event_type"] == "market", ["step", "side", "fill_qty", "fills"]]

print(market_fills.tail())
```

`get_labeled_event_history()`는 실제 적용 순서 그대로의 event stream에 participant, meta-order, shock label을 함께 붙여 반환합니다. 그래서 흔한 event/debug merge 단계를 생략할 수 있습니다.

## Latent Debug 확인

```python
joined = market.get_labeled_event_history()

print(joined[["step", "event_idx", "event_type", "participant_type", "meta_order_id", "shock_state"]].tail())
```

`get_debug_history()`도 그대로 제공되지만, 탐색 분석에서는 joined helper가 더 짧은 기본 경로입니다.

## Compact history-only 실행

```python
fast_market = Market(seed=11, preset="balanced", logging_mode="history_only")
result = fast_market.run(steps=20_000)

summary = result.history
figure = fast_market.plot(title="Compact overview")
figure.savefig("orderwave-history-only.png")
```

`history_only`는 compact history, visible-book plotting, 체결량 기반 imbalance만 필요할 때 쓰는 경량 모드입니다. 이 모드에서는 `get_event_history()`, `get_debug_history()`, `plot_diagnostics()`가 의도적으로 `RuntimeError`를 발생시킵니다.

## CLI 예제

저장소에는 [`examples/plot_market_heatmap.py`](https://github.com/smturtle2/quoteflow/blob/main/examples/plot_market_heatmap.py) 예제가 있고, 이제 내부적으로 `Market.plot()`을 직접 호출합니다.

```bash
python -m examples.plot_market_heatmap --steps 2000 --preset trend --output artifacts/orderwave_heatmap.png
```

## 성능 측정

엔진 변경 후 처리량, floor check, `full` vs `history_only` 비교를 한 번에 확인하려면 성능 측정 스크립트를 사용하면 됩니다.

```bash
python -m scripts.measure_performance --preset balanced --seeds 20 --steps 20000 --outdir artifacts/performance
```

생성 산출물:

- `performance_metrics.csv`
- `performance_summary.csv`
- `performance_logging_modes.csv`
- `performance_summary.md`

## 검증 스윕

단일 throughput 측정이 아니라 시장상태 검증 파이프라인을 돌리려면 validation runner를 사용하면 됩니다.

```bash
python -m scripts.validate_orderwave --profile quality_regression --jobs 4 --outdir artifacts/validation
```

runner는 다음 산출물을 생성합니다.

- `validation_summary.md`
- `run_metrics.csv`
- `preset_summary.csv`
- `sensitivity_summary.csv`
- `invariant_failures.csv`
- `acceptance_decision.md`
- diagnostics 렌더링이 켜진 경우 `diagnostics_<preset>_<seed>.png`

release 빌드는 별도의 `Release Validation` job에서 더 짧은 `--profile release_smoke` 회귀를 돌리고 `tests/golden/validation_release_baseline.json`과 비교한 뒤 publish를 진행합니다.
이 smoke profile은 CI 릴리스 게이트를 빠르게 유지하도록 아주 작게 유지합니다.

현재 엔진 로드맵은 더 넓은 market-state fidelity입니다. preset 분리, 시간구조, sensitivity control, validation 품질을 함께 강화합니다.

## Preset 비교

![Preset comparison](../assets/orderwave-built-in-presets.png)

preset comparison 그림은 문서 전용이지만, 동일한 public simulation API를 다른 preset으로 실행해 생성합니다.

## 문서 이미지 다시 생성

```bash
python -m scripts.render_doc_images
```
