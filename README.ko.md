# orderwave

[![PyPI version](https://img.shields.io/pypi/v/orderwave.svg)](https://pypi.org/project/orderwave/)
[![Python versions](https://img.shields.io/pypi/pyversions/orderwave.svg)](https://pypi.org/project/orderwave/)
[![Release workflow](https://github.com/smturtle2/quoteflow/actions/workflows/workflow.yml/badge.svg)](https://github.com/smturtle2/quoteflow/actions/workflows/workflow.yml)

내장 시각화를 포함한 aggregate order book 기반 시장상태 시뮬레이터입니다.

`orderwave`는 가격을 직접 random walk 하지 않습니다. 희소한 aggregate order book, 상태조건부 limit flow, marketable flow, 취소, latent meta-order, 외생 shock, session-aware state change를 시뮬레이션하고, 그 결과로 가격이 형성되게 만듭니다. 같은 `Market` 객체에서 경로, 현재 호가 스냅샷, built-in 시장상태 diagnostics까지 바로 그릴 수 있습니다.

`orderwave`의 공식 정체성은 aggregate order-book market-state simulator입니다.
연구, 시각화, 샌드박스 실험을 위한 intraday market path와 book state를 만드는 데 초점을 둡니다.
주문 단위 매칭이나 체결 정밀도를 목표로 내세우는 라이브러리는 아닙니다.

[English README](https://github.com/smturtle2/quoteflow/blob/main/README.md) | [English Docs](https://github.com/smturtle2/quoteflow/tree/main/docs/en) | [한국어 문서](https://github.com/smturtle2/quoteflow/tree/main/docs/ko)

![orderwave overview](docs/assets/orderwave-built-in-overview.png)

## 왜 orderwave인가

- 공개 진입점은 `from orderwave import Market` 하나로 단순합니다
- 지원되는 공개 API는 `Market`이고, engine/model 내부는 구현 세부사항입니다
- 가격 변화는 오직 호가장 변화의 결과로만 발생합니다
- hidden fair value, session clock, shock, meta-order가 주문 흐름을 편향시키지만 가격을 직접 덮어쓰지 않습니다
- 같은 seed면 같은 경로가 재현됩니다
- overview, 현재 호가, diagnostics를 내장 시각화로 제공합니다
- 얇은 public event history와 고급 latent debug history를 함께 제공합니다

## 설치

```bash
pip install orderwave
```

로컬 개발용 설치:

```bash
pip install -e .[dev]
```

## 빠른 시작

```python
from orderwave import Market

market = Market(seed=42, preset="trend")
result = market.run(steps=1_000)

snapshot = market.get_snapshot()
history = market.get_history()
events = market.get_labeled_event_history()
overview = market.plot()
book = market.plot_book()
diagnostics = market.plot_diagnostics()

print(snapshot.session_phase, snapshot.mid_price, snapshot.best_bid, snapshot.best_ask)
print(history.tail())
print(events.tail())
print(result.debug_history.tail())

overview.savefig("orderwave-overview.png")
```

가격 경로, visible book, 체결 강도만 필요하고 event/debug 로그가 불필요한 장기 실행이라면:

```python
fast_market = Market(seed=7, preset="balanced", logging_mode="history_only")
summary = fast_market.run(steps=10_000).history
figure = fast_market.plot()
```

## 성능 측정

빠른 처리량 점검과 `full` vs `history_only` logging 비교는 성능 측정 스크립트 하나로 확인할 수 있습니다.

```bash
python -m scripts.measure_performance --preset balanced --seeds 20 --steps 20000 --outdir artifacts/performance
```

생성 산출물:

- `performance_metrics.csv`
- `performance_summary.csv`
- `performance_logging_modes.csv`
- `performance_summary.md`

## 검증 스윕

baseline preset sweep, knob sensitivity, 재현성, soak test를 묶어서 돌리는 validation runner도 포함되어 있습니다.

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

release 빌드는 별도의 `Release Validation` job에서 더 짧은 `--profile release_smoke` 회귀를 돌리고 `tests/golden/validation_release_baseline.json`과 비교한 뒤에만 PyPI publish를 진행합니다.
이 smoke profile은 CI 검증 게이트를 빠르게 유지하도록 의도적으로 작게 유지합니다.

현재 엔진 로드맵은 더 넓은 market-state fidelity입니다. preset 분리, 시간구조, sensitivity control, validation artifact 품질을 함께 강화합니다.

## API 표면

| API | 설명 |
| --- | --- |
| `step()` | 한 번의 micro-batch를 실행하고 최신 snapshot을 반환 |
| `gen(steps=n)` | `n`번 진행하고 마지막 snapshot을 반환 |
| `run(steps=n)` | `n`번 진행하고 typed result bundle을 반환 |
| `get()` | 현재 snapshot 반환 |
| `get_snapshot()` | 현재 snapshot을 typed dataclass로 반환 |
| `get_history()` | compact `pandas.DataFrame` history 반환 |
| `get_event_history()` | 적용된 event log를 `pandas.DataFrame`으로 반환 |
| `get_debug_history()` | 고급 검증용 event-aligned latent debug history 반환 |
| `get_labeled_event_history()` | event history와 latent debug label을 join해서 반환 |
| `plot()` | 가격, 스프레드, 체결 강도, visible-book heatmap 렌더 |
| `plot_book()` | 현재 order book을 실제 가격축으로 렌더 |
| `plot_diagnostics()` | session, excitation, imbalance, spread/volatility, resiliency, occupancy diagnostics 렌더 |

고급 설정은 `orderwave.config.MarketConfig`를 통해 사용할 수 있습니다.
자주 쓰는 설정은 `Market(..., preset="trend", logging_mode="history_only", liquidity_backstop="off")`처럼 바로 넘길 수도 있습니다.

`logging_mode="history_only"`를 쓰면 summary history와 overview/book plotting 데이터만 남기고, `get_event_history()`, `get_debug_history()`, `plot_diagnostics()`는 `RuntimeError`를 발생시킵니다.
기본값 `liquidity_backstop="on_empty"`는 한쪽 호가가 사라진 경우만 복구하고, 매 step마다 최소 visible depth를 강제로 채우지는 않습니다.
더 강하게 안정화된 책이 필요하면 `"always"`를, 더 얇은 post-step liquidity를 허용하려면 `"off"`를 사용할 수 있습니다.

## 내장 시각화

모든 plotting 메서드는 `matplotlib.figure.Figure`를 반환하고, 저장이나 표시 시점은 호출자가 직접 제어합니다.

- `plot()`은 가격, 스프레드, 체결량 기반 imbalance, signed visible-depth heatmap을 한 번에 렌더링합니다
- `plot_book()`은 현재 order book을 실제 가격축으로 렌더링합니다
- `plot_diagnostics()`는 session phase profile, imbalance lead, market-flow excitation, spread-volatility coupling, depletion resiliency, regime/shock occupancy를 렌더링합니다

![orderwave current book](docs/assets/orderwave-built-in-current-book.png)

![orderwave diagnostics](docs/assets/orderwave-built-in-diagnostics.png)

overview heatmap은 signed depth를 유지합니다. ask는 빨강, bid는 파랑, `0`은 연회색 midpoint, 존재하지 않는 레벨은 검은색이 아니라 blank background로 렌더링합니다.

## Preset 비교

![orderwave presets](docs/assets/orderwave-built-in-presets.png)

`balanced`, `trend`, `volatile` preset은 같은 API를 유지하면서 스프레드 성향, 주문흐름 압력, 취소 압력, hidden fair-price 동학을 다르게 만듭니다.

## 핵심 의미

`Market.get()`은 session clock 필드, 가격, 스프레드, visible depth, 공격 주문 거래량, 체결 강도, depth imbalance, regime을 담은 compact dict를 반환합니다.

`trade_strength`는 체결량 기반 signed imbalance입니다. 실제 aggressor buy/sell 체결량의 EWMA로 계산되므로, quote-only book 변화만으로는 바뀌지 않습니다.

`Market.run()`은 typed snapshot과 logging mode별로 사용 가능한 테이블을 묶은 `SimulationResult`를 반환합니다.

핵심 차이:

- `mid_price`는 호가 개선, 취소, 최우선 호가 소진만으로도 움직일 수 있습니다
- `last_price`는 실제 체결이 발생했을 때만 바뀝니다
- `day`, `session_step`, `session_phase`는 synthetic intraday clock을 보여줍니다

핵심 보장:

- 가격을 직접 random walk 하지 않습니다
- 시장가 체결, 최우선 호가 소진, inside-spread 개선호가만 가격을 움직입니다
- history는 seeded initial book을 포함한 `step == 0`부터 시작합니다
- 적용된 limit/market/cancel 이벤트는 `get_event_history()`로 확인할 수 있습니다
- participant type, meta-order progress, burst state, shock state는 `get_debug_history()`로 확인할 수 있습니다
- v1에서는 per-order FIFO 대신 aggregate depth만 모델링합니다

## 문서

- [문서 인덱스](https://github.com/smturtle2/quoteflow/blob/main/docs/ko/README.md)
- [시작하기](https://github.com/smturtle2/quoteflow/blob/main/docs/ko/getting-started.md)
- [API 레퍼런스](https://github.com/smturtle2/quoteflow/blob/main/docs/ko/api.md)
- [예제](https://github.com/smturtle2/quoteflow/blob/main/docs/ko/examples.md)
- [릴리스 가이드](https://github.com/smturtle2/quoteflow/blob/main/docs/ko/releasing.md)

`orderwave.validation`은 재현성 점검, sensitivity sweep, validation pipeline을 위한 지원되는 고급 Python API입니다.
