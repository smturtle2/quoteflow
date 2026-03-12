# 시작하기

[문서 인덱스](https://github.com/smturtle2/quoteflow/blob/main/docs/ko/README.md) | [English](https://github.com/smturtle2/quoteflow/blob/main/docs/en/getting-started.md)

## 설치

```bash
pip install orderwave
```

개발용 설치:

```bash
pip install -e .[dev]
```

## 최소 예제

```python
from orderwave import Market

market = Market(seed=42, preset="trend")
result = market.run(steps=1_000)

snapshot = market.get_snapshot()
history = market.get_history()
events = market.get_labeled_event_history()
figure = market.plot()
```

장기 실행에서 compact history와 plotting만 필요하다면:

```python
fast_market = Market(seed=7, preset="balanced", logging_mode="history_only")
summary = fast_market.run(steps=10_000).history
overview = fast_market.plot()
```

## 생성자

```python
Market(
    init_price=100.0,
    tick_size=0.01,
    levels=5,
    seed=None,
    config=None,
    *,
    preset=None,
    logging_mode=None,
    liquidity_backstop=None,
)
```

- `init_price`: 초기 기준 가격. 가장 가까운 tick으로 스냅됩니다
- `tick_size`: 호가 단위
- `levels`: `get()`이 반환하는 visible depth 수이자 plot 기본 depth
- `seed`: deterministic random seed
- `config`: `dict` 또는 `orderwave.config.MarketConfig`
- `preset`: preset 선택 shortcut
- `logging_mode`: `config["logging_mode"]` shortcut
- `liquidity_backstop`: `config["liquidity_backstop"]` shortcut

## 엔진이 시뮬레이션하는 것

- per-order FIFO가 아닌 aggregate visible depth
- 참가자 조건부 `limit`, `market`, `cancel` flow
- `calm`, `directional`, `stressed` regime
- session phase와 내부 microphase 시간구조
- aggressive burst 전에 나타나는 structural pre-withdrawal과 depletion 뒤 passive refill
- latent stress diagnostics: `flow_toxicity`, `maker_stress`, `quote_revision_wave`, `refill_pressure`

## 내장 플롯

```python
overview = market.plot()
book = market.plot_book()
diagnostics = market.plot_diagnostics()
```

- `plot()`은 가격, 스프레드, 체결 강도, signed visible-book heatmap을 렌더합니다
- `plot_book()`은 현재 order book을 실제 가격축으로 렌더합니다
- `plot_diagnostics()`는 session profile, imbalance lead, market-flow excitation, spread-volatility coupling, depletion resiliency, regime/shock occupancy, microphase stress profile, revision/refill pressure를 렌더합니다
- `plot_diagnostics()`는 `logging_mode="full"`이 필요합니다

## Preset

- `balanced`: 더 부드러운 refill, 중간 수준의 directional pressure, 낮은 stress persistence
- `trend`: stronger directional dwell과 meta-order persistence
- `volatile`: 더 무거운 cancel/revision pressure, 넓은 spread tail, 느린 refill recovery

## 고급 점검

- `get_event_history()`는 적용된 `limit`, `market`, `cancel` 이벤트를 반환합니다
- `get_debug_history()`는 event-aligned latent label과 stress field를 반환합니다
- `get_labeled_event_history()`는 event/debug joined table을 반환합니다
- `run()`은 typed snapshot과 available tables를 묶은 `SimulationResult`를 반환합니다
- `history_only` 모드는 `get_history()`, `plot()`, `plot_book()`만 유지하고 event/debug API와 diagnostics를 끕니다

## 재현성

시뮬레이터는 생성 시점의 NumPy random generator seed를 사용합니다.
같은 인자와 seed로 생성한 두 `Market`은 같은 path, event log, debug history를 만들어야 합니다.
