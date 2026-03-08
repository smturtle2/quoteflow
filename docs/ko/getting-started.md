# 시작하기

[문서 인덱스](https://github.com/smturtle2/quoteflow/blob/main/docs/ko/README.md) | [English](https://github.com/smturtle2/quoteflow/blob/main/docs/en/getting-started.md)

## 설치

```bash
pip install orderwave
```

개발 환경 설치:

```bash
pip install -e .[dev]
```

## 최소 예제

```python
from orderwave import Market

market = Market(seed=42, config={"preset": "trend"})
market.gen(steps=1_000)

snapshot = market.get()
history = market.get_history()
events = market.get_event_history()
debug = market.get_debug_history()
figure = market.plot()
```

compact history와 overview/book plot만 필요한 장기 실행이라면:

```python
fast_market = Market(seed=7, config={"preset": "balanced", "logging_mode": "history_only"})
fast_market.gen(steps=10_000)
summary = fast_market.get_history()
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
)
```

- `init_price`: nearest tick으로 스냅되는 초기 기준 가격
- `tick_size`: 내부 호가장의 가격 단위
- `levels`: `get()`이 반환하는 visible depth이자 기본 plot depth
- `seed`: 재현 가능한 난수 시드
- `config`: `dict` 또는 `orderwave.config.MarketConfig`
- `config["logging_mode"]`: `"full"` 또는 `"history_only"`
- `config["liquidity_backstop"]`: `"always"`(기본값), `"on_empty"`, `"off"`

## 내장 플롯

```python
overview = market.plot()
book = market.plot_book()
diagnostics = market.plot_diagnostics()
```

- `plot()`은 가격, 스프레드, 체결 강도, signed visible-book heatmap을 렌더링합니다
- `plot_book()`은 현재 order book을 실제 가격축으로 렌더링합니다
- `plot_diagnostics()`는 session profile, market-flow excitation, imbalance lead, spread-volatility coupling, resiliency, regime/shock occupancy를 렌더링합니다
- `plot_diagnostics()`는 `logging_mode="full"`에서만 동작합니다

모든 plotting 메서드는 `matplotlib.figure.Figure`를 반환합니다. 저장이나 표시 시점은 사용자가 직접 제어합니다.

## Preset

- `balanced`: 기본값, 비교적 균형 잡힌 흐름과 스프레드 성향
- `trend`: 방향성과 fair-value 압력이 더 강함
- `volatile`: 스프레드 확대, 취소, 공격 주문 압력이 더 강함

## Snapshot 동작

`get()`이 반환하는 현재 상태는 의도적으로 compact합니다.

- `mid_price`는 최우선 bid/ask를 반영합니다
- `last_price`는 실제 체결 때만 갱신됩니다
- `day`, `session_step`, `session_phase`는 synthetic session clock을 보여줍니다
- `trade_strength`는 대칭형 `[-1, 1]` signed flow 지표입니다
- `bids`, `asks`는 최대 `levels`개의 visible price level을 담습니다

## 고급 검증

- `get_event_history()`는 실제 적용된 event stream만 반환합니다
- `get_debug_history()`는 같은 `step`, `event_idx` 키로 participant type, meta-order progress, burst state, shock state를 반환합니다
- `history_only` 모드는 `get_history()`, `plot()`, `plot_book()`은 유지하지만 `get_event_history()`, `get_debug_history()`, `plot_diagnostics()`는 비활성화합니다
- 기본값 `liquidity_backstop="always"`는 synthetic book이 양방향으로 보이고 최소 visible depth를 유지하도록 돕습니다

## 재현성

시뮬레이터는 생성 시점의 NumPy random generator를 사용합니다. 같은 인자와 같은 seed로 만든 두 시장은 같은 경로를 생성해야 합니다.
