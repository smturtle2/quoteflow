# API 레퍼런스

[문서 인덱스](https://github.com/smturtle2/quoteflow/blob/main/docs/ko/README.md) | [English](https://github.com/smturtle2/quoteflow/blob/main/docs/en/api.md)

## 공개 import

```python
from orderwave import Market
```

## `Market`

```python
Market(
    init_price=100.0,
    tick_size=0.01,
    levels=5,
    seed=None,
    config=None,
)
```

`Market`는 메인 공개 진입점입니다. `step == 0`에서 초기 aggregate order book을 시드하고 compact history를 즉시 기록하며, built-in plotting을 위한 private visual history도 함께 유지합니다.

### `step() -> dict`

한 번의 micro-batch를 진행하고 최신 snapshot을 반환합니다.

### `gen(steps: int) -> dict`

`steps`번 진행하고 마지막 snapshot을 반환합니다.

### `get() -> dict`

현재 snapshot을 반환합니다.

주요 필드:

- `step`
- `day`
- `session_step`
- `session_phase`
- `last_price`
- `mid_price`
- `microprice`
- `best_bid`
- `best_ask`
- `spread`
- `bids`
- `asks`
- `last_trade_side`
- `last_trade_qty`
- `buy_aggr_volume`
- `sell_aggr_volume`
- `trade_strength`
- `depth_imbalance`
- `regime`

`trade_strength`는 execution-only signed imbalance입니다. 실제 aggressor buy/sell 체결량의 EWMA로 계산되며, quote-only 변화로는 바뀌지 않습니다.

`config={"logging_mode": "history_only"}`를 쓰면 snapshot과 compact history는 그대로 유지되지만, event/debug API는 의도적으로 비활성화됩니다.

### `get_history() -> pandas.DataFrame`

초기 시드 상태부터 현재 step까지의 compact history를 반환합니다.

최소 컬럼:

- `step`
- `day`
- `session_step`
- `session_phase`
- `last_price`
- `mid_price`
- `microprice`
- `best_bid`
- `best_ask`
- `spread`
- `buy_aggr_volume`
- `sell_aggr_volume`
- `trade_strength`
- `depth_imbalance`
- `regime`

추가로 요약 depth, 변동성 컬럼이 포함될 수 있습니다.

### `get_event_history() -> pandas.DataFrame`

`step == 1`부터 현재 step까지의 적용 이벤트 로그를 반환합니다.

컬럼:

- `step`
- `event_idx`
- `day`
- `session_step`
- `session_phase`
- `event_type`
- `side`
- `level`
- `price`
- `requested_qty`
- `applied_qty`
- `fill_qty`
- `fills`
- `best_bid_after`
- `best_ask_after`
- `mid_price_after`
- `last_trade_price_after`
- `regime`

이 로그는 샘플링된 의도 이벤트가 아니라 실제로 적용된 이벤트만 기록합니다. `market` 행의 `fills`에는 전체 sweep 경로가 `(price, qty)` 튜플 리스트로 들어갑니다.

이 메서드는 `logging_mode="full"`에서만 동작하며, `history_only`에서는 `RuntimeError`를 발생시킵니다.

### `get_debug_history() -> pandas.DataFrame`

event-aligned latent debug stream을 반환합니다.

컬럼:

- `step`
- `event_idx`
- `day`
- `session_step`
- `session_phase`
- `source`
- `participant_type`
- `meta_order_id`
- `meta_order_side`
- `meta_order_progress`
- `burst_state`
- `shock_state`

`get_debug_history()`는 `get_event_history()`와 같은 `step`, `event_idx` 키를 공유합니다. 기본 사용 흐름보다는 고급 검증과 diagnostics용 API입니다.

이 메서드는 `logging_mode="full"`에서만 동작하며, `history_only`에서는 `RuntimeError`를 발생시킵니다.

### `plot(*, levels: int | None = None, title: str | None = None, figsize: tuple[float, float] | None = None) -> matplotlib.figure.Figure`

다음 요소를 포함한 built-in overview figure를 렌더링합니다.

- `mid_price`
- `last_price`
- bid/ask spread band
- `trade_strength`
- signed visible-depth heatmap

`levels`는 기본 visible depth를 사용하고, 내부 book buffer를 넘기면 자동으로 clamp 됩니다.

### `plot_book(*, levels: int | None = None, title: str | None = None, figsize: tuple[float, float] | None = None) -> matplotlib.figure.Figure`

현재 order book을 실제 가격축으로 렌더링합니다. bid/ask depth는 0을 기준으로 좌우 mirrored bar로 표시되고, best bid, best ask, microprice를 함께 강조합니다.

### `plot_diagnostics(*, imbalance_bins: int = 8, max_lag: int = 12, title: str | None = None, figsize: tuple[float, float] | None = None) -> matplotlib.figure.Figure`

다음 3x2 diagnostics figure를 렌더링합니다.

- session phase spread / filled-volume profile
- depth imbalance -> next mid return 관계
- market-flow excitation profile
- spread-volatility coupling
- depletion resiliency
- regime / shock occupancy

이 메서드는 최소 두 개 이상의 history row가 필요합니다.
또한 `logging_mode="full"`에서만 동작하며, `history_only`에서는 `RuntimeError`를 발생시킵니다.

## `orderwave.config.MarketConfig`

```python
from orderwave.config import MarketConfig
```

`MarketConfig`는 고급 설정 타입입니다. 외부 표면은 아래 필드들로만 제한합니다.

- `preset`
- `book_buffer_levels`
- `flow_window`
- `vol_window`
- `limit_rate_scale`
- `market_rate_scale`
- `cancel_rate_scale`
- `fair_price_vol_scale`
- `regime_transition_scale`
- `steps_per_day`
- `seasonality_scale`
- `excitation_scale`
- `meta_order_scale`
- `shock_scale`
- `logging_mode`

`Market`에 전달하는 `config`는 `MarketConfig` 인스턴스나 같은 키를 가진 `dict` 둘 다 가능합니다.
