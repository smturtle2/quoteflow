# API 레퍼런스

[문서 인덱스](https://github.com/smturtle2/quoteflow/blob/main/docs/ko/README.md) | [English](https://github.com/smturtle2/quoteflow/blob/main/docs/en/api.md)

## 공개 import

```python
from orderwave import Market
```

`Market`이 지원되는 공개 진입점입니다.
`orderwave.model`, `orderwave._model` 같은 내부 모듈은 안정적인 라이브러리 API로 취급하지 않습니다.

typed helper가 필요하면:

```python
from orderwave.market import BookLevel, MarketSnapshot, SimulationResult
```

## `Market`

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

`Market`은 `step == 0`에서 초기 aggregate order book을 seed하고, 바로 compact history를 기록하며, plotting용 private visual history도 함께 유지합니다.
여전히 aggregate order-book market-state simulator로 읽는 것이 맞고, order-level fill precision은 목표가 아닙니다.

### `step() -> dict`

한 번의 micro-batch를 실행하고 최신 snapshot을 반환합니다.

### `gen(steps: int) -> dict`

`steps`번 진행하고 최신 snapshot을 반환합니다.

### `run(steps: int) -> SimulationResult`

`steps`번 진행하고 bundled result를 반환합니다.

`SimulationResult`에는 다음이 들어 있습니다.

- `snapshot`
- `history`
- `event_history`
- `debug_history`
- `labeled_event_history`

`history_only` 모드에서는 `history`만 남고 event/debug 계열은 `None`입니다.

### `get() -> dict`

현재 snapshot을 반환합니다.

### `get_snapshot() -> MarketSnapshot`

현재 snapshot을 typed dataclass로 반환합니다.

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
- `visible_levels_bid`
- `visible_levels_ask`
- `drought_age`
- `recovery_pressure`
- `impact_residue`
- `regime_dwell`
- `inventory_pressure`

`trade_strength`는 aggressor buy/sell 체결량 EWMA 기반 realized-trade signed imbalance입니다.
quote-only 변화만으로는 움직이지 않습니다.

### `get_history() -> pandas.DataFrame`

초기 seeded book부터 현재 step까지 compact history를 반환합니다.

중요 컬럼:

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
- `top_n_bid_qty`
- `top_n_ask_qty`
- `realized_vol`
- `signed_flow`
- `visible_levels_bid`
- `visible_levels_ask`
- `drought_age`
- `recovery_pressure`
- `impact_residue`
- `regime_dwell`
- `inventory_pressure`

### `get_event_history() -> pandas.DataFrame`

`step == 1` 이후의 applied event log를 반환합니다.

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

`market` row의 `fills`는 sweep path 전체를 `(price, qty)` tuple list로 담습니다.

### `get_debug_history() -> pandas.DataFrame`

event-aligned latent debug stream을 반환합니다.

컬럼:

- `step`
- `event_idx`
- `day`
- `session_step`
- `session_phase`
- `microphase`
- `source`
- `participant_type`
- `meta_order_id`
- `meta_order_side`
- `meta_order_progress`
- `burst_state`
- `shock_state`
- `drought_age`
- `recovery_pressure`
- `impact_residue`
- `regime_dwell`
- `inventory_pressure`
- `flow_toxicity`
- `maker_stress`
- `quote_revision_wave`
- `refill_pressure`
- `visible_levels_bid`
- `visible_levels_ask`

해석 포인트:

- `microphase`: 엔진 내부 시간구조 버킷
- `flow_toxicity`: 최근 aggressive flow가 passive liquidity에 얼마나 adverse한지
- `maker_stress`: passive side가 얼마나 방어적으로 변했는지
- `quote_revision_wave`: structural pre-withdrawal revision 이벤트 여부
- `refill_pressure`: depletion 이후 passive replenishment 압력

### `get_labeled_event_history() -> pandas.DataFrame`

`step`과 `event_idx` 기준으로 event history와 debug history를 join한 테이블을 반환합니다.

### `plot(*, levels: int | None = None, title: str | None = None, figsize: tuple[float, float] | None = None) -> matplotlib.figure.Figure`

다음을 포함한 overview figure를 렌더합니다.

- `mid_price`
- `last_price`
- bid/ask spread band
- `trade_strength`
- signed visible-depth heatmap

### `plot_book(*, levels: int | None = None, title: str | None = None, figsize: tuple[float, float] | None = None) -> matplotlib.figure.Figure`

현재 order book을 실제 가격축으로 렌더합니다.
bid/ask depth는 0을 기준으로 좌우 대칭으로 놓이고, best bid, best ask, microprice를 함께 표시합니다.

### `plot_diagnostics(*, imbalance_bins: int = 8, max_lag: int = 12, title: str | None = None, figsize: tuple[float, float] | None = None) -> matplotlib.figure.Figure`

기존 market-state 패널에 더해, full debug 데이터가 있으면 microphase와 revision/refill 패널까지 포함한 diagnostics figure를 렌더합니다.

### Logging Mode

- `logging_mode="full"`: summary, event, debug, plotting history 유지
- `logging_mode="history_only"`: summary history와 overview/book plotting state만 유지

`history_only` 모드에서는:

- `get_history()`는 계속 동작합니다
- `plot()`과 `plot_book()`은 계속 동작합니다
- `get_event_history()`, `get_debug_history()`, `get_labeled_event_history()`, `plot_diagnostics()`는 `RuntimeError`를 발생시킵니다
