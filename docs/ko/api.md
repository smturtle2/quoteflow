# API 레퍼런스

[문서 인덱스](https://github.com/smturtle2/quoteflow/blob/main/docs/ko/README.md) | [English](https://github.com/smturtle2/quoteflow/blob/main/docs/api.md)

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

### `get_history() -> pandas.DataFrame`

초기 시드 상태부터 현재 step까지의 compact history를 반환합니다.

최소 컬럼:

- `step`
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

다음 2x2 diagnostics figure를 렌더링합니다.

- spread distribution
- depth imbalance -> next mid return 관계
- absolute return autocorrelation
- regime occupancy

이 메서드는 최소 두 개 이상의 history row가 필요합니다.

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

`Market`에 전달하는 `config`는 `MarketConfig` 인스턴스나 같은 키를 가진 `dict` 둘 다 가능합니다.
