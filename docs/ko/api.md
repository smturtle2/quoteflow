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

`Market`는 메인 공개 진입점입니다. `step == 0`에서 초기 aggregate order book을 시드하고 즉시 history를 기록합니다.

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
