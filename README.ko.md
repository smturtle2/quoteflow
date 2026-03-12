# orderwave

Python용 간결한 aggregate order-book 시뮬레이터입니다.

`orderwave`는 sparse bid/ask book을 유지하면서 limit/market/cancel flow를 명시적인 분포에서 샘플링하고, 결과를 작은 summary history로 기록합니다. 공개 API는 의도적으로 좁습니다. `Market`을 만들고, `step` 또는 `run`으로 진행시키고, 최신 snapshot과 history만 읽으면 됩니다.

![Overview](docs/assets/orderwave-built-in-overview.png)

## 설치

```bash
pip install orderwave
```

## 빠른 시작

```python
from orderwave import Market

market = Market(seed=42)
result = market.run(steps=1_000)

snapshot = result.snapshot
history = result.history
```

자주 쓰는 설정은 `config`에 넣습니다.

```python
from orderwave import Market, MarketConfig

config = MarketConfig(
    market_rate=3.0,
    fair_price_vol=0.45,
    max_spread_ticks=4,
)

market = Market(seed=7, config=config)
market.gen(steps=500)
snapshot = market.get()
```

## 공개 API

- `Market(...)`: 초기 가격, tick size, visible depth, seed, 선택적 `MarketConfig`로 시뮬레이터 생성
- `step()`: 한 step 진행 후 최신 snapshot 반환
- `gen(steps)`: 여러 step 진행 후 최신 snapshot 반환
- `run(steps)`: `SimulationResult(snapshot=..., history=...)` 반환
- `get()`: 현재 snapshot을 `dict`로 반환
- `get_history()`: summary history를 `pandas.DataFrame`으로 반환

Snapshot field:

- `step`
- `last_price`
- `mid_price`
- `best_bid`
- `best_ask`
- `spread`
- `bids`
- `asks`
- `bid_depth`
- `ask_depth`
- `depth_imbalance`
- `buy_aggr_volume`
- `sell_aggr_volume`
- `fair_price`

History column:

- `step`
- `last_price`
- `mid_price`
- `best_bid`
- `best_ask`
- `spread`
- `bid_depth`
- `ask_depth`
- `depth_imbalance`
- `buy_aggr_volume`
- `sell_aggr_volume`
- `fair_price`

## 모델

내부 모델은 하나만 남깁니다.

- fair price는 bounded mean-reverting Gaussian process로 움직입니다.
- limit, market, cancel 이벤트 개수는 Poisson 분포에서 샘플링합니다.
- 이벤트 side는 fair-value gap과 현재 depth imbalance로 결정합니다.
- 이벤트 level은 truncated decay 분포에서 샘플링합니다.
- 이벤트 size는 bounded lognormal 분포에서 샘플링합니다.

runtime 패키지에는 preset, participant taxonomy, latent regime, validation pipeline, plotting API가 더 이상 없습니다.

## 문서 이미지

![Book](docs/assets/orderwave-built-in-current-book.png)

![Diagnostics](docs/assets/orderwave-built-in-diagnostics.png)

![Variants](docs/assets/orderwave-built-in-presets.png)

문서 이미지는 아래 명령으로 다시 만듭니다.

```bash
python -m scripts.render_doc_images
```

추가 문서:

- English: [docs/en/README.md](docs/en/README.md)
- 한국어 release 문서: [docs/ko/releasing.md](docs/ko/releasing.md)
