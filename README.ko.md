# orderwave

가독성 좋은 built-in heatmap을 포함한 compact aggregate order-book 시뮬레이터입니다.

`orderwave`는 runtime 모델을 작게 유지합니다. sparse bid/ask book, bounded mean-reverting fair value, 그리고 큰 heuristic 트리 대신 latent-liquidity Cox kernel 하나로 hidden state에서 visible depth, cancel, market sweep을 확률적으로 드러냅니다.

![Overview](docs/assets/orderwave-built-in-overview.png)

## 설치

```bash
pip install orderwave
```

## 빠른 시작

```python
from orderwave import Market

market = Market(seed=42, capture="visual")
result = market.run(steps=1_000)

snapshot = result.snapshot
history = result.history
overview = market.plot()
heatmap = market.plot_heatmap(anchor="price")
book = market.plot_book()
```

## 공개 API

- `Market(...)`: 초기 가격, tick size, visible depth, seed, 선택적 `MarketConfig`, `capture="summary" | "visual"`로 시뮬레이터 생성
- `step()`: 한 step 진행 후 최신 snapshot 반환
- `gen(steps)`: 여러 step 진행 후 최신 snapshot 반환
- `run(steps)`: `SimulationResult(snapshot=..., history=...)` 반환
- `get()`: 현재 snapshot을 `dict`로 반환
- `get_history()`: summary history를 `pandas.DataFrame`으로 반환
- `plot()`: price path와 고정 level-rank signed-depth heatmap 렌더. `capture="visual"` 필요
- `plot_heatmap(anchor="mid" | "price")`: 고정 level 좌표계 기반 standalone heatmap 렌더. `capture="visual"` 필요
- `plot_book()`: 현재 order book 렌더

`capture="summary"`는 fast path를 최대한 가볍게 유지합니다. `capture="visual"`은 움직이는 시장 중심 주변의 fixed signed-depth window를 저장해서, heatmap에서 sweep, void, refill 구조가 보이게 합니다. heatmap row는 항상 `ask N ... ask 1 | bid 1 ... bid N` 형태의 고정 visible rank라서 가격 움직임 때문에 위아래로 밀리지 않습니다.

## Snapshot과 History

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

- fair price는 약한 flow coupling이 섞인 bounded mean-reverting Gaussian process로 움직입니다.
- hidden liquidity 상태가 먼저 진화하고, 그 상태에서 visible limit/cancel/market flow를 Cox-Poisson 계열 intensity로 샘플링합니다.
- 얇아진 side의 회복은 hard floor가 아니라 shortage-aware reveal budget, connected queue score, smooth cancel thinning에서 나옵니다.
- repair는 safety-only입니다. one-sided/crossed book과 spread cap만 다루고, visible rank를 보기 좋게 강제로 채우지는 않습니다.

## Realism 프로파일링

아래 명령으로 generic microstructure 지표를 점검할 수 있습니다.

```bash
python -m scripts.profile_realism --steps 5000
```

출력에는 spread/impact persistence, trade-sign autocorrelation, 상위 rank gap 빈도, rank별 depth shape, visible/full-book one-sidedness, near-touch connectivity, pair-distribution entropy가 포함됩니다.

## 문서 이미지

![Book](docs/assets/orderwave-built-in-current-book.png)

![Diagnostics](docs/assets/orderwave-built-in-diagnostics.png)

![Variants](docs/assets/orderwave-built-in-presets.png)

문서 이미지는 아래 명령으로 다시 만듭니다.

```bash
python -m scripts.render_doc_images
```

standalone heatmap 예제:

```bash
python -m examples.plot_market_heatmap --output artifacts/orderwave_heatmap.png
```

추가 문서:

- English: [docs/en/README.md](docs/en/README.md)
- 한국어 release 문서: [docs/ko/releasing.md](docs/ko/releasing.md)
