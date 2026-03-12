# orderwave

가독성 좋은 built-in heatmap을 다시 포함한 compact aggregate order-book 시뮬레이터입니다.

`orderwave`는 runtime 모델을 작게 유지합니다. sparse bid/ask book, Poisson limit/market/cancel flow, bounded mean-reverting fair value, 그리고 sweep/recovery 구조를 만드는 가벼운 liquidity-state kernel만 둡니다. 예전의 큰 heuristic 트리는 복구하지 않습니다.

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
- `plot()`: price path와 mid-anchor signed-depth heatmap 렌더. `capture="visual"` 필요
- `plot_heatmap(anchor="mid" | "price")`: standalone heatmap 렌더. `capture="visual"` 필요
- `plot_book()`: 현재 order book 렌더

`capture="summary"`는 fast path를 최대한 가볍게 유지합니다. `capture="visual"`은 움직이는 시장 중심 주변의 fixed signed-depth window를 저장해서, heatmap에서 sweep, void, refill 구조가 보이게 합니다.

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

- fair price는 bounded mean-reverting Gaussian process로 움직입니다.
- limit, market, cancel 이벤트 개수는 Poisson 분포에서 샘플링합니다.
- 이벤트 side는 fair-value gap, depth imbalance, 최근 signed flow로 결정합니다.
- limit placement는 inside join/improve, best-level refill, deeper wall placement의 mixture로 배치합니다.
- aggressive flow는 side-specific stress와 refill pressure를 올려 heatmap에 비대칭 withdrawal/recovery 구조가 보이게 합니다.

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
