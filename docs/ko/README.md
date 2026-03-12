# orderwave 문서

`orderwave`는 두 가지 runtime mode를 가진 compact event-based aggregate book 시뮬레이터입니다.

- `capture="summary"`: fast path
- `capture="visual"`: overview/heatmap plot용 visual capture 포함

## Runtime 표면

- `Market`
- `MarketConfig`
- `SimulationResult`

## MarketConfig

`MarketConfig`는 flow intensity, fair-value movement, price-level decay, order-size bounds, spread/fair-move limit만 계속 노출합니다.

## Runtime 모델

- 엔진은 aggregate-book 구조를 유지합니다. per-order FIFO queue는 시뮬레이션하지 않습니다.
- 현실성은 latent distribution synthesis에서 나옵니다. hidden stochastic state가 total liquidity, side skew, side별 depth distribution을 먼저 합성하고, 그 뒤 visible limit/cancel/market flow가 샘플링됩니다.
- 얇아진 side의 회복은 hard floor가 아니라 shortage distribution과 near-touch distribution이 다시 합성되면서 만들어집니다.
- repair는 safety-only라서 visible hole과 delayed refill이 매 step 지워지지 않습니다.

## Plotting

plot surface:

- `plot()` overview figure
- `plot_heatmap(anchor="mid" | "price")` standalone signed-depth heatmap
- `plot_book()` 현재 ladder snapshot

Heatmap 의미:

- heatmap row는 항상 `ask N ... ask 1 | bid 1 ... bid N` 순서의 고정 visible rank입니다.
- y축은 가격으로 바뀌지 않고, 시장 가격이 움직여도 row가 위아래로 드리프트하지 않습니다.
- `anchor="mid"`와 `anchor="price"`는 API 호환성 때문에 유지하지만, 둘 다 같은 고정 level-rank heatmap을 그립니다.
- color는 robust asinh scaling이 적용된 signed depth라서 큰 wall 하나가 전체 contrast를 망치지 않습니다.

## 문서 이미지

아래 명령으로 문서용 이미지를 전부 다시 생성합니다.

```bash
python -m scripts.render_doc_images
```

standalone example:

```bash
python -m examples.plot_market_heatmap --output artifacts/orderwave_heatmap.png
```

realism profile:

```bash
python -m scripts.profile_realism --steps 5000
```

출력에는 spread/impact persistence, rank별 depth shape, visible/full-book one-sidedness, near-touch connectivity, pair-distribution entropy가 포함됩니다.
