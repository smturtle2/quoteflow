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

## Plotting

plot surface:

- `plot()` overview figure
- `plot_heatmap(anchor="mid" | "price")` standalone signed-depth heatmap
- `plot_book()` 현재 ladder snapshot

Heatmap 의미:

- `anchor="mid"`는 움직이는 시장 중심 기준으로 row를 정렬하므로 sweep/refill 구조를 읽기 좋습니다.
- `anchor="price"`는 실제 가격축 기준이라 drift와 persistent wall을 읽기 좋습니다.
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
