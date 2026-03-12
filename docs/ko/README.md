# orderwave 문서

`orderwave`는 event-based aggregate book 시뮬레이터입니다. synthetic order-book path 생성, 가벼운 실험, 재현 가능한 smoke-scale 연구 실행을 목표로 합니다.

## Runtime 표면

- `Market`
- `MarketConfig`
- `SimulationResult`

나머지는 전부 내부 구현입니다. plotting helper, validation helper, preset, latent market-state label은 더 이상 공개하지 않습니다.

## MarketConfig

`MarketConfig`는 아래 12개 항목만 노출합니다.

- `limit_rate`
- `market_rate`
- `cancel_rate`
- `fair_price_vol`
- `mean_reversion`
- `level_decay`
- `size_mean`
- `size_dispersion`
- `min_order_qty`
- `max_order_qty`
- `max_spread_ticks`
- `max_fair_move_ticks`

검증 규칙:

- rate는 모두 `0`보다 커야 합니다
- `mean_reversion`은 `[0, 1]` 범위여야 합니다
- `level_decay`는 `(0, 1)` 범위여야 합니다
- `size_dispersion`은 `0`보다 커야 합니다
- `min_order_qty`는 `1` 이상이어야 합니다
- `max_order_qty`는 `min_order_qty` 이상이어야 합니다
- `max_spread_ticks`는 `1` 이상이어야 합니다
- `max_fair_move_ticks`는 `1` 이상이어야 합니다

## Snapshot과 History

`get()`은 최신 snapshot을 plain dictionary로 반환합니다. `get_history()`는 같은 핵심 필드를 시계열로 반환합니다.

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

현재 visible book view는 snapshot의 `bids`, `asks`에 포함됩니다.

## 문서 이미지

아래 명령으로 문서용 이미지를 전부 다시 생성합니다.

```bash
python -m scripts.render_doc_images
```

생성 파일:

- `docs/assets/orderwave-built-in-overview.png`
- `docs/assets/orderwave-built-in-current-book.png`
- `docs/assets/orderwave-built-in-diagnostics.png`
- `docs/assets/orderwave-built-in-presets.png`
