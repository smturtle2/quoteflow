# 예제

[문서 인덱스](https://github.com/smturtle2/quoteflow/blob/main/docs/ko/README.md) | [English](https://github.com/smturtle2/quoteflow/blob/main/docs/examples.md)

## 빠른 진단

```python
from orderwave import Market

market = Market(seed=7, config={"preset": "trend"})
market.gen(steps=5_000)
history = market.get_history()

mid_ret = history["mid_price"].diff().fillna(0.0)
abs_ret = mid_ret.abs()

print("spread mean:", history["spread"].mean())
print("imbalance -> next return corr:", history["depth_imbalance"].corr(mid_ret.shift(-1).fillna(0.0)))
print("|return| lag-1 autocorr:", abs_ret.autocorr(lag=1))
```

## 가격 + 체결 강도 + Heatmap

저장소에는 [`examples/plot_market_heatmap.py`](https://github.com/smturtle2/quoteflow/blob/main/examples/plot_market_heatmap.py) 예제가 포함되어 있습니다.

![Overview image](../assets/orderwave-overview.png)

로컬 실행:

```bash
pip install matplotlib
python examples/plot_market_heatmap.py --steps 2000 --preset trend
```

파일 저장:

```bash
python examples/plot_market_heatmap.py --steps 2000 --preset trend --output artifacts/orderwave_heatmap.png
```

예제는 다음을 렌더링합니다.

- `mid_price`, `last_price`
- `trade_strength`
- `ask n ... ask 1, bid 1 ... bid n` 구조의 signed visible-book heatmap

heatmap은 signed depth를 유지하면서 0은 정확히 검은색으로 렌더링합니다.

## Preset 비교

![Preset comparison](../assets/orderwave-presets.png)

같은 seed와 비슷한 조건에서 preset만 바꿔서 돌리면 어떤 체감 차이가 생기는지 빠르게 볼 수 있습니다.

## 진단 스냅샷

![Diagnostics snapshot](../assets/orderwave-diagnostics.png)

이 이미지는 synthetic market path를 볼 때 자주 확인하는 세 가지를 묶어 보여줍니다.

- spread가 단일값에 고정되지 않는지
- depth imbalance가 다음 움직임에 방향 정보를 주는지
- 절대수익률이 양의 자기상관을 가지는지

문서 이미지를 다시 만들려면:

```bash
python scripts/render_doc_images.py
```
