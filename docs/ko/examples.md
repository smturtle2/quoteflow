# 예제

[문서 인덱스](https://github.com/smturtle2/quoteflow/blob/main/docs/ko/README.md) | [English](https://github.com/smturtle2/quoteflow/blob/main/docs/examples.md)

## Built-in Overview Plot

```python
from orderwave import Market

market = Market(seed=7, config={"preset": "trend"})
market.gen(steps=2_000)

figure = market.plot(levels=8, title="orderwave overview")
figure.savefig("orderwave-overview.png")
```

![Overview image](../assets/orderwave-overview.png)

## 현재 호가 스냅샷

```python
book_figure = market.plot_book(levels=8, title="Current order book")
book_figure.savefig("orderwave-current-book.png")
```

![Current book](../assets/orderwave-current-book.png)

## Diagnostics

```python
diagnostics = market.plot_diagnostics(max_lag=12, title="Diagnostics")
diagnostics.savefig("orderwave-diagnostics.png")
```

![Diagnostics snapshot](../assets/orderwave-diagnostics.png)

이 built-in figure들은 서로 다른 질문에 답하도록 설계했습니다.

- 시뮬레이터가 어떤 경로를 만들었는가?
- 현재 호가장은 어떤 모양인가?
- 생성된 경로가 유용한 미시구조 신호를 갖는가?

## CLI 예제

저장소에는 [`examples/plot_market_heatmap.py`](https://github.com/smturtle2/quoteflow/blob/main/examples/plot_market_heatmap.py) 예제가 있고, 이제 내부적으로 `Market.plot()`을 직접 호출합니다.

```bash
python examples/plot_market_heatmap.py --steps 2000 --preset trend --output artifacts/orderwave_heatmap.png
```

## Preset 비교

![Preset comparison](../assets/orderwave-presets.png)

preset comparison 그림은 문서 전용이지만, 동일한 public simulation API를 다른 preset으로 실행해 생성합니다.

## 문서 이미지 다시 생성

```bash
python scripts/render_doc_images.py
```
