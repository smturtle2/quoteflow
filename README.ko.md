# orderwave

[![PyPI version](https://img.shields.io/pypi/v/orderwave.svg)](https://pypi.org/project/orderwave/)
[![Python versions](https://img.shields.io/pypi/pyversions/orderwave.svg)](https://pypi.org/project/orderwave/)
[![Release workflow](https://github.com/smturtle2/quoteflow/actions/workflows/workflow.yml/badge.svg)](https://github.com/smturtle2/quoteflow/actions/workflows/workflow.yml)

내장 시각화를 포함한 주문흐름 기반 합성 시장 시뮬레이터입니다.

`orderwave`는 가격을 직접 random walk 하지 않습니다. 희소한 지정가 호가장, 신규 지정가 유입, 시장가성 공격 주문, 취소, 스프레드 안쪽 개선호가를 시뮬레이션하고, 그 결과로 가격이 형성되게 만듭니다. 같은 `Market` 객체에서 경로, 현재 호가 스냅샷, 미시구조 진단까지 바로 그릴 수 있습니다.

[English README](https://github.com/smturtle2/quoteflow/blob/main/README.md) | [English Docs](https://github.com/smturtle2/quoteflow/tree/main/docs/en) | [한국어 문서](https://github.com/smturtle2/quoteflow/tree/main/docs/ko)

![orderwave overview](docs/assets/orderwave-built-in-overview.png)

## 왜 orderwave인가

- 공개 진입점은 `from orderwave import Market` 하나로 단순합니다
- 가격 변화는 오직 호가장 변화의 결과로만 발생합니다
- hidden fair value가 주문 흐름을 편향시키지만 가격을 직접 덮어쓰지 않습니다
- 같은 seed면 같은 경로가 재현됩니다
- overview, 현재 호가, diagnostics를 내장 시각화로 제공합니다

## 설치

```bash
pip install orderwave
```

로컬 개발용 설치:

```bash
pip install -e .[dev]
```

## 빠른 시작

```python
from orderwave import Market

market = Market(seed=42, config={"preset": "trend"})
market.gen(steps=1_000)

snapshot = market.get()
history = market.get_history()
overview = market.plot()
book = market.plot_book()
diagnostics = market.plot_diagnostics()

print(snapshot["mid_price"], snapshot["best_bid"], snapshot["best_ask"])
print(history.tail())

overview.savefig("orderwave-overview.png")
```

## API 표면

| API | 설명 |
| --- | --- |
| `step()` | 한 번의 micro-batch를 실행하고 최신 snapshot을 반환 |
| `gen(steps=n)` | `n`번 진행하고 마지막 snapshot을 반환 |
| `get()` | 현재 snapshot 반환 |
| `get_history()` | compact `pandas.DataFrame` history 반환 |
| `plot()` | 가격, 스프레드, 체결 강도, visible-book heatmap 렌더 |
| `plot_book()` | 현재 order book을 실제 가격축으로 렌더 |
| `plot_diagnostics()` | spread, imbalance, volatility, regime diagnostics 렌더 |

고급 설정은 `orderwave.config.MarketConfig`를 통해 사용할 수 있습니다.

## 내장 시각화

모든 plotting 메서드는 `matplotlib.figure.Figure`를 반환하고, 저장이나 표시 시점은 호출자가 직접 제어합니다.

- `plot()`은 가격, 스프레드, 체결 강도, signed visible-depth heatmap을 한 번에 렌더링합니다
- `plot_book()`은 현재 order book을 실제 가격축으로 렌더링합니다
- `plot_diagnostics()`는 spread, imbalance, volatility, regime diagnostics를 렌더링합니다

![orderwave current book](docs/assets/orderwave-built-in-current-book.png)

![orderwave diagnostics](docs/assets/orderwave-built-in-diagnostics.png)

overview heatmap은 signed depth를 유지합니다. ask는 빨강, bid는 파랑, `0`은 연회색 midpoint, 존재하지 않는 레벨은 검은색이 아니라 blank background로 렌더링합니다.

## Preset 비교

![orderwave presets](docs/assets/orderwave-built-in-presets.png)

`balanced`, `trend`, `volatile` preset은 같은 API를 유지하면서 스프레드 성향, 주문흐름 압력, 취소 압력, hidden fair-price 동학을 다르게 만듭니다.

## 핵심 의미

`Market.get()`은 가격, 스프레드, visible depth, 공격 주문 거래량, 체결 강도, depth imbalance, regime을 담은 compact dict를 반환합니다.

핵심 차이:

- `mid_price`는 호가 개선, 취소, 최우선 호가 소진만으로도 움직일 수 있습니다
- `last_price`는 실제 체결이 발생했을 때만 바뀝니다

핵심 보장:

- 가격을 직접 random walk 하지 않습니다
- 시장가 체결, 최우선 호가 소진, inside-spread 개선호가만 가격을 움직입니다
- history는 seeded initial book을 포함한 `step == 0`부터 시작합니다
- v1에서는 per-order FIFO 대신 aggregate depth만 모델링합니다

## 문서

- [문서 인덱스](https://github.com/smturtle2/quoteflow/blob/main/docs/ko/README.md)
- [시작하기](https://github.com/smturtle2/quoteflow/blob/main/docs/ko/getting-started.md)
- [API 레퍼런스](https://github.com/smturtle2/quoteflow/blob/main/docs/ko/api.md)
- [예제](https://github.com/smturtle2/quoteflow/blob/main/docs/ko/examples.md)
- [릴리스 가이드](https://github.com/smturtle2/quoteflow/blob/main/docs/ko/releasing.md)
