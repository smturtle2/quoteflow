# orderwave

[![PyPI version](https://img.shields.io/pypi/v/orderwave.svg)](https://pypi.org/project/orderwave/)
[![Python versions](https://img.shields.io/pypi/pyversions/orderwave.svg)](https://pypi.org/project/orderwave/)
[![Release workflow](https://github.com/smturtle2/quoteflow/actions/workflows/workflow.yml/badge.svg)](https://github.com/smturtle2/quoteflow/actions/workflows/workflow.yml)

파이썬용 주문흐름 기반 합성 시장 시뮬레이터입니다.

`orderwave`는 가격을 먼저 랜덤 생성한 뒤 이유를 붙이지 않습니다. 희소한 지정가 호가장, 신규 지정가 유입, 시장가성 공격 주문, 취소, 스프레드 안쪽 개선호가를 시뮬레이션하고, 그 결과로 가격이 형성되게 만듭니다.

[English README](https://github.com/smturtle2/quoteflow/blob/main/README.md) | [English Docs](https://github.com/smturtle2/quoteflow/tree/main/docs) | [한국어 문서](https://github.com/smturtle2/quoteflow/tree/main/docs/ko)

![orderwave overview](docs/assets/orderwave-overview.png)

## 왜 orderwave인가

- 공개 API는 `from orderwave import Market` 하나로 단순합니다
- 가격 변화는 오직 호가장 변화의 결과로만 발생합니다
- hidden fair value가 주문 흐름을 편향시키지만 가격을 직접 덮어쓰지 않습니다
- `calm`, `directional`, `stressed` regime 전환을 지원합니다
- 같은 seed면 같은 경로가 재현됩니다

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

market = Market(seed=42, config={"preset": "balanced"})

market.step()
market.gen(steps=1_000)

snapshot = market.get()
history = market.get_history()

print(snapshot["mid_price"], snapshot["best_bid"], snapshot["best_ask"])
print(history.tail())
```

## 얻을 수 있는 것

- 호가 기반 주 가격 경로인 `mid_price`
- 실제 체결이 있을 때만 바뀌는 `last_price`
- aggregate depth 기반 bid/ask ladder
- `pandas.DataFrame` 형태의 compact history
- `balanced`, `trend`, `volatile` preset

## 공개 API

```python
from orderwave import Market

market = Market(
    init_price=100.0,
    tick_size=0.01,
    levels=5,
    seed=42,
    config={"preset": "trend"},
)
```

| API | 설명 |
| --- | --- |
| `step()` | 한 번의 micro-batch를 실행하고 최신 snapshot을 반환 |
| `gen(steps=n)` | `n`번 진행하고 마지막 snapshot을 반환 |
| `get()` | 현재 snapshot 반환 |
| `get_history()` | compact `pandas.DataFrame` history 반환 |

고급 설정은 `orderwave.config.MarketConfig`를 통해 사용할 수 있습니다.

## Snapshot 의미

`Market.get()`은 가격, 스프레드, visible depth, 최근 공격 주문 거래량, 체결 강도, depth imbalance, regime을 담은 dict를 반환합니다.

핵심 차이:

- `mid_price`는 호가 개선, 취소, 최우선 호가 소진만으로도 움직일 수 있습니다
- `last_price`는 체결이 실제로 발생했을 때만 바뀝니다

## 시각화 예제

저장소에는 가격, 체결 강도, visible-book heatmap을 그리는 matplotlib 예제가 포함되어 있습니다.

```bash
pip install matplotlib
python examples/plot_market_heatmap.py --steps 2000 --preset trend
```

파일로 바로 저장:

```bash
python examples/plot_market_heatmap.py --steps 2000 --preset trend --output artifacts/orderwave_heatmap.png
```

## Preset 비교

![orderwave presets](docs/assets/orderwave-presets.png)

`balanced`, `trend`, `volatile` preset은 같은 API를 유지하면서 스프레드 성향, 주문흐름 압력, hidden fair-price 동학을 다르게 만듭니다.

## 진단 뷰

![orderwave diagnostics](docs/assets/orderwave-diagnostics.png)

시뮬레이터는 spread 변화, imbalance lead, 변동성 군집, regime 점유율 같은 미시구조 진단을 함께 확인하기 좋게 설계했습니다.

## 문서

- [문서 인덱스](https://github.com/smturtle2/quoteflow/blob/main/docs/ko/README.md)
- [시작하기](https://github.com/smturtle2/quoteflow/blob/main/docs/ko/getting-started.md)
- [API 레퍼런스](https://github.com/smturtle2/quoteflow/blob/main/docs/ko/api.md)
- [예제](https://github.com/smturtle2/quoteflow/blob/main/docs/ko/examples.md)
- [릴리스 가이드](https://github.com/smturtle2/quoteflow/blob/main/docs/ko/releasing.md)

## 설계 보장

- 가격을 직접 random walk 하지 않습니다
- 시장가 체결, 최우선 호가 소진, inside-spread 개선호가만 가격을 움직입니다
- history는 seeded initial book을 포함한 `step == 0`부터 시작합니다
- v1에서는 per-order FIFO 대신 aggregate depth만 모델링합니다
