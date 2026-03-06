# 시작하기

[문서 인덱스](https://github.com/smturtle2/quoteflow/blob/main/docs/ko/README.md) | [English](https://github.com/smturtle2/quoteflow/blob/main/docs/getting-started.md)

## 설치

```bash
pip install orderwave
```

개발 환경 설치:

```bash
pip install -e .[dev]
```

## 최소 예제

```python
from orderwave import Market

market = Market(seed=42)
snapshot = market.step()
history = market.get_history()
```

## 생성자

```python
Market(
    init_price=100.0,
    tick_size=0.01,
    levels=5,
    seed=None,
    config=None,
)
```

- `init_price`: nearest tick으로 스냅되는 초기 기준 가격
- `tick_size`: 내부 호가장의 가격 단위
- `levels`: `get()`이 반환하는 visible depth
- `seed`: 재현 가능한 난수 시드
- `config`: `dict` 또는 `orderwave.config.MarketConfig`

## Preset

- `balanced`: 기본값, 비교적 균형 잡힌 흐름과 스프레드 성향
- `trend`: 방향성과 fair-value 압력이 더 강함
- `volatile`: 스프레드 확대, 취소, 공격 주문 압력이 더 강함

## Snapshot 동작

`get()`이 반환하는 현재 상태는 의도적으로 compact합니다.

- `mid_price`는 최우선 bid/ask를 반영합니다
- `last_price`는 실제 체결 때만 갱신됩니다
- `trade_strength`는 대칭형 `[-1, 1]` signed flow 지표입니다
- `bids`, `asks`는 최대 `levels`개의 visible price level을 담습니다

## 재현성

시뮬레이터는 생성 시점의 NumPy random generator를 사용합니다. 같은 인자와 같은 seed로 만든 두 시장은 같은 경로를 생성해야 합니다.
