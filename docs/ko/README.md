# orderwave 문서

[English Docs](https://github.com/smturtle2/quoteflow/tree/main/docs/en) | [한국어 README](https://github.com/smturtle2/quoteflow/blob/main/README.ko.md)

`orderwave`는 session-aware 상태조건부 aggregate limit order book을 시뮬레이션하고, 같은 `Market` 객체에서 결과를 바로 시각화할 수 있는 파이썬 라이브러리입니다.
지원되는 공개 API는 `Market`이며, 내부 engine/model 모듈은 안정적인 import 경로로 문서화하지 않습니다.

공식 정체성은 aggregate order-book market-state simulator입니다.
주문 단위 매칭이나 체결 정밀도를 목표로 하는 시뮬레이터는 아닙니다.

![Overview](../assets/orderwave-built-in-overview.png)

## 문서 목록

- [시작하기](https://github.com/smturtle2/quoteflow/blob/main/docs/ko/getting-started.md)
- [API 레퍼런스](https://github.com/smturtle2/quoteflow/blob/main/docs/ko/api.md)
- [예제](https://github.com/smturtle2/quoteflow/blob/main/docs/ko/examples.md)
- [릴리스](https://github.com/smturtle2/quoteflow/blob/main/docs/ko/releasing.md)

## 현재 엔진에서 달라진 점

- 내부 microphase가 open release, midday lull, power hour, closing imbalance 같은 시간구조를 만듭니다
- event cycle은 market-first 실행, adverse quote revision, passive refill을 강조합니다
- debug history에 `microphase`, `flow_toxicity`, `maker_stress`, `quote_revision_wave`, `refill_pressure`가 추가됐습니다
- diagnostics는 microphase와 revision/refill pressure 패널을 추가로 렌더합니다

## 내장 시각화

- `Market.plot()`으로 메인 overview figure 렌더
- `Market.plot_book()`으로 현재 order book을 실제 가격축으로 렌더
- `Market.plot_diagnostics()`로 session, excitation, imbalance, resiliency, regime/shock, microphase, revision/refill 진단 렌더

![Current book](../assets/orderwave-built-in-current-book.png)

![Diagnostics](../assets/orderwave-built-in-diagnostics.png)
