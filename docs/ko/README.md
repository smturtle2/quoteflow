# orderwave 문서

[English Docs](https://github.com/smturtle2/quoteflow/tree/main/docs) | [한국어 README](https://github.com/smturtle2/quoteflow/blob/main/README.ko.md)

`orderwave`는 상태조건부 aggregate limit order book을 시뮬레이션하고, 같은 `Market` 객체에서 결과를 바로 시각화할 수 있는 파이썬 라이브러리입니다.

![Overview](../assets/orderwave-overview.png)

## 문서 목록

- [시작하기](https://github.com/smturtle2/quoteflow/blob/main/docs/ko/getting-started.md)
- [API 레퍼런스](https://github.com/smturtle2/quoteflow/blob/main/docs/ko/api.md)
- [예제](https://github.com/smturtle2/quoteflow/blob/main/docs/ko/examples.md)
- [릴리스](https://github.com/smturtle2/quoteflow/blob/main/docs/ko/releasing.md)

## 내장 시각화

- `Market.plot()`으로 메인 overview figure 렌더
- `Market.plot_book()`으로 현재 order book을 실제 가격축으로 렌더
- `Market.plot_diagnostics()`로 spread, imbalance, volatility, regime 진단 렌더

![Current book](../assets/orderwave-current-book.png)

![Diagnostics](../assets/orderwave-diagnostics.png)
