# orderwave 문서

[English Docs](https://github.com/smturtle2/quoteflow/tree/main/docs) | [한국어 README](https://github.com/smturtle2/quoteflow/blob/main/README.ko.md)

`orderwave`는 상태조건부 aggregate limit order book을 시뮬레이션하는 간결한 파이썬 라이브러리입니다.

![orderwave overview](../assets/orderwave-overview.png)

## 문서 목록

- [시작하기](https://github.com/smturtle2/quoteflow/blob/main/docs/ko/getting-started.md)
- [API 레퍼런스](https://github.com/smturtle2/quoteflow/blob/main/docs/ko/api.md)
- [예제](https://github.com/smturtle2/quoteflow/blob/main/docs/ko/examples.md)
- [릴리스](https://github.com/smturtle2/quoteflow/blob/main/docs/ko/releasing.md)

## 핵심 아이디어

이 시뮬레이터는 가격을 원인으로 두지 않고 결과로 둡니다.

- 지정가 주문은 상태조건부 레벨 분포에서 생성됩니다
- 시장가성 주문은 호가장 구조, hidden fair value, 최근 흐름, 스프레드에 반응합니다
- 취소는 최우선 호가를 소진시킬 수 있습니다
- inside-spread 개선호가는 체결 없이도 스프레드를 줄일 수 있습니다

즉, 결과 가격 경로는 직접 random walk 된 값이 아니라 호가장 변화의 산물입니다.

## 미리 보기

![Preset comparison](../assets/orderwave-presets.png)

![Diagnostics](../assets/orderwave-diagnostics.png)
