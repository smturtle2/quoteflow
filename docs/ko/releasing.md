# 릴리스

[문서 인덱스](https://github.com/smturtle2/quoteflow/blob/main/docs/ko/README.md) | [English](https://github.com/smturtle2/quoteflow/blob/main/docs/en/releasing.md)

PyPI 배포는 [`workflow.yml`](https://github.com/smturtle2/quoteflow/blob/main/.github/workflows/workflow.yml)에서 처리합니다.

## 릴리스 절차

1. `pyproject.toml`의 `version`을 올립니다
2. 커밋 후 `main`에 푸시합니다
3. GitHub `Releases`를 엽니다
4. `vX.Y.Z` 같은 태그로 새 릴리스를 작성합니다
5. 릴리스를 publish 합니다
6. GitHub Actions가 테스트, 빌드, PyPI 업로드를 수행할 때까지 기다립니다

PyPI publish 전에 워크플로는 아래 짧은 validation 명령을 실행합니다.

```bash
python scripts/validate_orderwave.py --profile release --outdir artifacts/validation-release --baseline-json tests/golden/validation_release_baseline.json --fail-on-baseline-drift
```

이 release profile은 full validation sweep보다 훨씬 작아서 CI 릴리스 게이트를 빠르게 유지합니다.

## Trusted Publisher 설정

- PyPI project name: `orderwave`
- Repository owner: `smturtle2`
- Repository name: `quoteflow`
- Workflow filename: `.github/workflows/workflow.yml`
- Environment name: `pypi`

## 참고

- 릴리스 워크플로는 `release.published`에서 동작합니다
- 초안만 만들어서는 배포되지 않습니다
- PyPI trusted publishing에는 GitHub Actions job의 `id-token: write` 권한이 필요합니다
- release validation은 `tests/golden/validation_release_baseline.json`과 비교합니다
