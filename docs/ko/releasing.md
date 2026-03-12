# 릴리스

[문서 인덱스](https://github.com/smturtle2/quoteflow/blob/main/docs/ko/README.md) | [English](https://github.com/smturtle2/quoteflow/blob/main/docs/en/releasing.md)

PyPI 배포는 [`workflow.yml`](https://github.com/smturtle2/quoteflow/blob/main/.github/workflows/workflow.yml)에서 처리합니다.

## 릴리스 절차

1. 로컬 검증을 실행합니다.

   ```bash
   python -m pytest -q
   python -m scripts.validate_orderwave --profile release_smoke --outdir artifacts/validation-release
   python -m scripts.render_doc_images
   ```

2. smoke validation drift가 의도된 변경이라면 release baseline을 갱신합니다.

   ```bash
   python -m scripts.validate_orderwave \
     --profile release_smoke \
     --outdir artifacts/validation-release \
     --write-baseline-json tests/golden/validation_release_baseline.json
   ```

3. `pyproject.toml`의 `version`을 올립니다
4. 코드, 테스트, 문서, 재생성된 이미지, 의도적으로 갱신한 golden baseline을 커밋합니다
5. `main`에 푸시합니다
6. GitHub `Releases`를 엽니다
7. `vX.Y.Z` 같은 태그로 새 release를 작성합니다
8. release를 publish 합니다
9. `Quality`, `Test`, `Build distributions`, `Release Validation`, `Publish to PyPI` job이 끝날 때까지 기다립니다

## Publish 전에 CI가 실행하는 것

release validation job은 아래 명령을 실행합니다.

```bash
python -m scripts.validate_orderwave \
  --profile release_smoke \
  --outdir artifacts/validation-release \
  --baseline-json tests/golden/validation_release_baseline.json \
  --fail-on-baseline-drift
```

이 profile은 full quality regression sweep보다 훨씬 작아서 CI 릴리스 게이트를 빠르게 유지합니다.

## Trusted Publisher 설정

- PyPI project name: `orderwave`
- Repository owner: `smturtle2`
- Repository name: `quoteflow`
- Workflow filename: `.github/workflows/workflow.yml`
- Environment name: `pypi`

## 참고

- 릴리스 워크플로는 `release.published`에서 동작합니다
- draft만 만들어서는 배포되지 않습니다
- PyPI trusted publishing에는 GitHub Actions job의 `id-token: write` 권한이 필요합니다
- feature-level simulator 변경에는 재생성된 문서 이미지까지 함께 포함하는 것을 전제로 합니다
- release validation은 `tests/golden/validation_release_baseline.json`과 비교합니다
