# orderwave 릴리즈

이 저장소는 GitHub Release가 publish될 때만 동작하는 GitHub Actions workflow 하나만 유지합니다.

## 로컬 체크리스트

1. `python -m ruff check orderwave tests scripts examples` 실행
2. `python -m mypy` 실행
3. `python -m pytest -q` 실행
4. `python -m scripts.render_doc_images` 실행
5. 필요하면 `python -m scripts.profile_realism --steps 5000`로 aggregate profile 확인
6. 코드, 문서, 재생성된 asset을 커밋
7. `main`에 푸시

## 배포

1. `pyproject.toml`의 버전을 릴리즈 버전으로 갱신
2. 버전 커밋을 `main`에 푸시
3. `vX.Y.Z` 태그의 GitHub Release를 생성하고 publish
4. `.github/workflows/workflow.yml`가 distribution build와 PyPI publish를 끝낼 때까지 확인

## Workflow 구조

workflow는 아래 단계만 수행합니다.

1. 저장소 checkout
2. Python 3.12 설정
3. `build` 설치
4. `python -m build` 실행
5. GitHub trusted publisher 흐름으로 PyPI publish
