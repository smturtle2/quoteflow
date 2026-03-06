# Releasing

[Docs index](https://github.com/smturtle2/quoteflow/blob/main/docs/en/README.md) | [한국어](https://github.com/smturtle2/quoteflow/blob/main/docs/ko/releasing.md)

PyPI publishing is driven by [`workflow.yml`](https://github.com/smturtle2/quoteflow/blob/main/.github/workflows/workflow.yml).

## Release Flow

1. Update `version` in `pyproject.toml`
2. Commit and push to `main`
3. Open GitHub `Releases`
4. Draft a new release with a tag like `v0.2.1`
5. Publish the release
6. Wait for the GitHub Actions workflow to test, build, and publish

## Trusted Publisher Settings

- PyPI project name: `orderwave`
- Repository owner: `smturtle2`
- Repository name: `quoteflow`
- Workflow filename: `.github/workflows/workflow.yml`
- Environment name: `pypi`

## Notes

- The release workflow triggers on `release.published`
- Drafts alone do not publish
- PyPI trusted publishing requires `id-token: write` in the GitHub Actions job
