# Releasing

[Docs index](https://github.com/smturtle2/quoteflow/blob/main/docs/en/README.md) | [한국어](https://github.com/smturtle2/quoteflow/blob/main/docs/ko/releasing.md)

PyPI publishing is driven by [`workflow.yml`](https://github.com/smturtle2/quoteflow/blob/main/.github/workflows/workflow.yml).

## Release Flow

1. Run the local checks:

   ```bash
   python -m pytest -q
   python -m scripts.validate_orderwave --profile release_smoke --outdir artifacts/validation-release
   python -m scripts.render_doc_images
   ```

2. If the smoke validation changed intentionally, refresh the release baseline:

   ```bash
   python -m scripts.validate_orderwave \
     --profile release_smoke \
     --outdir artifacts/validation-release \
     --write-baseline-json tests/golden/validation_release_baseline.json
   ```

3. Update `version` in `pyproject.toml`
4. Commit code, tests, docs, regenerated images, and any intentionally refreshed golden baseline
5. Push to `main`
6. Open GitHub `Releases`
7. Draft a new release with a tag like `vX.Y.Z`
8. Publish the release
9. Wait for `Quality`, `Test`, `Build distributions`, `Release Validation`, and `Publish to PyPI`

## What CI Runs Before Publish

The release validation job executes:

```bash
python -m scripts.validate_orderwave \
  --profile release_smoke \
  --outdir artifacts/validation-release \
  --baseline-json tests/golden/validation_release_baseline.json \
  --fail-on-baseline-drift
```

This profile is intentionally much smaller than the full quality regression sweep so the gate stays fast in CI.
It also uses a looser throughput floor than the full validation profile so GitHub-hosted runner variance does not block a structurally identical release. Use `scripts.measure_performance` for machine-local throughput tracking.

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
- Regenerated doc images are expected to ship with feature-level simulator changes
- Release validation compares against `tests/golden/validation_release_baseline.json`
