# Releasing orderwave

The repository keeps one GitHub Actions workflow and it only runs when a GitHub Release is published.

## Local Checklist

1. Run `python -m ruff check orderwave tests scripts examples`.
2. Run `python -m mypy`.
3. Run `python -m pytest -q`.
4. Run `python -m scripts.render_doc_images`.
5. Optionally run `python -m scripts.profile_realism --steps 5000` and review the aggregate profile.
6. Commit the code, docs, and regenerated assets.
7. Push the branch to `main`.

## Publish

1. Update `pyproject.toml` with the release version.
2. Push the version commit to `main`.
3. Create and publish a GitHub Release tagged `vX.Y.Z`.
4. Wait for `.github/workflows/workflow.yml` to build the distributions and publish to PyPI.

## Workflow Shape

The workflow does only this:

1. Check out the repo.
2. Set up Python 3.12.
3. Install `build`.
4. Run `python -m build`.
5. Publish the built distributions to PyPI through the GitHub trusted publisher flow.
