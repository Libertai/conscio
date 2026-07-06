# Release Process

One-time setup (already done or do before the first tag):

- Create a PyPI **Trusted Publisher** for project `conscio-agent`: repository
  `Libertai/conscio`, workflow `release.yml`, environment `pypi`.
- Create the `pypi` environment in the GitHub repo settings (optionally with
  required reviewers — that makes PyPI publishes a manual approval).

## Cutting a release

1. Make sure `main` is green and, if `web/**` changed recently, that the
   `build-web` bot commit with rebuilt static assets has landed — the release
   gate fails if `src/conscio/static/index.html` is missing from the tag.
2. Roll `CHANGELOG.md`: rename `## [Unreleased]` to `## [X.Y.Z]` with the date
   and start a fresh empty `## [Unreleased]` above it.
3. Bump `version` in `pyproject.toml` to `X.Y.Z` and run `uv lock`.
4. Commit, then tag and push:

   ```bash
   git tag vX.Y.Z
   git push origin main vX.Y.Z
   ```

5. Watch the `release` workflow: the `gate` job re-runs lint/type/tests/docs
   checks and refuses a tag that does not match `pyproject.toml`.
6. Verify artifacts:

   ```bash
   pip install conscio-agent==X.Y.Z && conscio --version
   docker pull ghcr.io/libertai/conscio:X.Y.Z
   ```

## Rollback

A bad release is rolled back by tagging a fixed `vX.Y.(Z+1)` — PyPI does not
allow re-uploading a deleted version. `docker pull ghcr.io/libertai/conscio:latest`
follows the newest tag, so shipping the fix supersedes the bad image.
