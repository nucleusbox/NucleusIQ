# Release Process

## Branching

We use **GitHub Flow** -- one main branch, feature branches, pull requests.

```
main                           -- single source of truth, always releasable
username/short-description     -- working branches (PR into main)
release/vX.Y.Z                -- release prep (PR into main)
```

Branch names use `username/description` so it's clear who owns the branch:

```
brijesh/streaming-support
alice/add-gemini-provider
release/v0.2.0
```

Maintainers can push directly to `main` for small fixes. External contributors must use PRs.

---

## How to Release

### 1. Prepare

```bash
git checkout main
git pull origin main
git checkout -b release/vX.Y.Z
```

- Bump `version` in `src/nucleusiq/pyproject.toml`
- Bump `version` in `src/providers/llms/openai/pyproject.toml`
- Update `CHANGELOG.md` with release notes
- Verify builds:
  ```bash
  cd src/nucleusiq && python -m build && twine check dist/*
  cd src/providers/llms/openai && python -m build && twine check dist/*
  ```

### 2. Merge

Push the branch, open a PR into `main`, wait for CI to pass, and merge.

### 3. Release

- Go to [GitHub Releases](https://github.com/nucleusbox/NucleusIQ/releases/new)
- Create a new release with tag `vX.Y.Z` targeting `main`
- Write release notes (or copy from CHANGELOG)
- Click **Publish release**

The `publish.yml` workflow triggers automatically -- runs CI, builds both packages, and publishes to PyPI via Trusted Publishing (OIDC). No API tokens or manual uploads needed.

### 4. Verify

- https://pypi.org/project/nucleusiq/
- https://pypi.org/project/nucleusiq-openai/
- `pip install nucleusiq==X.Y.Z nucleusiq-openai==X.Y.Z`
