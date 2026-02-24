# Release Process

## Branching Strategy

We use **GitHub Flow** -- a simple, trunk-based model used by most major open-source projects (Google ADK, LangChain, etc.).

```
main                           --- single source of truth, always releasable
username/short-description     --- working branches (PR into main)
release/vX.Y.Z                --- release prep (PR into main)
```

### Branch naming

Use `username/short-description` so it's clear who owns the branch:

```
brijesh/streaming-support
brijesh/fix-validation-bug
alice/add-gemini-provider
release/v0.2.0
```

### Rules

| Branch | Merge via | Protected |
|--------|-----------|-----------|
| `main` | PR with CI checks passing | Yes |
| `username/*` | PR into `main` | No |
| `release/*` | PR into `main` | No |

Maintainers can push directly to `main` for small fixes. External contributors must use PRs.

### Workflow

**New feature or fix:**
```bash
git checkout main
git pull origin main
git checkout -b yourname/streaming-support
# ... work ...
git push -u origin yourname/streaming-support
# Open PR -> main
```

**Prepare a release:**
```bash
git checkout main
git pull origin main
git checkout -b release/v0.2.0
# Bump versions, update changelog
git push -u origin release/v0.2.0
# Open PR -> main
# After merge: create GitHub Release with tag v0.2.0
```

---

## Branch Protection -- `main`

Apply in GitHub -> Settings -> Branches -> Add rule for `main`:

- [x] **Require a pull request before merging** (for external contributors)
  - [x] Dismiss stale pull request approvals when new commits are pushed
- [x] **Require status checks to pass before merging**
  - Required checks:
    - `ci / Core (Python 3.12)`
    - `ci / Lint`
    - `ci / Import check`
    - `ci / Build nucleusiq`
    - `ci / Build nucleusiq-openai`
- [x] **Require branches to be up to date before merging**
- [ ] **Do not allow bypassing** -- unchecked so maintainers can push directly
- [ ] Require signed commits (optional, enable later)

---

## GitHub Setup Checklist (One-Time)

### Repository Settings

- [x] `main` as default branch
- [x] Apply branch protection rules for `main` (see above)

### PyPI Setup

- [x] Create PyPI account at https://pypi.org
- [x] Set up Trusted Publisher (OIDC) for `nucleusiq`
- [x] Set up Trusted Publisher (OIDC) for `nucleusiq-openai`
  - Repository: `nucleusbox/NucleusIQ`
  - Workflow: `publish.yml`
  - Environment: `pypi`

### GitHub Environments

- [x] Create environment `pypi` in GitHub -> Settings -> Environments

### GitHub Secrets

No secrets needed -- using Trusted Publisher (OIDC).

---

## Release Checklist

Use this checklist for every release:

### Pre-Release

- [ ] All tests pass: `python -m pytest` in both core and openai
- [ ] Version bumped in:
  - [ ] `src/nucleusiq/pyproject.toml`
  - [ ] `src/providers/llms/openai/pyproject.toml`
- [ ] `CHANGELOG.md` updated with release notes
- [ ] No uncommitted changes
- [ ] Build verification:
  ```bash
  cd src/nucleusiq && python -m build && twine check dist/*
  cd src/providers/llms/openai && python -m build && twine check dist/*
  ```

### Release

- [ ] Merge release PR into `main` (or push directly for maintainers)
- [ ] Create GitHub Release with tag `vX.Y.Z` pointing at `main`
- [ ] Publish workflow triggers automatically and uploads to PyPI
- [ ] Verify packages on PyPI:
  - https://pypi.org/project/nucleusiq/
  - https://pypi.org/project/nucleusiq-openai/

### Post-Release

- [ ] Verify install: `pip install nucleusiq==X.Y.Z nucleusiq-openai==X.Y.Z`
- [ ] Announce release
