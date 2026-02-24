# Release Process

## Branching Strategy

```
main          ─── stable releases only, protected
                    │
develop       ─── integration branch, all features merge here first
                    │
feature/*     ─── individual features (branch from develop, PR into develop)
fix/*         ─── bug fixes (branch from develop, PR into develop)
release/*     ─── release prep (branch from develop, PR into main + back-merge to develop)
hotfix/*      ─── critical prod fixes (branch from main, PR into main + back-merge to develop)
```

### Rules

| Branch | Who can push | Merge via | Protected |
|--------|-------------|-----------|-----------|
| `main` | Nobody (PR only) | PR from `release/*` or `hotfix/*` | Yes |
| `develop` | Nobody (PR only) | PR from `feature/*`, `fix/*`, `release/*` | Yes |
| `feature/*` | Developer | PR into `develop` | No |
| `fix/*` | Developer | PR into `develop` | No |
| `release/*` | Release manager | PR into `main` (then back-merge to `develop`) | No |
| `hotfix/*` | Developer | PR into `main` (then back-merge to `develop`) | No |

### Workflow

**New feature:**
```bash
git checkout develop
git pull origin develop
git checkout -b feature/streaming-support
# ... work ...
git push -u origin feature/streaming-support
# Open PR: feature/streaming-support → develop
```

**Prepare a release:**
```bash
git checkout develop
git pull origin develop
git checkout -b release/v0.2.0
# Bump versions, update changelog, final fixes
git push -u origin release/v0.2.0
# Open PR: release/v0.2.0 → main
# After merge to main: tag v0.2.0
# Back-merge: main → develop
```

**Hotfix in production:**
```bash
git checkout main
git pull origin main
git checkout -b hotfix/v0.1.1-critical-fix
# Fix the issue
git push -u origin hotfix/v0.1.1-critical-fix
# Open PR: hotfix/v0.1.1-critical-fix → main
# After merge: tag v0.1.1
# Back-merge: main → develop
```

---

## Branch Protection — `main`

Apply these settings in GitHub → Settings → Branches → Add rule for `main`:

- [x] **Require a pull request before merging**
  - [x] Require approvals: 1
  - [x] Dismiss stale pull request approvals when new commits are pushed
- [x] **Require status checks to pass before merging**
  - Required checks:
    - `Core (Python 3.12)`
    - `OpenAI (Python 3.12)`
    - `Lint`
    - `Import check`
    - `uv install check`
    - `Build nucleusiq`
    - `Build nucleusiq-openai`
    - `Security scan`
- [x] **Require branches to be up to date before merging**
- [x] **Do not allow bypassing the above settings**
- [ ] Require signed commits (optional, enable later)
- [ ] Require linear history (optional)

## Branch Protection — `develop`

- [x] **Require a pull request before merging**
  - [x] Require approvals: 1
- [x] **Require status checks to pass before merging**
  - Required checks:
    - `Core (Python 3.12)`
    - `OpenAI (Python 3.12)`
    - `Lint`
    - `Import check`

---

## GitHub Setup Checklist (One-Time)

### Repository Settings

- [ ] Create `develop` branch from `main`
- [ ] Set `develop` as default branch
- [ ] Apply branch protection rules for `main` (see above)
- [ ] Apply branch protection rules for `develop` (see above)

### PyPI Setup

- [ ] Create PyPI account at https://pypi.org
- [ ] Create project `nucleusiq` on PyPI (first upload claims the name)
- [ ] Create project `nucleusiq-openai` on PyPI
- [ ] Set up Trusted Publisher (OIDC) in PyPI for GitHub Actions:
  - PyPI → Manage Project → Publishing → Add publisher
  - Repository: `nucleusbox/NucleusIQ`
  - Workflow: `publish.yml`
  - Environment: `pypi`

### GitHub Environments

- [ ] Create environment `pypi` in GitHub → Settings → Environments
- [ ] Add protection rule: require reviewers (optional)

### GitHub Secrets

No secrets needed if using Trusted Publisher (OIDC). Otherwise:
- [ ] `PYPI_API_TOKEN` — PyPI API token (if not using OIDC)

---

## Release Checklist

Use this checklist for every release:

### Pre-Release

- [ ] All tests pass on `develop`: `python -m pytest` in both core and openai
- [ ] Version bumped in:
  - [ ] `src/nucleusiq/pyproject.toml` → `version = "X.Y.Z"`
  - [ ] `src/nucleusiq/core/__init__.py` → `__version__ = "X.Y.Z"`
  - [ ] `src/providers/llms/openai/pyproject.toml` → `version = "X.Y.Z"`
- [ ] `CHANGELOG.md` updated with release notes
- [ ] `docs/IMPLEMENTATION_TRACKER.md` updated
- [ ] No uncommitted changes
- [ ] Build verification:
  ```bash
  cd src/nucleusiq && python -m build && twine check dist/*
  cd src/providers/llms/openai && python -m build && twine check dist/*
  ```

### Release

- [ ] Create `release/vX.Y.Z` branch from `develop`
- [ ] Open PR: `release/vX.Y.Z` → `main`
- [ ] CI passes, PR approved and merged
- [ ] Create GitHub Release with tag `vX.Y.Z` pointing at `main`
- [ ] Publish workflow triggers and uploads to PyPI
- [ ] Verify packages on PyPI:
  - https://pypi.org/project/nucleusiq/
  - https://pypi.org/project/nucleusiq-openai/
- [ ] Back-merge `main` → `develop`

### Post-Release

- [ ] Verify install: `pip install nucleusiq nucleusiq-openai`
- [ ] Announce release
- [ ] Bump versions on `develop` to next dev version (e.g., `0.2.0.dev0`)
