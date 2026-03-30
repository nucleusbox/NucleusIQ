# Release Plan — NucleusIQ v0.7.1

> **Date:** 2026-03-30
> **Type:** Patch — critical fix for `.env` loading
> **Previous release:** v0.7.0 (already on PyPI)

---

## What ships in v0.7.1

| Category | Detail |
|----------|--------|
| **Critical fix** | `.env` loading broken for pip-installed consumers — `core/__init__.py` used `Path(__file__).parents[2]` which resolves into `site-packages/`. Replaced with `load_dotenv(override=False)` (searches from CWD upward). |

### What shipped in v0.7.0 (already released)

| Category | Detail |
|----------|--------|
| **Security** | `requests>=2.33.0` (CVE-2026-25645), `pygments>=2.20.0` constraint (CVE-2026-4539) |
| **Packaging fix** | `nucleusiq.tools.builtin` included in wheel |
| **CI** | `actions/upload-artifact` v6 → v7; wheel smoke test + `verify_core_package_layout.py` |
| **Lockfile refresh** | `requests` 2.32→2.33, `pygments` 2.19→2.20, `ruff` 0.15.2→0.15.8, `openai` SDK→2.30.0 |

### Package versions

| Package | Version | Action |
|---------|---------|--------|
| `nucleusiq` | **0.7.1** | Publish new patch |
| `nucleusiq-openai` | 0.6.0 | No change (requires `nucleusiq>=0.7.0`, accepts 0.7.1) |
| `nucleusiq-gemini` | 0.2.0 | No change (requires `nucleusiq>=0.7.0`, accepts 0.7.1) |

---

## Pre-release checklist

### 1. Code verification

- [x] `core/__init__.py` — `__version__ = "0.7.1"`, `.env` fix uses `load_dotenv(override=False)`
- [x] `src/nucleusiq/pyproject.toml` — `version = "0.7.1"`
- [x] Provider `pyproject.toml` files — unchanged at 0.6.0 / 0.2.0 (their `nucleusiq>=0.7.0` accepts 0.7.1)
- [x] `CHANGELOG.md` — 0.7.1 section added above 0.7.0
- [x] `docs/marketing/TRACKER.md` — v0.7.1 release table added
- [x] `docs/BACKLOG.md` — all feature targets at v0.8.0+
- [x] `integration_test/` — in `.gitignore` (internal only)

### 2. Run test suites

- [ ] **Core unit tests** — `cd src/nucleusiq && uv run pytest tests/ -x -q`
- [ ] **OpenAI provider tests** — `cd src/providers/llms/openai && uv run pytest tests/ -x -q`
- [ ] **Gemini provider tests** — `cd src/providers/llms/gemini && uv run pytest tests/ -x -q`
- [ ] **Integration test** — `cd integration_test && python run_integration.py` (internal, not committed)

### 3. Build and verify wheel

```bash
cd src/nucleusiq
uv build

# Verify wheel contains nucleusiq.tools.builtin
python -m zipfile -l dist/nucleusiq-0.7.1-py3-none-any.whl | findstr builtin
```

### 4. Smoke test — clean install from wheel

```bash
python -m venv /tmp/smoke071
/tmp/smoke071/Scripts/activate   # Windows
pip install dist/nucleusiq-0.7.1-py3-none-any.whl
python -c "from nucleusiq.tools.builtin import FileReadTool; print('OK')"
python -c "import nucleusiq; print(nucleusiq.__version__)"
```

### 5. Verify .env loading from consumer perspective

```bash
mkdir /tmp/consumer_test && cd /tmp/consumer_test
echo "OPENAI_API_KEY=test123" > .env
pip install /path/to/nucleusiq-0.7.1-py3-none-any.whl
python -c "import nucleusiq; import os; assert os.getenv('OPENAI_API_KEY') == 'test123', 'FAIL'; print('PASS: .env loaded')"
```

---

## Release steps

### Step 1 — Commit and push

```bash
git add -A
git commit -m "release: v0.7.1 — fix .env loading for pip-installed consumers

core/__init__.py used Path(__file__).parents[2] to locate .env, which
resolved into site-packages/ for pip-installed consumers. Replaced with
load_dotenv(override=False) which searches from CWD upward.

Only nucleusiq core is bumped (0.7.0 → 0.7.1). Providers unchanged
(nucleusiq-openai 0.6.0, nucleusiq-gemini 0.2.0 — their >=0.7.0
constraint accepts 0.7.1)."

git push origin main
```

### Step 2 — Create GitHub release + tag

```bash
gh release create v0.7.1 \
  --title "v0.7.1 — Fix .env loading for pip-installed consumers" \
  --notes "### Fixed

- **\`.env\` loading broken for pip-installed consumers** — \`core/__init__.py\` used \`Path(__file__).parents[2]\` to locate \`.env\`, which resolved into \`site-packages/\`. Replaced with \`load_dotenv(override=False)\` (searches from CWD upward).

### Packages

| Package | Version | Note |
|---------|---------|------|
| \`nucleusiq\` | **0.7.1** | Patch fix |
| \`nucleusiq-openai\` | 0.6.0 | No change |
| \`nucleusiq-gemini\` | 0.2.0 | No change |

**Upgrade:** \`pip install --upgrade nucleusiq\`" \
  --target main
```

### Step 3 — Publish ONLY the core package to PyPI

```bash
cd src/nucleusiq
uv publish dist/nucleusiq-0.7.1*
```

> Do NOT re-publish `nucleusiq-openai` or `nucleusiq-gemini` — they are unchanged.

### Step 4 — Verify on PyPI

```bash
pip install nucleusiq==0.7.1
python -c "import nucleusiq; print(nucleusiq.__version__)"
# Should print: 0.7.1
```

### Step 5 — Close Dependabot PRs

Close all 4 Dependabot PRs with a comment:
> Resolved in v0.7.0 + v0.7.1 (manually applied and verified).

### Step 6 — Post-release

- [ ] Monitor GitHub issues for breakage reports (48 hours)
- [ ] Brief note in any open issue threads about the `.env` fix

---

## Rollback plan

If a critical issue is found post-publish:

1. **PyPI**: `pip install nucleusiq==0.7.0` still works
2. **Git**: `git revert` the release commit and publish `0.7.2`
3. **Yank**: If the wheel itself is broken: `uv publish --yank 0.7.1 "packaging issue"`

---

## What's next (v0.8.0+)

See `docs/BACKLOG.md`. High-priority items:

1. Agent DX: String argument support for `execute()` (small, high impact)
2. Comprehensive Exception Handling Framework (medium-large)
3. Agent Types: ReAct + CoT integration (medium-large)
4. Anthropic provider (large)
5. CostTracker Agent integration (low-medium)
