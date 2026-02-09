# NucleusIQ First Release – TODO Checklist

**Target:** v0.1.0 first public release
**Status:** Pre-release audit (Feb 2025)

---

## Executive Summary

NucleusIQ has a solid core: agents (DIRECT/STANDARD/AUTONOMOUS), planning, tools, prompts, and OpenAI integration. Before first release, fix packaging, deprecations, docs, and a few code gaps.

---

## 1. CRITICAL – Must Fix Before Release

### 1.1 Dependency & Packaging

| Item | Current | Action |
|------|---------|--------|
| **pyproject.toml dependencies** | Only `requests`, `flask` | Add: `pydantic>=2.0`, `openai`, `tiktoken`, `python-dotenv`, `PyYAML` (from requirements.txt) |
| **setup.py vs pyproject.toml** | Both exist, different `install_requires` | Use pyproject.toml as source of truth; align setup.py or deprecate |
| **Flask** | In deps but unused in `src/` | Remove from core deps or document planned use |
| **Python version** | pyproject: 3.6+, pyright: 3.12, CONTRIBUTING: 3.8+ | Standardize: recommend 3.10+ for first release |

### 1.2 Pydantic Deprecations

| Location | Issue | Fix |
|----------|-------|-----|
| `agent.py:903, 966` | `self.metrics.dict()` | Replace with `self.metrics.model_dump()` |
| `prompts/meta_prompt.py:101` | `@model_validator(mode='after')` on classmethod | Migrate to instance method per Pydantic V2.12 |

### 1.3 Documentation Gaps

| Item | Status | Action |
|------|--------|--------|
| **docs/** | Empty | Add minimal docs: installation, quickstart, API overview |
| **ROADMAP.md** | Referenced in README, missing | Create or remove reference |
| **CHANGELOG.md** | Not present | Add CHANGELOG.md for v0.1.0 |
| **README "Current Status"** | Lists "Coming Soon" | Update to match actual implementation |

---

## 2. HIGH – Should Fix Before Release

### 2.1 Code Quality

| Item | Details |
|------|---------|
| **Linter/type consistency** | Run `pyright` / `mypy` and fix critical issues |
| **Test coverage** | 194+ tests; run `pytest-cov` and document coverage |
| **Unused imports** | Audit and remove |

### 2.2 Known Limitations (Document, Don’t Block)

| Area | Status | Document in README |
|------|--------|--------------------|
| **Memory** | Interface only; no implementation | State "Memory: interface defined, implementation planned" |
| **Structured Output TOOL/PROMPT modes** | Not implemented | Already documented in `OutputMode` |
| **Ollama, Groq, Gemini, Chroma, Pinecone** | Stub packages only | List as "Planned providers" |

### 2.3 Package Metadata

| Item | Action |
|------|--------|
| **pyproject authors** | Replace "Your Name" with real author/org |
| **Version** | Confirm 0.1.0 for first release |

---

## 3. MEDIUM – Nice to Have

### 3.1 Documentation

- [ ] API reference (e.g. Sphinx or mkdocs)
- [ ] Migration guide if there were breaking changes
- [ ] `docs/TOOL_DESIGN.md` (referenced in CONTRIBUTING)

### 3.2 Developer Experience

- [ ] `pip install nucleusiq` works from PyPI (or test PyPI)
- [ ] `python -c "from nucleusiq import Agent; print(Agent)"` succeeds
- [ ] CI: run tests on push/PR (GitHub Actions)

### 3.3 Examples

- [ ] One minimal "Hello World" example in README
- [ ] Ensure `src/examples/` examples run with `pip install -e .`

---

## 4. POST-RELEASE – Future Work

- [ ] CI: GitHub Actions to run tests on push/PR
- [ ] Add `github.vscode-github-actions` to `.vscode/extensions.json` after CI setup
- Memory implementation (e.g. in-memory, SQLite)
- Ollama, Groq, Gemini LLM providers
- Chroma, Pinecone vector DB integrations
- Structured Output TOOL and PROMPT modes
- Multi-agent orchestration
- Observability dashboard

---

## 5. Implementation Checklist (Copy-Paste)

```
[x] 1. Update pyproject.toml dependencies (pydantic, openai, tiktoken, python-dotenv, PyYAML)
[x] 2. Replace metrics.dict() with metrics.model_dump() in agent.py
[x] 3. Fix meta_prompt.py @model_validator deprecation
[x] 4. Create docs/ with: INSTALL.md, QUICKSTART.md (or equivalent)
[x] 5. Create ROADMAP.md or remove README reference
[x] 6. Create CHANGELOG.md with v0.1.0 section
[x] 7. Update README "Current Status" and "Coming Soon"
[x] 8. Remove or justify Flask in dependencies
[x] 9. Standardize Python version (3.10+ recommended)
[x] 10. Update pyproject authors
[x] 11. Verify: pip install -e . && pytest tests/ -q
[ ] 12. Publish to PyPI (or Test PyPI) and validate install
```

---

## 6. What’s Already Solid (No Action)

- **Agents**: Agent, ReActAgent, execution modes (DIRECT, STANDARD, AUTONOMOUS)
- **Planning**: LLM-based planning, basic fallback, `$step_N` resolution
- **Tools**: BaseTool, Executor, OpenAI integration
- **Prompts**: 7+ techniques (Zero-shot, Few-shot, CoT, Auto-CoT, RAG, Meta-prompting, Composer)
- **Structured Output**: NATIVE mode with OpenAI `response_format`
- **Tests**: 194+ tests across agents, tasks, plans, executor, prompts
- **Examples**: Multiple agent and OpenAI examples in `src/examples/`

---

*Generated from codebase audit. Update this file as items are completed.*
