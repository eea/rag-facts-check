# AGENTS.md — Instructions for AI coding sessions

## Greenfield project — no backward compatibility

This is a greenfield project with no external consumers yet. **Do not preserve
backward compatibility.** Feel free to:

- Break public APIs, rename functions, change signatures
- Refactor data models, drop fields, change serialization formats
- Rewrite tests from scratch if the design changes
- Drop unused code without deprecation warnings
- Change prompt formats, output schemas, response structures

The only "users" are the internal test suite and the FastAPI server endpoints.
If tests break, fix them. If the server contract changes, update the client.

## Requirements interviews

Before implementing anything underspecified, follow this protocol:

1. **Explore the codebase first** — if a question can be answered by reading
   existing code, docs, or config, do that instead of asking.
2. **One question at a time** — never stack multiple clarifications. Wait for
   the user's answer before moving on.
3. **Always recommend** — never present a bare question. Propose your
   recommended answer with reasoning.
4. **Be concise** — questions should be short. Context should be brief.
5. **Track decisions** — note agreed decisions so they can be referenced later.
6. **Respect the user's direction** — if they give a clear answer, accept it
   and move on. Do not re-litigate settled decisions.

Skip this protocol when instructions are already specific and actionable,
the task is trivial, or the user says "just do it".

## Project conventions

### pyproject.toml is the single source of truth

All project configuration lives in `pyproject.toml`:

- **Project metadata**: `[project]`, `[project.optional-dependencies]`
- **Ruff config**: `[tool.ruff]`, `[tool.ruff.lint]`, `[tool.ruff.lint.per-file-ignores]`
- **Pytest config**: `[tool.pytest.ini_options]`

No `pytest.ini`, no `setup.py`, no `.ruff.toml`, no `setup.cfg`.

### Ruff per-file ignores

Prompt strings and mock responses contain long literals that can't be
broken. Use per-file ignores instead of `# noqa`:

```toml
[tool.ruff.lint.per-file-ignores]
"rag_facts_check/prompts.py" = ["E501"]
"tests/mocks/*.py" = ["E501"]
```

### Optional dependency groups

- `[test]` — pytest, pytest-cov
- `[dev]` — ruff
- `[server]` — fastapi, uvicorn, httpx, python-dotenv, atomic-agents, instructor

Install with `pip install -e ".[test,dev,server]"` or `make setup-server`.

### Markers

- `@pytest.mark.llm` — tests that require a live LLM. Skipped by default
  (`addopts = ["-m", "not llm"]`). Run with `pytest -m llm`.
