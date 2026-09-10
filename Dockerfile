FROM python:3.14-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# ---------------------------------------------------------------------------
# test — lint + full test suite. Not the default build target (the runtime
# stage below is last), so `docker build .` is unaffected. CI builds this
# with `--target test`.
# ---------------------------------------------------------------------------
FROM base AS test

# Source + tests first, then a single editable install. (setuptools'
# package discovery needs rag_facts_check/ present at install time.)
COPY pyproject.toml ./
COPY rag_facts_check/ ./rag_facts_check/
COPY prompts/ ./prompts/
COPY scripts/ ./scripts/
COPY tests/ ./tests/
COPY mock_datasets/ ./mock_datasets/
RUN pip install -e ".[test,dev,server]"

# llm-marked tests are excluded via pyproject addopts.
CMD ["sh", "-c", "ruff check rag_facts_check/ tests/ scripts/; ruff format --check rag_facts_check/ tests/ scripts/; pytest --junitxml=/app/junit.xml"]

# ---------------------------------------------------------------------------
# runtime — minimal production image (default build target).
# ---------------------------------------------------------------------------
FROM base AS runtime

# Install dependencies first (layer caching)
COPY pyproject.toml ./
RUN pip install --no-deps -e ".[server]"

# Copy application code
COPY rag_facts_check/ ./rag_facts_check/
# prompts.py loads these at import time (Path(__file__).parent.parent / "prompts").
COPY prompts/ ./prompts/

# Run as non-root user
RUN useradd --create-home appuser && chown -R appuser:appuser /app
USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=5s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

CMD ["uvicorn", "rag_facts_check.server:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "4"]
