FROM python:3.11-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    CCA_ENVIRONMENT=production \
    CCA_LOG_FORMAT=json \
    CCA_CHROMA_PERSIST_DIR=/app/data/chroma \
    CCA_ARTIFACT_DIR=/app/data/artifacts \
    CCA_WORKSPACE_SESSION_DB_PATH=/app/data/workspace_sessions.sqlite3

WORKDIR /app

RUN adduser --disabled-password --gecos "" --uid 10001 appuser

COPY pyproject.toml README.md ./
COPY src ./src

RUN python -m pip install --no-cache-dir --upgrade pip \
    && python -m pip install --no-cache-dir ".[server]" \
    && mkdir -p /app/data/chroma /app/data/artifacts \
    && chown -R appuser:appuser /app

USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/api/health/live', timeout=5).read()"

CMD ["gunicorn", "creative_coding_assistant.api.wsgi:application", "--bind", "0.0.0.0:8000", "--workers", "2", "--threads", "4", "--timeout", "120", "--access-logfile", "-", "--error-logfile", "-"]
