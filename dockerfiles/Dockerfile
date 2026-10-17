
ARG PYTHON_VERSION=3.14
FROM python:${PYTHON_VERSION}-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# Install only the API runtime dependencies.
COPY requirements-api.txt ./requirements-api.txt

RUN python -m pip install \
    --no-cache-dir \
    -r requirements-api.txt

# Match the UID/GID specified by Kubernetes.
RUN groupadd --gid 10001 appuser \
    && useradd \
       --uid 10001 \
       --gid 10001 \
       --no-create-home \
       --shell /usr/sbin/nologin \
       appuser

# Copy the application and trained model.
COPY --chown=10001:10001 src/ ./src/
COPY --chown=10001:10001 models/ ./models/

USER 10001:10001

EXPOSE 8000

HEALTHCHECK --interval=30s \
            --timeout=5s \
            --start-period=15s \
            --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=3)" || exit 1

CMD ["uvicorn", "src.app:app", "--host", "0.0.0.0", "--port", "8000"]
