FROM ghcr.io/meta-pytorch/openenv-base:latest

WORKDIR /app

COPY . /app/env
WORKDIR /app/env

RUN pip install --no-cache-dir openenv-core numpy pydantic

ENV PYTHONPATH=/app/env

HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

CMD ["python", "-m", "server.app"]