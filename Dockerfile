FROM python:3.11.14-slim AS base
#base minimal common stage

COPY --from=ghcr.io/astral-sh/uv:0.9.5 /uv /uvx /bin/
ENV UV_LINK_MODE=copy \
    UV_COMPILE_BYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /src

FROM base AS builder

COPY pyproject.toml uv.lock .

RUN uv sync --frozen --no-cache --no-dev --no-install-project

COPY app/ ./app/

RUN uv pip install -e . --no-cache

FROM builder AS server

CMD ["uv", "run", "--no-sync", "python", "-m", "app.api.main"]

FROM builder AS client

CMD ["uv", "run", "--no-sync", "streamlit", "run", "app/client/main.py", "--server.address", "0.0.0.0"]