FROM python:3.13-slim-bookworm

COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv

WORKDIR /app

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    PATH="/app/.venv/bin:$PATH" \
    RL_CHECKPOINT_PATH=/app/rl/checkpoints/production_model.zip

RUN apt-get update \
    && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --extra rl --no-dev --no-install-project

COPY api ./api
COPY rl ./rl
COPY entrypoint.sh ./entrypoint.sh
RUN chmod +x ./entrypoint.sh \
    && mkdir -p /app/rl/checkpoints

EXPOSE 8000

ENTRYPOINT ["./entrypoint.sh"]
