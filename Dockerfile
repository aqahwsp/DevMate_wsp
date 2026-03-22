FROM python:3.13-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_LINK_MODE=copy \
    UV_PROJECT_ENVIRONMENT=/app/.venv

RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential curl \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir uv

WORKDIR /app

COPY pyproject.toml uv.lock* README.md config.toml AGENTS.md requirements.md checklist.md ./
COPY docs ./docs
COPY src ./src
COPY tests ./tests
COPY .skills ./.skills
COPY workspace ./workspace
COPY data ./data

RUN uv sync --frozen --no-dev -v

ENV PATH="/app/.venv/bin:${PATH}" \
    DEVMATE_CONFIG_PATH=/app/config.toml

EXPOSE 8080 8001

CMD ["devmate", "web", "--host", "0.0.0.0", "--port", "8080"]
