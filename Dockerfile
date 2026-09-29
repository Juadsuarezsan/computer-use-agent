# API image: FastAPI + LangGraph loop. The sandbox (browser / X desktop) lives in
# docker/sandbox.Dockerfile; this image talks to it or runs the Playwright
# sandbox in-process when VM_MODE=playwright.
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PLAYWRIGHT_BROWSERS_PATH=/opt/pw-browsers \
    CHROMIUM_PATH=/opt/pw-browsers/chromium

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential libpq-dev curl \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml README.md /app/
COPY src /app/src
RUN pip install --upgrade pip && pip install -e . \
    && python -m playwright install --with-deps chromium \
    && ln -s "$(ls -d /opt/pw-browsers/chromium-*/chrome-linux/chrome | head -n1)" /opt/pw-browsers/chromium

COPY . /app

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s CMD curl -f http://localhost:8000/health || exit 1
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
