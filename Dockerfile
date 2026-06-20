# PeatLearn Web API — production container for the public website backend.
# Serves the 9.64/10 adaptive RAG pipeline (app/web_api.py) as a JSON API.
#
# Build:  docker build -t peatlearn-api .
# Run:    docker run --rm -p 8080:8080 --env-file .env peatlearn-api

FROM python:3.12-slim AS base

# - PYTHONUNBUFFERED: stream logs immediately (no buffering)
# - PIP_NO_CACHE_DIR: smaller image
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PORT=8080

WORKDIR /app

# Install slim dependencies first (better layer caching).
COPY requirements-api.txt ./
RUN pip install --upgrade pip && pip install -r requirements-api.txt

# Copy only what the serve path needs (no data/, no venv/, no tests/).
COPY app/ ./app/
COPY peatlearn/ ./peatlearn/
COPY config/ ./config/

# Run as a non-root user.
RUN useradd --create-home --uid 10001 appuser
USER appuser

EXPOSE 8080

# Single worker keeps the in-process rate limiter / singletons coherent.
# Scale horizontally by running more containers behind the host's load balancer.
CMD ["sh", "-c", "uvicorn app.web_api:app --host 0.0.0.0 --port ${PORT}"]
