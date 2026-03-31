# syntax=docker/dockerfile:1.7

FROM node:20-bookworm-slim AS frontend-builder
WORKDIR /build/frontend

COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci --no-audit --no-fund

COPY frontend/ ./
ARG VITE_BASE_PATH=/gaia_agent/
ENV VITE_BASE_PATH=${VITE_BASE_PATH}
RUN npm run build

FROM python:3.11-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

COPY requirements.txt ./
RUN pip install --upgrade pip && pip install -r requirements.txt

COPY . ./
COPY --from=frontend-builder /build/frontend/dist /app/frontend/dist

ENV WEB_BASE_PATH=/gaia_agent \
    WEB_STATIC_DIR=/app/frontend/dist \
    WEB_PUBLIC_MODE=true \
    AGENT_ALLOW_UNSAFE_TOOLS=false \
    WEB_CORS_ORIGINS=*

EXPOSE 8000

CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000"]
