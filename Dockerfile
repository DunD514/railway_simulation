# Serves the app for any scenario: the precomputed default results (results/), plus live simulation, evaluation
# and training for user scenarios, run as background jobs (api/jobs.py) into results/scenarios/. Set RAIL_WORKERS
# to the number of CPU cores the service really has.

# ---- web build
FROM node:22-alpine AS web
WORKDIR /web
COPY web/package.json web/package-lock.json ./
RUN npm ci
COPY web/ ./
RUN npm run build

# ---- runtime
FROM python:3.11-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
WORKDIR /app
COPY requirements-serve.txt ./
RUN pip install --no-cache-dir -r requirements-serve.txt
COPY config.yaml ./
COPY sim/ sim/
COPY agents/ agents/
COPY experiments/ experiments/
COPY api/ api/
COPY results/ results/
COPY --from=web /web/dist web/dist
RUN useradd --create-home app && chown -R app /app
USER app
# Render injects PORT; default for local `docker run`.
ENV PORT=8000
EXPOSE 8000
CMD ["sh", "-c", "exec uvicorn api.main:app --host 0.0.0.0 --port ${PORT} --proxy-headers --forwarded-allow-ips='*'"]
