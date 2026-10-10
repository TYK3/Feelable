# Feelable — one container: builds the web app, then serves web + API with FastAPI.
# Works on Hugging Face Spaces (Docker), Render, Railway, Fly.io. Listens on $PORT (default 7860).

FROM node:22-slim AS web
WORKDIR /src/web
COPY web/package.json web/package-lock.json ./
RUN npm ci --no-audit --no-fund
COPY web/ ./
RUN npm run build

FROM python:3.12-slim
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY backend/requirements.txt backend/requirements.txt
RUN pip install --no-cache-dir -r backend/requirements.txt
COPY backend/ backend/
COPY samples/ samples/
COPY --from=web /src/web/dist web/dist
# warm the OCR model so the first request is fast
RUN cd backend && python -c "from rapidocr_onnxruntime import RapidOCR; RapidOCR()"
# pre-make every sample so clicking one is instant even on a slow free-tier CPU
RUN cd backend && FEELABLE_DATA=/tmp/build-jobs FEELABLE_SAMPLE_CACHE=/app/sample_cache python precompute_samples.py
ENV FEELABLE_DATA=/tmp/feelable-jobs PORT=7860 PYTHONUNBUFFERED=1 \
    FEELABLE_ISOLATE=1 FEELABLE_LAZY_STL=1 FEELABLE_THREADS=1 OMP_NUM_THREADS=1 MALLOC_ARENA_MAX=2 FEELABLE_SAMPLE_CACHE=/app/sample_cache
RUN useradd -m -u 1000 user && chown -R user /app
USER user
EXPOSE 7860
WORKDIR /app/backend
CMD ["sh", "-c", "python -m uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-7860}"]
