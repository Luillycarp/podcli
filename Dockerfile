# podcli — Hugging Face Spaces Docker image
FROM python:3.11-slim

# System deps: ffmpeg, Node.js 20, build tools
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    curl \
    ca-certificates \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && curl -fsSL https://deb.nodesource.com/setup_20.x | bash - \
    && apt-get install -y nodejs \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Python deps first (better layer caching)
COPY backend/requirements.txt backend/requirements.txt
RUN pip install --no-cache-dir \
    -r backend/requirements.txt \
    fastapi \
    uvicorn[standard] \
    python-multipart \
    yt-dlp \
    youtube-transcript-api

# Node deps + build TypeScript
COPY package.json package-lock.json ./
RUN npm ci --omit=dev

COPY . .
RUN npm run build 2>/dev/null || true  # non-fatal: API works without the web UI build

# HF Spaces: port 7860, non-root user
RUN useradd -m -u 1000 appuser \
    && mkdir -p /tmp/podcli_output /app/.podcli \
    && chown -R appuser:appuser /app /tmp/podcli_output
USER appuser

EXPOSE 7860

CMD ["python", "app.py"]
