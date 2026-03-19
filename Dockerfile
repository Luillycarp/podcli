# podcli — Hugging Face Spaces Docker image
FROM python:3.11-slim

# System deps: ffmpeg, Node.js 20, build tools
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    curl \
    ca-certificates \
    libgl1 \
    libglib2.0-0 \
    && curl -fsSL https://deb.nodesource.com/setup_20.x | bash - \
    && apt-get install -y nodejs \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy everything first
COPY . .

# Python deps
RUN pip install --no-cache-dir \
    -r backend/requirements.txt \
    fastapi \
    "uvicorn[standard]" \
    python-multipart \
    yt-dlp \
    youtube-transcript-api

# Node deps + build TypeScript (non-fatal)
RUN npm ci --omit=dev && npm run build 2>/dev/null || true

# HF Spaces: port 7860, non-root user
RUN useradd -m -u 1000 appuser \
    && mkdir -p /tmp/podcli_output /app/.podcli \
    && chown -R appuser:appuser /app /tmp/podcli_output
USER appuser

EXPOSE 7860

CMD ["python", "app.py"]
