# ==========================================
# Stage 1: Build & Dependencies Builder
# ==========================================
FROM python:3.11-slim as builder

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Install temporary build tools needed for wheels compiling
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /install

# Copy requirements
COPY backend/requirements.txt /install/requirements.txt

# Install PyTorch CPU-only (~180MB vs 2.5GB CUDA) and dependencies into isolated prefix
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir --prefix=/install/local torch torchaudio --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir --prefix=/install/local -r /install/requirements.txt


# ==========================================
# Stage 2: Clean Minimal Final Runtime Image
# ==========================================
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PORT=8000
ENV PYTHONPATH=/app

# Install ONLY runtime shared libraries (no compilers, no gcc, no build-essential)
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libsndfile1 \
    espeak-ng \
    curl \
    && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

WORKDIR /app

# Copy installed Python packages from builder stage
COPY --from=builder /install/local /usr/local

# Copy ONLY backend application code
COPY backend /app/backend

# Create runtime storage directories if missing
RUN mkdir -p /app/backend/storage/uploads \
             /app/backend/storage/cloned \
             /app/backend/storage/analysis

# Expose backend service port
EXPOSE 8000

# Health check endpoint
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
  CMD curl -f http://localhost:8000/api/health || exit 1

# Launch FastAPI app with Uvicorn
CMD ["uvicorn", "backend.app.main:app", "--host", "0.0.0.0", "--port", "8000"]
