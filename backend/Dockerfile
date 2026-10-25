# ==========================================
# Stage 1: Build & Dependencies Builder
# ==========================================
FROM python:3.11-slim AS builder

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV COQUI_TOS_AGREED=1

# Install temporary build tools needed for C-extensions/wheels
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /install

# Copy requirements from root build context
COPY backend/requirements.txt /install/requirements.txt

# Install Python dependencies with CPU-only PyTorch and Neural TTS
RUN pip install --no-cache-dir --upgrade pip setuptools wheel && \
    pip install --no-cache-dir --prefix=/install/local \
        --extra-index-url https://download.pytorch.org/whl/cpu \
        -r /install/requirements.txt


# ==========================================
# Stage 2: Clean Minimal Final Runtime Image
# ==========================================
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV COQUI_TOS_AGREED=1
ENV PORT=8000
ENV PYTHONPATH=/app

# Install ONLY runtime shared libraries (no compilers, no gcc, no build-essential)
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libsndfile1 \
    espeak-ng \
    curl \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

WORKDIR /app

# Copy installed Python packages from builder stage
COPY --from=builder /install/local /usr/local

# Copy ONLY backend application code
COPY backend /app/backend

# Create runtime storage directories
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
