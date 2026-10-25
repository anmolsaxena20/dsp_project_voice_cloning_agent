"""Configuration settings for the Synthetic Speech & DSP Lab Backend."""
from __future__ import annotations

import os
from pathlib import Path

# Paths
BASE_DIR = Path(__file__).resolve().parent.parent.parent  # backend/
STORAGE_DIR = BASE_DIR / "storage"
UPLOADS_DIR = STORAGE_DIR / "uploads"
CLONED_DIR = STORAGE_DIR / "cloned"
ANALYSIS_DIR = STORAGE_DIR / "analysis"
MODELS_DIR = BASE_DIR / "models"
DATA_DIR = BASE_DIR / "data"
SAMPLE_AUDIO_DIR = BASE_DIR / "sample_audio"
FRONTEND_DIR = BASE_DIR.parent / "frontend"

for directory in (STORAGE_DIR, UPLOADS_DIR, CLONED_DIR, ANALYSIS_DIR, MODELS_DIR, DATA_DIR, SAMPLE_AUDIO_DIR):
    directory.mkdir(parents=True, exist_ok=True)

# Audio Parameters
SAMPLE_RATE = 16_000
N_MELS = 128
N_FFT = 1024
HOP_LENGTH = 256
N_MFCC = 13
FIXED_DURATION_SECONDS = 4.0

# Server settings
HOST = os.getenv("HOST", "0.0.0.0")
PORT = int(os.getenv("PORT", "8000"))
CORS_ORIGINS = ["*"]
