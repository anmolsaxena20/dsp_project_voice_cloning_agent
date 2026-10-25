"""API routes for audio upload, DSP feature extraction, and playback streaming."""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict

from fastapi import APIRouter, File, HTTPException, UploadFile
from fastapi.responses import FileResponse

from backend.app.core.config import ANALYSIS_DIR, BASE_DIR, CLONED_DIR, SAMPLE_RATE, UPLOADS_DIR
from backend.app.dsp.features import (
    extract_dsp_features,
    get_feature_explanations,
    load_and_preprocess_audio,
    save_audio_file,
)
from backend.app.dsp.visualizer import generate_all_analysis_plots

router = APIRouter(prefix="/api/audio", tags=["Audio & DSP Analysis"])


def _process_and_save_audio(file_path: Path, audio_id: str, original_filename: str) -> Dict[str, Any]:
    """Process audio file, extract DSP features, generate plots, and cache results."""
    audio, sr = load_and_preprocess_audio(file_path, target_sr=SAMPLE_RATE)
    
    # Save standardized 16kHz WAV copy
    wav_path = UPLOADS_DIR / f"{audio_id}.wav"
    save_audio_file(wav_path, audio, sr=sr)

    # Extract DSP features
    features = extract_dsp_features(audio, sr=sr)
    
    # Generate visualization plots
    plots = generate_all_analysis_plots(audio, sr=sr)

    metadata = {
        "audio_id": audio_id,
        "original_filename": original_filename,
        "wav_filename": f"{audio_id}.wav",
        "audio_url": f"/api/audio/stream/uploads/{audio_id}.wav",
        "features": features,
        "plots": plots,
        "explanations": get_feature_explanations(),
        "created_at": int(time.time())
    }

    # Cache analysis JSON
    analysis_cache_file = ANALYSIS_DIR / f"{audio_id}.json"
    with open(analysis_cache_file, "w", encoding="utf-8") as f:
        json.dump(metadata, f)

    return metadata


@router.post("/upload")
async def upload_audio(file: UploadFile = File(...)):
    """Upload a user voice recording and immediately extract all DSP features and plots."""
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file selected.")

    extension = Path(file.filename).suffix.lower()
    if extension not in [".wav", ".mp3", ".ogg", ".flac", ".m4a", ".aac"]:
        raise HTTPException(status_code=400, detail=f"Unsupported audio format: {extension}")

    audio_id = f"voice_{int(time.time() * 1000)}"
    temp_path = UPLOADS_DIR / f"raw_{audio_id}{extension}"

    try:
        content = await file.read()
        with open(temp_path, "wb") as f:
            f.write(content)

        metadata = _process_and_save_audio(temp_path, audio_id, file.filename)
        return metadata
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Audio processing failed: {str(exc)}")
    finally:
        if temp_path.exists():
            try:
                temp_path.unlink()
            except OSError:
                pass


@router.get("/sample")
async def load_bundled_sample():
    from backend.app.core.config import SAMPLE_AUDIO_DIR
    sample_file = SAMPLE_AUDIO_DIR / "reference_voice.ogg"
    if not sample_file.exists():
        raise HTTPException(status_code=404, detail="Sample reference audio not found.")

    audio_id = "sample_reference_voice"
    analysis_cache = ANALYSIS_DIR / f"{audio_id}.json"
    
    if analysis_cache.exists():
        try:
            with open(analysis_cache, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass

    metadata = _process_and_save_audio(sample_file, audio_id, "Sample_Reference_Voice.ogg")
    return metadata


@router.get("/analysis/{audio_id}")
async def get_audio_analysis(audio_id: str):
    """Retrieve cached analysis, plots, and feature explanations for an audio file."""
    cache_file = ANALYSIS_DIR / f"{audio_id}.json"
    if not cache_file.exists():
        raise HTTPException(status_code=404, detail=f"Analysis not found for ID: {audio_id}")
    
    with open(cache_file, "r", encoding="utf-8") as f:
        return json.load(f)


@router.get("/stream/{folder}/{filename}")
async def stream_audio_file(folder: str, filename: str):
    """Stream audio files with proper headers for web audio playback."""
    if folder == "uploads":
        target = UPLOADS_DIR / filename
    elif folder == "cloned":
        target = CLONED_DIR / filename
    else:
        raise HTTPException(status_code=400, detail="Invalid folder specified.")

    if not target.exists():
        raise HTTPException(status_code=404, detail="Audio file not found.")

    media_type = "audio/wav"
    if target.suffix.lower() == ".ogg":
        media_type = "audio/ogg"
    elif target.suffix.lower() == ".mp3":
        media_type = "audio/mpeg"

    return FileResponse(path=str(target), media_type=media_type, filename=filename)
