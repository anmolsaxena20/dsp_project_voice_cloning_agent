"""API routes for AI Voice Deepfake Detection and Forensic Security Analysis."""
from __future__ import annotations

import time
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from pydantic import BaseModel

from backend.app.core.config import CLONED_DIR, UPLOADS_DIR
from backend.app.dsp.features import load_and_preprocess_audio
from backend.app.ml.cnn_detector import detect_voice_authenticity
from backend.app.ml.voice_cloner import cloner_engine

router = APIRouter(prefix="/api/security", tags=["AI Security & Fake Detection"])


class CompareRequest(BaseModel):
    reference_id: str
    target_id: str


@router.post("/detect")
async def detect_deepfake(
    file: Optional[UploadFile] = File(None),
    audio_id: Optional[str] = Form(None)
):
    """
    Classify an audio recording as Authentic Human Voice vs AI-Generated Voice.
    Analyzes Mel-Spectrogram features via CNN and inspects acoustic vocoder anomalies.
    Can be run via direct file upload OR by providing an existing audio_id.
    """
    target_path = None
    temp_path = None

    try:
        if file is not None and file.filename:
            extension = Path(file.filename).suffix.lower()
            temp_path = UPLOADS_DIR / f"detect_{int(time.time() * 1000)}{extension}"
            content = await file.read()
            with open(temp_path, "wb") as f:
                f.write(content)
            target_path = temp_path
            source_name = file.filename
        elif audio_id:
            # Check uploads first, then cloned
            candidate_upload = UPLOADS_DIR / f"{audio_id}.wav"
            candidate_cloned = CLONED_DIR / f"{audio_id}"
            if not candidate_cloned.exists():
                candidate_cloned = CLONED_DIR / f"{audio_id}.wav"

            if candidate_upload.exists():
                target_path = candidate_upload
                source_name = f"Uploaded Voice ({audio_id})"
            elif candidate_cloned.exists():
                target_path = candidate_cloned
                source_name = f"Cloned Audio ({audio_id})"
            else:
                raise HTTPException(status_code=404, detail=f"Audio ID '{audio_id}' not found.")
        else:
            raise HTTPException(status_code=400, detail="Must provide an audio file upload or a valid audio_id.")

        # Run authenticity detector
        report = detect_voice_authenticity(target_path)
        report["source_name"] = source_name
        return report

    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Deepfake detection analysis failed: {str(exc)}")
    finally:
        if temp_path and temp_path.exists():
            try:
                temp_path.unlink()
            except OSError:
                pass


@router.post("/compare")
async def compare_voices(request: CompareRequest):
    """
    Compare two voice samples (e.g. Original Reference Voice vs Cloned Voice).
    Returns acoustic similarity distance and individual metric comparison.
    """
    ref_file = UPLOADS_DIR / f"{request.reference_id}.wav"
    target_file = CLONED_DIR / request.target_id
    if not target_file.exists():
        target_file = CLONED_DIR / f"{request.target_id}.wav"
    if not target_file.exists():
        target_file = UPLOADS_DIR / f"{request.target_id}.wav"

    if not ref_file.exists():
        raise HTTPException(status_code=404, detail=f"Reference audio '{request.reference_id}' not found.")
    if not target_file.exists():
        raise HTTPException(status_code=404, detail=f"Target audio '{request.target_id}' not found.")

    try:
        similarity = cloner_engine.compute_voice_similarity(ref_file, target_file)
        ref_profile = cloner_engine.extract_speaker_profile(ref_file)
        target_profile = cloner_engine.extract_speaker_profile(target_file)

        return {
            "similarity_score_percentage": similarity,
            "reference_profile": ref_profile,
            "target_profile": target_profile,
            "pitch_difference_hz": round(abs(ref_profile["mean_f0_hz"] - target_profile["mean_f0_hz"]), 1),
            "spectral_centroid_diff_hz": round(abs(ref_profile["spectral_centroid_hz"] - target_profile["spectral_centroid_hz"]), 1)
        }
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Comparison failed: {str(exc)}")
