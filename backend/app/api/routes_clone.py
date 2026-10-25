"""API routes for voice cloning synthesis."""
from __future__ import annotations

from pathlib import Path
from typing import Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from backend.app.core.config import UPLOADS_DIR
from backend.app.ml.voice_cloner import cloner_engine

router = APIRouter(prefix="/api/clone", tags=["Voice Cloning"])


class CloneRequest(BaseModel):
    audio_id: str = Field(..., description="Audio ID of the uploaded speaker reference")
    text_prompt: str = Field(..., min_length=1, max_length=1500, description="Text prompt to synthesize")
    consent: bool = Field(..., description="Confirmation of voice ownership or consent")


@router.post("/generate")
async def generate_cloned_voice(request: CloneRequest):
    """
    Generate speech in the user's voice based on their uploaded reference audio profile.
    Requires explicit consent confirmation.
    """
    if not request.consent:
        raise HTTPException(
            status_code=400,
            detail="Consent confirmation required. You must verify ownership or authorization to clone this voice."
        )

    clean_text = request.text_prompt.strip()
    if not clean_text:
        raise HTTPException(status_code=400, detail="Text prompt cannot be empty.")

    # Locate reference audio file
    reference_file = UPLOADS_DIR / f"{request.audio_id}.wav"
    if not reference_file.exists():
        raise HTTPException(
            status_code=404,
            detail=f"Reference voice audio file for ID '{request.audio_id}' not found. Please upload your voice first."
        )

    try:
        result = cloner_engine.clone_voice(
            reference_path=reference_file,
            text=clean_text,
            consent_confirmed=request.consent
        )
        
        result["audio_url"] = f"/api/audio/stream/cloned/{result['filename']}"
        return result
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Voice cloning synthesis failed: {str(exc)}")
