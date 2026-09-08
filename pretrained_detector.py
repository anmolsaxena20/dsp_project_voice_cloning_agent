"""Inference wrapper for an off-the-shelf audio deepfake detector."""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import numpy as np

MODEL_ID = "Shanmugapriya6/voice-fake-detector-v1"


@lru_cache(maxsize=1)
def _classifier():
    from transformers import pipeline
    return pipeline("audio-classification", model=MODEL_ID)


def classify_audio(path: str | Path) -> dict[str, float]:
    """Return normalised Human and AI-generated probabilities from a pretrained model."""
    results = _classifier()(str(path))
    scores = {"Human": 0.0, "AI-generated": 0.0}
    for result in results:
        label = str(result["label"]).lower()
        if any(token in label for token in ("fake", "spoof", "ai", "deepfake", "generated")):
            scores["AI-generated"] += float(result["score"])
        else:
            scores["Human"] += float(result["score"])
    total = sum(scores.values())
    if total <= 0:
        raise RuntimeError(f"The model returned unrecognised labels: {results}")
    return {name: value / total for name, value in scores.items()}
