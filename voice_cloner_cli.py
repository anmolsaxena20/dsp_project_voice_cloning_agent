"""Consent-gated interactive zero-shot voice cloning with Coqui XTTS-v2.

Run this in a Python 3.10/3.11 environment where `TTS` is installed:
    python voice_cloner_cli.py --reference path/to/my_authorised_voice.wav
"""
from __future__ import annotations

import argparse
import re
import sys
import time
from pathlib import Path


def get_tts():
    try:
        from TTS.api import TTS
    except ImportError as exc:
        raise RuntimeError(
            "Voice cloning requires Coqui TTS/XTTS-v2, which is not compatible with this project's Python 3.13 runtime. "
            "Install Python 3.11, create a separate virtual environment, then install `TTS` and run this command there."
        ) from exc
    return TTS


def main() -> int:
    parser = argparse.ArgumentParser(description="Interactive, authorised voice-cloning TTS")
    parser.add_argument("--reference", type=Path, required=True, help="Authorised 6–30 second clean WAV reference")
    parser.add_argument("--language", default="en", help="XTTS language code (default: en)")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/cloned"))
    args = parser.parse_args()
    # PowerShell copy/paste can insert a newline or spaces inside a long quoted
    # filename. Try the original path first, then a compacted form.
    if not args.reference.is_file():
        reference_text = re.sub(r"\s+", " ", str(args.reference)).strip()
        # Repair a wrapped ISO date such as "2026-09-\n 08".
        reference_text = re.sub(r"(\d{4}-\d{2}-)\s+(\d{2})", r"\1\2", reference_text)
        compact_reference = Path(reference_text)
        if compact_reference.is_file():
            args.reference = compact_reference
    if not args.reference.is_file():
        parser.error(f"Reference file not found: {args.reference}")
    TTS = get_tts()
    use_gpu = False
    try:
        import torch
        use_gpu = torch.cuda.is_available()
    except ImportError:
        pass
    print("Loading XTTS-v2 and extracting the reference speaker embedding (first run downloads model files)...")
    tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2", progress_bar=True).to("cuda" if use_gpu else "cpu")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    print("Ready. Enter text to synthesize. Type /quit to exit.")
    while True:
        text = input("text> ").strip()
        if text.lower() in {"/quit", "/exit", "quit", "exit"}:
            break
        if not text:
            continue
        destination = args.output_dir / f"cloned_{int(time.time())}.wav"
        tts.tts_to_file(text=text, speaker_wav=str(args.reference), language=args.language, file_path=str(destination))
        print(f"Generated synthetic audio: {destination}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except RuntimeError as error:
        print(f"Error: {error}", file=sys.stderr)
        raise SystemExit(2)
