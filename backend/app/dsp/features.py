"""Audio preprocessing, feature extraction, and DSP signal metrics."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Tuple

import librosa
import numpy as np
import soundfile as sf

from backend.app.core.config import FIXED_DURATION_SECONDS, HOP_LENGTH, N_FFT, N_MELS, N_MFCC, SAMPLE_RATE


def load_and_preprocess_audio(
    source: str | Path | Any,
    target_sr: int = SAMPLE_RATE,
    trim_silence: bool = True
) -> Tuple[np.ndarray, int]:
    """
    Load an audio file, resample to target_sr, convert to mono, 
    optionally trim leading/trailing silence, and peak normalize.
    """
    audio, sr = librosa.load(source, sr=target_sr, mono=True)
    if trim_silence and audio.size > 0:
        audio, _ = librosa.effects.trim(audio, top_db=25)
    
    # Peak normalization to [-1.0, 1.0]
    peak = float(np.max(np.abs(audio))) if audio.size > 0 else 0.0
    if peak > 0:
        audio = audio / peak
        
    return audio.astype(np.float32), sr


def get_fixed_length_audio(
    audio: np.ndarray,
    target_sr: int = SAMPLE_RATE,
    seconds: float = FIXED_DURATION_SECONDS
) -> np.ndarray:
    """Pad or truncate audio to an exact fixed duration for consistent ML tensor shapes."""
    target_len = int(target_sr * seconds)
    if len(audio) >= target_len:
        return audio[:target_len]
    return np.pad(audio, (0, target_len - len(audio)))


def compute_mel_spectrogram(
    audio: np.ndarray,
    sr: int = SAMPLE_RATE,
    fixed_duration: bool = True
) -> np.ndarray:
    """Compute 128-band log Mel-spectrogram in decibels."""
    if fixed_duration:
        audio = get_fixed_length_audio(audio, sr)
    mel = librosa.feature.melspectrogram(
        y=audio, sr=sr, n_fft=N_FFT, hop_length=HOP_LENGTH, n_mels=N_MELS
    )
    return librosa.power_to_db(mel, ref=np.max).astype(np.float32)


def extract_dsp_features(audio: np.ndarray, sr: int = SAMPLE_RATE) -> Dict[str, Any]:
    """
    Extract comprehensive DSP parameters from speech signal:
    - Pitch / F0 via pYIN
    - MFCCs (13 coefficients)
    - Spectral Centroid (brightness)
    - Spectral Rolloff (85% energy frequency)
    - Zero Crossing Rate
    - Root Mean Square (RMS) energy
    - Harmonic vs Percussive ratio
    """
    duration = float(len(audio) / sr)
    rms = float(np.sqrt(np.mean(audio**2))) if len(audio) > 0 else 0.0
    peak_amp = float(np.max(np.abs(audio))) if len(audio) > 0 else 0.0
    
    # 1. Pitch / Fundamental Frequency (F0) tracking via pYIN
    f0, voiced_flag, voiced_probs = librosa.pyin(
        audio,
        fmin=librosa.note_to_hz("C2"),  # ~65 Hz
        fmax=librosa.note_to_hz("C7"),  # ~2093 Hz
        sr=sr,
        hop_length=HOP_LENGTH
    )
    voiced_f0 = f0[voiced_flag] if voiced_flag is not None else np.array([])
    mean_pitch = float(np.mean(voiced_f0)) if len(voiced_f0) > 0 else 0.0
    min_pitch = float(np.min(voiced_f0)) if len(voiced_f0) > 0 else 0.0
    max_pitch = float(np.max(voiced_f0)) if len(voiced_f0) > 0 else 0.0
    voiced_ratio = float(len(voiced_f0) / len(f0)) if len(f0) > 0 else 0.0

    # 2. MFCCs (13 coefficients)
    mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=N_MFCC, hop_length=HOP_LENGTH)
    mfcc_means = [float(v) for v in np.mean(mfccs, axis=1)]
    mfcc_stds = [float(v) for v in np.std(mfccs, axis=1)]

    # 3. Spectral Centroid (Center of mass of spectrum - timbral brightness)
    spectral_centroids = librosa.feature.spectral_centroid(y=audio, sr=sr, hop_length=HOP_LENGTH)[0]
    mean_centroid = float(np.mean(spectral_centroids)) if len(spectral_centroids) > 0 else 0.0

    # 4. Spectral Rolloff (Frequency below which 85% of spectral energy lies)
    spectral_rolloff = librosa.feature.spectral_rolloff(y=audio, sr=sr, roll_percent=0.85, hop_length=HOP_LENGTH)[0]
    mean_rolloff = float(np.mean(spectral_rolloff)) if len(spectral_rolloff) > 0 else 0.0

    # 5. Zero Crossing Rate (Measures noisiness / unvoiced phonemes like 's', 'sh')
    zcr = librosa.feature.zero_crossing_rate(audio, hop_length=HOP_LENGTH)[0]
    mean_zcr = float(np.mean(zcr)) if len(zcr) > 0 else 0.0

    # 6. Harmonic-Percussive Source Separation
    harmonic, percussive = librosa.effects.hpss(audio)
    harm_energy = float(np.sum(harmonic**2))
    perc_energy = float(np.sum(percussive**2))
    hnr_ratio = float(harm_energy / (perc_energy + 1e-9))

    return {
        "duration_seconds": round(duration, 2),
        "sample_rate": sr,
        "rms_energy": round(rms, 4),
        "peak_amplitude": round(peak_amp, 4),
        "pitch": {
            "mean_f0_hz": round(mean_pitch, 1),
            "min_f0_hz": round(min_pitch, 1),
            "max_f0_hz": round(max_pitch, 1),
            "voiced_percentage": round(voiced_ratio * 100, 1),
        },
        "timbre": {
            "spectral_centroid_hz": round(mean_centroid, 1),
            "spectral_rolloff_hz": round(mean_rolloff, 1),
            "zero_crossing_rate": round(mean_zcr, 4),
            "harmonic_ratio": round(hnr_ratio, 2)
        },
        "mfcc_summary": {
            "n_coefficients": N_MFCC,
            "means": [round(x, 2) for x in mfcc_means],
            "stds": [round(x, 2) for x in mfcc_stds]
        }
    }


def get_feature_explanations() -> Dict[str, Dict[str, str]]:
    """Return scientific and practical explanations for all extracted DSP features."""
    return {
        "waveform": {
            "name": "Time-Domain Waveform",
            "description": "Represents instantaneous acoustic air pressure fluctuations over time. Shows amplitude dynamics, speech phrasing, silence pauses, and overall energy envelope.",
            "dsp_significance": "Essential for inspecting clipping distortion, speech cadence, and envelope contours."
        },
        "mel_spectrogram": {
            "name": "Mel-Scale Spectrogram",
            "description": "Decomposes the audio signal into time-frequency energy representations mapped to the psychoacoustic Mel scale, reflecting human non-linear pitch perception.",
            "dsp_significance": "The primary input representation used by modern neural vocoders and deepfake detection CNNs."
        },
        "mfcc": {
            "name": "Mel-Frequency Cepstral Coefficients (MFCC)",
            "description": "Compact representation of the short-term spectral envelope, computed by taking the Discrete Cosine Transform (DCT) of log mel energies.",
            "dsp_significance": "Captures vocal tract resonance shapes (formants) while discarding glottal excitation, making it the golden standard for speaker identification."
        },
        "pitch_f0": {
            "name": "Fundamental Frequency (Pitch F0)",
            "description": "The rate of vocal cord vibration during voiced speech segments, estimated using the probabilistic YIN (pYIN) algorithm.",
            "dsp_significance": "Reveals speaker intonation, prosody, and emotional inflection. Synthetic speech often lacks micro-jitter naturalness."
        },
        "spectral_centroid": {
            "name": "Spectral Centroid & Rolloff",
            "description": "The spectral centroid corresponds to the brightness or sharpness of the sound. Spectral rolloff marks the frequency cutoff containing 85% of total signal energy.",
            "dsp_significance": "Human vocal tracts have natural high-frequency roll-offs, whereas synthetic vocoders frequently exhibit telltale high-frequency suppression or artifacts."
        },
        "zero_crossing_rate": {
            "name": "Zero Crossing Rate & HPSS",
            "description": "Tracks how frequently the signal changes sign, differentiating unvoiced fricatives from voiced vowels, alongside Harmonic-Percussive source decomposition.",
            "dsp_significance": "Crucial for analyzing vocal stability and background noise consistency."
        }
    }


def save_audio_file(path: str | Path, audio: np.ndarray, sr: int = SAMPLE_RATE) -> None:
    """Save normalized audio as 16-bit PCM WAV."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), audio, sr, subtype="PCM_16")
