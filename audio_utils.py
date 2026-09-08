"""Audio loading, preprocessing, features, and analysis plots."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf

SAMPLE_RATE = 16_000
N_MELS = 128
DURATION_SECONDS = 4.0


def load_audio(source: str | Path | Any, sample_rate: int = SAMPLE_RATE, trim: bool = True) -> tuple[np.ndarray, int]:
    """Load audio as normalised mono float32 audio at a common sample rate."""
    audio, _ = librosa.load(source, sr=sample_rate, mono=True)
    if trim and audio.size:
        audio, _ = librosa.effects.trim(audio, top_db=30)
    peak = float(np.max(np.abs(audio))) if audio.size else 0.0
    if peak > 0:
        audio = audio / peak
    return audio.astype(np.float32), sample_rate


def fixed_length(audio: np.ndarray, sample_rate: int = SAMPLE_RATE, seconds: float = DURATION_SECONDS) -> np.ndarray:
    target = int(sample_rate * seconds)
    if len(audio) >= target:
        return audio[:target]
    return np.pad(audio, (0, target - len(audio)))


def mel_spectrogram(audio: np.ndarray, sample_rate: int = SAMPLE_RATE) -> np.ndarray:
    audio = fixed_length(audio, sample_rate)
    mel = librosa.feature.melspectrogram(y=audio, sr=sample_rate, n_fft=1024, hop_length=256, n_mels=N_MELS)
    return librosa.power_to_db(mel, ref=np.max).astype(np.float32)


def analysis_figure(audio: np.ndarray, sample_rate: int = SAMPLE_RATE):
    """Return a matplotlib figure with the four analyses requested in the abstract."""
    fig, axes = plt.subplots(2, 2, figsize=(13, 8), constrained_layout=True)
    times = np.arange(len(audio)) / sample_rate
    axes[0, 0].plot(times, audio, linewidth=0.55, color="#2563eb")
    axes[0, 0].set(title="Waveform", xlabel="Time (s)", ylabel="Amplitude")

    spec = librosa.stft(audio, n_fft=1024, hop_length=256)
    librosa.display.specshow(librosa.amplitude_to_db(np.abs(spec), ref=np.max), sr=sample_rate, hop_length=256, x_axis="time", y_axis="hz", ax=axes[0, 1])
    axes[0, 1].set_title("Spectrogram")

    mfcc = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=13)
    librosa.display.specshow(mfcc, sr=sample_rate, hop_length=256, x_axis="time", ax=axes[1, 0])
    axes[1, 0].set(title="MFCC", ylabel="Coefficient")

    pitch, voiced, _ = librosa.pyin(audio, fmin=librosa.note_to_hz("C2"), fmax=librosa.note_to_hz("C7"), sr=sample_rate, hop_length=256)
    pitch_time = librosa.times_like(pitch, sr=sample_rate, hop_length=256)
    axes[1, 1].plot(pitch_time[voiced], pitch[voiced], ".", markersize=3, color="#dc2626")
    axes[1, 1].set(title="Estimated pitch (F0)", xlabel="Time (s)", ylabel="Hz", ylim=(50, 500))
    return fig


def save_wav(path: str | Path, audio: np.ndarray, sample_rate: int = SAMPLE_RATE) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    sf.write(path, audio, sample_rate)

