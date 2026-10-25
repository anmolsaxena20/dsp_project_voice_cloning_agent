"""Matplotlib visualizer for DSP acoustic features with Cloudflare dark theme."""
from __future__ import annotations

import base64
import io
from typing import Dict

import librosa
import librosa.display
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
import numpy as np

from backend.app.core.config import HOP_LENGTH, N_FFT, N_MELS, N_MFCC, SAMPLE_RATE

# Styling constants matching Cloudflare Dark Theme
BG_COLOR = "#18191E"         # Card background
PLOT_BG = "#131419"          # Canvas background
GRID_COLOR = "#282A33"       # Subtle divider grid
TEXT_COLOR = "#F9FAFB"       # Crisp title text
MUTED_TEXT = "#9CA3AF"       # Axis labels & ticks
PRIMARY_ORANGE = "#F6821F"   # Cloudflare signature orange
ACCENT_GREEN = "#10B981"     # Emerald for pitch / natural
ACCENT_BLUE = "#3B82F6"      # Deep blue


def _apply_dark_styling(ax: plt.Axes, title: str, xlabel: str = "Time (seconds)", ylabel: str = ""):
    """Apply Cloudflare dark theme styling to axes."""
    ax.set_facecolor(PLOT_BG)
    ax.set_title(title, fontsize=12, fontweight="600", color=TEXT_COLOR, pad=10)
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=10, color=MUTED_TEXT)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=10, color=MUTED_TEXT)
    ax.tick_params(colors=MUTED_TEXT, labelsize=9)
    ax.grid(True, linestyle="--", linewidth=0.5, color=GRID_COLOR, alpha=0.7)
    for spine in ax.spines.values():
        spine.set_color(GRID_COLOR)
        spine.set_linewidth(1.0)


def _fig_to_base64(fig: plt.Figure) -> str:
    """Convert matplotlib figure directly to Base64 encoded PNG string."""
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", dpi=130, bbox_inches="tight", facecolor=BG_COLOR, edgecolor="none")
    plt.close(fig)
    buffer.seek(0)
    return base64.b64encode(buffer.read()).decode("utf-8")


def generate_waveform_plot(audio: np.ndarray, sr: int = SAMPLE_RATE) -> str:
    """Generate time-domain waveform plot."""
    fig, ax = plt.subplots(figsize=(9, 3.2), facecolor=BG_COLOR)
    times = np.arange(len(audio)) / sr
    ax.plot(times, audio, linewidth=0.8, color=PRIMARY_ORANGE, alpha=0.95)
    ax.axhline(0, color="#363945", linestyle="-", linewidth=0.8, alpha=0.7)
    _apply_dark_styling(ax, "Acoustic Waveform (Amplitude vs Time)", "Time (seconds)", "Normalized Amplitude")
    ax.set_ylim(-1.05, 1.05)
    return _fig_to_base64(fig)


def generate_mel_spectrogram_plot(audio: np.ndarray, sr: int = SAMPLE_RATE) -> str:
    """Generate 128-band Mel-Spectrogram plot."""
    fig, ax = plt.subplots(figsize=(9, 3.8), facecolor=BG_COLOR)
    mel = librosa.feature.melspectrogram(y=audio, sr=sr, n_fft=N_FFT, hop_length=HOP_LENGTH, n_mels=N_MELS)
    mel_db = librosa.power_to_db(mel, ref=np.max)
    img = librosa.display.specshow(
        mel_db,
        sr=sr,
        hop_length=HOP_LENGTH,
        x_axis="time",
        y_axis="mel",
        ax=ax,
        cmap="magma"
    )
    _apply_dark_styling(ax, "Mel-Scale Spectrogram (Time-Frequency Energy)", "Time (seconds)", "Frequency (Hz)")
    cbar = fig.colorbar(img, ax=ax, format="%+2.0f dB", pad=0.02)
    cbar.ax.tick_params(labelsize=8, colors=MUTED_TEXT)
    cbar.outline.set_edgecolor(GRID_COLOR)
    return _fig_to_base64(fig)


def generate_mfcc_plot(audio: np.ndarray, sr: int = SAMPLE_RATE) -> str:
    """Generate MFCC 13-coefficient heatmap."""
    fig, ax = plt.subplots(figsize=(9, 3.5), facecolor=BG_COLOR)
    mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=N_MFCC, hop_length=HOP_LENGTH)
    img = librosa.display.specshow(
        mfccs,
        sr=sr,
        hop_length=HOP_LENGTH,
        x_axis="time",
        ax=ax,
        cmap="plasma"
    )
    _apply_dark_styling(ax, "Mel-Frequency Cepstral Coefficients (MFCC 1-13)", "Time (seconds)", "MFCC Index")
    cbar = fig.colorbar(img, ax=ax, pad=0.02)
    cbar.ax.tick_params(labelsize=8, colors=MUTED_TEXT)
    cbar.outline.set_edgecolor(GRID_COLOR)
    return _fig_to_base64(fig)


def generate_pitch_plot(audio: np.ndarray, sr: int = SAMPLE_RATE) -> str:
    """Generate pitch F0 contour plot via pYIN."""
    fig, ax = plt.subplots(figsize=(9, 3.2), facecolor=BG_COLOR)
    f0, voiced_flag, _ = librosa.pyin(
        audio,
        fmin=librosa.note_to_hz("C2"),
        fmax=librosa.note_to_hz("C7"),
        sr=sr,
        hop_length=HOP_LENGTH
    )
    times = librosa.times_like(f0, sr=sr, hop_length=HOP_LENGTH)
    
    if voiced_flag is not None and np.any(voiced_flag):
        ax.scatter(times[voiced_flag], f0[voiced_flag], s=9, color=ACCENT_GREEN, label="Voiced Pitch (F0)", zorder=3)
        ax.plot(times[voiced_flag], f0[voiced_flag], color=ACCENT_GREEN, alpha=0.45, linewidth=1.0)
    
    _apply_dark_styling(ax, "Fundamental Frequency Contour (Pitch F0)", "Time (seconds)", "Frequency (Hz)")
    ax.set_ylim(50, 450)
    ax.legend(loc="upper right", frameon=True, facecolor=BG_COLOR, edgecolor=GRID_COLOR, fontsize=8, labelcolor=TEXT_COLOR)
    return _fig_to_base64(fig)


def generate_spectral_features_plot(audio: np.ndarray, sr: int = SAMPLE_RATE) -> str:
    """Generate Spectral Centroid and Spectral Rolloff tracking plot."""
    fig, ax = plt.subplots(figsize=(9, 3.2), facecolor=BG_COLOR)
    centroid = librosa.feature.spectral_centroid(y=audio, sr=sr, hop_length=HOP_LENGTH)[0]
    rolloff = librosa.feature.spectral_rolloff(y=audio, sr=sr, roll_percent=0.85, hop_length=HOP_LENGTH)[0]
    times = librosa.times_like(centroid, sr=sr, hop_length=HOP_LENGTH)

    ax.plot(times, rolloff, color="#9CA3AF", linewidth=1.2, linestyle="--", label="Spectral Rolloff (85% energy)")
    ax.plot(times, centroid, color=PRIMARY_ORANGE, linewidth=1.5, label="Spectral Centroid (Brightness)")
    
    _apply_dark_styling(ax, "Timbral Dynamics: Spectral Centroid & Rolloff", "Time (seconds)", "Frequency (Hz)")
    ax.legend(loc="upper right", frameon=True, facecolor=BG_COLOR, edgecolor=GRID_COLOR, fontsize=8, labelcolor=TEXT_COLOR)
    return _fig_to_base64(fig)


def generate_all_analysis_plots(audio: np.ndarray, sr: int = SAMPLE_RATE) -> Dict[str, str]:
    """Generate all distinct DSP visual plots as Base64 images."""
    return {
        "waveform": generate_waveform_plot(audio, sr),
        "mel_spectrogram": generate_mel_spectrogram_plot(audio, sr),
        "mfcc": generate_mfcc_plot(audio, sr),
        "pitch_f0": generate_pitch_plot(audio, sr),
        "spectral_features": generate_spectral_features_plot(audio, sr)
    }
