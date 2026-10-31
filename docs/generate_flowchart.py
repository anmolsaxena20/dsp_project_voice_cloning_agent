"""
Generate high-resolution publication-quality flowchart for VoxLab Project.
Exports both high-DPI PNG (300 DPI) and SVG for direct insertion into Microsoft Word.
"""
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pathlib import Path

# Setup figure canvas
fig = plt.figure(figsize=(20, 14), dpi=300)
ax = fig.add_subplot(111)
ax.set_xlim(0, 200)
ax.set_ylim(0, 140)
ax.axis("off")

# Palette definition
COLOR_BG = "#F8FAFC"
COLOR_TEXT_MAIN = "#0F172A"
COLOR_TEXT_MUTED = "#475569"

# Module Theme Colors
THEMES = {
    "frontend": {"header": "#2563EB", "bg": "#EFF6FF", "border": "#93C5FD"},
    "api":      {"header": "#0D9488", "bg": "#F0FDFA", "border": "#99F6E4"},
    "storage":  {"header": "#4F46E5", "bg": "#EEF2FF", "border": "#C7D2FE"},
    "dsp":      {"header": "#7C3AED", "bg": "#FAF5FF", "border": "#DDD6FE"},
    "clone":    {"header": "#D97706", "bg": "#FFFBEB", "border": "#FDE68A"},
    "security": {"header": "#DC2626", "bg": "#FEF2F2", "border": "#FECACA"},
}

def draw_group(x, y, w, h, title, theme_key):
    """Draw a bounded subsystem enclosure."""
    t = THEMES[theme_key]
    rect = patches.FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=1.5,rounding_size=2.2",
        facecolor=t["bg"],
        edgecolor=t["border"],
        linewidth=2.0,
        linestyle="--",
        zorder=1
    )
    ax.add_patch(rect)
    
    # Title badge with proper width calculation
    badge_width = max(len(title) * 1.55 + 6, 26)
    badge = patches.FancyBboxPatch(
        (x + 3, y + h - 6.0), badge_width, 5.2,
        boxstyle="round,pad=0.4,rounding_size=1.2",
        facecolor=t["header"],
        edgecolor="none",
        zorder=3
    )
    ax.add_patch(badge)
    ax.text(
        x + 4.5, y + h - 3.4, title,
        color="#FFFFFF", fontsize=9.2, fontweight="bold",
        verticalalignment="center", zorder=4
    )

def draw_node(x, y, w, h, title, subtitle="", color_accent="#2563EB", shape="round", font_size=8.4):
    """Draw a process node."""
    box_style = "round,pad=0.7,rounding_size=1.2" if shape == "round" else "square,pad=0.7"
    card = patches.FancyBboxPatch(
        (x, y), w, h,
        boxstyle=box_style,
        facecolor="#FFFFFF",
        edgecolor="#CBD5E1",
        linewidth=1.4,
        zorder=3
    )
    ax.add_patch(card)
    
    # Left accent indicator bar
    bar = patches.FancyBboxPatch(
        (x, y + 0.6), 1.2, h - 1.2,
        boxstyle="round,pad=0.1,rounding_size=0.4",
        facecolor=color_accent,
        edgecolor="none",
        zorder=4
    )
    ax.add_patch(bar)

    if subtitle:
        ax.text(
            x + w/2 + 0.6, y + h/2 + 1.8, title,
            color=COLOR_TEXT_MAIN, fontsize=font_size, fontweight="bold",
            horizontalalignment="center", verticalalignment="center", zorder=5
        )
        ax.text(
            x + w/2 + 0.6, y + h/2 - 1.8, subtitle,
            color=COLOR_TEXT_MUTED, fontsize=font_size - 1.5,
            horizontalalignment="center", verticalalignment="center", zorder=5
        )
    else:
        ax.text(
            x + w/2 + 0.6, y + h/2, title,
            color=COLOR_TEXT_MAIN, fontsize=font_size, fontweight="bold",
            horizontalalignment="center", verticalalignment="center", zorder=5
        )

def draw_arrow(x1, y1, x2, y2, label="", color="#475569", connectionstyle="arc3,rad=0", label_offset=(0, 0)):
    """Draw a directional arrow with optional centered badge label."""
    arrow = patches.FancyArrowPatch(
        (x1, y1), (x2, y2),
        arrowstyle="-|>",
        mutation_scale=13,
        linewidth=1.6,
        color=color,
        connectionstyle=connectionstyle,
        zorder=2
    )
    ax.add_patch(arrow)
    if label:
        mid_x = (x1 + x2) / 2 + label_offset[0]
        mid_y = (y1 + y2) / 2 + label_offset[1]
        bbox_props = dict(boxstyle="round,pad=0.3", fc="#FFFFFF", ec=color, lw=1.0, alpha=0.95)
        ax.text(
            mid_x, mid_y, label,
            fontsize=7.0, color=color, fontweight="bold",
            ha="center", va="center", bbox=bbox_props, zorder=6
        )

# ----------------- 0. Main Diagram Header -----------------
title_box = patches.FancyBboxPatch(
    (5, 127), 190, 10.0,
    boxstyle="round,pad=0.8,rounding_size=1.8",
    facecolor="#0F172A",
    edgecolor="none",
    zorder=10
)
ax.add_patch(title_box)
ax.text(
    100, 134.0, "VOXLAB: COMPLETE SYSTEM ARCHITECTURE & DSP FORENSICS PIPELINE",
    color="#FFFFFF", fontsize=13.5, fontweight="bold", ha="center", va="center", zorder=11
)
ax.text(
    100, 129.5, "Digital Speech Processing (DSP)  •  Zero-Shot Voice Cloning (Coqui XTTS-v2 & Edge)  •  CNN Deepfake Authenticity Detection",
    color="#94A3B8", fontsize=8.8, ha="center", va="center", zorder=11
)

# ----------------- Subsystem Enclosures -----------------
# Row 1: Frontend (Left), API Gateway (Center), Persistent Storage (Right)
draw_group(5, 87, 56, 36, "1. USER INTERFACE (REACT SPA)", "frontend")
draw_group(67, 87, 62, 36, "2. FASTAPI BACKEND GATEWAY", "api")
draw_group(135, 87, 60, 36, "3. STORAGE & RUNTIME CACHE", "storage")

# Row 2: DSP Engine (Left), Voice Cloning Studio (Center), Deepfake Forensics (Right)
draw_group(5, 6, 60, 75, "4. DIGITAL SIGNAL PROCESSING (DSP) ENGINE", "dsp")
draw_group(71, 6, 58, 75, "5. NEURAL VOICE CLONING STUDIO", "clone")
draw_group(135, 6, 60, 75, "6. AI VOICE DEEPFAKE FORENSICS", "security")

# ----------------- 1. Frontend Subsystem Nodes -----------------
draw_node(8, 111, 50, 7.5, "Step 1: Voice Onboarding", "Upload Audio (.wav/.mp3/.ogg) or Load Sample", THEMES["frontend"]["header"])
draw_node(8, 101.5, 50, 7.5, "Step 2: Acoustic DSP Gallery", "5 Interactive Visual Plots & Signal Metrics", THEMES["frontend"]["header"])
draw_node(8, 92, 50, 7.5, "Steps 3-5: Studio Actions", "Cloning Prompt • Authenticity Check • Forensics", THEMES["frontend"]["header"])

# ----------------- 2. FastAPI Gateway Nodes -----------------
draw_node(70, 111, 56, 7.5, "routes_audio.py", "POST /api/audio/upload • GET /sample • /stream", THEMES["api"]["header"])
draw_node(70, 101.5, 56, 7.5, "routes_clone.py", "POST /api/clone/generate (Ethical Consent Guard)", THEMES["api"]["header"])
draw_node(70, 92, 56, 7.5, "routes_security.py", "POST /api/security/detect • POST /compare", THEMES["api"]["header"])

# ----------------- 3. Persistent Storage Nodes -----------------
draw_node(138, 111, 26, 7.5, "uploads/", "Standardized 16kHz WAV", THEMES["storage"]["header"])
draw_node(166, 111, 26, 7.5, "cloned/", "Zero-Shot Synthesized WAV", THEMES["storage"]["header"])
draw_node(138, 93, 26, 12.0, "analysis/", "Cached JSON Profiles\n& Base64 Plots", THEMES["storage"]["header"])
draw_node(166, 93, 26, 12.0, "models/", "PyTorch Checkpoint\n(cnn_detector.pt)", THEMES["storage"]["header"])

# ----------------- 4. DSP Engine Nodes -----------------
draw_node(8, 66, 54, 8.5, "Audio Normalization & Preprocessing", "16 kHz Mono Resample • Silence Trim • Peak Norm [-1, 1]", THEMES["dsp"]["header"])
draw_node(8, 54, 25, 9.0, "Pitch Tracking (F0)", "Probabilistic YIN (pYIN)\nMean, Range, Voiced %", THEMES["dsp"]["header"], font_size=8.0)
draw_node(37, 54, 25, 9.0, "13-Band MFCCs", "Formant Cepstral Heatmap\nDCT over Mel Filterbank", THEMES["dsp"]["header"], font_size=8.0)
draw_node(8, 41.5, 25, 9.0, "Mel-Spectrogram", "128-Band Log-dB Energy\nSTFT (N_FFT=1024, Hop=256)", THEMES["dsp"]["header"], font_size=8.0)
draw_node(37, 41.5, 25, 9.0, "Spectral Dynamics", "Centroid (Brightness), Rolloff,\nRMS Energy, ZCR & HPSS", THEMES["dsp"]["header"], font_size=8.0)
draw_node(8, 26.5, 54, 11.5, "Acoustic Feature Aggregator", "Compiles Comprehensive Speaker Acoustic Vector:\nF0 Mean/Std • Timbre Centroid • Formant MFCC Means", THEMES["dsp"]["header"])
draw_node(8, 11, 54, 11.5, "Matplotlib Neutral Visualizer", "Generates 5 High-Resolution Diagnostic Plots\nEncoded to Base64 URI Strings for Instant Web Rendering", THEMES["dsp"]["header"])

# ----------------- 5. Neural Voice Cloning Nodes -----------------
draw_node(74, 66, 52, 8.5, "Speaker Acoustic Profiler", "Ingests Reference Audio • Evaluates Pitch, Register & Timbre", THEMES["clone"]["header"])
draw_node(74, 53, 52, 9.5, "Tier 1: Coqui XTTS-v2 Foundation Model", "Zero-Shot Autoregressive Latent Speaker Embeddings\nProduces 24 kHz Studio-Quality Cloned Voice", THEMES["clone"]["header"])
draw_node(74, 40, 52, 9.5, "Tier 2: Neural Prosody TTS Transfer", "Edge Neural Vocoder (Indian English HD Voices)\nProsody Conditioned on Speaker Gender & F0 Profile", THEMES["clone"]["header"])
draw_node(74, 27, 52, 9.5, "Tier 3: Parametric Offline Fallback", "pyttsx3 Synthesis + DSP Phase Vocoder Pitch-Shifting\nor Harmonic Oscillator Carrier Reconstruction", THEMES["clone"]["header"])
draw_node(74, 11, 52, 12.0, "Acoustic Cosine Similarity Scorer", "Calculates Multidimensional Cosine Similarity Metric\nBetween Reference and Synthesized MFCC Vectors (0-100%)", THEMES["clone"]["header"])

# ----------------- 6. AI Deepfake Forensics Nodes -----------------
draw_node(138, 66, 54, 8.5, "Mel-Spectrogram Tensor Preparation", "4D PyTorch Tensor: [Batch=1, Ch=1, Mels=128, T=188]", THEMES["security"]["header"])
draw_node(138, 51.5, 54, 11.5, "SpectrogramCNN Architecture", "3x [Conv2D(3x3) + BatchNorm2D + ReLU + MaxPool(2x2)]\nAdaptiveAvgPool2D(4x4) + Linear(2048->128) + Dropout(0.35)\nFinal Linear(128->2) Classifier [Human vs Synthetic]", THEMES["security"]["header"])
draw_node(138, 34.5, 54, 13.5, "Forensic Acoustic DSP Heuristics", "1. High-Freq Vocoder Cutoff Anomaly (7-8 kHz vs Mid-Band)\n2. Pitch Micro-Jitter (F0 Perturbation Instability Index)\n3. Spectral Flatness & Harmonic Formant Resonances\n4. Envelope Smoothness & Phase Continuity Checks", THEMES["security"]["header"])
draw_node(138, 11, 54, 19.5, "Multi-Modal Ensemble Decision Engine", "Weighted Fusion: 65% CNN Logits + 35% DSP Penalties\nOutputs:\n• Final Authenticity Score (0 - 100%)\n• Binary Verdict ('Authentic Human' vs 'AI-Generated')\n• 4-Point Forensic Diagnostic Verification Checklist\n• Comparative Pitch Difference & Spectral Shift Metrics", THEMES["security"]["header"])

# ----------------- Arrows & Routing -----------------
# Frontend -> API Gateway
draw_arrow(58, 114.7, 70, 114.7, "Voice File", THEMES["frontend"]["header"])
draw_arrow(58, 105.2, 70, 105.2, "Clone Prompt", THEMES["frontend"]["header"])
draw_arrow(58, 95.7, 70, 95.7, "Verify Audio", THEMES["frontend"]["header"])

# Gateway -> Persistent Storage
draw_arrow(126, 114.7, 138, 114.7, "Save WAV", THEMES["storage"]["header"])

# Gateway -> DSP Engine
draw_arrow(80, 111, 35, 74.5, "Raw Audio Data", THEMES["dsp"]["header"], connectionstyle="arc3,rad=-0.12")

# DSP Internal Flow
draw_arrow(35, 66, 20.5, 63, "", THEMES["dsp"]["header"])
draw_arrow(35, 66, 49.5, 63, "", THEMES["dsp"]["header"])
draw_arrow(20.5, 54, 20.5, 50.5, "", THEMES["dsp"]["header"])
draw_arrow(49.5, 54, 49.5, 50.5, "", THEMES["dsp"]["header"])
draw_arrow(20.5, 41.5, 27, 38, "", THEMES["dsp"]["header"])
draw_arrow(49.5, 41.5, 43, 38, "", THEMES["dsp"]["header"])
draw_arrow(35, 26.5, 35, 22.5, "Features", THEMES["dsp"]["header"])

# DSP -> Storage Cache & Frontend
draw_arrow(62, 32, 138, 96, "Cache JSON & Base64 Plots", THEMES["storage"]["header"], connectionstyle="arc3,rad=-0.22", label_offset=(0, 2))

# Gateway -> Voice Cloning Engine
draw_arrow(98, 101.5, 98, 74.5, "Text & Reference ID", THEMES["clone"]["header"])
draw_arrow(100, 66, 100, 62.5, "Profile", THEMES["clone"]["header"])
draw_arrow(100, 53, 100, 49.5, "Fallback", THEMES["clone"]["header"])
draw_arrow(100, 40, 100, 36.5, "Fallback", THEMES["clone"]["header"])
draw_arrow(100, 27, 100, 23, "", THEMES["clone"]["header"])

# Cloning Output -> Storage
draw_arrow(126, 58, 166, 111, "Synthesized Audio", THEMES["storage"]["header"], connectionstyle="arc3,rad=-0.22")

# Gateway -> Deepfake Security Engine
draw_arrow(116, 92, 148, 74.5, "Audio Stream / ID", THEMES["security"]["header"], connectionstyle="arc3,rad=-0.1")

# Deepfake Internal Flow
draw_arrow(165, 66, 165, 63, "Mel Tensor", THEMES["security"]["header"])
draw_arrow(165, 51.5, 165, 48, "Logits", THEMES["security"]["header"])
draw_arrow(165, 34.5, 165, 30.5, "Acoustic Penalties", THEMES["security"]["header"])

# Checkpoint model to CNN
draw_arrow(179, 93, 179, 63, "Model Weights", THEMES["storage"]["header"], connectionstyle="arc3,rad=0.15")

# Forensics to Gateway Return
draw_arrow(138, 20, 110, 89, "Authenticity Verdict & Report", THEMES["security"]["header"], connectionstyle="arc3,rad=0.22", label_offset=(-2, 0))

# Save outputs in docs/
output_dir = Path("docs")
output_dir.mkdir(parents=True, exist_ok=True)
png_path = output_dir / "voxlab_project_flowchart.png"
svg_path = output_dir / "voxlab_project_flowchart.svg"

plt.tight_layout()
plt.savefig(str(png_path), dpi=300, bbox_inches="tight", facecolor=COLOR_BG)
plt.savefig(str(svg_path), format="svg", bbox_inches="tight", facecolor=COLOR_BG)
plt.close()

print(f"[SUCCESS] Flowchart generated:\n- {png_path.resolve()}\n- {svg_path.resolve()}")
