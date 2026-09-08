# AI Voice Cloning and AI-Generated Voice Detection — Implementation Plan

## Goal

Build a local academic prototype that (a) turns text into clearly labelled synthetic speech using an authorised voice workflow, (b) classifies WAV audio as human or AI-generated, and (c) produces the signal-analysis plots and metrics specified in the abstract.

## MVP scope (this delivery)

1. **Local Streamlit interface** with separate Synthesis, Detection, and Analysis tabs.
2. **Working offline TTS baseline** using the Windows-installed voice engine. Generated files are saved as WAV, labelled as synthetic, and playback/download is available.
3. **Consent guard**: synthesis requires confirmation that the user owns the voice or has written permission. The baseline does not imitate a reference voice.
4. **Shared audio preprocessing**: mono conversion, 16 kHz resampling, amplitude normalization, and optional silence trimming.
5. **Mel-spectrogram pipeline and CNN** in PyTorch, with scripts to prepare data, train, evaluate, and run inference.
6. **Honest detector state**: before a trained checkpoint exists, the UI explains that a prediction cannot be made rather than presenting an unreliable score.
7. **Python plots plus MATLAB scripts** for waveform, spectrogram, MFCC, and pitch inspection.

## Architecture

```text
authorised WAV + text ──> TTS adapter ──> generated WAV ──> outputs/
                                      │
input WAV ─> preprocess ─> mel tensor ─> CNN checkpoint ─> class + confidence
                         └───────────> analysis plots / MATLAB scripts
```

## Data layout

```text
data/raw/human/<speaker-or-source>/*.wav
data/raw/ai/<generator-or-source>/*.wav
data/processed/                 # optional prepared artefacts
models/                          # saved CNN checkpoints
outputs/                         # generated WAVs, plots, reports
```

Keep speakers separated between training, validation, and test sets. Balance classes and retain the source/generator identifier in a CSV manifest. Do not use recordings without explicit permission or a compatible dataset licence.

## Delivery milestones

1. **MVP foundation** — project structure, local UI, audio utilities, baseline TTS, plots, MATLAB script. *(implemented now)*
2. **Dataset preparation** — collect permitted human speech and generate/lawfully source AI speech; create speaker-independent split manifest.
3. **Detector training** — train the CNN, select the best validation checkpoint, then evaluate once on test data using accuracy, precision, recall, F1, and a confusion matrix.
4. **Voice-cloning adapter** — install a compatible open-source speaker-conditioned model and connect it only after verifying consent, hardware, licence, and Python/model compatibility. The proposed candidate is XTTS-v2, normally with Python 3.10–3.11 and adequate disk/GPU resources.
5. **Demo and report** — compare known human/AI samples in the UI and MATLAB, disclose limitations, and label all synthetic audio.

## Decisions still needed

- Is MATLAB available on the intended presentation machine?
- Do you have an NVIDIA GPU, and if so, which model/VRAM? This determines whether local voice cloning can be practical.
- Will you provide authorised reference WAVs and a permitted human-speech dataset, or should the project include download/preparation instructions only?
- Is English-only synthesis acceptable for the first demonstration?

## Acceptance checks

- A text prompt produces a playable synthetic WAV locally.
- A WAV upload generates waveform, mel-spectrogram, MFCC, and pitch plots.
- With a trained checkpoint, the detector returns both class probabilities.
- `evaluate.py` writes the required metrics and confusion matrix from held-out labelled data.
- MATLAB analysis runs on any selected WAV after changing one path variable.

