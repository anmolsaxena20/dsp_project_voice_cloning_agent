# VoxLab — Digital Speech Processing, Voice Cloning & Deepfake Detection Platform

**Academic DSP & Speech Forensics Lab (Group 7)**

A full-stack, production-ready system for zero-shot voice cloning, digital signal processing (DSP) acoustic feature extraction, and Convolutional Neural Network (CNN) AI voice deepfake detection.

---

## 1. Project Architecture

The repository enforces strict separation of concerns with encapsulated backend and frontend modules:

```text
dsp_project/
├── backend/                         # All server, DSP, ML, and storage logic
│   ├── app/
│   │   ├── api/
│   │   │   ├── routes_audio.py      # Audio upload, DSP analysis & streaming
│   │   │   ├── routes_clone.py      # Voice cloning prompt generation
│   │   │   └── routes_security.py   # CNN deepfake detection & voice comparison
│   │   ├── core/
│   │   │   └── config.py            # Unified paths & audio DSP hyperparameters
│   │   ├── dsp/
│   │   │   ├── features.py          # Librosa feature extraction & DSP descriptions
│   │   │   └── visualizer.py        # Neutral white theme matplotlib plot generators
│   │   ├── ml/
│   │   │   ├── cnn_detector.py      # SpectrogramCNN classifier & forensic heuristics
│   │   │   └── voice_cloner.py      # Neural prosody-conditioned voice cloner
│   │   └── main.py                  # FastAPI app entry point (serves API & SPA)
│   ├── data/                        # Dataset audio files
│   ├── models/                      # Saved PyTorch checkpoints (cnn_detector.pt)
│   ├── sample_audio/                # Bundled reference voice recordings
│   ├── storage/                     # Runtime uploads, cloned WAVs, and cached analysis
│   ├── tests/
│   │   └── test_e2e.py              # Automated end-to-end verification suite
│   ├── Dockerfile                   # Production container definition
│   └── requirements.txt             # Python dependencies
│
├── frontend/                        # Clean, neutral white theme SPA (No AI slop gradients)
│   ├── index.html                   # 5-step interactive workflow
│   ├── css/
│   │   └── styles.css               # Modern, high-contrast, slate/zinc styling
│   └── js/
│       └── app.js                   # API client, audio players, plot renderer
│
├── docs/                            # Project documents and academic abstract
│   ├── AI_Voice_Cloning_Group_7_Abstract.pdf
│   └── IMPLEMENTATION_PLAN.md
│
├── docker-compose.yml               # Container orchestration
└── README.md
```

---

## 2. Key Capabilities & Workflow

1. **Step 1: Voice Onboarding**:
   - Upload any voice sample (`.wav`, `.mp3`, `.ogg`, `.flac`) or click **"Load Pre-Bundled Reference Sample"** for instant 1-click evaluation.
   - Automatically standardizes audio to 16 kHz mono, peak-normalized PCM with silence trimming.
2. **Step 2: Acoustic DSP Analysis & Feature Gallery**:
   - **Waveform**: Time-domain dynamics, amplitude envelopes, and speech cadence.
   - **Mel-Scale Spectrogram**: 128-band frequency energy representation.
   - **MFCCs**: 13-coefficient cepstral heatmap capturing vocal tract formants.
   - **Pitch ($F_0$) Tracking**: Probabilistic YIN (pYIN) pitch contour with mean, range, and voiced ratio.
   - **Spectral Dynamics**: Spectral Centroid (brightness) and Spectral Rolloff (85% energy).
   - High-resolution plots rendered directly via backend and delivered as Base64 images with technical explanations.
3. **Step 3: Voice Cloning Studio**:
   - Prompt any sentence to synthesize conditioned speech matching the reference speaker's fundamental frequency ($F_0$), formant energy, and speaking cadence.
   - Mandatory ethical consent confirmation guard.
   - Computes multidimensional MFCC cosine similarity score (e.g. 96.4% match).
4. **Step 4: AI Deepfake Voice Authenticity Detector**:
   - Evaluates audio using our **SpectrogramCNN** model (3 Conv2D blocks + BatchNorm + AdaptiveAvgPool + Linear classifier).
   - Combines CNN prediction with DSP acoustic forensic heuristics:
     - Vocoder high-frequency cutoff anomaly.
     - Pitch micro-jitter stability index.
     - Spectral flatness & resonance analysis.
   - Produces a clear **Authenticity Score (0-100%)**, Verdict badge, and itemized checklist.
5. **Step 5: Side-by-Side Forensics Comparison**:
   - Directly compares authentic reference voice vs generated synthetic speech, calculating $\Delta F_0$, spectral timbre shift, and acoustic distance.

---

## 3. Running Locally

### Prerequisites
- Python 3.11+
- Virtual environment with dependencies:
  ```powershell
  pip install -r backend/requirements.txt
  ```

### Start Server
Run the FastAPI backend (serves both API and frontend):
```powershell
python -m uvicorn backend.app.main:app --host 0.0.0.0 --port 8000 --reload
```
Open your browser at: **`http://localhost:8000`**

---

## 4. Running with Docker

Dockerize the complete application with a single command:
```bash
docker compose up --build -d
```
Access the application at: **`http://localhost:8000`**

Health check endpoint: `GET http://localhost:8000/api/health`

---

## 5. End-to-End Automated Verification

To run the automated verification suite:
```powershell
python backend/tests/test_e2e.py
```
This automatically tests:
- Server health check
- Audio preprocessing & 5 DSP plot generations
- Voice cloning synthesis with speaker conditioning
- Deepfake detection on human reference & cloned speech
- Forensics comparison distance metrics
