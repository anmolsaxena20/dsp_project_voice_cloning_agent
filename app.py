from __future__ import annotations

import tempfile
import time
from pathlib import Path

import streamlit as st

from audio_utils import analysis_figure, load_audio
from pretrained_detector import MODEL_ID, classify_audio

ROOT = Path(__file__).parent
OUTPUTS = ROOT / "outputs"

st.set_page_config(page_title="Synthetic Speech Lab", page_icon="🔊", layout="wide")
st.title("Synthetic Speech Lab")
st.caption("Python-only academic MVP — labelled synthesis, audio analysis, and trained-model detection.")


def uploaded_path(upload) -> str:
    suffix = Path(upload.name).suffix or ".wav"
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    tmp.write(upload.getvalue())
    tmp.close()
    return tmp.name


tab_synth, tab_detect, tab_analyse = st.tabs(["Synthesis", "Detection", "Audio analysis"])

with tab_synth:
    st.subheader("Offline text-to-speech")
    st.info("This MVP produces clearly labelled synthetic audio with a local Windows voice. It does not imitate a person from an uploaded sample.")
    text = st.text_area("Text to synthesize", "Hello. This is a clearly labelled synthetic speech demonstration.", max_chars=800)
    consent = st.checkbox("I confirm I will not use this system to impersonate or deceive anyone.")
    if st.button("Generate labelled WAV", type="primary", disabled=not consent or not text.strip()):
        try:
            import pyttsx3
            OUTPUTS.mkdir(exist_ok=True)
            destination = OUTPUTS / f"synthetic_{int(time.time())}.wav"
            engine = pyttsx3.init()
            engine.setProperty("rate", 165)
            engine.save_to_file(text.strip(), str(destination))
            engine.runAndWait()
            if not destination.exists() or destination.stat().st_size == 0:
                raise RuntimeError("The local speech engine did not produce an audio file.")
            st.success(f"Saved labelled synthetic speech to `{destination.name}`")
            st.audio(destination.read_bytes(), format="audio/wav")
            st.download_button("Download WAV", destination.read_bytes(), file_name=destination.name, mime="audio/wav")
        except Exception as exc:
            st.error(f"Synthesis failed: {exc}")

with tab_detect:
    st.subheader("Human vs AI-generated speech (pretrained detector)")
    upload = st.file_uploader("Upload a WAV/MP3/FLAC recording", type=["wav", "mp3", "flac", "ogg"], key="detect")
    if upload:
        st.audio(upload.getvalue())
        st.caption(f"Model: `{MODEL_ID}`. Its weights download once on first use.")
        if st.button("Run detector"):
            try:
                probabilities = classify_audio(uploaded_path(upload))
                label = "AI-Generated Voice" if probabilities["AI-generated"] >= probabilities["Human"] else "Human Voice"
                st.metric("Prediction", label, f"{max(probabilities.values()) * 100:.1f}% confidence")
                st.bar_chart(probabilities)
                st.caption("Prediction only — not forensic proof. Reliability is limited to data and generators represented in training.")
            except Exception as exc:
                st.error(f"Detection failed: {exc}")

with tab_analyse:
    st.subheader("Waveform, spectrogram, MFCC, and pitch")
    upload = st.file_uploader("Upload audio to analyse", type=["wav", "mp3", "flac", "ogg"], key="analysis")
    if upload:
        try:
            audio, sr = load_audio(uploaded_path(upload))
            st.caption(f"Preprocessed to mono, {sr:,} Hz; duration {len(audio)/sr:.2f} seconds.")
            st.pyplot(analysis_figure(audio, sr), clear_figure=True)
        except Exception as exc:
            st.error(f"Analysis failed: {exc}")
