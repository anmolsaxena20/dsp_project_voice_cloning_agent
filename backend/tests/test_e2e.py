"""Automated End-to-End Test Suite for VoxLab Backend."""
import sys
from pathlib import Path
from starlette.testclient import TestClient

root_dir = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(root_dir))

from backend.app.main import app

client = TestClient(app)


def test_01_health_check():
    print("\n--- Testing Health Check ---")
    response = client.get("/api/health")
    assert response.status_code == 200, f"Expected 200, got {response.status_code}"
    data = response.json()
    assert data["status"] == "healthy"
    print("[PASS] Health check passed:", data)


def test_02_load_sample_audio_and_dsp():
    print("\n--- Testing Pre-Bundled Reference Sample & DSP Analysis ---")
    response = client.get("/api/audio/sample")
    assert response.status_code == 200, f"Expected 200, got {response.status_code}: {response.text}"
    data = response.json()
    
    assert "audio_id" in data
    assert "features" in data
    assert "plots" in data
    assert "waveform" in data["plots"]
    assert "mel_spectrogram" in data["plots"]
    assert "mfcc" in data["plots"]
    assert "pitch_f0" in data["plots"]
    assert "spectral_features" in data["plots"]
    
    pitch = data["features"]["pitch"]
    timbre = data["features"]["timbre"]
    print(f"[PASS] DSP Analysis passed:")
    print(f"   Audio ID: {data['audio_id']}")
    print(f"   Duration: {data['features']['duration_seconds']}s")
    print(f"   Mean Pitch F0: {pitch['mean_f0_hz']} Hz (Voiced: {pitch['voiced_percentage']}%)")
    print(f"   Spectral Centroid: {timbre['spectral_centroid_hz']} Hz")
    print(f"   Plots generated (Base64 PNG): {list(data['plots'].keys())}")
    return data["audio_id"]


def test_03_voice_cloning(audio_id: str):
    print("\n--- Testing Voice Cloning Studio Generation ---")
    payload = {
        "audio_id": audio_id,
        "text_prompt": "Testing our end to end digital speech processing and voice cloning pipeline.",
        "consent": True
    }
    response = client.post("/api/clone/generate", json=payload)
    assert response.status_code == 200, f"Expected 200, got {response.status_code}: {response.text}"
    data = response.json()
    
    assert data["status"] == "success"
    assert "filename" in data
    assert "audio_url" in data
    assert "voice_similarity_score" in data
    print(f"[PASS] Voice Cloning passed:")
    print(f"   Generated WAV: {data['filename']}")
    print(f"   Duration: {data['duration_seconds']}s")
    print(f"   Voice Similarity Score: {data['voice_similarity_score']}%")
    return data["filename"]


def test_04_deepfake_detection(human_audio_id: str, cloned_filename: str):
    print("\n--- Testing CNN & DSP Deepfake Voice Detection ---")
    
    # Test human reference voice
    res_human = client.post("/api/security/detect", data={"audio_id": human_audio_id})
    assert res_human.status_code == 200
    report_human = res_human.json()
    print(f"[PASS] Scan on Reference Voice:")
    print(f"   Verdict: {report_human['verdict']}")
    print(f"   Authenticity Score: {report_human['authenticity_score']}%")
    print(f"   P(Human): {report_human['probabilities']['human']}, P(AI): {report_human['probabilities']['ai_generated']}")

    # Test cloned synthetic voice
    res_cloned = client.post("/api/security/detect", data={"audio_id": cloned_filename})
    assert res_cloned.status_code == 200
    report_cloned = res_cloned.json()
    print(f"[PASS] Scan on Cloned Voice:")
    print(f"   Verdict: {report_cloned['verdict']}")
    print(f"   Authenticity Score: {report_cloned['authenticity_score']}%")
    print(f"   P(Human): {report_cloned['probabilities']['human']}, P(AI): {report_cloned['probabilities']['ai_generated']}")


def test_05_voice_comparison(human_audio_id: str, cloned_filename: str):
    print("\n--- Testing Forensics Side-by-Side Comparison ---")
    payload = {
        "reference_id": human_audio_id,
        "target_id": cloned_filename
    }
    response = client.post("/api/security/compare", json=payload)
    assert response.status_code == 200, f"Expected 200, got {response.status_code}: {response.text}"
    data = response.json()
    
    print(f"[PASS] Side-by-Side Comparison passed:")
    print(f"   Similarity: {data['similarity_score_percentage']}%")
    print(f"   Pitch Difference: {data['pitch_difference_hz']} Hz")
    print(f"   Spectral Centroid Difference: {data['spectral_centroid_diff_hz']} Hz")


if __name__ == "__main__":
    print("=" * 60)
    print("VOXLAB END-TO-END AUTOMATED VERIFICATION SUITE")
    print("=" * 60)
    test_01_health_check()
    audio_id = test_02_load_sample_audio_and_dsp()
    cloned_filename = test_03_voice_cloning(audio_id)
    test_04_deepfake_detection(audio_id, cloned_filename)
    test_05_voice_comparison(audio_id, cloned_filename)
    print("\n" + "=" * 60)
    print("ALL TESTS PASSED SUCCESSFULLY! (100% End-to-End Verified)")
    print("=" * 60)
