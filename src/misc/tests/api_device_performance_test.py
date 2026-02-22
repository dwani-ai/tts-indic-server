import pytest
import torch
import nemo.collections.asr as nemo_asr
from fastapi.testclient import TestClient
from src.asr_api import app, ASRModelManager
import time

# Initialize the FastAPI test client
client = TestClient(app)

# Mock audio file paths for testing
AUDIO_FILE_PATH_WAV = "path/to/your/test_audio.wav"
AUDIO_FILE_PATH_MP3 = "path/to/your/test_audio.mp3"

# Function to measure the time taken for a request
def measure_time(func, *args, **kwargs):
    start_time = time.time()
    result = func(*args, **kwargs)
    end_time = time.time()
    return result, end_time - start_time

# Test case for CUDA mode
def test_transcribe_audio_cuda():
    asr_manager = ASRModelManager(default_language="kn", device_type="cuda")
    app.dependency_overrides[ASRModelManager] = lambda: asr_manager

    with open(AUDIO_FILE_PATH_WAV, "rb") as audio_file:
        response, duration = measure_time(
            client.post,
            "/transcribe/",
            files={"file": ("test_audio.wav", audio_file, "audio/wav")},
            params={"language": "kannada"}
        )

    assert response.status_code == 200
    assert "text" in response.json()
    print(f"CUDA mode transcription time: {duration:.2f} seconds")

# Test case for CPU mode
def test_transcribe_audio_cpu():
    asr_manager = ASRModelManager(default_language="kn", device_type="cpu")
    app.dependency_overrides[ASRModelManager] = lambda: asr_manager

    with open(AUDIO_FILE_PATH_WAV, "rb") as audio_file:
        response, duration = measure_time(
            client.post,
            "/transcribe/",
            files={"file": ("test_audio.wav", audio_file, "audio/wav")},
            params={"language": "kannada"}
        )

    assert response.status_code == 200
    assert "text" in response.json()
    print(f"CPU mode transcription time: {duration:.2f} seconds")

if __name__ == "__main__":
    pytest.main()