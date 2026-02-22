from transformers import AutoModel
import numpy as np
import soundfile as sf
import torch

# ----------------------------
# Configuration
# ----------------------------

REPO_ID = "ai4bharat/IndicF5"
REVISION = "b82d286220e3070e171f4ef4b4bd047b9a447c9a"  # pinned commit

# Text to synthesize
TEXT_TO_SYNTH = (
    "ಬೆಂಗಳೂರು ಕರ್ನಾಟಕ ರಾಜ್ಯದ ರಾಜಧಾನಿ ಆಗಿದೆ, "
    "ಕರ್ನಾಟಕದಲ್ಲಿ ನಾವು ಕನ್ನಡ ಮಾತನಾಡುತ್ತೇವೆ"
)

# Reference audio and its transcript
REF_AUDIO_PATH = "prompts/KAN_F_HAPPY_00001.wav"
REF_TEXT = (
    "ನಮ್‌ ಫ್ರಿಜ್ಜಲ್ಲಿ  ಕೂಲಿಂಗ್‌ ಸಮಸ್ಯೆ ಆಗಿ ನಾನ್‌ ಭಾಳ ದಿನದಿಂದ ಒದ್ದಾಡ್ತಿದ್ದೆ, "
    "ಆದ್ರೆ ಅದ್ನೀಗ ಮೆಕಾನಿಕ್ ಆಗಿರೋ ನಿಮ್‌ ಸಹಾಯ್ದಿಂದ ಬಗೆಹರಿಸ್ಕೋಬೋದು ಅಂತಾಗಿ ನಿರಾಳ ಆಯ್ತು ನಂಗೆ."
)

OUTPUT_WAV = "namaste.wav"
OUTPUT_SR = 24000

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ----------------------------
# Load model
# ----------------------------

print(f"Loading model '{REPO_ID}' at revision '{REVISION}' on {DEVICE}...")
model = AutoModel.from_pretrained(
    REPO_ID,
    trust_remote_code=True,
    revision=REVISION,
)
model = model.to(DEVICE)
print("Model loaded.")

# ----------------------------
# Generate speech
# ----------------------------

print("Generating speech...")
with torch.no_grad():
    audio = model(
        TEXT_TO_SYNTH,
        ref_audio_path=REF_AUDIO_PATH,
        ref_text=REF_TEXT,
    )

# Convert to numpy array
audio = np.array(audio)

# Normalize if int16, and ensure float32
if audio.dtype == np.int16:
    audio = audio.astype(np.float32) / 32768.0
else:
    audio = audio.astype(np.float32)

# ----------------------------
# Save to WAV
# ----------------------------

sf.write(OUTPUT_WAV, audio, samplerate=OUTPUT_SR)
print(f"Saved synthesized audio to '{OUTPUT_WAV}' at {OUTPUT_SR} Hz.")
