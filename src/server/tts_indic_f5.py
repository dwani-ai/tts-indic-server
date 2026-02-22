from transformers import AutoModel
import numpy as np
import soundfile as sf

# Load INF5 from Hugging Face
repo_id = "ai4bharat/IndicF5"
model = AutoModel.from_pretrained(repo_id, trust_remote_code=True)


# Generate speech
audio = model(
    "ಬೆಂಗಳೂರು ಕರ್ನಾಟಕ ರಾಜ್ಯದ ರಾಜಧಾನಿ ಆಗಿದೆ, ಕರ್ನಾಟಕದಲ್ಲಿ ನಾವು ಕನ್ನಡ ಮಾತನಾಡುತ್ತೇವೆ",
    ref_audio_path="prompts/KAN_F_HAPPY_00001.wav",
    ref_text="ನಮ್‌ ಫ್ರಿಜ್ಜಲ್ಲಿ  ಕೂಲಿಂಗ್‌ ಸಮಸ್ಯೆ ಆಗಿ ನಾನ್‌ ಭಾಳ ದಿನದಿಂದ ಒದ್ದಾಡ್ತಿದ್ದೆ, ಆದ್ರೆ ಅದ್ನೀಗ ಮೆಕಾನಿಕ್ ಆಗಿರೋ ನಿಮ್‌ ಸಹಾಯ್ದಿಂದ ಬಗೆಹರಿಸ್ಕೋಬೋದು ಅಂತಾಗಿ ನಿರಾಳ ಆಯ್ತು ನಂಗೆ."
)


# Normalize and save output
if audio.dtype == np.int16:
    audio = audio.astype(np.float32) / 32768.0
sf.write("namaste.wav", np.array(audio, dtype=np.float32), samplerate=24000)
