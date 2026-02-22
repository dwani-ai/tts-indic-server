import io
import torch
import requests
import tempfile
import numpy as np
import soundfile as sf
from fastapi import FastAPI, HTTPException
from transformers import AutoModel
from pydantic import BaseModel
from typing import Optional
from starlette.responses import StreamingResponse
import types
import sys
import importlib

original_find_spec = importlib.util.find_spec
def mocked_find_spec(name, *args, **kwargs):
    if name == "torchvision":
        return None
    return original_find_spec(name, *args, **kwargs)
importlib.util.find_spec = mocked_find_spec

sys.modules['torchvision'] = types.ModuleType('torchvision')
sys.modules['torchvision.io'] = types.ModuleType('torchvision.io')
sys.modules['torchvision.transforms'] = types.ModuleType('torchvision.transforms')
sys.modules['torchvision.transforms.v2'] = types.ModuleType('torchvision.transforms.v2')
sys.modules['torchvision.transforms.v2'].functional = types.ModuleType('torchvision.transforms.v2.functional')

# Mock InterpolationMode with required attributes
class InterpolationMode:
    NEAREST = "nearest"
    BILINEAR = "bilinear"
    BICUBIC = "bicubic"
    BOX = "box"
    HAMMING = "hamming"
    LANCZOS = "lanczos"

sys.modules['torchvision.transforms'].InterpolationMode = InterpolationMode

# Initialize FastAPI app
app = FastAPI(title="IndicF5 Text-to-Speech API", description="High-quality TTS for Indian languages with Kannada output")

# Load TTS model
repo_id = "ai4bharat/IndicF5"
model = AutoModel.from_pretrained(repo_id, trust_remote_code=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)
model = model.to(device)

# Example Data (Multiple Examples with URLs)
EXAMPLES = [
    {
        "audio_name": "PAN_F (Happy)",
        "audio_url": "https://github.com/AI4Bharat/IndicF5/raw/refs/heads/main/prompts/PAN_F_HAPPY_00002.wav",
        "ref_text": "ਇੱਕ ਗ੍ਰਾਹਕ ਨੇ ਸਾਡੀ ਬੇਮਿਸਾਲ ਸੇਵਾ ਬਾਰੇ ਦਿਲੋਂਗਵਾਹੀ ਦਿੱਤੀ ਜਿਸ ਨਾਲ ਸਾਨੂੰ ਅਨੰದ ಮਹಿಸೂਸ ਹੋਇਆ।",
        "synth_text": "ನಾನು ಯಾವುದೇ ಚಿಂತೆ ಇಲ್ಲದೆ ನನ್ನ ಸ್ನೇಹಿತರನ್ನು ನನ್ನ ಆಟೋಮೊಬೈಲ್ ತಜ್ಞರ ಬಳಿಗೆ ಕಳುಹಿಸುತ್ತೇನೆ ಏಕೆಂದರೆ ಅವರು ಖಂಡಿತವಾಗಿಯೂ ಅವರ ಎಲ್ಲಾ ಅಗತ್ಯಗಳನ್ನು ಪೂರೈಸುತ್ತಾರೆ ಎಂದು ನನಗೆ ಗೊತ್ತು."
    },
    {
        "audio_name": "TAM_F (Happy)",
        "audio_url": "https://github.com/AI4Bharat/IndicF5/raw/refs/heads/main/prompts/TAM_F_HAPPY_00001.wav",
        "ref_text": "நான் நெனச்ச மாதிரியே அமேசான்ல பெரிய தள்ளுபடி வந்திருக்கு. கம்மி காசுக்கே அந்தப் புது சேம்சங் மாடல வாங்கிடலாம்.",
        "synth_text": "ಊಟದ ನಂತರ ಮೊಸರು ಅನ್ನ ತಿಂದರೆ ಒಂದು ಉತ್ಸಾಹವಾಗುತ್ತದೆ!"
    },
    {
        "audio_name": "MAR_F (WIKI)",
        "audio_url": "https://github.com/AI4Bharat/IndicF5/raw/refs/heads/main/prompts/MAR_F_WIKI_00001.wav",
        "ref_text": "दिगंतराव्दारे अंतराळ कक्षेतला कचरा चिन्हित करण्यासाठी प्रयत्न केले जात आहे.",
        "synth_text": "ಪ್ರಾರಂಭಿಕ ಬೀಜ ಚಿಗುರೊಡೆಯುವಿಕೆ. ನಾನು ಸೋಲಾಪುರ ಜಿಲ್ಲೆಯ ಮಾಲಶಿರಸ್ ತಾಲೂಕಿನ ರೈತ ಗಣಪತ್ ಪಾಟೀಲ್ ಮಾತನಾಡುತ್ತಿದ್ದೇನೆ. ನನ್ನ ಕಬ್ಬಿನ ಬೆಳೆಯಲ್ಲಿ ಪ್ರಾರಂಭಿಕ ಬೀಜ ಚಿಗುರೊಡೆಯುವ ಕೀಟ ಕಂಡುಬರುತ್ತಿದೆ. ಕ್ಲೋರಂಟ್ರಾನಿಲಿಪ್ರೋಲ್ (ಕೊರಾಜೆನ್) ಬಳಸುವುದು ಸೂಕ್ತವೇ? ಅದರ ಪ್ರಮಾಣ ಎಷ್ಟಿರಬೇಕು?"
    },
    {
        "audio_name": "MAR_M (WIKI)",
        "audio_url": "https://github.com/AI4Bharat/IndicF5/raw/refs/heads/main/prompts/MAR_M_WIKI_00001.wav",
        "ref_text": "या प्रथाला एकोणीसशे पंचातर ईसवी पासून भारतीय दंड संहिताची धारा चारशे अठ्ठावीस आणि चारशे एकोणतीसच्या अंतर्गत निषেধ केला.",
        "synth_text": "ಜೀವಾಣು ಕೊಳೆತ. ನಾನು ಅಹಮದ್‌ನಗರ ಜಿಲ್ಲೆಯ ರಾಹುರಿ ಗ್ರಾಮದಿಂದ ಬಾಳಾಸಾಹೇಬ್ ಜಾಧವ್ ಮಾತನಾಡುತ್ತಿದ್ದೇನೆ. ನನ್ನ ದಾಳಿಂಬೆ ತೋಟದಲ್ಲಿ ಜೀವಾಣು ಕೊಳೆತ ಹೆಚ್ಚಾಗಿ ಕಾಣಿಸುತ್ತಿದೆ. ಸ್ಟ್ರೆಪ್ಟೋಸೈಕ್ಲಿನ್ ಮತ್ತು ಕಾಪರ್ ಆಕ್ಸಿಕ್ಲೋರೈಡ್ ಸಿಂಪಡಣೆಗೆ ಸೂಕ್ತ ಪ್ರಮಾಣ ಎಷ್ಟು?"
    },
    {
        "audio_name": "KAN_F (Happy)",
        "audio_url": "https://github.com/AI4Bharat/IndicF5/raw/refs/heads/main/prompts/KAN_F_HAPPY_00001.wav",
        "ref_text": "ನಮ್‌ ಫ್ರಿಜ್ಜಲ್ಲಿ  ಕೂಲಿಂಗ್‌ ಸಮಸ್ಯೆ ಆಗಿ ನಾನ್‌ ಭಾಳ ದಿನದಿಂದ ಒದ್ದಾಡ್ತಿದ್ದೆ, ಆದ್ರೆ ಅದ್ನೀಗ ಮೆಕಾನಿಕ್ ಆಗಿರೋ ನಿಮ್‌ ಸಹಾಯ್ದಿಂದ ಬಗೆಹರಿಸ್ಕೋಬೋದು ಅಂತಾಗಿ ನಿರಾಳ ಆಯ್ತು ನಂಗೆ.",
        "synth_text": "ಚೆನ್ನೈನ ಶೇರ್ ಆಟೋ ಪ್ರಯಾಣಿಕರ ನಡುವೆ ಆಹಾರವನ್ನು ಹಂಚಿಕೊಂಡು ತಿನ್ನುವುದು ನನಗೆ ಮನಸ್ಸಿಗೆ ತುಂಬಾ ಒಳ್ಳೆಯದೆನಿಸುವ ವಿಷಯ."
    },
]

# Pydantic models for request bodies
class SynthesizeRequest(BaseModel):
    text: str  # Text to synthesize (expected in Kannada)
    ref_audio_name: str  # Dropdown of audio names from EXAMPLES
    ref_text: Optional[str] = None  # Optional, defaults to example ref_text if not provided

class KannadaSynthesizeRequest(BaseModel):
    text: str  # Text to synthesize (must be in Kannada)

# Function to load audio from URL
def load_audio_from_url(url: str):
    response = requests.get(url)
    if response.status_code == 200:
        audio_data, sample_rate = sf.read(io.BytesIO(response.content))
        return sample_rate, audio_data
    raise HTTPException(status_code=500, detail="Failed to load reference audio from URL.")

# Function to synthesize speech
def synthesize_speech(text: str, ref_audio_name: str, ref_text: str):
    # Find the matching example
    ref_audio_url = None
    for example in EXAMPLES:
        if example["audio_name"] == ref_audio_name:
            ref_audio_url = example["audio_url"]
            if not ref_text:
                ref_text = example["ref_text"]
            break
    
    if not ref_audio_url:
        raise HTTPException(status_code=400, detail="Invalid reference audio name.")

    if not text.strip():
        raise HTTPException(status_code=400, detail="Text to synthesize cannot be empty.")
    
    if not ref_text or not ref_text.strip():
        raise HTTPException(status_code=400, detail="Reference text cannot be empty.")

    # Load reference audio from URL
    sample_rate, audio_data = load_audio_from_url(ref_audio_url)

    # Save reference audio to a temporary file
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_audio:
        sf.write(temp_audio.name, audio_data, samplerate=sample_rate, format='WAV')
        temp_audio.flush()

        # Generate speech
        audio = model(text, ref_audio_path=temp_audio.name, ref_text=ref_text)

    # Normalize output
    if audio.dtype == np.int16:
        audio = audio.astype(np.float32) / 32768.0

    # Save generated audio to a BytesIO buffer
    buffer = io.BytesIO()
    sf.write(buffer, audio, 24000, format='WAV')
    buffer.seek(0)

    return buffer
'''
# Original endpoint
@app.post("/synthesize", response_class=StreamingResponse)
async def synthesize(request: SynthesizeRequest):
    audio_buffer = synthesize_speech(request.text, request.ref_audio_name, request.ref_text)
    return StreamingResponse(
        audio_buffer,
        media_type="audio/wav",
        headers={"Content-Disposition": "attachment; filename=synthesized_speech.wav"}
    )
'''
# New endpoint for Kannada-only synthesis
@app.post("/audio/speech", response_class=StreamingResponse)
async def synthesize_kannada(request: KannadaSynthesizeRequest):
    # Use the Kannada example as fixed reference
    kannada_example = next(ex for ex in EXAMPLES if ex["audio_name"] == "KAN_F (Happy)")
    
    if not request.text.strip():
        raise HTTPException(status_code=400, detail="Text to synthesize cannot be empty.")
    
    # Use the fixed Kannada reference audio and text
    audio_buffer = synthesize_speech(
        text=request.text,
        ref_audio_name="KAN_F (Happy)",
        ref_text=kannada_example["ref_text"]
    )
    
    return StreamingResponse(
        audio_buffer,
        media_type="audio/wav",
        headers={"Content-Disposition": "attachment; filename=synthesized_kannada_speech.wav"}
    )

# Root endpoint with basic info
@app.get("/")
async def root():
    return {
        "message": "Welcome to IndicF5 Text-to-Speech API",
        "description": "High-quality TTS for Indian languages with output in Kannada. Provide Kannada text for synthesis.",
        "endpoints": {
            "/synthesize": "General synthesis with customizable reference audio",
            "/synthesize_kannada": "Kannada-specific synthesis using KAN_F (Happy) as reference"
        },
        "available_ref_audio_names": [ex["audio_name"] for ex in EXAMPLES],
        "example_synth_texts_in_kannada": {ex["audio_name"]: ex["synth_text"] for ex in EXAMPLES}
    }

# Run the app
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)