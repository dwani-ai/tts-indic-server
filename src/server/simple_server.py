# tts_server.py - Complete single-file FastAPI TTS server
"""
IndicF5 Text-to-Speech API Server (Single File)
Pins model to safe commit b82d286220e3070e171f4ef4b4bd047b9a447c9a
"""

"""
curl -X POST "http://localhost:8000/tts/synthesize/file" \
  -H "Content-Type: application/json" \
  -d '{"text": "ಬೆಂಗಳೂರು ಕರ್ನಾಟಕ ರಾಜ್ಯದ ರಾಜಧಾನಿ ಆಗಿದೆ"}' \
  --output output.wav

"""

import os
import time
import torch
import numpy as np
import soundfile as sf
from pathlib import Path
from typing import Optional
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel
import uvicorn
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global TTS Manager (singleton-like)
class TTSManager:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = None
        self.repo_id = "ai4bharat/IndicF5"
        self.revision = "b82d286220e3070e171f4ef4b4bd047b9a447c9a"
        self.ref_audio_path = "prompts/KAN_F_HAPPY_00001.wav"
        self.ref_text = (
            "ನಮ್‌ ಫ್ರಿಜ್ಜಲ್ಲಿ  ಕೂಲಿಂಗ್‌ ಸಮಸ್ಯೆ ಆಗಿ ನಾನ್‌ ಭಾಳ ದಿನದಿಂದ ಒದ್ದಾಡ್ತಿದ್ದೆ, "
            "ಆದ್ರೆ ಅದ್ನೀಗ ಮೆಕಾನಿಕ್ ಆಗಿರೋ ನಿಮ್‌ ಸಹಾಯ್ದಿಂದ ಬಗೆಹರಿಸ್ಕೋಬೋದು ಅಂತಾಗಿ ನಿರಾಳ ಆಯ್ತು ನಂಗೆ."
        )

    def load_model(self):
        if self.model is not None:
            return
            
        logger.info(f"Loading IndicF5 from {self.repo_id}@{self.revision} on {self.device}...")
        from transformers import AutoModel
        
        try:
            self.model = AutoModel.from_pretrained(
                self.repo_id,
                trust_remote_code=True,
                revision=self.revision,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            )
            self.model = self.model.to(self.device)
            logger.info("✅ Model loaded successfully")
        except Exception as e:
            logger.error(f"❌ Failed to load model: {e}")
            raise

    def synthesize(self, text: str, ref_audio_path: Optional[str] = None, ref_text: Optional[str] = None):
        if self.model is None:
            raise ValueError("Model not loaded")
        
        ref_path = ref_audio_path or self.ref_audio_path
        ref_txt = ref_text or self.ref_text
        
        if not os.path.exists(ref_path):
            raise HTTPException(status_code=400, detail=f"Reference audio not found: {ref_path}")
        
        logger.info(f"🎤 Synthesizing: {text[:50]}...")
        start_time = time.time()
        
        with torch.no_grad():
            audio = self.model(text, ref_audio_path=ref_path, ref_text=ref_txt)
        
        # Normalize audio
        audio_np = np.array(audio, dtype=np.float32)
        if audio_np.dtype == np.int16:
            audio_np = audio_np.astype(np.float32) / 32768.0
        
        duration = time.time() - start_time
        logger.info(f"✅ Synthesis complete: {duration:.2f}s")
        
        return audio_np

# Initialize global manager
tts_manager = TTSManager()

# Pydantic models
class TTSRequest(BaseModel):
    text: str
    ref_audio_path: Optional[str] = None
    ref_text: Optional[str] = None

class TTSResponse(BaseModel):
    success: bool
    message: str
    duration: Optional[float] = None

# FastAPI app
app = FastAPI(
    title="IndicF5 TTS API",
    description="Single-file TTS server using AI4Bharat IndicF5",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("startup")
async def startup():
    try:
        tts_manager.load_model()
    except Exception as e:
        logger.error(f"Startup failed: {e}")

@app.get("/")
async def root():
    return {
        "message": "IndicF5 TTS API 🚀",
        "model": f"{tts_manager.repo_id}@{tts_manager.revision}",
        "device": tts_manager.device,
        "status": "ready" if tts_manager.model else "loading..."
    }

@app.post("/tts/synthesize", response_model=TTSResponse)
async def synthesize_speech(request: TTSRequest):
    """Generate speech from text"""
    try:
        audio = tts_manager.synthesize(
            text=request.text,
            ref_audio_path=request.ref_audio_path,
            ref_text=request.ref_text
        )
        
        # Save temp file
        timestamp = int(time.time())
        wav_path = f"temp_audio_{timestamp}.wav"
        sf.write(wav_path, audio, 24000)
        
        return TTSResponse(
            success=True,
            message=f"Generated {len(audio)/24000:.2f}s of audio",
            duration=len(audio)/24000
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/tts/synthesize/file")
async def synthesize_file(request: TTSRequest):
    """Generate and return audio file directly"""
    try:
        audio = tts_manager.synthesize(
            text=request.text,
            ref_audio_path=request.ref_audio_path,
            ref_text=request.ref_text
        )
        
        # Create in-memory WAV file
        timestamp = int(time.time())
        wav_path = f"temp_audio_{timestamp}.wav"
        sf.write(wav_path, audio, 24000)
        
        def iterfile():
            with open(wav_path, mode="rb") as file:
                yield from file
            os.unlink(wav_path)  # Cleanup
        
        return StreamingResponse(
            iterfile(),
            media_type="audio/wav",
            headers={"Content-Disposition": f"attachment; filename=tts_output_{timestamp}.wav"}
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/tts/health")
async def health():
    return {
        "status": "healthy",
        "model_loaded": tts_manager.model is not None,
        "device": tts_manager.device
    }

if __name__ == "__main__":
    # Create prompts directory if missing
    os.makedirs("prompts", exist_ok=True)
    print("🎵 IndicF5 TTS Server Starting...")
    print("🌐 Open http://localhost:8000/docs for interactive API")
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
