# routes/speech.py
from fastapi import APIRouter, HTTPException, Depends
from starlette.responses import StreamingResponse
import io
import tempfile
import torch
import torchaudio
import soundfile as sf
import numpy as np
from logging_config import logger
from config.constants import LANGUAGE_TO_SCRIPT
from utils.tts_utils import load_audio_from_url, synthesize_speech, SynthesizeRequest, KannadaSynthesizeRequest, EXAMPLES
from core.dependencies import get_tts_manager,  get_model_manager, get_settings

router = APIRouter(prefix="/v1", tags=["speech"])

@router.post("/audio/speech", response_class=StreamingResponse)
async def synthesize_kannada(
    request: KannadaSynthesizeRequest,
    tts_manager=Depends(get_tts_manager)
):
    if not tts_manager.model:
        raise HTTPException(status_code=503, detail="TTS model not loaded")
    kannada_example = next(ex for ex in EXAMPLES if ex["audio_name"] == "KAN_F (Happy)")
    if not request.text.strip():
        raise HTTPException(status_code=400, detail="Text to synthesize cannot be empty.")
    
    audio_buffer = synthesize_speech(
        tts_manager,
        text=request.text,
        ref_audio_name="KAN_F (Happy)",
        ref_text=kannada_example["ref_text"]
    )
    
    return StreamingResponse(
        audio_buffer,
        media_type="audio/wav",
        headers={"Content-Disposition": "attachment; filename=synthesized_kannada_speech.wav"}
    )

