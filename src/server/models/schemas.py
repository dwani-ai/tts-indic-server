# models/schemas.py
from pydantic import BaseModel, field_validator
from typing import List



class SynthesizeRequest(BaseModel):
    text: str
    ref_audio_name: str
    ref_text: str = None

class KannadaSynthesizeRequest(BaseModel):
    text: str