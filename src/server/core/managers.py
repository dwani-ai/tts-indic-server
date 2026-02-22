# core/managers.py
# Phase 1 speedup: bfloat16 on GPU (see docs/indic-f5-tts-speed-plan.md).
import torch
from transformers import AutoModel
from logging_config import logger
from config.settings import Settings
from utils.device_utils import setup_device

# Device setup (bfloat16 on GPU for ~3.5x speedup vs float32)
device, torch_dtype = setup_device()

# Initialize settings
settings = Settings()


# Manager Registry
class ManagerRegistry:
    def __init__(self):
        self.tts_manager = None


# Singleton registry instance
registry = ManagerRegistry()


class TTSManager:
    def __init__(self, device_type=device):
        self.device_type = device_type
        self.model = None
        self.repo_id = "ai4bharat/IndicF5"
        # Pin to a known-safe commit you have reviewed
        self.revision = "b82d286220e3070e171f4ef4b4bd047b9a447c9a"

    def load(self):
        if self.model is not None:
            return

        logger.info(
            f"Loading TTS model IndicF5 from '{self.repo_id}' "
            f"at revision '{self.revision}' on device '{self.device_type}' ({torch_dtype})..."
        )
        try:
            self.model = AutoModel.from_pretrained(
                self.repo_id,
                trust_remote_code=True,  # required for IndicF5
                revision=self.revision,
                torch_dtype=torch_dtype,  # bfloat16 on GPU for Phase 1 speedup
            )
            self.model = self.model.to(self.device_type)
            logger.info("TTS model IndicF5 loaded successfully (%s)", torch_dtype)
        except Exception as e:
            logger.error(f"Failed to load TTS model IndicF5: {e}")
            self.model = None
            raise

    def synthesize(self, text, ref_audio_path, ref_text):
        if self.model is None:
            raise ValueError("TTS model not loaded. Call load() before synthesize().")

        # You can add logging and basic validation here if you want
        logger.debug(
            f"Synthesizing TTS with text length={len(text)}, "
            f"ref_audio_path='{ref_audio_path}'"
        )

        with torch.no_grad():
            audio = self.model(
                text,
                ref_audio_path=ref_audio_path,
                ref_text=ref_text,
            )
        return audio


def initialize_managers():
    registry.tts_manager = TTSManager()
