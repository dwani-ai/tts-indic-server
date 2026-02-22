# core/managers.py
import torch
from transformers import AutoModel
from logging_config import logger
from config.settings import Settings
from utils.device_utils import setup_device


# Device setup
device, torch_dtype = setup_device()

# Initialize settings
settings = Settings()

# Manager Registry
class ManagerRegistry:
    def __init__(self):
        self.tts_manager = None

# Singleton registry instance
registry = ManagerRegistry()

# Updated TTS Manager
class TTSManager:
    def __init__(self, device_type=device):
        self.device_type = device_type
        self.model = None
        self.repo_id = "ai4bharat/IndicF5"

    def load(self):
        if not self.model:
            logger.info("Loading TTS model IndicF5...")
            try:
                self.model = AutoModel.from_pretrained(
                    self.repo_id,
                    trust_remote_code=True
                )
                self.model = self.model.to(self.device_type)
                logger.info("TTS model IndicF5 loaded")
            except Exception as e:
                logger.error(f"Failed to load TTS model: {str(e)}")
                raise

    def synthesize(self, text, ref_audio_path, ref_text):
        if not self.model:
            raise ValueError("TTS model not loaded")
        return self.model(text, ref_audio_path=ref_audio_path, ref_text=ref_text)


def initialize_managers():

    registry.tts_manager = TTSManager()

    