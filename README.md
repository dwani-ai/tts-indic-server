# TTS Indic Server

## Overview
Text to Speech (TTS) for Indian languages using [ai4bharat/IndicF5](https://huggingface.co/ai4bharat/IndicF5)  model.

## Table of Contents
- [Getting Started - Development](#getting-started---development)
  - [For Development (Local)](#for-development-local)
- [Downloading Indic TTS Model](#downloading-indic-tts-model)
- [Running with FastAPI Server](#running-with-fastapi-server)
- [Evaluating Results](#evaluating-results)
  - [Examples](#examples)
    - [Kannada](#kannada)
    - [Hindi](#hindi)
  - [Specifying a Different Format](#specifying-a-different-format)
  - [Playing Back the Audio](#playing-back-the-audio)
  - [Describing the Voice](#describing-the-voice)
    - [Describing the Voice with Specific Speaker - Suresh](#describing-the-voice-with-specific-speaker---suresh)
    - [Describing the Voice with Specific Speaker - Anu](#describing-the-voice-with-specific-speaker---anu)
- [Building Docker Image](#building-docker-image)
- [Run the Docker Image](#run-the-docker-image)
- [Available Speakers](#available-speakers)
- [Tips](#tips)
- [Description Examples](#description-examples)
- [Citations](#citations)


## Getting Started

### For Development (Local)
- **Prerequisites**: Python 3.10, Ubuntu 22.04
- **Steps**:
  1. **Create a virtual environment**:
  ```bash
  python3.10 -m venv venv
  ```
  2. **Activate the virtual environment**:
  ```bash
  source venv/bin/activate
  ```
  On Windows, use:
  ```bash
  venv\Scripts\activate
  ```
  3. **Install dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

## Downloading Indic TTS Model


Login to HuggingFace account 

Request permission for access from https://huggingface.co/ai4bharat/IndicF5

export HF_TOKEN="HF_TOKEN_Account"

```bash download_model.sh
huggingface_cli download ai4bharat/IndicF5
```

### Local Model Run
```python
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
```

- Or Run the python code
```bash
cd src/server
python tts_indic_f5.py
```

<!-- 
 docker build -t dwani/tts-indic-server:latest .


sudo docker run --runtime nvidia -it --rm -p 7864:7864 slabstech/dwani-tts

sudo docker run --runtime nvidia -it --rm -p 7864:7864 -e HF_TOKEN=$HF_TOKEN slabstech/dwani-tts

sudo docker run --runtime nvidia -it --rm \
    -p 7864:7864 \
    -v ~/hf-cache:/data/hf-cache \
    -e HF_HOME=/data/hf-cache \
    -e HF_TOKEN=$HF_TOKEN \
    slabstech/dwani-tts

-->

#### Running with FastAPI Server


Run the server using FastAPI with the desired language (e.g., Kannada):
- With GPU
  ```bash
    python src/server/main.py --host 0.0.0.0 --port 10804
  ```

### Evaluating Results
You can evaluate the TTS generation results using `curl` commands. Below are examples for Kannada audio samples.

#### Kannada

```bash kannada_example.sh
curl -X 'POST' \
  'http://localhost:10804/v1/audio/speech' \
  -H 'accept: */*' \
  -H 'Content-Type: application/json' \
  -d '{
  "text": "ಉದ್ಯಾನದಲ್ಲಿ ಮಕ್ಕಳ ಆಟವಾಡುತ್ತಿದ್ದಾರೆ ಮತ್ತು ಪಕ್ಷಿಗಳು ಚಿಲಿಪಿಲಿ ಮಾಡುತ್ತಿವೆ."
}' -o kannada-tts-out.wav
```

With the default server (Phase 1 bfloat16), a sentence like the above typically returns in about 7–8 seconds on GPU.

**Faster inference (fewer NFE steps):** Start the server with `TTS_NFE_STEPS=16` to reduce latency (e.g. ~4–6 s per request instead of ~7–8 s):
  ```bash
  TTS_NFE_STEPS=16 python src/server/main.py --host 0.0.0.0 --port 10804
  ```
  There is a small quality trade-off; try 24 if 16 is too low. The reference script (`tts_indic_f5.py --steps 16`) can be ~2 s for the same text because it runs a single batch; the API may chunk longer texts so end-to-end latency can be higher. For more speed options (ONNX, TensorRT-LLM), see [docs/indic-f5-tts-speed-plan.md](docs/indic-f5-tts-speed-plan.md).

#### Hindi

```bash hindi_example.sh
curl -X 'POST' \
  'http://localhost:10804/v1/audio/speech' \
  -H 'accept: */*' \
  -H 'Content-Type: application/json' \
  -d '{
  "text": "अरे, तुम आज कैसे हो?"
}' -o hindi-tts-output.wav
```


### Specifying a Different Format

```bash specify_format.sh
curl -s -H "content-type: application/json" localhost:7860/v1/audio/speech -d '{"input": "Hey, how are you?", "response_type": "wav"}' -o audio.wav
```


## Contributing

We welcome contributions! Please read the [CONTRIBUTING.md](CONTRIBUTING.md) file for guidelines on how to contribute to this project.

Also you can join the [discord group](https://discord.gg/WZMCerEZ2P) to collaborate

- Reference
    - [ai4bharat/indic-parler-tts](https://huggingface.co/ai4bharat/indic-parler-tts)

## Citations

```bibtex citation_1.bib
@misc{lacombe-etal-2024-parler-tts,
  author = {Yoach Lacombe and Vaibhav Srivastav and Sanchit Gandhi},
  title = {Parler-TTS},
  year = {2024},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/huggingface/parler-tts}}
}
```

```bibtex citation_2.bib
@misc{lyth2024natural,
  title = {Natural language guidance of high-fidelity text-to-speech with synthetic annotations},
  author = {Dan Lyth and Simon King},
  year = {2024},
  eprint = {2402.01912},
  archivePrefix = {arXiv},
  primaryClass = {cs.SD}
}
```


```bibtex
@misc{AI4Bharat_IndicF5_2025,
  author       = {Praveen S V and Srija Anand and Soma Siddhartha and Mitesh M. Khapra},
  title        = {IndicF5: High-Quality Text-to-Speech for Indian Languages},
  year         = {2025},
  url          = {https://github.com/AI4Bharat/IndicF5},
}

```


<!--



## Alternate forms of Development  

- Check the torch.compile option for fast inference.
  - Suitable on Nvidia L4 GPU
  - Source for fast inference - [torch.compile](torch_compile.py) example

- Streaming example
  - [source code](tts_streaming.py)




## Usage

### How to Use the Service

You can test the service using `curl` commands. Below are examples for both service modes:

#### Paused, On-demand, $.05 /hour, Fast

```bash kannada_example.sh
curl -X 'POST' \
  'https://gaganyatri-tts-indic-server.hf.space/v1/audio/speech' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{"input": "ಉದ್ಯಾನದಲ್ಲಿ ಮಕ್ಕಳ ಆಟವಾಡುತ್ತಿದ್ದಾರೆ ಮತ್ತು ಪಕ್ಷಿಗಳು ಚಿಲಿಪಿಲಿ ಮಾಡುತ್ತಿವೆ.", "voice": "A female speaker delivers a slightly expressive and animated speech with a moderate speed and pitch. The recording is of very high quality, with the speakers voice sounding clear and very close up."}'  -o audio_kannada_gpu_cloud.mp3
```
-->

<!-- 
  - [For Production (Docker)](#for-production-docker)
### For Production (Docker)
- **Prerequisites**: Docker and Docker Compose
- **Steps**:
  1. **Start the server**:
  For GPU
  ```bash
  docker compose -f compose.yaml up -d
  ```
  For CPU only
  ```bash
  docker compose -f cpu-compose.yaml up -d
  ```


## Building Docker Image
Build the Docker image locally:
```bash
docker build -t slabstech/tts_indic_server -f Dockerfile .
```

### Run the Docker Image
```bash
docker run --gpus all -it --rm -p 7860:7860 slabstech/tts_indic_server
```

-->