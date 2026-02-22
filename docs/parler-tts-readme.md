# Parler-TTS-Server

This repository provides a server with an [OpenAI compatible API](https://platform.openai.com/docs/api-reference/audio/createSpeech) interface for [Parler-TTS](https://github.com/huggingface/parler-tts).

## Table of Contents

- [Quick Start](#quick-start)
  - [Prerequisites](#prerequisites)
  - [Docker](#docker)
  - [Docker Compose](#docker-compose)
- [Usage](#usage)
  - [Examples](#examples)
    - [Kannada](#kannada)
    - [Hindi](#hindi)
  - [Saving to File](#saving-to-file)
  - [Specifying a Different Format](#specifying-a-different-format)
  - [Playing Back the Audio](#playing-back-the-audio)
  - [Describing the Voice](#describing-the-voice)
  - [OpenAI SDK Usage Example](#openai-sdk-usage-example)
- [Citations](#citations)
- [License](#license)

## Quick Start

### Prerequisites

- Docker
- cURL

### Docker

Run the server with the default model:
```bash
docker run --detach --volume ~/.cache/huggingface:/root/.cache/huggingface --publish 8000:8000 slabstech/parler-tts-server
```

Run the server with a fine-tuned model:
```bash
docker run --detach --volume ~/.cache/huggingface:/root/.cache/huggingface --publish 8000:8000 --env MODEL="ai4bharat/indic-parler-tts" slabstech/parler-tts-server
```

### Docker Compose

Download the `compose.yaml` file and start the server:
```bash
curl -sO https://raw.githubusercontent.com/sachinsshetty/parler-tts-server/refs/heads/master/compose.yaml
docker compose up --detach parler-tts-server
```

## Usage

### Examples

#### Kannada
```bash
curl -s -H "content-type: application/json" localhost:8000/v1/audio/speech -d '{"input": "ಉದ್ಯಾನದಲ್ಲಿ ಮಕ್ಕಳ ಆಟವಾಡುತ್ತಿದ್ದಾರೆ ಮತ್ತು ಪಕ್ಷಿಗಳು ಚಿಲಿಪಿಲಿ ಮಾಡುತ್ತಿವೆ."}' -o audio_kannada.mp3
```


#### Hindi
```bash
curl -s -H "content-type: application/json" localhost:8000/v1/audio/speech -d '{"input": "अरे, तुम आज कैसे हो?"}' -o audio_hindi.mp3
```

### Saving to File
```bash
curl -s -H "content-type: application/json" localhost:8000/v1/audio/speech -d '{"input": "Hey, how are you?"}' -o audio.mp3
```

### Specifying a Different Format
```bash
curl -s -H "content-type: application/json" localhost:8000/v1/audio/speech -d '{"input": "Hey, how are you?", "response_type": "wav"}' -o audio.wav
```

### Playing Back the Audio
```bash
curl -s -H "content-type: application/json" localhost:8000/v1/audio/speech -d '{"input": "Hey, how are you?"}' | ffplay -hide_banner -autoexit -nodisp -loglevel quiet -
```

### Describing the Voice
```bash
curl -s -H "content-type: application/json" localhost:8000/v1/audio/speech -d '{"input": "Hey, how are you?", "voice": "Feminine, speedy, and cheerful"}' | ffplay -hide_banner -autoexit -nodisp -loglevel quiet -
```

### OpenAI SDK Usage Example
An example of using the OpenAI SDK can be found [here](./examples/openai_sdk.py).

## Citations

```bibtex
@misc{lacombe-etal-2024-parler-tts,
  author = {Yoach Lacombe and Vaibhav Srivastav and Sanchit Gandhi},
  title = {Parler-TTS},
  year = {2024},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/huggingface/parler-tts}}
}
```

```bibtex
@misc{lyth2024natural,
  title = {Natural language guidance of high-fidelity text-to-speech with synthetic annotations},
  author = {Dan Lyth and Simon King},
  year = {2024},
  eprint = {2402.01912},
  archivePrefix = {arXiv},
  primaryClass = {cs.SD}
}
```

## License

This project is licensed under the MIT License. See the [LICENSE](./LICENSE) file for details.