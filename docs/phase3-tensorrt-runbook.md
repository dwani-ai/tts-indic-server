# Phase 3: TensorRT-LLM for IndicF5 (Runbook)

This runbook covers **Phase 3 only** (skip Phase 2 ONNX). Goal: replace the F5 **Transformer** with a TensorRT-LLM engine for ~4× speedup (e.g. 3 s → 0.7 s on RTX 3090). Preprocess and vocoder stay in PyTorch.

**Reference:** [WGS-note/F5_TTS_Faster](https://github.com/WGS-note/F5_TTS_Faster). Build can take ~3 h; run in **tmux** on an Ubuntu machine with NVIDIA GPU.

---

## Prerequisites

- Ubuntu 22.04, NVIDIA GPU, CUDA 12.x, enough disk (~10 GB for env + engine)
- HuggingFace access to `ai4bharat/IndicF5` (gated)
- This repo and F5_TTS_Faster repo

---

## Step 1: Export IndicF5 checkpoint for TensorRT-LLM

The F5_TTS_Faster converter expects a **PyTorch .pt** file with key `ema_model_state_dict` and keys like `ema_model.transformer.*`. IndicF5 is distributed as `model.safetensors` on HuggingFace.

From this repo (with venv active, from repo root):

```bash
# Download from HuggingFace (set HF_TOKEN if model is gated)
python scripts/export_indicf5_for_trtllm.py --output phase3_ckpts/indicf5_transformer.pt

# Or with explicit checkpoint path (e.g. after huggingface-cli download ai4bharat/IndicF5)
python scripts/export_indicf5_for_trtllm.py \
  --checkpoint ~/.cache/huggingface/hub/models--ai4bharat--IndicF5/snapshots/<revision>/model.safetensors \
  --output phase3_ckpts/indicf5_transformer.pt
```

This produces `indicf5_transformer.pt` containing only the transformer weights in the format expected by `convert_checkpoint.py`.

---

## Step 2: Environment for TensorRT-LLM

Use a **dedicated env** (TensorRT-LLM has specific versions).

```bash
conda create -n f5_tts_faster python=3.10 -y
conda activate f5_tts_faster
```

Install PyTorch with CUDA 12.4 (match F5_TTS_Faster):

```bash
conda install pytorch==2.5.0 torchvision==0.20.0 torchaudio==2.5.0 pytorch-cuda=12.4 -c pytorch -c nvidia -y
```

System deps for TensorRT-LLM:

```bash
sudo apt-get -y install libopenmpi-dev
```

Optional (for OpenMPI build):

```bash
conda install -c conda-forge ffmpeg cmake openmpi -y
export OMPI_CC=$(which gcc)
export OMPI_CXX=$(which g++)
```

Install TensorRT-LLM (version used by F5_TTS_Faster):

```bash
pip install tensorrt_llm==0.15.0
```

Verify:

```bash
python -c "import tensorrt_llm; import tensorrt_llm.bindings"
```

Install F5_TTS_Faster dependencies (from their repo):

```bash
git clone https://github.com/WGS-note/F5_TTS_Faster.git /tmp/F5_TTS_Faster
pip install -r /tmp/F5_TTS_Faster/requirements.txt
```

(Adjust versions if you hit conflicts; you need `safetensors`, `torch`, etc.)

---

## Step 3: Add F5TTS model to TensorRT-LLM

TensorRT-LLM needs a model definition for F5. Copy it from F5_TTS_Faster into the installed `tensorrt_llm` package:

```bash
# Find TensorRT-LLM site-packages
TRTLLM_DIR=$(python -c "import tensorrt_llm; print(tensorrt_llm.__path__[0])")
mkdir -p "$TRTLLM_DIR/models/f5tts"
cp -r /tmp/F5_TTS_Faster/export_trtllm/model/* "$TRTLLM_DIR/models/f5tts/"
```

Register the model in TensorRT-LLM:

Edit `$TRTLLM_DIR/models/__init__.py`:

- Add: `from .f5tts.model import F5TTS`
- Add `'F5TTS': F5TTS` to `MODEL_MAP` (or the dict used for model lookup)

(Exact names depend on the TensorRT-LLM version; check existing entries in that file.)

---

## Step 4: Convert checkpoint to TensorRT-LLM format

Use F5_TTS_Faster’s `convert_checkpoint.py` with the **IndicF5** .pt from Step 1:

```bash
cd /tmp/F5_TTS_Faster
python export_trtllm/convert_checkpoint.py \
  --timm_ckpt /path/to/tts-indic-server/phase3_ckpts/indicf5_transformer.pt \
  --output_dir ./ckpts/trtllm_indicf5 \
  --hidden_size 1024 \
  --depth 22 \
  --num_heads 16 \
  --dtype float16
```

If the model was trained in fp32, use `--dtype float32`. The script expects transformer keys; our export script produces the right format.

---

## Step 5: Build the TensorRT-LLM engine

```bash
trtllm-build --checkpoint_dir ./ckpts/trtllm_indicf5 \
  --remove_input_padding disable \
  --bert_attention_plugin disable \
  --output_dir ./ckpts/engine_indicf5
```

If you see dtype errors (e.g. fp16 vs fp32), adjust the default in `tensorrt_llm/parameter.py` as noted in F5_TTS_Faster (e.g. `_DEFAULT_DTYPE`).

---

## Step 6: Run fast inference (F5_TTS_Faster sample)

F5_TTS_Faster’s `sample.py` runs the full pipeline (preprocess → TRT-LLM Transformer → vocoder). You need to point it at the **IndicF5** vocab and ref files, and the engine we built:

```bash
cd /tmp/F5_TTS_Faster
python export_trtllm/sample.py --tllm_model_dir ./ckpts/engine_indicf5
```

Their script may expect their own checkpoint paths and vocab; you may need to copy or adapt `sample.py` to use:

- Vocab: IndicF5’s `checkpoints/vocab.txt` (from HF snapshot)
- Ref audio/text: your own or from IndicF5 prompts
- Engine dir: `./ckpts/engine_indicf5`

---

## Step 7: Integrate into this server (optional)

To use the TRT-LLM engine from the FastAPI server:

1. Add a backend option (e.g. `TTS_BACKEND=trtllm` or `--backend trtllm`).
2. At startup, if backend is `trtllm`, load the TensorRT-LLM engine once (path to `engine_indicf5`).
3. For each request: run existing preprocess (ref_audio, ref_text, text → conditioning), run the **Transformer** step via the TRT-LLM engine instead of PyTorch, then run the existing vocoder (Vocos) and return audio.
4. Keep the existing PyTorch path when `TTS_BACKEND` is not `trtllm`.

The heavy part to replace is the ODE sampling (the DiT/Transformer forward); preprocess and vocoder can stay as in `f5_tts/infer/utils_infer.py`.

---

## Checklist

- [ ] Export IndicF5 transformer to `.pt` with `scripts/export_indicf5_for_trtllm.py`
- [ ] Create conda env, install PyTorch 2.5 + CUDA 12.4, TensorRT-LLM 0.15.0, F5_TTS_Faster deps
- [ ] Copy F5TTS model definition into `tensorrt_llm/models/f5tts/` and register in `__init__.py`
- [ ] Run `convert_checkpoint.py` with IndicF5 .pt → `ckpts/trtllm_indicf5`
- [ ] Run `trtllm-build` → `ckpts/engine_indicf5`
- [ ] Run `sample.py` (or adapted script) with IndicF5 vocab and engine
- [ ] (Optional) Add `trtllm` backend to this server and benchmark

---

## References

- [F5_TTS_Faster](https://github.com/WGS-note/F5_TTS_Faster)
- [NVIDIA TensorRT-LLM](https://github.com/NVIDIA/TensorRT-LLM)
- [indic-f5-tts-speed-plan.md](indic-f5-tts-speed-plan.md) (full plan)
