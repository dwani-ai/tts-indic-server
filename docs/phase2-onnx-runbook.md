# Phase 2: ONNX Runtime for IndicF5 (Runbook)

This runbook walks through running IndicF5 with ONNX Runtime on GPU for better speed. It is a stepping stone toward Phase 3 (TensorRT-LLM) which gives the largest gain (~4×).

## Prerequisites

- Ubuntu with NVIDIA GPU, CUDA 12.x
- HuggingFace access to `ai4bharat/IndicF5` (gated model)
- Enough disk space for ONNX models (~2–3 GB)

## Quick win: fewer NFE steps (no ONNX)

Before doing ONNX, you can get ~2× speedup with the current PyTorch server:

- **Server:** Start with `TTS_NFE_STEPS=16`:
  ```bash
  TTS_NFE_STEPS=16 python src/server/main.py --host 0.0.0.0 --port 10804
  ```
- **Reference script:** `python3 src/server/tts_indic_f5.py --steps 16`

Quality may be slightly lower; try and tune (e.g. 24) if needed.

---

## Phase 2: Export and run with ONNX

Existing ONNX export targets **SWivid F5-TTS**. IndicF5 uses the **same architecture** but different checkpoint and vocab. You need to adapt paths and optionally checkpoint layout.

### Step 1: Environment

```bash
# Optional: dedicated env
conda create -n indicf5-onnx python=3.10 -y
conda activate indicf5-onnx

# From repo root
pip install -r requirements.txt
pip install onnx onnxruntime-gpu
```

### Step 2: Get export code and adapt for IndicF5

1. Clone the ONNX export repo:
   ```bash
   git clone https://github.com/DakeQQ/F5-TTS-ONNX.git /tmp/F5-TTS-ONNX
   ```
2. Their export lives under `Export_ONNX/F5_TTS/`. You need to:
   - Point the export script to **IndicF5** weights. Options:
     - Use the HuggingFace snapshot path (after `huggingface-cli download ai4bharat/IndicF5`): e.g. `~/.cache/huggingface/hub/models--ai4bharat--IndicF5/snapshots/<revision>/` contains `model.safetensors` and `checkpoints/`.
     - Or load IndicF5 in Python, extract the underlying F5 `ema_model` state dict, save it, and point the export to that file.
   - Use the **Indic vocab** from the IndicF5 repo (`checkpoints/vocab.txt` in the HF snapshot).
3. In their `Export_F5.py` (or equivalent), set `use_fp16_transformer = True` for FP16 export (see DakeQQ README).
4. Run the export and confirm you get:
   - `F5_Preprocess.onnx`
   - `F5_Transformer.onnx`
   - `F5_Decode.onnx`

### Step 3: Run inference with ONNX Runtime (GPU)

- Use `onnxruntime-gpu` with `CUDAExecutionProvider`.
- Optional: use [I/O binding](https://onnxruntime.ai/docs/performance/tune-performance/io-binding.html) to keep tensors on GPU.
- Pipeline: preprocess (PyTorch or ONNX) → run `F5_Transformer.onnx` → decode (PyTorch or ONNX). You can replace only the Transformer at first.

### Step 4: Integrate into this repo (optional)

- Add a backend flag (e.g. `--backend onnx`) to `src/server/tts_indic_f5.py` and/or to the FastAPI server config.
- When `onnx` is selected, load the ONNX sessions and the same preprocess/vocoder logic; call the ONNX Transformer instead of the PyTorch model’s sample step.

### References

- [DakeQQ/F5-TTS-ONNX](https://github.com/DakeQQ/F5-TTS-ONNX) – export and ONNX Runtime usage
- [WGS-note/F5_TTS_Faster](https://github.com/WGS-note/F5_TTS_Faster) – ONNX export + TensorRT-LLM (~4× speedup)
- [indic-f5-tts-speed-plan.md](indic-f5-tts-speed-plan.md) – full Phase 2/3 plan

---

## Phase 3 (real-time target)

For **real-time** use (e.g. &lt;1 s for a short sentence), Phase 3 (TensorRT-LLM) is the target: see [indic-f5-tts-speed-plan.md](indic-f5-tts-speed-plan.md) and [F5_TTS_Faster](https://github.com/WGS-note/F5_TTS_Faster). Build takes ~3 h; result is ~4× faster than PyTorch (e.g. 3.2 s → 0.72 s on RTX 3090).
