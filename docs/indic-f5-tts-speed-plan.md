# IndicF5 TTS Speed Improvement Plan (Ubuntu + NVIDIA GPU)

This document provides a **step-by-step detailed plan** to achieve faster speech generation with the IndicF5 model (ai4bharat/IndicF5), based on the sources listed in [indic-f5-tts-speed.md](indic-f5-tts-speed.md). The plan is ordered from **quick wins** to **maximum speedup**, so you can stop at the level that fits your effort vs. gain trade-off.

---

## Reference implementation

**All changes are to be made on top of:** [`src/server/tts_indic_f5.py`](../src/server/tts_indic_f5.py).

That script is the canonical reference. It:

1. Sets **config**: `REPO_ID = "ai4bharat/IndicF5"`, `REVISION`, `DEVICE` (cuda/cpu).
2. Loads the model: `AutoModel.from_pretrained(REPO_ID, trust_remote_code=True, revision=REVISION)` then `model.to(DEVICE)`.
3. Generates speech: `with torch.no_grad(): audio = model(TEXT_TO_SYNTH, ref_audio_path=REF_AUDIO_PATH, ref_text=REF_TEXT)`.
4. Normalizes and saves: convert to float32 (handle int16), `sf.write(OUTPUT_WAV, audio, samplerate=OUTPUT_SR)`.

Speed-related changes (torch.compile, dtype, NFE, ONNX/TRT backends) should be implemented in this script first; the same logic can then be mirrored into `src/server/core/managers.py` for the FastAPI server.

---

## Current Setup Summary

- **Model**: `ai4bharat/IndicF5` loaded via `transformers.AutoModel.from_pretrained(..., trust_remote_code=True)`.
- **Reference script**: [`src/server/tts_indic_f5.py`](../src/server/tts_indic_f5.py) (standalone); **server**: `src/server/core/managers.py` — `TTSManager` uses the same load/call pattern.
- **Device**: `DEVICE = "cuda" if torch.cuda.is_available() else "cpu"` in the reference script; server uses `bfloat16` on GPU via `utils/device_utils.py`.
- **F5 backbone**: Flow-matching TTS with **NFE steps** (e.g. 32 in f5_tts infer), **cfg_strength**, and a **vocoder** (e.g. Vocos). IndicF5 is based on [F5-TTS](https://github.com/SWivid/F5-TTS); same architecture, Indian-language checkpoint and vocab.

---

## Source Summary (from indic-f5-tts-speed.md)

| Source | Approach | Reported gain | Effort |
|--------|----------|---------------|--------|
| [F5_TTS_Faster](https://github.com/WGS-note/F5_TTS_Faster) | ONNX export → TensorRT-LLM for Transformer | ~4× (e.g. 3.2s → 0.72s on RTX 3090) | High |
| [SWivid F5-TTS triton_trtllm](https://github.com/SWivid/F5-TTS/blob/main/src/f5_tts/runtime/triton_trtllm/README.md) | Triton + TensorRT-LLM engines | RTF 0.04 (TRT) vs 0.15 (PyTorch) (~3.7×) | High |
| [DakeQQ/F5-TTS-ONNX](https://github.com/DakeQQ/F5-TTS-ONNX) | ONNX Runtime (CUDA), FP16, I/O binding | Moderate (no exact IndicF5 number) | Medium |
| [F5-TTS infer](https://github.com/SWivid/F5-TTS/tree/main/src/f5_tts/infer) | Lower NFE (e.g. 16), same PyTorch path | Fewer steps = faster, slight quality trade-off | Low |
| [TTS-StressTest](https://github.com/WGS-note/TTS-StressTest) | Benchmarking | For measuring before/after | Low |

---

## Phase 1: Quick Wins (No New Infra)

**Goal**: Faster inference with the **existing** PyTorch + Transformers pipeline on Ubuntu with an NVIDIA GPU.

### Step 1.1: Enable `torch.compile` (reduce-overhead)

- Your codebase already uses `torch.compile` for the Parler-TTS path (e.g. `src/misc/others/torch_compile.py`, `reduce-overhead` mode). Apply the same idea to IndicF5.
- **Where**: In **`src/server/tts_indic_f5.py`** (reference), after `model = model.to(DEVICE)`, add compile and warmup. Mirror the same logic in `src/server/core/managers.py` for the server.
- **Actions** (in `tts_indic_f5.py`):
  1. After `model = model.to(DEVICE)`, add:
     - Optional guard: only compile when `DEVICE.startswith("cuda")` (and optionally an env var like `TTS_COMPILE=1`), so CPU/local dev stays simple.
     - `model.forward = torch.compile(model.forward, mode="reduce-overhead")`.
  2. Run **warmup** before the real "Generating speech..." call: one or two `model(TEXT_TO_SYNTH, ref_audio_path=REF_AUDIO_PATH, ref_text=REF_TEXT)` (or a short dummy text) under `torch.no_grad()` so the first real request doesn’t pay compilation cost.
- **Caveat**: First call after compile can be slow; warmup is important. If you see recompilation issues, use `TORCH_LOGS="recompiles"` to debug (see [PyTorch compiler troubleshooting](https://pytorch.org/docs/stable/torch.compiler_troubleshooting.html)).
- **Reference**: [README.md](../README.md) (torch.compile) and [indic-parler-tts-speed.md](indic-parler-tts-speed.md) (L4 reduce-overhead helped).

### Step 1.2: Expose and reduce NFE steps (if API allows)

- F5 uses a flow-matching ODE; fewer steps = faster, with a possible small quality trade-off. The f5_tts infer uses `nfe_step = 32` (or 16); see `src/server/f5_tts/infer/utils_infer.py`.
- **Check**: Whether the HuggingFace IndicF5 model’s `forward()` (from `trust_remote_code`) accepts a `steps` or similar argument. If it does:
  - Add an optional parameter to your API (e.g. `steps=16` for “fast” mode).
  - Run A/B tests (e.g. 16 vs 32) for latency and quality on your target hardware.
- **If the HF model does not expose steps**: You can still try loading IndicF5 **via the original F5-TTS inference path** (same architecture, IndicF5 checkpoint + Indic vocab). [IndicF5 HF discussion](https://huggingface.co/ai4bharat/IndicF5/discussions/1) and [GitHub issue #10](https://github.com/AI4Bharat/IndicF5/issues/10) mention using the model with the original F5-TTS; that path typically exposes `steps` and `cfg_strength` in `infer_process()`.

### Step 1.3: Ensure GPU memory and dtype

- Confirm `torch_dtype` is `bfloat16` on GPU (already in `device_utils.py`). No change needed unless you add a float32 fallback for debugging.
- Ensure enough GPU memory so the model and a small batch don’t trigger CPU offload or OOM (which would slow things down).

**Deliverables**: Optional config for `torch.compile`, warmup on startup, and (if available) a “fast” mode with fewer NFE steps. Measure latency (e.g. time-to-first-byte and end-to-end) before/after with a fixed test script or [TTS-StressTest](https://github.com/WGS-note/TTS-StressTest).

### Phase 1 benchmark results (implemented)

Using **bfloat16 + warmup** (no torch.compile; compile is off for IndicF5). Same text/ref, 5.79 s audio, 2 warmup + 3 timed runs.

| Config   | time (avg) | time (min) | RTF    |
|----------|------------|------------|--------|
| bfloat16 | 3.67 s     | 3.67 s     | 0.634  |
| float32  | 12.73 s    | 12.69 s    | 2.198  |

**Speedup: ~3.5×** (bfloat16 vs float32). Run: `python3 tts_indic_f5.py` (default bfloat16) vs `python3 tts_indic_f5.py --float32`, or `bash scripts/bench_phase1_compile.sh`.

---

## Phase 2: ONNX Runtime (GPU) Path

**Goal**: Run the F5 Transformer (and optionally preprocess/decode) with ONNX Runtime using CUDA, for better GPU utilization and a path toward TensorRT-LLM later.

**Important**: Existing export scripts target **SWivid F5-TTS** checkpoints. IndicF5 uses the **same architecture** but different **checkpoint and vocab**. You will need to either adapt the export for IndicF5 or confirm compatibility (e.g. same tensor names and shapes).

### Step 2.1: Environment (Ubuntu + NVIDIA GPU)

- Python 3.10 recommended (matches F5_TTS_Faster / ONNX repos).
- Create a dedicated env, e.g. `conda create -n indicf5-onnx python=3.10 -y` and activate.
- Install:
  - PyTorch with CUDA (e.g. 2.5.x or match your current `requirements.txt`).
  - `onnx`, `onnxruntime-gpu` (or the version used by [DakeQQ/F5-TTS-ONNX](https://github.com/DakeQQ/F5-TTS-ONNX)).
  - F5-TTS / IndicF5 dependencies (transformers, etc.) so you can load the model and run the export script.

### Step 2.2: Export IndicF5 to ONNX

- Clone or reference [DakeQQ/F5-TTS-ONNX](https://github.com/DakeQQ/F5-TTS-ONNX). Their export produces:
  - `F5_Preprocess.onnx`
  - `F5_Transformer.onnx`
  - `F5_Decode.onnx`
- **Adaptation for IndicF5**:
  1. Point the export script to **IndicF5** weights and **Indic vocab** (from ai4bharat/IndicF5 repo or HuggingFace).
  2. If the export script assumes SWivid F5-TTS checkpoint layout, adjust config/keys to match IndicF5 (same architecture, so likely minimal changes).
  3. Enable FP16 for the Transformer if supported (DakeQQ mentions `use_fp16_transformer = True` in `Export_F5.py` line 21); verify no silence output with FP16.
- Run export and confirm the three ONNX files are generated and that input/output names and shapes match what you expect for your inputs (ref_audio, ref_text, text).

### Step 2.3: Run inference with ONNX Runtime (GPU)

- Use `onnxruntime-gpu` with `CUDAExecutionProvider`.
- Optional: use **I/O binding** (as in DakeQQ README) to keep tensors on GPU and reduce transfers.
- Keep **preprocess** and **decode** (vocoder) in PyTorch or ONNX as in the reference; only the heavy Transformer part may need to be ONNX for a first iteration.
- Integrate this path into your server: e.g. a flag or config to choose “PyTorch” vs “ONNX” backend, then call the ONNX pipeline when enabled. Prefer adding a backend option in **`tts_indic_f5.py`** first, then mirror in `TTSManager`.

### Step 2.4: Benchmark

- Compare end-to-end latency and quality (same text + ref_audio + ref_text) for:
  - Current PyTorch (and after Phase 1),
  - ONNX GPU path.
- Use a fixed test set and, if possible, [TTS-StressTest](https://github.com/WGS-note/TTS-StressTest) for repeatability.

**Deliverables**: Scripts to export IndicF5 to ONNX, a small inference script using ONNX Runtime GPU (and optionally I/O binding), and latency/quality numbers vs baseline.

---

## Phase 3: TensorRT-LLM Acceleration (Maximum Speedup)

**Goal**: Replace the F5 **Transformer** with a TensorRT-LLM engine to achieve the ~4× speedup reported by [F5_TTS_Faster](https://github.com/WGS-note/F5_TTS_Faster) (e.g. 3.2s → 0.72s on RTX 3090). Preprocess and vocoder can remain ONNX/PyTorch.

**Note**: Build and conversion are heavy (~3h build mentioned in F5_TTS_Faster). Recommended to run in a **tmux** session on an Ubuntu machine with NVIDIA GPU and sufficient disk space.

### Step 3.1: Environment (Ubuntu + NVIDIA GPU)

- **OS**: Ubuntu (e.g. 22.04) with NVIDIA drivers and CUDA 12.x.
- **Conda env**: e.g. `conda create -n f5_tts_faster python=3.10 -y` and activate.
- Install:
  - PyTorch 2.5.x with CUDA 12.4 (or match TensorRT-LLM’s requirements), e.g.:
    - `conda install pytorch==2.5.0 torchvision==0.20.0 torchaudio==2.5.0 pytorch-cuda=12.4 -c pytorch -c nvidia -y`
  - System: `sudo apt-get -y install libopenmpi-dev` (for TensorRT-LLM).
  - For OpenMPI build: `conda install -c conda-forge ffmpeg cmake openmpi` and set `OMPI_CC`/`OMPI_CXX` to your `gcc`/`g++` if needed.
  - Pip: install dependencies from F5_TTS_Faster’s `requirements.txt` (and f5-tts if needed for export).
  - TensorRT-LLM: e.g. `pip install tensorrt_llm==0.15.0` (version used in F5_TTS_Faster; confirm compatibility with your CUDA/cuDNN).

### Step 3.2: Export F5 to ONNX (same as Phase 2)

- Use F5_TTS_Faster’s `export_onnx/Export_F5.py` (or DakeQQ’s export adapted for IndicF5).
- If you use F5_TTS_Faster’s export, note their warning: exporting modifies F5/vocos code, so you may need to reinstall vocos/f5-tts after export if you still need the original PyTorch path.
- Output: `F5_Preprocess.onnx`, `F5_Transformer.onnx`, `F5_Decode.onnx` (or equivalent for IndicF5).

### Step 3.3: Convert checkpoint to TensorRT-LLM format

- F5_TTS_Faster provides `export_trtllm/convert_checkpoint.py`. You need a **TensorRT-LLM model definition** for F5 (they use `tensorrt_llm/models/f5tts/`).
- **For IndicF5**:
  1. Use the same F5 Transformer architecture in TensorRT-LLM as in F5_TTS_Faster (they copy `export_trtllm/model/*` into `tensorrt_llm/models/f5tts/`).
  2. Run `convert_checkpoint.py` with **IndicF5 checkpoint** (and correct dtype if you trained in fp32: `--dtype float32`).
  3. Output: a checkpoint directory that TensorRT-LLM’s build step can consume (e.g. `./ckpts/trtllm_ckpt`).

### Step 3.4: Build TensorRT-LLM engine

- Install TensorRT-LLM so that `tensorrt_llm/models` is available (e.g. in site-packages). Add the F5TTS model module under `tensorrt_llm/models/f5tts/` (model.py, modules.py) and register it in `tensorrt_llm/models/__init__.py` (e.g. `from .f5tts.model import F5TTS` and add to `MODEL_MAP`).
- Run the **build** step (F5_TTS_Faster uses `trtllm-build`), e.g.:
  - `trtllm-build --checkpoint_dir ./ckpts/trtllm_ckpt --remove_input_padding disable --bert_attention_plugin disable --output_dir ./ckpts/engine_outputs`
- If you hit dtype errors (e.g. fp16 vs fp32), adjust default dtype in TensorRT-LLM’s parameter module as noted in F5_TTS_Faster (e.g. `_DEFAULT_DTYPE` in `tensorrt_llm/parameter.py`).
- **Optional**: Tensor parallelism for multi-GPU (`--tp_size`).

### Step 3.5: Run fast inference

- Use F5_TTS_Faster’s `export_trtllm/sample.py` (or equivalent) pointing to the built engine directory (`--tllm_model_dir ./ckpts/engine_outputs`).
- Pipeline: preprocess (ONNX or PyTorch) → TensorRT-LLM Transformer → decode (ONNX or PyTorch vocoder). Replace only the Transformer forward with the TRT-LLM engine in your inference code.

### Step 3.6: Integrate into your server

- Add a “backend” option (e.g. `pytorch` | `onnx` | `trtllm`). When `trtllm` is selected:
  - Load the TRT-LLM engine once at startup.
  - For each request: run preprocess → TRT-LLM step → decode, then return audio.
- Keep Phase 1 (torch.compile) and Phase 2 (ONNX) available for environments where TensorRT-LLM is not installed or not desired.

### Step 3.7: Benchmark and validate

- Measure latency (and RTF if available) for the same test set. Compare:
  - PyTorch baseline (and Phase 1),
  - ONNX (Phase 2),
  - TensorRT-LLM (Phase 3).
- Check quality (subjective or objective) at each stage so that speed gains don’t come with unacceptable quality loss.

**Deliverables**: TensorRT-LLM engine build for IndicF5 (or F5-compatible) Transformer, sample inference script, and optional integration behind a backend flag in your FastAPI server; benchmark report.

---

## Phase 4: Triton Inference Server (Production-style)

**Goal**: Serve the model using [NVIDIA Triton](https://github.com/triton-inference-server) with TensorRT-LLM backends, as in [SWivid F5-TTS triton_trtllm](https://github.com/SWivid/F5-TTS/blob/main/src/f5_tts/runtime/triton_trtllm/README.md). This is most relevant for **multi-instance**, **scaling**, and **consistent latency** under load.

### Step 4.1: Build Triton + F5-TTS image

- Use the Dockerfile from F5-TTS:  
  `https://github.com/SWivid/F5-TTS/blob/main/src/f5_tts/runtime/triton_trtllm/Dockerfile.server`
- Build: e.g. `docker build . -f Dockerfile.server -t triton-f5-tts:24.12` (adjust path to the Dockerfile if you clone only a subdir).
- Run container with GPU and sufficient shared memory: e.g. `docker run -it --name f5-server --gpus all --net host -v /mnt:/mnt --shm-size=2g triton-f5-tts:24.12`.

### Step 4.2: Build TensorRT-LLM engines inside the container

- Follow the Triton README: use the provided `run.sh` to build F5-TTS TensorRT-LLM engines (e.g. `bash run.sh 0 4 F5TTS_v1_Base`). For **IndicF5**, use a **custom checkpoint and vocab**: set `ckpt_file` and `vocab_file` in `run.sh` to your IndicF5 checkpoint and vocab; use the model version that matches your checkpoint (e.g. `F5TTS_*` for v0-style).
- If the checkpoint has a different structure, use or adapt `scripts/convert_checkpoint.py` as in the README.

### Step 4.3: Launch Triton and run client

- Start the Triton server with the built engines as per the repo’s instructions.
- Use the HTTP client from the repo: e.g. `python3 client_http.py` to hit the service.
- Benchmark: e.g. `client_grpc.py` with a dataset (README mentions `yuekai/seed_tts` and `wenetspeech4tts`), or your own list of (ref_audio, ref_text, text) for Indic languages.

### Step 4.4: Call Triton from your FastAPI server

- From your existing TTS server (Ubuntu), add an optional “remote” backend that sends requests to the Triton F5-TTS endpoint (HTTP/GRPC) instead of running the model in-process. This allows you to scale Triton independently and keep your API server lightweight.

**Deliverables**: Docker image for Triton + F5-TTS, runbook for building engines with IndicF5 checkpoint, and optional FastAPI client that proxies to Triton for TTS.

---

## Checklist (Ubuntu + NVIDIA GPU)

All implementation is based on **`src/server/tts_indic_f5.py`**; server integration follows in `core/managers.py` as needed.

- [ ] **Phase 1**: In `tts_indic_f5.py`: enable `torch.compile` (reduce-overhead) + warmup; optionally expose NFE steps and add “fast” mode (16 steps); measure latency.
- [ ] **Phase 2**: Set up ONNX export for IndicF5; run ONNX Runtime with CUDA (and I/O binding); add optional backend to `tts_indic_f5.py`; benchmark.
- [ ] **Phase 3**: Install TensorRT-LLM and F5TTS model plugin; convert IndicF5 checkpoint; build TRT-LLM engine; run `sample.py`; optionally add TRT-LLM backend to reference script; benchmark.
- [ ] **Phase 4** (optional): Build Triton+F5-TTS Docker image; build engines with IndicF5; run Triton server; add remote backend in FastAPI; benchmark under load.

---

## References

- [indic-f5-tts-speed.md](indic-f5-tts-speed.md) — list of sources.
- [AI4Bharat/IndicF5](https://github.com/AI4Bharat/IndicF5) — model repo.
- [SWivid/F5-TTS](https://github.com/SWivid/F5-TTS) — upstream F5-TTS (infer, triton_trtllm).
- [SWivid F5-TTS triton_trtllm README](https://github.com/SWivid/F5-TTS/blob/main/src/f5_tts/runtime/triton_trtllm/README.md) — Triton + TRT-LLM.
- [DakeQQ/F5-TTS-ONNX](https://github.com/DakeQQ/F5-TTS-ONNX) — ONNX export and runtime.
- [WGS-note/F5_TTS_Faster](https://github.com/WGS-note/F5_TTS_Faster) — ONNX + TensorRT-LLM, ~4× speedup.
- [WGS-note/TTS-StressTest](https://github.com/WGS-note/TTS-StressTest) — benchmarking.
- [NVIDIA TensorRT-LLM](https://github.com/NVIDIA/TensorRT-LLM).
- [Bigfishering/f5-tts-trtllm](https://github.com/Bigfishering/f5-tts-trtllm) — TRT-LLM model definition for F5-TTS.
