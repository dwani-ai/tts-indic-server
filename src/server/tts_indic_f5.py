"""
IndicF5 TTS reference script with Phase 1 speed optimizations:
- bfloat16 on GPU (faster, less memory)
- Warmup before timed run (stable latency)
- Timing and optional --compile for experimentation

Note: torch.compile is OFF by default. The IndicF5 forward does I/O and uses
torchdiffeq.odeint; Dynamo tracing breaks the ODE solver (AssertionError: t must
be strictly increasing or decreasing). Use --compile only to experiment; it may fail.
"""
import argparse
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from transformers import AutoModel

# ----------------------------
# Configuration
# ----------------------------

REPO_ID = "ai4bharat/IndicF5"
REVISION = "b82d286220e3070e171f4ef4b4bd047b9a447c9a"  # pinned commit

# Text to synthesize
TEXT_TO_SYNTH = (
    "ಬೆಂಗಳೂರು ಕರ್ನಾಟಕ ರಾಜ್ಯದ ರಾಜಧಾನಿ ಆಗಿದೆ, "
    "ಕರ್ನಾಟಕದಲ್ಲಿ ನಾವು ಕನ್ನಡ ಮಾತನಾಡುತ್ತೇವೆ"
)

# Reference audio and its transcript
REF_AUDIO_PATH = "prompts/KAN_F_HAPPY_00001.wav"
REF_TEXT = (
    "ನಮ್‌ ಫ್ರಿಜ್ಜಲ್ಲಿ  ಕೂಲಿಂಗ್‌ ಸಮಸ್ಯೆ ಆಗಿ ನಾನ್‌ ಭಾಳ ದಿನದಿಂದ ಒದ್ದಾಡ್ತಿದ್ದೆ, "
    "ಆದ್ರೆ ಅದ್ನೀಗ ಮೆಕಾನಿಕ್ ಆಗಿರೋ ನಿಮ್‌ ಸಹಾಯ್ದಿಂದ ಬಗೆಹರಿಸ್ಕೋಬೋದು ಅಂತಾಗಿ ನಿರಾಳ ಆಯ್ತು ನಂಗೆ."
)

OUTPUT_WAV = "namaste.wav"
OUTPUT_SR = 24000

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TORCH_DTYPE = torch.bfloat16 if DEVICE != "cpu" else torch.float32


def parse_args():
    p = argparse.ArgumentParser(
        description="IndicF5 TTS with Phase 1 optimizations (bfloat16, warmup) and timing."
    )
    p.add_argument(
        "--compile",
        action="store_true",
        help="Enable torch.compile (experimental; often fails with IndicF5/torchdiffeq).",
    )
    p.add_argument(
        "--warmup-runs",
        type=int,
        default=2,
        metavar="N",
        help="Number of warmup generations before timed run (default: 2).",
    )
    p.add_argument(
        "--timed-runs",
        type=int,
        default=3,
        metavar="N",
        help="Number of timed generations to average (default: 3).",
    )
    p.add_argument(
        "--float32",
        action="store_true",
        help="Use float32 on GPU (for baseline comparison; default is bfloat16 on GPU).",
    )
    return p.parse_args()


def main():
    args = parse_args()
    use_compile = (DEVICE != "cpu") and args.compile
    dtype = torch.float32 if args.float32 else TORCH_DTYPE

    # ----------------------------
    # Load model
    # ----------------------------
    print(f"Loading model '{REPO_ID}' at revision '{REVISION}' on {DEVICE} ({dtype})...")
    load_start = time.perf_counter()
    model = AutoModel.from_pretrained(
        REPO_ID,
        trust_remote_code=True,
        revision=REVISION,
        torch_dtype=dtype,
    )
    model = model.to(DEVICE)

    if use_compile:
        print("Applying torch.compile(model.forward, mode='reduce-overhead')...")
        model.forward = torch.compile(model.forward, mode="reduce-overhead")

    print(f"Model loaded in {time.perf_counter() - load_start:.2f} s.")

    # ----------------------------
    # Warmup (so first timed run is not paying compilation cost)
    # ----------------------------
    if args.warmup_runs > 0:
        print(f"Warmup: {args.warmup_runs} run(s)...")
        warmup_start = time.perf_counter()
        for i in range(args.warmup_runs):
            with torch.no_grad():
                _ = model(
                    TEXT_TO_SYNTH,
                    ref_audio_path=REF_AUDIO_PATH,
                    ref_text=REF_TEXT,
                )
        print(f"Warmup done in {time.perf_counter() - warmup_start:.2f} s.")

    # ----------------------------
    # Timed generation(s)
    # ----------------------------
    print(f"Timed generation: {args.timed_runs} run(s)...")
    times_s = []
    for run in range(args.timed_runs):
        with torch.no_grad():
            start = time.perf_counter()
            audio = model(
                TEXT_TO_SYNTH,
                ref_audio_path=REF_AUDIO_PATH,
                ref_text=REF_TEXT,
            )
            elapsed = time.perf_counter() - start
        times_s.append(elapsed)
        audio_np = np.array(audio)

    # ----------------------------
    # Normalize and save (using last run's audio)
    # ----------------------------
    if audio_np.dtype == np.int16:
        audio_np = audio_np.astype(np.float32) / 32768.0
    else:
        audio_np = audio_np.astype(np.float32)

    out_path = Path(OUTPUT_WAV)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(out_path), audio_np, samplerate=OUTPUT_SR)

    # ----------------------------
    # Report speed
    # ----------------------------
    audio_duration_s = len(audio_np) / OUTPUT_SR
    avg_s = sum(times_s) / len(times_s)
    min_s = min(times_s)
    rtf = avg_s / audio_duration_s if audio_duration_s > 0 else 0

    print()
    print("--- Speed (Phase 1) ---")
    print(f"  compile      : {'yes (reduce-overhead)' if use_compile else 'no'}")
    print(f"  dtype        : {dtype}")
    print(f"  warmup runs  : {args.warmup_runs}")
    print(f"  timed runs   : {args.timed_runs}")
    print(f"  audio length : {audio_duration_s:.2f} s ({len(audio_np)} samples @ {OUTPUT_SR} Hz)")
    print(f"  time (avg)   : {avg_s:.3f} s")
    print(f"  time (min)   : {min_s:.3f} s")
    print(f"  RTF          : {rtf:.4f} (lower is faster)")
    print()
    print(f"Saved to '{OUTPUT_WAV}' at {OUTPUT_SR} Hz.")
    if not use_compile and DEVICE != "cpu":
        print("(torch.compile is off by default; use --compile to try, may fail with this model)")


if __name__ == "__main__":
    main()
