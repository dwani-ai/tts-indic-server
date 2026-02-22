#!/usr/bin/env python3
"""
Export IndicF5 transformer weights to a .pt file in the format expected by
F5_TTS_Faster's convert_checkpoint.py (TensorRT-LLM Phase 3).

Usage:
  # From repo root, using HuggingFace cache (requires HF_TOKEN if model is gated)
  python scripts/export_indicf5_for_trtllm.py --output phase3_ckpts/indicf5_transformer.pt

  # With explicit checkpoint path (e.g. model.safetensors from HF snapshot)
  python scripts/export_indicf5_for_trtllm.py \\
    --checkpoint ~/.cache/huggingface/hub/models--ai4bharat--IndicF5/snapshots/<revision>/model.safetensors \\
    --output phase3_ckpts/indicf5_transformer.pt
"""
import argparse
import os
import sys

def main():
    parser = argparse.ArgumentParser(
        description="Export IndicF5 transformer for TensorRT-LLM convert_checkpoint.py"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to IndicF5 checkpoint (model.safetensors or .pt). If not set, downloads from HuggingFace.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="phase3_ckpts/indicf5_transformer.pt",
        help="Output .pt file path (default: phase3_ckpts/indicf5_transformer.pt)",
    )
    parser.add_argument(
        "--repo",
        type=str,
        default="ai4bharat/IndicF5",
        help="HuggingFace repo ID (default: ai4bharat/IndicF5)",
    )
    parser.add_argument(
        "--revision",
        type=str,
        default="b82d286220e3070e171f4ef4b4bd047b9a447c9a",
        help="Revision/commit for HF repo",
    )
    args = parser.parse_args()

    if args.checkpoint and not os.path.isfile(args.checkpoint):
        print(f"Error: checkpoint file not found: {args.checkpoint}", file=sys.stderr)
        sys.exit(1)

    # Load state dict
    if args.checkpoint:
        path = args.checkpoint
        if path.endswith(".safetensors"):
            import safetensors.torch
            state_dict = safetensors.torch.load_file(path)
        else:
            import torch
            ckpt = torch.load(path, map_location="cpu", weights_only=True)
            state_dict = ckpt.get("ema_model_state_dict", ckpt.get("model_state_dict", ckpt))
            if not isinstance(state_dict, dict):
                state_dict = ckpt
    else:
        from huggingface_hub import hf_hub_download
        import safetensors.torch
        path = hf_hub_download(
            repo_id=args.repo,
            filename="model.safetensors",
            revision=args.revision,
        )
        state_dict = safetensors.torch.load_file(path)

    # Normalize to transformer-only keys with "ema_model.transformer." prefix
    # convert_checkpoint.py expects ema_model_state_dict with keys like "ema_model.transformer.xxx"
    ema_model_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith("ema_model.transformer."):
            new_key = k
        elif k.startswith("transformer."):
            new_key = "ema_model." + k
        elif k.startswith("model.transformer."):
            new_key = "ema_model." + k
        else:
            continue
        ema_model_state_dict[new_key] = v

    if not ema_model_state_dict:
        print("Error: no transformer keys found. Keys in checkpoint:", list(state_dict.keys())[:20], file=sys.stderr)
        sys.exit(1)

    out_dir = os.path.dirname(args.output)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    import torch
    torch.save({"ema_model_state_dict": ema_model_state_dict}, args.output)
    print(f"Saved {len(ema_model_state_dict)} transformer parameters to {args.output}")
    print("Use this file with F5_TTS_Faster convert_checkpoint.py:")
    print(f"  --timm_ckpt {os.path.abspath(args.output)}")


if __name__ == "__main__":
    main()
