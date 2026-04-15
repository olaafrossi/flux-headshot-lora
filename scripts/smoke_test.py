"""End-to-end smoke test for the local FLUX stack.

Verifies that your install can:
  1. See the RTX 5070 via PyTorch (cu128 Blackwell build)
  2. Quantize FLUX.1-dev to NF4 with bitsandbytes
  3. Load the pipeline with CPU offload on 12GB VRAM
  4. Run a single generation end-to-end and save the output

Run this BEFORE a 45-minute training run so you find install problems in
two minutes instead of discovering them after training finishes.

    python scripts/smoke_test.py
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch
from diffusers import BitsAndBytesConfig as DiffusersBnbConfig
from diffusers import FluxPipeline, FluxTransformer2DModel
from transformers import BitsAndBytesConfig as TransformersBnbConfig
from transformers import T5EncoderModel

MODEL_ID = "black-forest-labs/FLUX.1-dev"
TEST_PROMPT = (
    "a photograph of a red apple on a wooden table, studio lighting, "
    "sharp focus, shallow depth of field, 85mm lens"
)


def check(label: str, ok: bool, detail: str = "") -> None:
    mark = "[ OK ]" if ok else "[FAIL]"
    suffix = f"  ({detail})" if detail else ""
    print(f"  {mark} {label}{suffix}")
    if not ok:
        raise SystemExit(1)


def gpu_memory_gb() -> float:
    return torch.cuda.memory_allocated() / (1024**3)


def main() -> None:
    root = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(description="FLUX stack smoke test.")
    parser.add_argument("--steps", type=int, default=12)
    parser.add_argument("--width", type=int, default=768)
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument(
        "--output",
        type=Path,
        default=root / "output" / "smoke_test.png",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("FLUX stack smoke test")
    print("=" * 60)

    print("\n[1/4] Environment")
    check("PyTorch CUDA available", torch.cuda.is_available(), torch.__version__)
    gpu_name = torch.cuda.get_device_name(0)
    gpu_vram = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    check("GPU detected", True, f"{gpu_name}, {gpu_vram:.1f} GB")
    cap = torch.cuda.get_device_capability(0)
    check("Compute capability", cap >= (8, 0), f"sm_{cap[0]}{cap[1]}")
    try:
        import bitsandbytes as bnb  # noqa: F401
        check("bitsandbytes importable", True)
    except Exception as e:
        check("bitsandbytes importable", False, str(e))

    dtype = torch.bfloat16

    print("\n[2/4] Loading FLUX.1-dev with NF4 quantization")
    print("  (first run downloads ~24GB — subsequent runs use HF cache)")
    t0 = time.time()

    transformer_quant = DiffusersBnbConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=dtype,
    )
    transformer = FluxTransformer2DModel.from_pretrained(
        MODEL_ID,
        subfolder="transformer",
        quantization_config=transformer_quant,
        torch_dtype=dtype,
    )
    check("Transformer quantized", True, f"{gpu_memory_gb():.2f} GB alloc")

    text_encoder_quant = TransformersBnbConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=dtype,
    )
    text_encoder_2 = T5EncoderModel.from_pretrained(
        MODEL_ID,
        subfolder="text_encoder_2",
        quantization_config=text_encoder_quant,
        torch_dtype=dtype,
    )
    check("T5 text encoder quantized", True, f"{gpu_memory_gb():.2f} GB alloc")

    pipe = FluxPipeline.from_pretrained(
        MODEL_ID,
        transformer=transformer,
        text_encoder_2=text_encoder_2,
        torch_dtype=dtype,
    )
    # Model-level offload for 8GB VRAM (RTX 5070 Laptop). Sequential
    # offload triggers a bitsandbytes Params4bit.to() meta-tensor bug on
    # NF4 weights during sub-module shuffles.
    pipe.enable_model_cpu_offload()
    check(
        "Pipeline assembled + model CPU offload",
        True,
        f"{time.time() - t0:.1f}s total load",
    )

    print("\n[3/4] Running one generation")
    print(f"  prompt: {TEST_PROMPT}")
    print(f"  {args.width}x{args.height}, {args.steps} steps")
    t0 = time.time()
    image = pipe(
        prompt=TEST_PROMPT,
        num_inference_steps=args.steps,
        guidance_scale=3.5,
        height=args.height,
        width=args.width,
        generator=torch.Generator(device="cuda").manual_seed(0),
    ).images[0]
    gen_time = time.time() - t0
    peak = torch.cuda.max_memory_allocated() / (1024**3)
    check("Generation completed", True, f"{gen_time:.1f}s, peak {peak:.2f} GB")

    print("\n[4/4] Saving output")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    image.save(args.output)
    check("Image saved", args.output.exists(), str(args.output))

    print("\n" + "=" * 60)
    print("All checks passed. You're clear to train.")
    print("=" * 60)


if __name__ == "__main__":
    main()
