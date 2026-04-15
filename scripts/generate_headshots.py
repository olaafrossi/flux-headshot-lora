"""Generate headshots with a trained FLUX LoRA on an RTX 5070 laptop GPU (8GB VRAM).

Loads FLUX.1-dev with NF4 quantization on the transformer and T5 text
encoder, attaches the trained LoRA, and runs a batch of prompts from
prompts.py with N variations each.

Run from the project root, after training:

    python scripts/generate_headshots.py \\
        --lora output/loras/my_headshot_lora/my_headshot_lora.safetensors \\
        --count 3
"""
from __future__ import annotations

import argparse
import gc
import sys
from datetime import datetime
from pathlib import Path

import torch
from diffusers import BitsAndBytesConfig as DiffusersBnbConfig
from diffusers import FluxPipeline, FluxTransformer2DModel
from tqdm import tqdm
from transformers import BitsAndBytesConfig as TransformersBnbConfig
from transformers import T5EncoderModel

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
from prompts import HEADSHOT_PROMPTS  # noqa: E402

MODEL_ID = "black-forest-labs/FLUX.1-dev"


def load_pipeline(lora_path: Path) -> FluxPipeline:
    dtype = torch.bfloat16

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

    pipe = FluxPipeline.from_pretrained(
        MODEL_ID,
        transformer=transformer,
        text_encoder_2=text_encoder_2,
        torch_dtype=dtype,
    )

    # Order matters: load the LoRA BEFORE enabling offload. If offload hooks
    # are already installed, load_lora_weights calls
    # _optionally_disable_offloading which tries to detach them, and
    # accelerate's hook removal collides with bitsandbytes NF4 meta-tensor
    # quant state ("Cannot copy out of meta tensor; no data!").
    pipe.load_lora_weights(str(lora_path))

    # Model-level offload (not sequential): swaps whole sub-models
    # (transformer, text_encoder, text_encoder_2, vae) in and out of VRAM
    # as units. Sequential offload's per-sub-module hook path triggers a
    # bitsandbytes bug where Params4bit.to() can't migrate QuantState
    # through meta tensors. Model offload avoids that code path.
    #
    # VRAM math on 8GB: NF4 transformer ~7GB is the peak, NF4 T5 ~4-5GB,
    # VAE + CLIP are small. Only one is resident at a time.
    pipe.enable_model_cpu_offload()
    return pipe


def slugify(text: str, max_len: int = 40) -> str:
    out = []
    for ch in text.lower():
        if ch.isalnum():
            out.append(ch)
        elif ch in " _-":
            out.append("_")
    slug = "".join(out).strip("_")
    while "__" in slug:
        slug = slug.replace("__", "_")
    return slug[:max_len]


def main() -> None:
    root = SCRIPT_DIR.parent
    parser = argparse.ArgumentParser(
        description="Batch-generate headshots from a trained FLUX LoRA."
    )
    parser.add_argument(
        "--lora",
        type=Path,
        required=True,
        help="Path to the trained LoRA .safetensors file.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=root / "output" / "images",
        help="Where to write generated images.",
    )
    parser.add_argument("--trigger", default="ohwx_person")
    parser.add_argument(
        "--count", type=int, default=3, help="Variations per prompt."
    )
    parser.add_argument("--steps", type=int, default=28)
    parser.add_argument("--guidance", type=float, default=3.5)
    # 768x1024 leaves activation headroom on 8GB with model_cpu_offload.
    # Bump to 896x1152 or 1024x1280 if you have VRAM to spare.
    parser.add_argument("--width", type=int, default=768)
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--lora-scale",
        type=float,
        default=1.0,
        help="LoRA influence strength. Drop to 0.8 if overtrained.",
    )
    args = parser.parse_args()

    if not args.lora.exists():
        raise SystemExit(f"LoRA not found: {args.lora}")

    run_dir = args.output_dir / datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading FLUX.1-dev (NF4 quantized) + LoRA from {args.lora.name}")
    pipe = load_pipeline(args.lora)

    total = len(HEADSHOT_PROMPTS) * args.count
    print(
        f"Generating {total} images "
        f"({len(HEADSHOT_PROMPTS)} prompts × {args.count}) → {run_dir}"
    )

    bar = tqdm(total=total, desc="Generating")
    for prompt_idx, template in enumerate(HEADSHOT_PROMPTS):
        prompt = template.format(trigger=args.trigger)
        for variation in range(args.count):
            seed = args.seed + prompt_idx * 1000 + variation
            generator = torch.Generator(device="cuda").manual_seed(seed)
            image = pipe(
                prompt=prompt,
                num_inference_steps=args.steps,
                guidance_scale=args.guidance,
                height=args.height,
                width=args.width,
                generator=generator,
                joint_attention_kwargs={"scale": args.lora_scale},
            ).images[0]

            slug = slugify(template.format(trigger="").lstrip(" ,"))
            out_path = run_dir / f"{prompt_idx:02d}_{slug}_s{seed}.png"
            image.save(out_path)
            bar.update(1)

            del image
            gc.collect()
            torch.cuda.empty_cache()
    bar.close()
    print(f"\nDone. {total} images saved to {run_dir}")


if __name__ == "__main__":
    main()
