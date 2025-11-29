#!/usr/bin/env python3
"""
Text2Image inference script for testing Difference/Copier LoRAs.
"""

import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
from diffusers import StableDiffusionXLPipeline
import argparse


def run_t2i_inference(
    model_path: str,
    lora_path: str,
    output_dir: str,
    prompts: list,
    negative_prompt: str = "worst quality, low quality, blurry, bad anatomy",
    guidance_scale: float = 7.5,
    num_inference_steps: int = 30,
    lora_scale: float = 1.0,
    seed: int = 42,
    width: int = 1024,
    height: int = 1024,
):
    """Run text2image inference with LoRA."""

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print(f"Loading base model from {model_path}...")

    pipe = StableDiffusionXLPipeline.from_single_file(
        model_path,
        torch_dtype=torch.float16,
        use_safetensors=True,
    )
    pipe = pipe.to("cuda")

    # Load LoRA if specified
    lora_name = "baseline"
    if lora_path and Path(lora_path).exists():
        lora_name = Path(lora_path).stem
        print(f"Loading LoRA: {lora_name} with scale {lora_scale}")
        pipe.load_lora_weights(lora_path, adapter_name="lora")
        pipe.set_adapters(["lora"], adapter_weights=[lora_scale])

    print(f"\n=== Generating images with {lora_name} ===")

    for i, prompt in enumerate(prompts):
        generator = torch.Generator(device="cuda").manual_seed(seed + i)

        print(f"  Generating: {prompt[:50]}...")

        output = pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            width=width,
            height=height,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
            generator=generator,
        ).images[0]

        # Create safe filename
        safe_prompt = "".join(c if c.isalnum() or c in " _-" else "_" for c in prompt[:30])
        output_file = output_path / f"{lora_name}_{width}_{safe_prompt}.png"
        output.save(output_file)
        print(f"  Saved: {output_file}")

    print(f"\n=== Complete. Results at {output_dir} ===")


def main():
    parser = argparse.ArgumentParser(description="Text2Image LoRA inference")
    parser.add_argument("--model", required=True, help="Base SDXL model path")
    parser.add_argument("--lora", default="", help="LoRA file path (optional)")
    parser.add_argument("--output-dir", required=True, help="Output directory")
    parser.add_argument("--prompts", nargs="+", default=[
        "1girl, red hair, gothic dress, standing, full body, masterpiece, best quality",
        "1girl, purple hair, yandere, holding knife, dark room, masterpiece, best quality",
        "1girl, long hair, looking away, standing by window, side view, masterpiece, best quality",
    ])
    parser.add_argument("--negative-prompt", default="worst quality, low quality, blurry, bad anatomy")
    parser.add_argument("--guidance-scale", type=float, default=7.5)
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--lora-scale", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--height", type=int, default=1024)

    args = parser.parse_args()

    run_t2i_inference(
        model_path=args.model,
        lora_path=args.lora,
        output_dir=args.output_dir,
        prompts=args.prompts,
        negative_prompt=args.negative_prompt,
        guidance_scale=args.guidance_scale,
        num_inference_steps=args.steps,
        lora_scale=args.lora_scale,
        seed=args.seed,
        width=args.width,
        height=args.height,
    )


if __name__ == "__main__":
    main()
