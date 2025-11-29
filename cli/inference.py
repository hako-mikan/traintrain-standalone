#!/usr/bin/env python3
"""
Inference script to test trained LoRAs on test images.
Generates comparison images for ADDifT and Copier LoRAs.
"""

import os
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
from PIL import Image
from diffusers import StableDiffusionXLImg2ImgPipeline, AutoencoderKL
from safetensors.torch import load_file
import argparse


def load_lora_weights(pipe, lora_path: str, alpha: float = 1.0):
    """Load LoRA weights into the pipeline."""
    state_dict = load_file(lora_path)

    # Check if this is a standard diffusers LoRA or kohya format
    # Kohya format uses keys like "lora_unet_..." and "lora_te_..."
    is_kohya = any(k.startswith("lora_") for k in state_dict.keys())

    if is_kohya:
        pipe.load_lora_weights(lora_path, adapter_name="lora")
        pipe.set_adapters(["lora"], adapter_weights=[alpha])
    else:
        # Try loading as generic LoRA
        pipe.load_lora_weights(lora_path, adapter_name="lora")
        pipe.set_adapters(["lora"], adapter_weights=[alpha])

    return pipe


def run_inference(
    model_path: str,
    lora_paths: list,
    test_images: list,
    output_dir: str,
    prompt: str = "1girl, masterpiece, best quality",
    negative_prompt: str = "worst quality, low quality, blurry",
    strength: float = 0.5,
    guidance_scale: float = 7.5,
    num_inference_steps: int = 30,
    lora_scale: float = 1.0,
    seed: int = 42,
):
    """Run inference with each LoRA on each test image."""

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print(f"Loading base model from {model_path}...")

    # Load pipeline from single file
    pipe = StableDiffusionXLImg2ImgPipeline.from_single_file(
        model_path,
        torch_dtype=torch.float16,
        use_safetensors=True,
    )
    pipe = pipe.to("cuda")
    pipe.enable_model_cpu_offload()

    generator = torch.Generator(device="cuda").manual_seed(seed)

    # First, generate images without LoRA as baseline
    print("\n=== Generating baseline images (no LoRA) ===")
    for img_path in test_images:
        img_name = Path(img_path).stem
        input_image = Image.open(img_path).convert("RGB")

        # Resize to 1024x1024 for SDXL
        input_image = input_image.resize((1024, 1024), Image.LANCZOS)

        output = pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            image=input_image,
            strength=strength,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
            generator=generator,
        ).images[0]

        output_file = output_path / f"baseline_{img_name}.png"
        output.save(output_file)
        print(f"  Saved: {output_file}")

        # Reset generator for consistency
        generator = torch.Generator(device="cuda").manual_seed(seed)

    # Process each LoRA
    for lora_path in lora_paths:
        lora_name = Path(lora_path).stem
        print(f"\n=== Processing LoRA: {lora_name} ===")

        try:
            # Load LoRA
            pipe.load_lora_weights(lora_path, adapter_name="current_lora")
            pipe.set_adapters(["current_lora"], adapter_weights=[lora_scale])
            print(f"  Loaded LoRA with scale {lora_scale}")
        except Exception as e:
            print(f"  Error loading LoRA: {e}")
            continue

        for img_path in test_images:
            img_name = Path(img_path).stem
            input_image = Image.open(img_path).convert("RGB")

            # Resize to 1024x1024 for SDXL
            input_image = input_image.resize((1024, 1024), Image.LANCZOS)

            # Reset generator for consistency
            generator = torch.Generator(device="cuda").manual_seed(seed)

            try:
                output = pipe(
                    prompt=prompt,
                    negative_prompt=negative_prompt,
                    image=input_image,
                    strength=strength,
                    guidance_scale=guidance_scale,
                    num_inference_steps=num_inference_steps,
                    generator=generator,
                ).images[0]

                output_file = output_path / f"{lora_name}_{img_name}.png"
                output.save(output_file)
                print(f"  Saved: {output_file}")
            except Exception as e:
                print(f"  Error generating image: {e}")

        # Unload LoRA for next iteration
        try:
            pipe.unload_lora_weights()
        except:
            # Reinitialize pipeline if unload fails
            pipe = StableDiffusionXLImg2ImgPipeline.from_single_file(
                model_path,
                torch_dtype=torch.float16,
                use_safetensors=True,
            )
            pipe = pipe.to("cuda")
            pipe.enable_model_cpu_offload()

    print(f"\n=== Inference complete. Results saved to {output_dir} ===")


def main():
    parser = argparse.ArgumentParser(description="Run inference with trained LoRAs")
    parser.add_argument("--model", required=True, help="Path to base SDXL model")
    parser.add_argument("--lora-dir", required=True, help="Directory containing LoRA files")
    parser.add_argument("--test-images", nargs="+", required=True, help="Test image paths")
    parser.add_argument("--output-dir", required=True, help="Output directory for results")
    parser.add_argument("--prompt", default="1girl, masterpiece, best quality", help="Prompt")
    parser.add_argument("--negative-prompt", default="worst quality, low quality, blurry")
    parser.add_argument("--strength", type=float, default=0.5, help="img2img strength")
    parser.add_argument("--guidance-scale", type=float, default=7.5)
    parser.add_argument("--steps", type=int, default=30, help="Inference steps")
    parser.add_argument("--lora-scale", type=float, default=1.0, help="LoRA weight scale")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--lora-filter", default="", help="Filter LoRA files by name pattern")

    args = parser.parse_args()

    # Find LoRA files
    lora_dir = Path(args.lora_dir)
    lora_files = sorted(lora_dir.glob("*.safetensors"))

    if args.lora_filter:
        lora_files = [f for f in lora_files if args.lora_filter in f.name]

    # Filter out _copy files and debug files
    lora_files = [f for f in lora_files if "_copy" not in f.name and "debug" not in f.name and "fixed" not in f.name]

    print(f"Found {len(lora_files)} LoRA files:")
    for f in lora_files:
        print(f"  - {f.name}")

    run_inference(
        model_path=args.model,
        lora_paths=[str(f) for f in lora_files],
        test_images=args.test_images,
        output_dir=args.output_dir,
        prompt=args.prompt,
        negative_prompt=args.negative_prompt,
        strength=args.strength,
        guidance_scale=args.guidance_scale,
        num_inference_steps=args.steps,
        lora_scale=args.lora_scale,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
