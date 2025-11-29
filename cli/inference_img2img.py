#!/usr/bin/env python3
"""
Img2Img inference script for testing Difference/Copier LoRAs on actual images.
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
from PIL import Image
from diffusers import StableDiffusionXLImg2ImgPipeline
import argparse


def run_img2img_inference(
    model_path: str,
    lora_path: str,
    input_images: list,
    output_dir: str,
    prompt: str = "1girl, masterpiece, best quality",
    negative_prompt: str = "worst quality, low quality, blurry, bad anatomy",
    strength: float = 0.3,
    guidance_scale: float = 7.5,
    num_inference_steps: int = 30,
    lora_scale: float = 1.0,
    seed: int = 42,
):
    """Run img2img inference with LoRA on input images."""

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print(f"Loading base model from {model_path}...")

    pipe = StableDiffusionXLImg2ImgPipeline.from_single_file(
        model_path,
        torch_dtype=torch.float16,
        use_safetensors=True,
    )
    pipe = pipe.to("cuda")

    # Load LoRA
    lora_name = "baseline"
    if lora_path and Path(lora_path).exists():
        lora_name = Path(lora_path).stem
        print(f"Loading LoRA: {lora_name} with scale {lora_scale}")
        pipe.load_lora_weights(lora_path, adapter_name="lora")
        pipe.set_adapters(["lora"], adapter_weights=[lora_scale])

    scale_str = str(lora_scale).replace(".", "")
    print(f"\n=== Generating images with {lora_name} (scale={lora_scale}) ===")

    for img_path in input_images:
        img_name = Path(img_path).stem
        input_image = Image.open(img_path).convert("RGB")

        # Get original size and resize to max 1024 while maintaining aspect ratio
        w, h = input_image.size
        max_size = 1024
        if max(w, h) > max_size:
            if w > h:
                new_w = max_size
                new_h = int(h * max_size / w)
            else:
                new_h = max_size
                new_w = int(w * max_size / h)
            # Round to nearest 64
            new_w = (new_w // 64) * 64
            new_h = (new_h // 64) * 64
            input_image = input_image.resize((new_w, new_h), Image.LANCZOS)

        generator = torch.Generator(device="cuda").manual_seed(seed)

        print(f"  Processing: {img_name} ({input_image.size[0]}x{input_image.size[1]})...")

        output = pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            image=input_image,
            strength=strength,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
            generator=generator,
        ).images[0]

        output_file = output_path / f"{lora_name}_scale{scale_str}_{img_name}.png"
        output.save(output_file)
        print(f"  Saved: {output_file}")

    print(f"\n=== Complete. Results at {output_dir} ===")


def main():
    parser = argparse.ArgumentParser(description="Img2Img LoRA inference")
    parser.add_argument("--model", required=True, help="Base SDXL model path")
    parser.add_argument("--lora", default="", help="LoRA file path")
    parser.add_argument("--input-images", nargs="+", required=True, help="Input image paths")
    parser.add_argument("--output-dir", required=True, help="Output directory")
    parser.add_argument("--prompt", default="1girl, masterpiece, best quality")
    parser.add_argument("--negative-prompt", default="worst quality, low quality, blurry, bad anatomy")
    parser.add_argument("--strength", type=float, default=0.3, help="img2img strength")
    parser.add_argument("--guidance-scale", type=float, default=7.5)
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--lora-scale", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    run_img2img_inference(
        model_path=args.model,
        lora_path=args.lora,
        input_images=args.input_images,
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
