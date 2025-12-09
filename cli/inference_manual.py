#!/usr/bin/env python3
"""
Manual LoRA inference without PEFT dependency.
Loads SDXL and manually applies Kohya-format LoRA weights.
"""

import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
from PIL import Image
from diffusers import StableDiffusionXLImg2ImgPipeline
from safetensors.torch import load_file
import argparse
from collections import defaultdict


def apply_lora_to_model(pipe, lora_path: str, alpha: float = 1.0):
    """Manually apply Kohya-format LoRA weights to SDXL pipeline."""
    print(f"  Loading LoRA weights from {Path(lora_path).name}...")
    state_dict = load_file(lora_path)

    # Parse Kohya LoRA keys and apply to UNet
    applied_count = 0
    unet = pipe.unet

    # Group weights by layer
    lora_weights = defaultdict(dict)
    for key, value in state_dict.items():
        if not key.startswith("lora_unet_"):
            continue
        # Parse key: lora_unet_<layer_path>.<lora_type>
        # e.g., lora_unet_down_blocks_1_attentions_0_proj_in.lora_down.weight
        parts = key.replace("lora_unet_", "").rsplit(".", 2)
        if len(parts) >= 2:
            layer_key = parts[0]
            weight_type = ".".join(parts[1:])
            lora_weights[layer_key][weight_type] = value

    # Apply LoRA weights to UNet layers
    for layer_key, weights in lora_weights.items():
        if "lora_down.weight" not in weights or "lora_up.weight" not in weights:
            continue

        lora_down = weights["lora_down.weight"]
        lora_up = weights["lora_up.weight"]
        lora_alpha = weights.get("alpha", torch.tensor(lora_down.shape[0]))
        if isinstance(lora_alpha, torch.Tensor):
            lora_alpha = lora_alpha.item()

        # Calculate scale
        rank = lora_down.shape[0]
        scale = alpha * lora_alpha / rank

        # Find the target module in UNet
        # Convert layer_key to module path
        # e.g., down_blocks_1_attentions_0_proj_in -> down_blocks.1.attentions.0.proj_in
        module_path = layer_key.replace("_", ".")
        # Fix double dots and numbered indices
        parts = module_path.split(".")
        fixed_parts = []
        i = 0
        while i < len(parts):
            part = parts[i]
            if part.isdigit():
                fixed_parts.append(part)
            elif i + 1 < len(parts) and parts[i + 1].isdigit():
                fixed_parts.append(f"{part}")
            else:
                fixed_parts.append(part)
            i += 1

        # Reconstruct path with proper formatting
        try:
            module = unet
            path_parts = layer_key.split("_")
            reconstructed = []
            i = 0
            while i < len(path_parts):
                part = path_parts[i]
                if part.isdigit():
                    reconstructed[-1] = f"{reconstructed[-1]}[{part}]"
                else:
                    reconstructed.append(part)
                i += 1

            # Navigate to the target module
            attr_path = "_".join(path_parts)
            # Try different path reconstruction strategies
            for attr in ["down_blocks", "up_blocks", "mid_block"]:
                if attr_path.startswith(attr):
                    break

            # Use direct attribute access with underscore-to-dot conversion
            module_attrs = []
            current = ""
            for p in path_parts:
                if p.isdigit():
                    module_attrs.append((current, int(p)))
                    current = ""
                else:
                    if current:
                        current += "_" + p
                    else:
                        current = p
            if current:
                module_attrs.append((current, None))

            target = unet
            for attr, idx in module_attrs:
                if hasattr(target, attr):
                    target = getattr(target, attr)
                    if idx is not None:
                        target = target[idx]
                else:
                    # Try with underscores replaced by nothing for compound names
                    found = False
                    for name in dir(target):
                        if name.replace("_", "") == attr.replace("_", ""):
                            target = getattr(target, name)
                            if idx is not None:
                                target = target[idx]
                            found = True
                            break
                    if not found:
                        raise AttributeError(f"Cannot find {attr} in {type(target)}")

            # Apply LoRA: W' = W + scale * (lora_up @ lora_down)
            if hasattr(target, 'weight'):
                delta = (lora_up @ lora_down).to(target.weight.device, dtype=target.weight.dtype)
                target.weight.data += scale * delta
                applied_count += 1

        except Exception as e:
            # Silently skip layers that can't be found
            pass

    print(f"  Applied {applied_count} LoRA weights with scale {alpha}")
    return applied_count > 0


def run_inference_simple(
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

    def load_fresh_pipeline():
        pipe = StableDiffusionXLImg2ImgPipeline.from_single_file(
            model_path,
            torch_dtype=torch.float16,
            use_safetensors=True,
        )
        pipe = pipe.to("cuda")
        return pipe

    pipe = load_fresh_pipeline()

    # Generate baseline images first
    print("\n=== Generating baseline images (no LoRA) ===")
    for img_path in test_images:
        img_name = Path(img_path).stem
        input_image = Image.open(img_path).convert("RGB")
        input_image = input_image.resize((1024, 1024), Image.LANCZOS)

        generator = torch.Generator(device="cuda").manual_seed(seed)
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

    # Process each LoRA
    for lora_path in lora_paths:
        lora_name = Path(lora_path).stem
        print(f"\n=== Processing LoRA: {lora_name} ===")

        # Reload fresh pipeline for each LoRA
        del pipe
        torch.cuda.empty_cache()
        pipe = load_fresh_pipeline()

        success = apply_lora_to_model(pipe, lora_path, lora_scale)
        if not success:
            print(f"  Warning: No weights applied for {lora_name}")

        for img_path in test_images:
            img_name = Path(img_path).stem
            input_image = Image.open(img_path).convert("RGB")
            input_image = input_image.resize((1024, 1024), Image.LANCZOS)

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
                print(f"  Error: {e}")

    print(f"\n=== Complete. Results at {output_dir} ===")


def main():
    parser = argparse.ArgumentParser(description="Manual LoRA inference")
    parser.add_argument("--model", required=True, help="Base SDXL model path")
    parser.add_argument("--lora-dir", required=True, help="LoRA directory")
    parser.add_argument("--test-images", nargs="+", required=True, help="Test images")
    parser.add_argument("--output-dir", required=True, help="Output directory")
    parser.add_argument("--prompt", default="1girl, masterpiece, best quality")
    parser.add_argument("--negative-prompt", default="worst quality, low quality, blurry")
    parser.add_argument("--strength", type=float, default=0.5)
    parser.add_argument("--guidance-scale", type=float, default=7.5)
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--lora-scale", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    lora_dir = Path(args.lora_dir)
    lora_files = sorted(lora_dir.glob("*.safetensors"))
    lora_files = [f for f in lora_files if "_copy" not in f.name and "debug" not in f.name and "fixed" not in f.name]

    print(f"Found {len(lora_files)} LoRAs:")
    for f in lora_files:
        print(f"  - {f.name}")

    run_inference_simple(
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
