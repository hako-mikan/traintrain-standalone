#!/usr/bin/env python3
"""
Command line runner for TrainTrain standalone features.

Provides subcommands for ADDifT, Multi-ADDifT, Difference (copy LoRA),
and standard LoRA creation. Each command loads a JSON preset,
applies CLI overrides, optionally generates the JSON for inspection,
and calls TrainTrain's trainer modules without launching the Gradio UI.
"""

from __future__ import annotations

import argparse
import ast
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
CONFIG_DIR = PROJECT_ROOT / "cli" / "configs"

DEFAULT_CONFIGS: Dict[str, Path] = {
    "addift": CONFIG_DIR / "addift_sdxl_action.json",
    "multi-addift": CONFIG_DIR / "multi_addift_sdxl.json",
    "difference": CONFIG_DIR / "difference_copy_sdxl.json",
    "lora": CONFIG_DIR / "lora_basic_sdxl.json",
}


def parse_key_values(items: List[str]) -> Dict[str, Any]:
    """Parse CLI --set key=value overrides with dotted keys for nested dicts."""
    parsed: Dict[str, Any] = {}
    for raw in items:
        if "=" not in raw:
            raise ValueError(f"Invalid override '{raw}'. Use key=value syntax.")
        key, value = raw.split("=", 1)
        key = key.strip()
        value = value.strip()
        try:
            parsed_val = ast.literal_eval(value)
        except Exception:
            parsed_val = value
        parsed[key] = parsed_val
    return parsed


def assign_nested(config: Dict[str, Any], dotted_key: str, value: Any) -> None:
    """Assign a value to config using dotted notation such as '2nd pass.network_rank'."""
    parts = dotted_key.split(".")
    cursor = config
    for part in parts[:-1]:
        if part not in cursor or not isinstance(cursor[part], dict):
            cursor[part] = {}
        cursor = cursor[part]
    cursor[parts[-1]] = value


def resolve_path(value: str | os.PathLike[str] | None) -> str | None:
    if value is None or value == "":
        return None
    return str(Path(value).expanduser().resolve())


def ensure_traintrain_repo(traintrain_path: Path, auto_setup: bool) -> Path:
    """
    Ensure the traintrain sources exist. When they are missing and auto_setup
    is enabled, clone and install them via modules.launch_utils.
    """
    if traintrain_path.exists():
        return traintrain_path

    if not auto_setup:
        raise FileNotFoundError(
            f"TrainTrain sources not found at {traintrain_path}. "
            "Run `python cli/train_cli.py setup` or pass --auto-setup."
        )

    from modules import launch_utils

    print("TrainTrain repo missing; running environment preparation...")
    launch_utils.prepare_environment()
    if not traintrain_path.exists():
        raise RuntimeError(
            f"prepare_environment finished but {traintrain_path} is still missing."
        )
    return traintrain_path


def import_train_modules(traintrain_path: Path):
    """Add the parent folder to sys.path and import trainer modules."""
    module_parent = traintrain_path.parent
    if traintrain_path.name != "traintrain":
        module_parent = traintrain_path
    if str(module_parent) not in sys.path:
        sys.path.insert(0, str(module_parent))

    from traintrain.trainer import trainer as trainer_module
    from traintrain.trainer import train as train_module
    import traintrain.scripts.traintrain  # populates trainer.all_configs

    return trainer_module, train_module


def build_config(args: argparse.Namespace) -> Dict[str, Any]:
    """Load preset config and apply CLI overrides."""
    config_path = Path(args.config or args.default_config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config {config_path} not found.")
    with open(config_path, "r", encoding="utf-8") as fh:
        config: Dict[str, Any] = json.load(fh)

    config["mode"] = args.mode_name
    config["model"] = resolve_path(args.model) or ""
    config["vae"] = resolve_path(args.vae) or config.get("vae", "None") or "None"

    if args.output_name:
        config["save_lora_name"] = args.output_name
    if args.data_dir:
        config["lora_data_directory"] = resolve_path(args.data_dir)
    if args.diff_target_suffix:
        config["diff_target_name"] = args.diff_target_suffix
    if args.trigger_word:
        config["lora_trigger_word"] = args.trigger_word
    if args.rank is not None:
        config["network_rank"] = args.rank
    if args.alpha is not None:
        config["network_alpha"] = args.alpha
    if args.batch_size is not None:
        config["train_batch_size"] = args.batch_size
    if args.iterations is not None:
        config["train_iterations"] = args.iterations
    if args.learning_rate is not None:
        config["train_learning_rate"] = args.learning_rate
    if args.seed is not None:
        config["train_seed"] = args.seed
    if args.timesteps_min is not None:
        config["train_min_timesteps"] = args.timesteps_min
    if args.timesteps_max is not None:
        config["train_max_timesteps"] = args.timesteps_max

    if args.orig_prompt:
        config["original prompt"] = args.orig_prompt
    if args.targ_prompt:
        config["target prompt"] = args.targ_prompt
    if args.negative_prompt:
        config["negative prompt"] = args.negative_prompt
    elif "negative prompt" not in config:
        config["negative prompt"] = ""

    if args.orig_image:
        config["original image"] = resolve_path(args.orig_image)
    if args.targ_image:
        config["target image"] = resolve_path(args.targ_image)

    overrides = parse_key_values(args.set_override or [])
    for key, value in overrides.items():
        assign_nested(config, key, value)

    return config


def dump_config(config: Dict[str, Any], target: Path) -> None:
    with open(target, "w", encoding="utf-8") as fh:
        json.dump(config, fh, indent=2, ensure_ascii=False)


def run_training(args: argparse.Namespace) -> None:
    config = build_config(args)

    if args.dry_run:
        print(json.dumps(config, indent=2, ensure_ascii=False))
        return

    traintrain_dir = ensure_traintrain_repo(args.traintrain_path, auto_setup=not args.no_auto_setup)
    trainer_module, train_module = import_train_modules(traintrain_dir)

    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as tmp:
        json.dump(config, tmp, indent=2, ensure_ascii=False)
        tmp_path = Path(tmp.name)

    try:
        inputs = trainer_module.import_json(str(tmp_path), cli=True)
        cfg_len = len(trainer_module.all_configs)
        prompt_start = 3 + cfg_len * 2
        prompts = [
            config.get("original prompt", ""),
            config.get("target prompt", ""),
            config.get("negative prompt", ""),
        ]
        for idx, prompt in enumerate(prompts):
            inputs[prompt_start + idx] = prompt

        if config.get("original image"):
            inputs[prompt_start + 3] = config["original image"]
        if config.get("target image"):
            inputs[prompt_start + 4] = config["target image"]

        paths = [
            resolve_path(args.models_dir),
            resolve_path(args.ckpt_dir),
            resolve_path(args.vae_dir_override),
            resolve_path(args.lora_dir_override),
        ]
        print(f"Starting {args.mode_name} training using preset {tmp_path.name} ...")
        result = train_module.train_main(paths, *inputs)
        print(result)
    finally:
        tmp_path.unlink(missing_ok=True)


def run_setup(_: argparse.Namespace) -> None:
    from modules import launch_utils

    launch_utils.prepare_environment()
    print("Environment preparation finished.")


def register_train_command(
    subparsers,
    name: str,
    mode_name: str,
    description: str,
    requires_images: bool,
    needs_dataset: bool,
):
    parser = subparsers.add_parser(name, help=description)
    parser.set_defaults(
        handler=run_training,
        mode_name=mode_name,
        default_config_path=DEFAULT_CONFIGS[name],
    )

    parser.add_argument("--config", type=str, help="Custom JSON preset (defaults to the CLI preset).")
    parser.add_argument("--model", required=True, help="Path to base checkpoint (.safetensors).")
    parser.add_argument("--vae", default="", help="Optional VAE file.")
    parser.add_argument("--output-name", help="Override save_lora_name.")
    parser.add_argument("--orig-prompt", help="Prompt describing the original image.")
    parser.add_argument("--targ-prompt", help="Prompt describing the target image.")
    parser.add_argument("--negative-prompt", help="Optional negative prompt.")
    parser.add_argument("--trigger-word", help="Set lora_trigger_word.")
    parser.add_argument("--rank", type=int, help="Override network_rank.")
    parser.add_argument("--alpha", type=float, help="Override network_alpha.")
    parser.add_argument("--iterations", type=int, help="Override train_iterations.")
    parser.add_argument("--batch-size", type=int, help="Override train_batch_size.")
    parser.add_argument("--learning-rate", type=float, help="Override train_learning_rate.")
    parser.add_argument("--seed", type=int, help="Override train_seed.")
    parser.add_argument("--timesteps-min", type=int, help="Override train_min_timesteps.")
    parser.add_argument("--timesteps-max", type=int, help="Override train_max_timesteps.")
    parser.add_argument("--set", dest="set_override", action="append", default=[], help="Arbitrary overrides key=value (supports dotted keys).")
    parser.add_argument("--dry-run", action="store_true", help="Print the merged config and exit.")
    parser.add_argument("--no-auto-setup", action="store_true", help="Fail if the traintrain repo is missing.")
    parser.add_argument("--traintrain-path", type=str, default=str(PROJECT_ROOT / "traintrain"), help="Location of the traintrain sources.")
    parser.add_argument("--models-dir", help="Root models directory (sets ckpt/vae/lora default roots).")
    parser.add_argument("--ckpt-dir", help="Explicit Stable Diffusion checkpoint directory.")
    parser.add_argument("--vae-dir", dest="vae_dir_override", help="Explicit VAE directory.")
    parser.add_argument("--lora-dir", dest="lora_dir_override", help="Explicit LoRA output directory.")

    if requires_images:
        parser.add_argument("--orig-image", required=True, help="Path to the original reference image.")
        parser.add_argument("--targ-image", required=True, help="Path to the target reference image.")
    else:
        parser.add_argument("--orig-image", help="Optional original image path.")
        parser.add_argument("--targ-image", help="Optional target image path.")

    if needs_dataset:
        parser.add_argument("--data-dir", required=True, help="Directory containing training images.")
        parser.add_argument(
            "--diff-target-suffix",
            default="_target",
            help="Suffix appended to base filenames to find the target image in Multi-ADDifT.",
        )
    else:
        parser.add_argument("--data-dir", help="Optional dataset directory.")
        parser.add_argument("--diff-target-suffix", help="Optional suffix for dataset pairing.")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="CLI runner for TrainTrain Standalone.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    setup_parser = subparsers.add_parser("setup", help="Clone/update the TrainTrain sources and install requirements.")
    setup_parser.set_defaults(handler=run_setup)

    register_train_command(
        subparsers,
        name="addift",
        mode_name="ADDifT",
        description="Train ADDifT from a single original/target pair.",
        requires_images=True,
        needs_dataset=False,
    )
    register_train_command(
        subparsers,
        name="multi-addift",
        mode_name="Multi-ADDifT",
        description="Train Multi-ADDifT by pairing images in a directory.",
        requires_images=False,
        needs_dataset=True,
    )
    register_train_command(
        subparsers,
        name="difference",
        mode_name="Difference",
        description="Train copy-machine (Difference) LoRA from a pair of images.",
        requires_images=True,
        needs_dataset=False,
    )
    register_train_command(
        subparsers,
        name="lora",
        mode_name="LoRA",
        description="Standard LoRA training from a dataset directory.",
        requires_images=False,
        needs_dataset=True,
    )
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    if hasattr(args, "traintrain_path"):
        args.traintrain_path = Path(args.traintrain_path).expanduser().resolve()
    else:
        args.traintrain_path = PROJECT_ROOT / "traintrain"
    if hasattr(args, "models_dir"):
        args.models_dir = resolve_path(args.models_dir)
    if hasattr(args, "ckpt_dir"):
        args.ckpt_dir = resolve_path(args.ckpt_dir)
    if hasattr(args, "vae_dir_override"):
        args.vae_dir_override = resolve_path(args.vae_dir_override)
    if hasattr(args, "lora_dir_override"):
        args.lora_dir_override = resolve_path(args.lora_dir_override)
    handler = getattr(args, "handler", None)
    if handler is None:
        parser.error("No handler registered for this command.")
    handler(args)


if __name__ == "__main__":
    main()
