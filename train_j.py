import argparse
import os
from traintrain.trainer.train import train_main
from traintrain.trainer import trainer as tt_trainer
from traintrain.trainer.trainer import import_json
# importing the UI script is what fills in trainer.all_configs
import traintrain.scripts.traintrain

# the head of what import_json returns, before the settings themselves
HEAD_KEYS = ["mode", "model", "vae", "te"]
# and the tail, after the two passes of settings
TAIL_KEYS = ["original prompt", "target prompt", "negative prompt", "original image", "target image"]


def apply_overrides(inputs, overrides):
    """Overwrite single values of a loaded json, so one file can drive a whole
    series of runs. Keys are the ones the json itself uses."""
    all_configs = tt_trainer.all_configs
    names = [c[0] for c in all_configs]
    passes = (len(inputs) - len(HEAD_KEYS) - len(TAIL_KEYS)) // len(names)

    for item in overrides:
        if ":" not in item:
            raise SystemExit(f'--override wants "key:value", got {item!r}')
        key, value = item.split(":", 1)
        key, value = key.strip(), value.strip()

        if key in HEAD_KEYS:
            inputs[HEAD_KEYS.index(key)] = value
        elif key in names:
            # both passes, so a 2nd pass that was following the 1st keeps doing so
            for p in range(passes):
                index = len(HEAD_KEYS) + p * len(names) + names.index(key)
                dtype = all_configs[names.index(key)][4]
                try:
                    inputs[index] = dtype(value) if dtype is not bool else value.lower() in ("1", "true", "yes", "on")
                except ValueError:
                    inputs[index] = value
        elif key in TAIL_KEYS:
            inputs[len(inputs) - len(TAIL_KEYS) + TAIL_KEYS.index(key)] = value
        else:
            raise SystemExit(f"--override does not know the key {key!r}")

        print(f"override {key} = {value}")

    return inputs


def main():
    parser = argparse.ArgumentParser(
        description="Load a json, optionally overwrite single values, and train.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument("json_path", type=str, help="Path to the JSON file")
    parser.add_argument("--models-dir", type=str, default=None, help="Directory for models")
    parser.add_argument("--ckpt-dir", type=str, default=None, help="Directory for StableDiffusion Models (overrides --models-dir)")
    parser.add_argument("--vae-dir", type=str, default=None, help="Directory for VAE (overrides --models-dir)")
    parser.add_argument("--lora-dir", type=str, default=None, help="Directory for LoRA (overrides --models-dir)")
    parser.add_argument("--te-dir", type=str, default=None, help="Directory for TextEncoders (overrides --models-dir)")
    parser.add_argument(
        "--override",
        nargs="+",
        metavar="KEY:VALUE",
        default=[],
        help='Overwrite a value from the json, for example --override "save_lora_name:test01" "train_iterations:200"',
    )

    args = parser.parse_args()
    paths = [args.models_dir, args.ckpt_dir, args.vae_dir, args.lora_dir, args.te_dir]

    inputs = import_json(args.json_path, cli=True)
    if args.override:
        inputs = apply_overrides(list(inputs), args.override)

    result = train_main(paths, *inputs)
    print(result)


if __name__ == "__main__":
    main()
