# Traintrain Standalone
　Text-to-Image Low Rank Adaption (LoRA) training tool. Stand alone environment for sd-webui-traintrain.  
　Stable Diffusion WebUI用のLoRA学習拡張であるsd-webui-traintrainをスタンドアロンで使用するための環境を構築します。

[<img src="https://img.shields.io/badge/lang-Egnlish-red.svg?style=plastic" height="25" />](#overview)
[<img src="https://img.shields.io/badge/言語-日本語-green.svg?style=plastic" height="25" />](#概要)
[<img src="https://img.shields.io/badge/Support-%E2%99%A5-magenta.svg?logo=github&style=plastic" height="25" />](https://github.com/sponsors/hako-mikan)

## Overview
This project sets up a standalone environment for using the sd-webui-traintrain extension for LoRA training with Stable Diffusion WebUI. Since it builds an independent Python environment (venv), it prevents dependency conflicts with other extensions. The tool supports execution from the TrainTrain WebUI and command-line execution using JSON configurations.

## Installation
To install, enter the directory where you want to install and run the following command in the command prompt:
```
git clone https://github.com/hako-mikan/traintrain-standalone
```
After installation, double-click `webui-user.bat` to start the program. The first launch will set up the environment, which may take some time. Even if you plan to use the command-line execution, you must run `webui-user.bat` once for environment setup.

## Usage
Refer to the [TrainTrain repository](https://github.com/hako-mikan/sd-webui-traintrain) for instructions on using TrainTrain.
In Standalone mode, models and VAEs must be specified using full paths. If you set variables like `--models-dir` in the command line, models will be selectable from a list.

## Additional Settings
You can add command-line variables in `webui-user.bat` to specify model directories and other parameters.
```
set COMMANDLINE_ARGS=--models-dir X:\StabilityMatrix\Models --branch dev
```
### Command-line Variables

| Variable | Description |
|----------|-------------|
| `--models-dir` | Specifies the root directory for models, LoRA, and VAE. |
| `--ckpt-dir` | Specifies the model directory (overrides `--models-dir` if set). |
| `--vae-dir` | Specifies the VAE directory (overrides `--models-dir` if set). |
| `--lora-dir` | Specifies the output directory for LoRA (overrides `--models-dir` if set). |
| `--branch` | Specifies the TrainTrain branch to use. |
| `--xformers` | Enables Xformers. |
| `--reinstall-xformers` | Reinstalls Xformers. |
| `--reinstall-torch` | Reinstalls Torch. |
| `--skip-python-version-check` | Skips Python version check at startup. |
| `--skip-torch-cuda-test` | Skips Torch CUDA test at startup. |
| `--skip-prepare-environment` | Disables environment setup at startup. |
| `--disable-update` | Disables update of TrainTrain. |

## Command-line Execution
If you want to run the tool from the command line without launching the WebUI, follow these steps.

### **Activate `venv`**
First, activate the virtual environment (`venv`). Open the command prompt in the TrainTrain-Standalone directory and run:

**Windows (PowerShell):**
```powershell
.\venv\Scripts\Activate
```

**Windows (Command Prompt):**
```cmd
venv\Scripts\activate
```

**Linux / macOS (bash):**
```bash
source G:/StabilityMatrix/Models/venv/bin/activate
```

### **Run `train_j.py`**
Once the virtual environment is active, execute the following command:

```cmd
python train_j.py test.json --models-dir X:\StabilityMatrix\Models
```

The `--models-dir X:\StabilityMatrix\Models` argument is optional. By default, models must be specified with full paths. However, if `--models-dir` is set, models can be referenced with subdirectory and filename formats like `xl\xl_base.safetensors`. Specifying `--models-dir` or `--lora-dir` changes the LoRA output directory.

| Variable | Description |
|----------|-------------|
| `--models-dir` | Specifies the root directory for models, LoRA, and VAE. |
| `--ckpt-dir` | Specifies the model directory (overrides `--models-dir` if set). |
| `--vae-dir` | Specifies the VAE directory (overrides `--models-dir` if set). |
| `--lora-dir` | Specifies the LoRA output directory (overrides `--models-dir` if set). |

## CLI (headless) training
Run TrainTrain trainers without launching Gradio. Presets live in `cli/configs/*.json` and can be overridden from the command line.

1. **Set up once**
   ```bash
   python cli/train_cli.py setup
   ```

2. **ADDifT (single pair)**
   ```bash
   python cli/train_cli.py addift \
     --model /models/sdxl/sd_xl_base_1.0.safetensors \
     --orig-image /data/sample/orig.png \
     --targ-image /data/sample/targ.png \
     --output-name sample_addift \
     --orig-prompt "source look" \
     --targ-prompt "target look"
   ```

3. **Multi-ADDifT (directory pairs)**
   ```bash
   python cli/train_cli.py multi-addift \
     --model /models/sdxl/sd_xl_base_1.0.safetensors \
     --data-dir /data/pairs \
     --diff-target-suffix _target \
     --output-name sample_multi
   ```

4. **Difference / Copy-machine LoRA**
   ```bash
   python cli/train_cli.py difference \
     --model /models/sdxl/sd_xl_base_1.0.safetensors \
     --orig-image /data/sample/orig.png \
     --targ-image /data/sample/targ.png \
     --output-name sample_diff
   ```

5. **Standard LoRA (dataset)**
   ```bash
   python cli/train_cli.py lora \
     --model /models/sdxl/sd_xl_base_1.0.safetensors \
     --data-dir /data/lora_dataset \
     --output-name sample_lora
   ```

Tips: `--set key=value` overrides any preset field (dotted keys allowed), `--dry-run` prints the merged JSON, and `--traintrain-path` lets you reuse an existing `traintrain` clone (e.g., `~/sd-webui-traintrain`).

## Acknowledgments
This repository references code from [Stable Diffusion WebUI Forge](https://github.com/lllyasviel/stable-diffusion-webui-forge).


## 概要
　Stable Diffusion WebUI用のLoRA学習拡張であるsd-webui-traintrainをスタンドアロンで使用するための環境を構築します。venv(独立したpython環境)を構築するため他の拡張との依存関係などの問題が起きません。TrainTrain用のWebUIからの実行および`json`を利用したコマンドライン実行に対応します。

## インストール
　コマンドプロンプトでインストールしたいディレクトリにて以下のコマンドを入力します。
```
git clone https://github.com/hako-mikan/traintrain-standalone
```
　インストール後、`webui-user.bat`をダブルクリックすることで起動します。初回起動時は環境構築が行われるので時間がかかります。コマンドライン実行を行いたい場合でも環境構築のため`webui-user.bat`一度起動してください。

## 使い方
　TrainTrainの使い方は[TrainTrain](https://github.com/hako-mikan/sd-webui-traintrain)のレポジトリを参照してください。
StandAloneではモデルやVAEの指定がフルパス指定になります。コマンドライン変数で`--models-dir`などを設定している場合にはモデルリストから選ぶ形式になります。

## 追加設定
　`webui-user.bat`にコマンドライン変数を追加することでモデルのディレクトリなどを指定できます。
```
set COMMANDLINE_ARGS=--models-dir X:\StabilityMatrix\Models --branch dev
```
　以下はコマンドライン変数です。

| 変数                          | 説明 |
|----------------------------------|----------------------------------------------|
| `--models-dir`                     | モデルなどのルートディレクトリを指定します。Model, LoRA, VAEディレクトリも設定されます。 |
| `--ckpt-dir`                   | モデルのディレクトリを指定します。`--models-dir`が指定されている場合でも優先されます。 |
| `--vae-dir`                      | VAEのディレクトリを指定します。`--models-dir`が指定されている場合でも優先されます。 |
| `--lora-dir`                     | LoRA出力先ディレクトリを指定します。`--models-dir`が指定されている場合でも優先されます。 |
| `--branch`                       | TrainTrainのブランチを指定します。 |
| `--xformers`                     | Xformersを有効化します。 |
| `--reinstall-xformers`           | Xformersを再インストールします。 |
| `--reinstall-torch`              | Torchを再インストールします。 |
| `--skip-python-version-check`    | 起動時にPythonのバージョンチェックをスキップします。 |
| `--skip-torch-cuda-test`         | 起動時にTorchのCUDAテストをスキップします。 |
| `--skip-prepare-environment`     | 起動時の環境構築を無効化します。 |
| `--disable-update` | TrainTrainのアップデートを無効化します。 |

## コマンドライン起動
　WebUIを起動せずにコマンドラインから実行したい場合には以下の手順を踏んでください。

### ** `venv` を有効化**
まず、仮想環境 (`venv`) をアクティブにする必要があります。TrainTrain-StaindAloneディレクトリでコマンドプロンプトを起動し、
仮想環境をアクティブ化します。

**Windows (PowerShell の場合):**
```powershell
.\venv\Scripts\Activate
```

**Windows (コマンドプロンプトの場合):**
```cmd
venv\Scripts\activate
```

**Linux / macOS (bash の場合):**
```bash
source G:/StabilityMatrix/Models/venv/bin/activate
```

### ** `train_j.py` を実行**
仮想環境をアクティブにした状態で、以下のコマンドを実行してください。

```cmd
python train_j.py test.json --models-dir X:\StabilityMatrix\Models
```

　`--models-dir X:\StabilityMatrix\Models`はオプションです。デフォルトではモデルはフルパス入力が必要ですが、`--models-dir`を指定するとサブディレクトリ＋モデル名`xl\xl_base.safetensors`のような指定もできます。`--models-dir`や`--lora-dir`を指定するとLoRAの出力先が変わります。

| 変数                          | 説明 |
|----------------------------------|----------------------------------------------|
| `--models-dir`                     | モデルなどのルートディレクトリを指定します。Model, LoRA, VAEディレクトリも設定されます。 |
| `--ckpt-dir`                   | モデルのディレクトリを指定します。`--models-dir`が指定されている場合でも優先されます。 |
| `--vae-dir`                      | VAEのディレクトリを指定します。`--models-dir`が指定されている場合でも優先されます。 |
| `--lora-dir`                     | LoRAのディレクトリを指定します。`--models-dir`が指定されている場合でも優先されます。 |

## 謝辞
　本レポジトリは[Stable Diffusion WebUI Forge](https://github.com/lllyasviel/stable-diffusion-webui-forge)のコードを参考にしています。
