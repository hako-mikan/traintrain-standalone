import re
import subprocess
import os
import shutil
import sys
import importlib.util
import importlib.metadata
import platform
from pathlib import Path
import shlex
import argparse

modules_path = os.path.dirname(os.path.realpath(__file__))
script_path = os.path.dirname(modules_path)
normalized_filepath = lambda filepath: str(Path(filepath).absolute())

parser = argparse.ArgumentParser()

parser.add_argument("--models-dir", type=str, default=None, help="base path where models are stored", )
parser.add_argument("--ckpt-dir", type=normalized_filepath, default=None, help="Path to directory with stable diffusion checkpoints")
parser.add_argument("--vae-dir", type=normalized_filepath, default=None, help="Path to directory with VAE files")
parser.add_argument("--lora-dir", type=normalized_filepath, default=None, help="Path to directory with LoRA files")
parser.add_argument("--branch", type=str, default="main", help="Branch name for TrainTrain")
parser.add_argument("--skip-python-version-check", action='store_true', help="launch.py argument: do not check python version")
parser.add_argument("--skip-torch-cuda-test", action='store_true', help="launch.py argument: do not check if CUDA is able to work properly")
parser.add_argument("--reinstall-xformers", action='store_true', help="launch.py argument: install the appropriate version of xformers even if you have some version already installed")
parser.add_argument("--reinstall-torch", action='store_true', help="launch.py argument: install the appropriate version of torch even if you have some version already installed")
parser.add_argument("--disable-update", action='store_true', help="Disable auto-update of TrainTrain")
parser.add_argument("--skip-prepare-environment", action='store_true', help="launch.py argument: skip all environment preparation")
parser.add_argument("--skip-install", action='store_true', help="launch.py argument: skip installation of packages")
parser.add_argument("--dump-sysinfo", action='store_true', help="launch.py argument: dump limited sysinfo file (without information about extensions, options) to disk and quit")
parser.add_argument("--ngrok", type=str, help="ngrok authtoken, alternative to gradio --share", default=None)
parser.add_argument("--xformers", action='store_true', help="enable xformers for cross attention layers")
parser.add_argument("--use-cpu", nargs='+', help="use CPU as torch device for specified modules", default=[], type=str.lower)
parser.add_argument("--use-ipex", action="store_true", help="use Intel XPU as torch device")
parser.add_argument("--thema", type=str, default="origin", help='change gradio thema, "base","default","origin","citrus","monochrome","soft","glass","ocean"')

args, _ = parser.parse_known_args()

python = sys.executable
git = os.environ.get('GIT', "git")
index_url = os.environ.get('INDEX_URL', "")

# Whether to default to printing command output
default_command_live = (os.environ.get('WEBUI_LAUNCH_LIVE_OUTPUT') == "1")

os.environ.setdefault('GRADIO_ANALYTICS_ENABLED', 'False')

def print_error_explanation(message):
    
    lines = message.strip().split("\n")
    max_len = max([len(x) for x in lines])

    print('=' * max_len, file=sys.stderr)
    for line in lines:
        print(line, file=sys.stderr)
    print('=' * max_len, file=sys.stderr)

def check_python_version():
    is_windows = platform.system() == "Windows"
    major = sys.version_info.major
    minor = sys.version_info.minor
    micro = sys.version_info.micro

    if is_windows:
        supported_minors = [10]
    else:
        supported_minors = [7, 8, 9, 10, 11]

    if not (major == 3 and minor in supported_minors):
        print_error_explanation(f"""
INCOMPATIBLE PYTHON VERSION

This program is tested with 3.10.6 Python, but you have {major}.{minor}.{micro}.
If you encounter an error with "RuntimeError: Couldn't install torch." message,
or any other error regarding unsuccessful package (library) installation,
please downgrade (or upgrade) to the latest version of 3.10 Python
and delete current Python and "venv" folder in WebUI's directory.

You can download 3.10 Python from here: https://www.python.org/downloads/release/python-3106/
""")

def run(command, desc=None, errdesc=None, custom_env=None, live: bool = default_command_live) -> str:
    if desc is not None:
        print(desc)

    run_kwargs = {
        "args": command,
        "shell": True,
        "env": os.environ if custom_env is None else custom_env,
        "encoding": 'utf8',
        "errors": 'ignore',
    }

    if not live:
        run_kwargs["stdout"] = run_kwargs["stderr"] = subprocess.PIPE

    result = subprocess.run(**run_kwargs)

    if result.returncode != 0:
        error_bits = [
            f"{errdesc or 'Error running command'}.",
            f"Command: {command}",
            f"Error code: {result.returncode}",
        ]
        if result.stdout:
            error_bits.append(f"stdout: {result.stdout}")
        if result.stderr:
            error_bits.append(f"stderr: {result.stderr}")
        raise RuntimeError("\n".join(error_bits))

    return (result.stdout or "")


def is_installed(package):
    try:
        dist = importlib.metadata.distribution(package)
    except importlib.metadata.PackageNotFoundError:
        try:
            spec = importlib.util.find_spec(package)
        except ModuleNotFoundError:
            return False

        return spec is not None

    return dist is not None


def run_pip(command, desc=None, live=default_command_live):
    if args.skip_install:
        return

    index_url_line = f' --index-url {index_url}' if index_url != '' else ''
    return run(f'"{python}" -m pip {command} --prefer-binary{index_url_line}', desc=f"Installing {desc}", errdesc=f"Couldn't install {desc}", live=live)


def check_run_python(code: str) -> bool:
    result = subprocess.run([python, "-c", code], capture_output=True, shell=False)
    return result.returncode == 0


def git_fix_workspace(dir, name):
    run(f'"{git}" -C "{dir}" fetch --refetch --no-auto-gc', f"Fetching all contents for {name}", f"Couldn't fetch {name}", live=True)
    run(f'"{git}" -C "{dir}" gc --aggressive --prune=now', f"Pruning {name}", f"Couldn't prune {name}", live=True)
    return


def run_git(dir, name, command, desc=None, errdesc=None, custom_env=None, live=True, autofix=True):
    try:
        return run(f'"{git}" -C "{dir}" {command}', desc=desc, errdesc=errdesc, custom_env=custom_env, live=live)
    except RuntimeError:
        if not autofix:
            raise

    print(f"{errdesc}, attempting autofix...")
    git_fix_workspace(dir, name)

    return run(f'"{git}" -C "{dir}" {command}', desc=desc, errdesc=errdesc, custom_env=custom_env, live=live)


def git_clone(url, dir, name, commithash=None, pull=False):
    # TODO clone into temporary dir and move if successful

    if os.path.exists(dir):
        if commithash is None:
            if pull:
                try:
                    result = subprocess.run(
                        [git, '-C', dir, 'pull'],
                        check=True,
                        capture_output=True,
                        text=True
                    )
                    print(f"Pull successful in {dir}:\n{result.stdout}")
                except subprocess.CalledProcessError as e:
                    print(f"Error pulling in {dir}:\n{e.stderr}")
            return

        current_hash = run_git(dir, name, 'rev-parse HEAD', None, f"Couldn't determine {name}'s hash: {commithash}", live=False).strip()
        if current_hash == commithash:
            print("current_hash == commithash")
            return

        if run_git(dir, name, 'config --get remote.origin.url', None, f"Couldn't determine {name}'s origin URL", live=False).strip() != url:
            run_git(dir, name, f'remote set-url origin "{url}"', None, f"Failed to set {name}'s origin URL", live=False)

        run_git(dir, name, 'fetch', f"Fetching updates for {name}...", f"Couldn't fetch {name}", autofix=False)

        run_git(dir, name, f'checkout {commithash}', f"Checking out commit for {name} with hash: {commithash}...", f"Couldn't checkout commit {commithash} for {name}", live=True)
            
        return

    try:
        run(f'"{git}" clone --config core.filemode=false "{url}" "{dir}"', f"Cloning {name} into {dir}...", f"Couldn't clone {name}", live=True)
    except RuntimeError:
        shutil.rmtree(dir, ignore_errors=True)
        raise

    if commithash is not None:
        run(f'"{git}" -C "{dir}" checkout {commithash}', None, "Couldn't checkout {name}'s hash: {commithash}")

re_requirement = re.compile(r"\s*([-_a-zA-Z0-9]+)\s*(?:==\s*([-+_.a-zA-Z0-9]+))?\s*")


def prepare_environment():
    tt_repo = "https://github.com/hako-mikan/sd-webui-traintrain.git"
    tt_branch = args.branch
    torch_index_url = os.environ.get('TORCH_INDEX_URL', "https://download.pytorch.org/whl/cu121")
    torch_command = os.environ.get('TORCH_COMMAND', f"pip install torch==2.3.1 torchvision==0.18.1 --extra-index-url {torch_index_url}")
    if args.use_ipex:
        if platform.system() == "Windows":
            # The "Nuullll/intel-extension-for-pytorch" wheels were built from IPEX source for Intel Arc GPU: https://github.com/intel/intel-extension-for-pytorch/tree/xpu-main
            # This is NOT an Intel official release so please use it at your own risk!!
            # See https://github.com/Nuullll/intel-extension-for-pytorch/releases/tag/v2.0.110%2Bxpu-master%2Bdll-bundle for details.
            #
            # Strengths (over official IPEX 2.0.110 windows release):
            #   - AOT build (for Arc GPU only) to eliminate JIT compilation overhead: https://github.com/intel/intel-extension-for-pytorch/issues/399
            #   - Bundles minimal oneAPI 2023.2 dependencies into the python wheels, so users don't need to install oneAPI for the whole system.
            #   - Provides a compatible torchvision wheel: https://github.com/intel/intel-extension-for-pytorch/issues/465
            # Limitation:
            #   - Only works for python 3.10
            url_prefix = "https://github.com/Nuullll/intel-extension-for-pytorch/releases/download/v2.0.110%2Bxpu-master%2Bdll-bundle"
            torch_command = os.environ.get('TORCH_COMMAND', f"pip install {url_prefix}/torch-2.0.0a0+gite9ebda2-cp310-cp310-win_amd64.whl {url_prefix}/torchvision-0.15.2a0+fa99a53-cp310-cp310-win_amd64.whl {url_prefix}/intel_extension_for_pytorch-2.0.110+gitc6ea20b-cp310-cp310-win_amd64.whl")
        else:
            # Using official IPEX release for linux since it's already an AOT build.
            # However, users still have to install oneAPI toolkit and activate oneAPI environment manually.
            # See https://intel.github.io/intel-extension-for-pytorch/index.html#installation for details.
            torch_index_url = os.environ.get('TORCH_INDEX_URL', "https://pytorch-extension.intel.com/release-whl/stable/xpu/us/")
            torch_command = os.environ.get('TORCH_COMMAND', f"pip install torch==2.0.0a0 intel-extension-for-pytorch==2.0.110+gitba7f6c1 --extra-index-url {torch_index_url}")
    requirements_file = os.environ.get('REQS_FILE', "requirements_versions.txt")

    xformers_package = os.environ.get('XFORMERS_PACKAGE', 'xformers==0.0.27')
    clip_package = os.environ.get('CLIP_PACKAGE', "https://github.com/openai/CLIP/archive/d50d76daa670286dd6cacf3bcd80b5e4823fc8e1.zip")
    openclip_package = os.environ.get('OPENCLIP_PACKAGE', "https://github.com/mlfoundations/open_clip/archive/bb6e834e9c70d9c27d0dc3ecedeebeaeb1ffad6b.zip")

    try:
        # the existence of this file is a signal to webui.sh/bat that webui needs to be restarted when it stops execution
        os.remove(os.path.join(script_path, "tmp", "restart"))
        os.environ.setdefault('SD_WEBUI_RESTARTING', '1')
    except OSError:
        pass

    if not args.skip_python_version_check:
        check_python_version()

    print(f"Python {sys.version}")
    if args.reinstall_torch or not is_installed("torch") or not is_installed("torchvision"):
        run(f'"{python}" -m {torch_command}', "Installing torch and torchvision", "Couldn't install torch", live=True)

    if args.use_ipex:
        args.skip_torch_cuda_test = True
        
    if not args.skip_torch_cuda_test and not check_run_python("import torch; assert torch.cuda.is_available()"):
        raise RuntimeError(
            'Your device does not support the current version of Torch/CUDA! Consider download another version: \n'
            'https://github.com/lllyasviel/stable-diffusion-webui-forge/releases/tag/latest'
            # 'Torch is not able to use GPU; '
            # 'add --skip-torch-cuda-test to COMMANDLINE_ARGS variable to disable this check'
        )
        
    if not is_installed("clip"):
        run_pip(f"install {clip_package}", "clip")

    if not is_installed("open_clip"):
        run_pip(f"install {openclip_package}", "open_clip")

    if (not is_installed("xformers") or args.reinstall_xformers) and args.xformers:
        run_pip(f"install -U -I --no-deps {xformers_package}", "xformers")

    if not is_installed("ngrok") and args.ngrok:
        run_pip("install ngrok", "ngrok")
        
    git_clone(tt_repo, os.path.join(script_path, "traintrain"), "traintrain", tt_branch, pull=True)
    if not args.disable_update:
        git_pull("traintrain")

    if not os.path.isfile(requirements_file):
        requirements_file = os.path.join(script_path, requirements_file)

    run_pip(f"install -r \"{requirements_file}\"", "requirements")

def start():
    print(f"Launching {'API server' if '--nowebui' in sys.argv else 'Web UI'} with arguments: {shlex.join(sys.argv[1:])}")
    import traintrain.scripts.traintrain as traintrain
    traintrain.launch()
    return

def dump_sysinfo():
    from modules import sysinfo
    import datetime

    text = sysinfo.get()
    filename = f"sysinfo-{datetime.datetime.utcnow().strftime('%Y-%m-%d-%H-%M')}.json"

    with open(filename, "w", encoding="utf8") as file:
        file.write(text)

    return filename
    
def git_pull(dirpath='.'):
    try:
        result = subprocess.run(
            ['git', '-C', dirpath, 'pull'],
            check=True,
            capture_output=True,
            text=True
        )
        print(f"Pull successful in {dirpath}:\n{result.stdout}")
    except subprocess.CalledProcessError as e:
        print(f"Error pulling in {dirpath}:\n{e.stderr}")