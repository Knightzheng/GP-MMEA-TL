import argparse
import platform
import shutil
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path.cwd()
ENV_DIR = PROJECT_ROOT / "env"

# Main environment aligned with your plan: modern PyTorch stack.
DEFAULT_MAIN_ENV = "bysj-main"
DEFAULT_MAIN_PY = "3.10"
DEFAULT_CUDA = "cu126"

CORE_PACKAGES = [
    "numpy==1.26.4",
    "pandas==2.2.3",
    "scipy==1.14.1",
    "scikit-learn==1.6.1",
    "matplotlib==3.10.0",
    "seaborn==0.13.2",
    "tqdm==4.67.1",
    "pyyaml==6.0.2",
    "opencv-python==4.10.0.84",
    "jupyterlab==4.3.4",
    "tensorboard==2.18.0",
    "wandb==0.19.1",
    "mlflow==2.19.0",
    "transformers==4.47.1",
    "datasets==3.2.0",
    "sentencepiece==0.2.0",
]

TORCH_PACKAGES = ["torch", "torchvision", "torchaudio"]


def run(cmd, check=True, capture=False):
    print(f"\n[RUN] {' '.join(cmd)}")
    result = subprocess.run(
        cmd,
        check=False,
        text=True,
        capture_output=capture,
        encoding="utf-8",
        errors="replace",
    )
    if check and result.returncode != 0:
        stderr = f"\nSTDERR:\n{result.stderr}" if capture else ""
        raise RuntimeError(f"Command failed ({result.returncode}): {' '.join(cmd)}{stderr}")
    return result


def command_exists(name):
    return shutil.which(name) is not None


def detect_manager(preferred):
    if preferred in ("conda", "venv"):
        return preferred
    return "conda" if command_exists("conda") else "venv"


def conda_env_exists(env_name):
    try:
        out = run(["conda", "env", "list"], check=True, capture=True).stdout
    except Exception:
        return False

    for line in out.splitlines():
        text = line.strip()
        if not text or text.startswith("#"):
            continue
        if text.split()[0] == env_name:
            return True
    return False


def ensure_conda_env(env_name, python_version):
    if conda_env_exists(env_name):
        print(f"[INFO] conda env '{env_name}' already exists, skip create.")
        return
    run(["conda", "create", "-y", "-n", env_name, f"python={python_version}"])


def ensure_venv_env(env_name, python_version):
    venv_dir = PROJECT_ROOT / ".venv" / env_name
    if venv_dir.exists():
        print(f"[INFO] venv '{venv_dir}' already exists, skip create.")
        return venv_dir

    if platform.system().lower().startswith("win"):
        run(["py", f"-{python_version}", "-m", "venv", str(venv_dir)])
    else:
        run(["python3", "-m", "venv", str(venv_dir)])
    return venv_dir


def get_venv_python(venv_dir):
    if platform.system().lower().startswith("win"):
        return str(venv_dir / "Scripts" / "python.exe")
    return str(venv_dir / "bin" / "python")


def conda_pip(env_name, args):
    run(["conda", "run", "-n", env_name, "python", "-m", "pip"] + args)


def venv_pip(venv_dir, args):
    run([get_venv_python(venv_dir), "-m", "pip"] + args)


def install_core(manager, env_name, venv_dir):
    if manager == "conda":
        conda_pip(env_name, ["install", "--no-user", "--upgrade", "pip", "setuptools", "wheel"])
        conda_pip(env_name, ["install", "--no-user"] + CORE_PACKAGES)
        return
    venv_pip(venv_dir, ["install", "--no-user", "--upgrade", "pip", "setuptools", "wheel"])
    venv_pip(venv_dir, ["install", "--no-user"] + CORE_PACKAGES)


def install_torch(manager, env_name, venv_dir, cuda_variant):
    if cuda_variant == "cpu":
        extra = ["--index-url", "https://download.pytorch.org/whl/cpu"]
    elif cuda_variant == "cu126":
        extra = ["--index-url", "https://download.pytorch.org/whl/cu126"]
    else:
        raise ValueError(f"Unsupported cuda option: {cuda_variant}")

    if manager == "conda":
        conda_pip(env_name, ["install", "--no-user"] + TORCH_PACKAGES + extra)
        return
    venv_pip(venv_dir, ["install", "--no-user"] + TORCH_PACKAGES + extra)


def install_pyg(manager, env_name, venv_dir):
    # Official installation endpoint based on torch+cuda wheel matrix.
    index = "https://data.pyg.org/whl/torch-2.5.0+cu126.html"
    packages = ["pyg_lib", "torch_scatter", "torch_sparse", "torch_cluster", "torch_spline_conv", "torch_geometric"]

    if manager == "conda":
        conda_pip(env_name, ["install", "--no-user"] + packages + ["-f", index])
        return
    venv_pip(venv_dir, ["install", "--no-user"] + packages + ["-f", index])


def check_and_report_tools():
    print("\n=== Host Tools Check ===")
    checks = ["git", "nvidia-smi", "nvcc", "conda"]
    for name in checks:
        print(f"{name}: {'FOUND' if command_exists(name) else 'MISSING'}")
    print("Note: system-level tools (Git, CUDA toolkit, driver) should be installed outside this script.")


def export_hardware_snapshot():
    ENV_DIR.mkdir(parents=True, exist_ok=True)
    snapshot_path = ENV_DIR / "hardware_snapshot.txt"

    lines = []
    lines.append("=== System ===")
    lines.append(f"Platform: {platform.platform()}")
    lines.append(f"Python executable: {sys.executable}")
    lines.append(f"Python version: {platform.python_version()}")
    lines.append("")

    for cmd in (["nvidia-smi"], ["nvcc", "--version"], ["conda", "--version"]):
        lines.append(f"=== {' '.join(cmd)} ===")
        try:
            out = run(cmd, check=False, capture=True)
            if out.stdout:
                lines.append(out.stdout.strip())
            if out.stderr:
                lines.append(out.stderr.strip())
            lines.append(f"exit_code={out.returncode}")
        except Exception as e:
            lines.append(f"error: {e}")
        lines.append("")

    snapshot_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"[INFO] Wrote hardware snapshot: {snapshot_path}")


def export_locks(manager, env_name, venv_dir):
    ENV_DIR.mkdir(parents=True, exist_ok=True)
    req_lock = ENV_DIR / "requirements.lock.txt"

    if manager == "conda":
        conda_yaml = ENV_DIR / "conda-pytorch.yaml"
        env_data = run(["conda", "env", "export", "-n", env_name, "--no-builds"], check=True, capture=True).stdout
        conda_yaml.write_text(env_data, encoding="utf-8")
        freeze = run(["conda", "run", "-n", env_name, "python", "-m", "pip", "freeze"], check=True, capture=True).stdout
        req_lock.write_text(freeze, encoding="utf-8")
        print(f"[INFO] Wrote conda env yaml: {conda_yaml}")
        print(f"[INFO] Wrote pip lock: {req_lock}")
        return

    freeze = run([get_venv_python(venv_dir), "-m", "pip", "freeze"], check=True, capture=True).stdout
    req_lock.write_text(freeze, encoding="utf-8")
    print(f"[INFO] Wrote pip lock: {req_lock}")


def verify_env(manager, env_name, venv_dir):
    code = (
        "import sys;print('Python:',sys.version.split()[0]);"
        "import torch;print('Torch:',torch.__version__);"
        "print('Torch CUDA:',torch.version.cuda);"
        "print('CUDA available:',torch.cuda.is_available())"
    )
    if manager == "conda":
        run(["conda", "run", "-n", env_name, "python", "-c", code], check=False)
        return
    run([get_venv_python(venv_dir), "-c", code], check=False)


def print_activation_hint(manager, env_name, venv_dir):
    print("\n[DONE] Main environment ready.")
    if manager == "conda":
        print(f"Activate: conda activate {env_name}")
    else:
        if platform.system().lower().startswith("win"):
            print(f"Activate: {venv_dir}\\Scripts\\activate")
        else:
            print(f"Activate: source {venv_dir}/bin/activate")

    print("Recommended next step:")
    print("  python test.py")


def print_legacy_note():
    print("\n[LEGACY NOTE] TF1.10/MMEA environment is intentionally not auto-installed.")
    print("Reason: legacy stack (Python 3.6 + TensorFlow 1.10) often conflicts with modern Windows setup.")
    print("Recommended: isolate with Docker/WSL2 or a separate dedicated conda env.")


def main():
    parser = argparse.ArgumentParser(
        description="Graduation-project bootstrap: create virtual env and install core packages."
    )
    parser.add_argument("--manager", choices=["auto", "conda", "venv"], default="auto")
    parser.add_argument("--env-name", default=DEFAULT_MAIN_ENV)
    parser.add_argument("--python-version", default=DEFAULT_MAIN_PY)
    parser.add_argument("--cuda", choices=["cu126", "cpu"], default=DEFAULT_CUDA)
    parser.add_argument("--skip-torch", action="store_true")
    parser.add_argument("--install-pyg", action="store_true")
    parser.add_argument("--with-legacy-note", action="store_true")
    args = parser.parse_args()

    manager = detect_manager(args.manager)
    print(f"[INFO] manager={manager}")
    print(f"[INFO] env_name={args.env_name}")
    print(f"[INFO] python={args.python_version}")
    print(f"[INFO] torch_variant={'skip' if args.skip_torch else args.cuda}")

    check_and_report_tools()

    venv_dir = None
    if manager == "conda":
        ensure_conda_env(args.env_name, args.python_version)
    else:
        venv_dir = ensure_venv_env(args.env_name, args.python_version)

    install_core(manager, args.env_name, venv_dir)
    if not args.skip_torch:
        install_torch(manager, args.env_name, venv_dir, args.cuda)
    if args.install_pyg:
        install_pyg(manager, args.env_name, venv_dir)

    verify_env(manager, args.env_name, venv_dir)
    export_hardware_snapshot()
    export_locks(manager, args.env_name, venv_dir)
    if args.with_legacy_note:
        print_legacy_note()
    print_activation_hint(manager, args.env_name, venv_dir)


if __name__ == "__main__":
    main()
