import platform
import re
import subprocess
import sys


def run_cmd(cmd):
    try:
        result = subprocess.run(
            cmd,
            shell=True,
            check=False,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        return result.returncode, (result.stdout or "").strip(), (result.stderr or "").strip()
    except Exception as e:
        return -1, "", str(e)


def check_python():
    print("=== Python 信息 ===")
    print(f"Python 可执行文件: {sys.executable}")
    print(f"Python 版本: {platform.python_version()}")
    print(f"完整版本信息: {sys.version}")
    print()


def check_conda():
    print("=== Anaconda/Conda 信息 ===")
    code, out, err = run_cmd("conda --version")
    if code != 0:
        print("未检测到 conda 命令，或 conda 未加入 PATH。")
        if err:
            print(f"错误信息: {err}")
        print()
        return

    print(f"Conda 版本: {out}")

    code, out, err = run_cmd("conda info")
    if code == 0:
        # 仅输出关键信息，避免太长
        install_line = ""
        env_line = ""
        for line in out.splitlines():
            if "active environment" in line:
                env_line = line.strip()
            if "base environment" in line:
                install_line = line.strip()
        if env_line:
            print(env_line)
        if install_line:
            print(install_line)
        print("结论: conda 可正常使用。")
    else:
        print("conda 命令可用，但执行 conda info 失败。")
        if err:
            print(f"错误信息: {err}")
    print()


def check_cuda_nvidia_smi():
    print("=== CUDA 信息 (nvidia-smi) ===")
    code, out, err = run_cmd("nvidia-smi")
    if code != 0:
        print("未检测到 nvidia-smi，或 NVIDIA 驱动未正确安装。")
        if err:
            print(f"错误信息: {err}")
        print()
        return

    # 从 nvidia-smi 输出中提取 CUDA Version
    m = re.search(r"CUDA Version:\s*([0-9.]+)", out)
    if m:
        print(f"驱动报告的 CUDA 版本: {m.group(1)}")
    else:
        print("未在 nvidia-smi 输出中解析到 CUDA Version。")
    print()


def check_cuda_nvcc():
    print("=== CUDA Toolkit 信息 (nvcc) ===")
    code, out, err = run_cmd("nvcc --version")
    if code != 0:
        print("未检测到 nvcc（可能未安装 CUDA Toolkit 或未加入 PATH）。")
        if err:
            print(f"错误信息: {err}")
        print()
        return

    # 常见格式: release 12.4, V12.4.131
    m = re.search(r"release\s+([0-9.]+)", out)
    if m:
        print(f"nvcc 报告的 CUDA Toolkit 版本: {m.group(1)}")
    else:
        print("未在 nvcc 输出中解析到 release 版本号。")
    print("nvcc 原始输出:")
    print(out)
    print()


def check_torch_cuda():
    print("=== PyTorch CUDA 信息 (可选) ===")
    try:
        import torch

        print(f"PyTorch 版本: {torch.__version__}")
        print(f"torch.version.cuda: {torch.version.cuda}")
        print(f"CUDA 是否可用: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"GPU 数量: {torch.cuda.device_count()}")
            print(f"当前 GPU: {torch.cuda.get_device_name(torch.cuda.current_device())}")
    except Exception as e:
        print(f"未能导入或读取 PyTorch 信息: {e}")
    print()


if __name__ == "__main__":
    check_python()
    check_conda()
    check_cuda_nvidia_smi()
    check_cuda_nvcc()
    check_torch_cuda()
