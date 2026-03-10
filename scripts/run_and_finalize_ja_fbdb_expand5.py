import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PY = r"D:\Anaconda_envs\envs\bysj-main\python.exe"


def run_cmd(cmd):
    print(f"[RUN] {' '.join(cmd)}")
    proc = subprocess.run(cmd, cwd=str(ROOT), check=False)
    if proc.returncode != 0:
        raise SystemExit(proc.returncode)


def main():
    run_cmd([PY, "scripts/run_transfer_adapt_ja_fbdb_expand5_next.py", "--run-missing", "1"])
    run_cmd([PY, "scripts/finalize_ja_fbdb_expand5_after_run.py"])
    print("[DONE] run + finalize completed.")


if __name__ == "__main__":
    main()
