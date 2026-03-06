import subprocess
import sys


def run_step(script_name):
    print(f"Running {script_name}...")
    result = subprocess.run([sys.executable, script_name], check=False)
    if result.returncode != 0:
        raise SystemExit(f"{script_name} failed with exit code {result.returncode}")


if __name__ == "__main__":
    run_step("train_text.py")
    run_step("train_audio.py")
    print("Done. Text and audio models trained separately.")
