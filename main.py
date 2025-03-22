import argparse
import subprocess
import platform
import torch
import psutil
import shutil
import sys
from colorama import Fore, Style, init

# Initialize colorama for colored output on Windows
init(autoreset=True)

# ROYGBIV colors
ROYGBIV = [
    Fore.RED,        # R
    Fore.YELLOW,     # O
    Fore.GREEN,      # Y
    Fore.BLUE,       # B
    Fore.MAGENTA,    # I
    Fore.CYAN,       # V
]

def color_text(text, color_index):
    """ Apply ROYGBIV color cycling """
    return ROYGBIV[color_index % len(ROYGBIV)] + text + Style.RESET_ALL

def get_system_info():
    """ Collect system information for debugging """
    info = {
        "Python Version": sys.version,
        "PIP Version": subprocess.getoutput("pip --version"),
        "OS": platform.system(),
        "OS Version": platform.version(),
        "Architecture": platform.architecture()[0],
        "CPU Cores": str(psutil.cpu_count(logical=True)),
        "Memory (Total)": f"{psutil.virtual_memory().total / 1e9:.2f} GB",
        "Memory (Available)": f"{psutil.virtual_memory().available / 1e9:.2f} GB",
        "GPU Detected": str(torch.cuda.is_available()),
    }

    # GPU details
    if torch.cuda.is_available():
        info["CUDA Version"] = torch.version.cuda
        info["PyTorch Version"] = torch.__version__
        info["GPU Name"] = torch.cuda.get_device_name(0)
        info["GPU Memory (Total)"] = f"{torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB"
    else:
        info["CUDA Version"] = "N/A"
        info["PyTorch Version"] = torch.__version__
        info["GPU Name"] = "None"
        info["GPU Memory (Total)"] = "N/A"

    # Print system info in ROYGBIV colors
    print("\n🌈 🚀 Initialization Debug Info:")
    for i, (key, value) in enumerate(info.items()):
        colored_key = color_text(f"  ➤ {key}:", i)
        colored_value = color_text(f"{value}", i + 1)
        print(f"{colored_key} {colored_value}")

    return info

def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Launch Gradio Interface")
    parser.add_argument("--listen", type=str, default="127.0.0.1", help="IP address to listen on")
    parser.add_argument("--port", type=int, default=7860, help="Port number")

    args = parser.parse_args()

    # Print system info with ROYGBIV colors
    get_system_info()

    # Check if Gradio is installed
    if not shutil.which("gradio"):
        print("⚠️ Gradio is not installed. Installing now...")
        subprocess.run("pip install gradio", shell=True, check=True)

    # Launch Gradio interface
    command = f"python interface.py --listen {args.listen} --port {args.port}"

    try:
        print(f"\n🚀 Launching Gradio interface at {args.listen}:{args.port}...")
        subprocess.run(command, shell=True, check=True)
    except KeyboardInterrupt:
        print("\n❌ Server interrupted by user.")
    except Exception as e:
        print(f"⚠️ Error: {e}")

if __name__ == "__main__":
    main()
