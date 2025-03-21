import argparse
import subprocess
import platform
import torch
import psutil
import shutil
import sys

def get_system_info():
    """ Collect system information for debugging """
    info = {
        "Python Version": sys.version,
        "PIP Version": subprocess.getoutput("pip --version"),
        "OS": platform.system(),
        "OS Version": platform.version(),
        "Architecture": platform.architecture()[0],
        "CPU Cores": psutil.cpu_count(logical=True),
        "Memory (Total)": f"{psutil.virtual_memory().total / 1e9:.2f} GB",
        "Memory (Available)": f"{psutil.virtual_memory().available / 1e9:.2f} GB",
        "GPU Detected": torch.cuda.is_available(),
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

    # Print system info
    print("\n🚀 Initialization Debug Info:")
    for key, value in info.items():
        print(f"  ➤ {key}: {value}")

    return info

def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Launch Gradio Interface")
    parser.add_argument("--listen", type=str, default="127.0.0.1", help="IP address to listen on")
    parser.add_argument("--port", type=int, default=7860, help="Port number")

    args = parser.parse_args()

    # Print system info
    get_system_info()

    # Check if Gradio is installed
    if not shutil.which("gradio"):
        print("⚠️ Gradio is not installed. Installing now...")
        subprocess.run("pip install gradio", shell=True, check=True)

    # Launch Gradio interface
    command = f"python interface.py --server_name {args.listen} --server_port {args.port}"

    try:
        print(f"\n🚀 Launching Gradio interface at {args.listen}:{args.port}...")
        subprocess.run(command, shell=True, check=True)
    except KeyboardInterrupt:
        print("\n❌ Server interrupted by user.")
    except Exception as e:
        print(f"⚠️ Error: {e}")

if __name__ == "__main__":
    main()
