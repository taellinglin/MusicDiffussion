@echo off

REM Install base dependencies
pip install -r requirements.txt

REM Check for CUDA support
python -c "import torch; print(torch.cuda.is_available())" > cuda_check.txt
findstr /C:"True" cuda_check.txt > nul

IF %ERRORLEVEL% EQU 0 (
    echo ✅ CUDA detected! Installing PyTorch with GPU support...
    pip install torch==2.2.2+cu124 -f https://download.pytorch.org/whl/torch_stable.html
) ELSE (
    echo ❌ No CUDA detected. Installing PyTorch for CPU...
    pip install torch==2.2.2+cpu -f https://download.pytorch.org/whl/torch_stable.html
)

del cuda_check.txt
