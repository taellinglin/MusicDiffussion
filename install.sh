#!/bin/bash

# Install base dependencies
pip install -r requirements.txt

# Check for CUDA support
if python -c "import torch; print(torch.cuda.is_available())" | grep -q "True"; then
    echo "✅ CUDA detected! Installing PyTorch with GPU support..."
    pip install torch==2.2.2+cu124 -f https://download.pytorch.org/whl/torch_stable.html
else
    echo "❌ No CUDA detected. Installing PyTorch for CPU..."
    pip install torch==2.2.2+cpu -f https://download.pytorch.org/whl/torch_stable.html
fi
