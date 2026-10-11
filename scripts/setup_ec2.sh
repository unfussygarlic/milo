#!/bin/bash
set -e

echo "=== Milo SLM v2 — EC2 Setup ==="

# The Deep Learning AMI has PyTorch in a conda env
source activate pytorch 2>/dev/null || conda activate pytorch 2>/dev/null || true

pip install tiktoken

# Verify
python3 -c "
import torch
print(f'PyTorch: {torch.__version__}')
print(f'CUDA: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name()}')
    print(f'VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB')
else:
    print('ERROR: No GPU!')
    exit(1)
import tiktoken
print('tiktoken: OK')
"

mkdir -p checkpoints_v2
echo ""
echo "=== Ready! ==="
echo "Run: tmux new -s train"
echo "Then: python3 train.py --data_dir data/pretrain_data/curriculum"