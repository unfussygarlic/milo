#!/bin/bash
# ═══════════════════════════════════════════════════════════════
# Milo — Codebase Restructuring Migration
# ═══════════════════════════════════════════════════════════════
# Run this from your milo project root:
#   chmod +x migrate.sh && ./migrate.sh
#
# What it does:
#   - Creates data/, infra/ directories
#   - Moves data generation scripts into data/
#   - Moves infra/deployment scripts into infra/
#   - Cleans up __pycache__ and training.log
#   - Does NOT touch model.py, train.py, finetune.py,
#     finetune_tools.py, serve.py, react_runtime.py (they stay in root)
# ═══════════════════════════════════════════════════════════════

set -e

echo "═══════════════════════════════════════════════"
echo "  Milo — Codebase Restructuring"
echo "═══════════════════════════════════════════════"
echo ""

# Safety check
if [ ! -f "model.py" ]; then
    echo "ERROR: model.py not found. Run this from the milo project root."
    exit 1
fi

# ── Create directories ──
echo "Creating directories..."
mkdir -p data
mkdir -p infra

# ── Move data scripts ──
echo "Moving data scripts → data/"
[ -f "collect_sft_data.py" ] && mv collect_sft_data.py data/
[ -f "generate_tool_data.py" ] && mv generate_tool_data.py data/
[ -f "data_v2.py" ] && mv data_v2.py data/

# ── Move infra scripts ──
echo "Moving infra scripts → infra/"
[ -f "setup_ec2.sh" ] && mv setup_ec2.sh infra/
[ -f "spot_training_guide.sh" ] && mv spot_training_guide.sh infra/
[ -f "continue_pretrain.py" ] && mv continue_pretrain.py infra/
[ -f "stream_more_data.py" ] && mv stream_more_data.py infra/

# ── Clean up ──
echo "Cleaning up..."
rm -rf __pycache__
[ -f "training.log" ] && rm training.log

echo ""
echo "Done! New layout:"
echo ""

# Show the new tree (if tree is installed, otherwise ls)
if command -v tree &> /dev/null; then
    tree -I 'experiments|.git|__pycache__|*.pyc|.venv' --dirsfirst
else
    echo "Root files:"
    ls -1 *.py 2>/dev/null || true
    echo ""
    echo "data/:"
    ls -1 data/ 2>/dev/null || true
    echo ""
    echo "infra/:"
    ls -1 infra/ 2>/dev/null || true
    echo ""
    echo "research/:"
    ls -1 research/ 2>/dev/null || true
    echo ""
    echo "static/:"
    ls -1 static/ 2>/dev/null || true
fi

echo ""
echo "═══════════════════════════════════════════════"
echo "  Next: review .gitignore and README.md"
echo "═══════════════════════════════════════════════"