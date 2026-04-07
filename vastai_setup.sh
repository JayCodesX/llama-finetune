#!/bin/bash
# vastai_setup.sh — Run this on your Vast.ai instance after SSH'ing in.
#
# Usage:
#   scp -r -P <port> /Users/jonathan.mascio/Projects/llama-finetune/* root@<ip>:/workspace/
#   ssh -p <port> root@<ip>
#   cd /workspace && bash vastai_setup.sh
#
# Then:
#   python train.py      # ~1-2 hours on RTX 3090
#   python evaluate.py   # ~15 min

set -euo pipefail

echo "=== Vast.ai QLoRA Setup ==="
echo ""

# ── 1. System check ─────────────────────────────────────────────────────────

echo "[1/5] Checking GPU..."
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo ""

# ── 2. Install dependencies ─────────────────────────────────────────────────

echo "[2/5] Installing Python packages..."
pip install --upgrade pip
pip install "unsloth[cu121-torch240] @ git+https://github.com/unslothai/unsloth.git"
pip install datasets trl peft bitsandbytes accelerate
echo ""

# ── 3. Prepare dataset (if not already present) ─────────────────────────────

if [ ! -f /workspace/dataset.jsonl ]; then
    echo "[3/5] Preparing dataset..."
    python /workspace/prepare_dataset.py
else
    echo "[3/5] dataset.jsonl already exists, skipping preparation"
    wc -l /workspace/dataset.jsonl
fi
echo ""

# ── 4. Verify dataset ───────────────────────────────────────────────────────

LINES=$(wc -l < /workspace/dataset.jsonl)
echo "[4/5] Dataset: ${LINES} examples"

if [ "$LINES" -lt 10 ]; then
    echo "WARNING: Very small dataset (${LINES} examples). Consider adding more."
fi
echo ""

# ── 5. Quick sanity check — can we load the model? ──────────────────────────

echo "[5/5] Testing model load (this downloads ~35GB on first run)..."
python -c "
from unsloth import FastLanguageModel
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name='unsloth/Llama-3.3-70B-Instruct-bnb-4bit',
    max_seq_length=2048,
    load_in_4bit=True,
)
print(f'Model loaded! Parameters: {sum(p.numel() for p in model.parameters()):,}')
del model
import torch; torch.cuda.empty_cache()
print('Sanity check passed.')
"

echo ""
echo "=== Setup complete ==="
echo ""
echo "Next steps:"
echo "  python train.py       # Start training (~1-2 hrs)"
echo "  python evaluate.py    # Compare base vs fine-tuned"
echo ""
echo "When done, download your adapter:"
echo "  # From your LOCAL machine:"
echo "  scp -r -P <port> root@<ip>:/workspace/output/final-adapter ./my-adapter"
echo ""
echo "Then DESTROY the instance on Vast.ai dashboard to stop billing."
