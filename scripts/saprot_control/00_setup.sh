#!/bin/bash
# Setup: Install Foldseek + cache SaProt model
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

echo "=== SaProt Control Experiment Setup ==="

# 1. Download Foldseek
if [ ! -f "foldseek" ]; then
    echo "Downloading Foldseek..."
    wget -q https://mmseqs.com/foldseek/foldseek-linux-avx2.tar.gz
    tar xzf foldseek-linux-avx2.tar.gz
    cp foldseek/bin/foldseek ./foldseek
    rm -rf foldseek-linux-avx2.tar.gz foldseek/
    chmod +x foldseek
    echo "  Foldseek installed: $(./foldseek version 2>/dev/null || echo 'OK')"
else
    echo "  Foldseek already installed"
fi

# 2. Cache SaProt 650M model (via hf-mirror)
export HF_ENDPOINT=https://hf-mirror.com
echo "Caching SaProt 650M model (mirror: $HF_ENDPOINT)..."
python3 -c "
import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
from transformers import AutoTokenizer, EsmForMaskedLM
print('  Downloading tokenizer...')
AutoTokenizer.from_pretrained('westlake-repl/SaProt_650M_AF2')
print('  Downloading model...')
EsmForMaskedLM.from_pretrained('westlake-repl/SaProt_650M_AF2')
print('  SaProt 650M cached successfully')
"

echo "=== Setup complete ==="
