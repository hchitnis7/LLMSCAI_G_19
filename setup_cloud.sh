#!/bin/bash
# One-shot setup for a fresh GPU machine (RunPod / Vast.ai / Lambda / Colab terminal ...).
# Usage:  bash setup_cloud.sh            # >= 80 GB GPU (A100-80GB / H100 / RTX PRO 6000): full bf16
#         QUANT=4bit bash setup_cloud.sh # 24-48 GB GPU (L4 / A10 / 4090 / A6000): 4-bit
# Before running: git clone your repo and copy the fixed files over it (api.py, chatbot.py, intents.py,
# src/*, app.py, fass_sections.csv -> llm_ayesha/). Run from the repo root.
set -e
nvidia-smi --query-gpu=name,memory.total --format=csv
pip install -q -r requirements.txt fastapi uvicorn python-multipart langdetect pyngrok
[ -n "$QUANT" ] && export MEDICHECK_QUANT=$QUANT
export HF_HOME=${HF_HOME:-$PWD/.hf}          # keep the 60 GB download on the big disk
export MEDICHECK_NO_CACHE=1
rm -rf demo_cache
python build_sections_db.py                   # Chroma index of the corrected sections
python - <<'PY'                               # download the model now, so the demo doesn't wait on it
from huggingface_hub import snapshot_download
snapshot_download("Qwen/Qwen3-30B-A3B-Instruct-2507")
PY
echo "Setup done. Now:  python api.py   (then in a 2nd terminal: ngrok http 8000)"
