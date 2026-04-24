#!/usr/bin/env bash
set -euo pipefail

VENV_DIR=".venv"

if [ ! -d "$VENV_DIR" ]; then
    echo "ERROR: Virtual environment not found. Run setup.sh first."
    exit 1
fi

source "$VENV_DIR/bin/activate"

echo "==> Authenticating with Modal..."
modal setup

echo "==> Creating HuggingFace secret (you'll need your HF token)..."
if modal secret list 2>/dev/null | grep -q huggingface-secret; then
    echo "    huggingface-secret already exists, skipping."
else
    echo "    Create a secret named 'huggingface-secret' with your HF_TOKEN:"
    echo "    modal secret create huggingface-secret HF_TOKEN=hf_..."
    echo "    (skipping automatic creation — set it manually)"
fi

echo ""
echo "==> Deploying Modal services..."

echo "  [1/4] Retrieval service..."
modal deploy modal_retrieval_app.py

echo "  [2/4] vLLM 8B (L4)..."
modal deploy src/serving/modal_vllm_8b.py

echo "  [3/4] vLLM 30B-A3B (A100-80GB)..."
modal deploy src/serving/modal_vllm_30b.py

echo "  [4/4] vLLM 32B (A100-80GB)..."
modal deploy src/serving/modal_vllm_32b.py

echo ""
echo "==> Syncing FAISS index to Modal volume..."
echo "    (requires a built index at data/indexes/default/)"
if [ -d "data/indexes/default" ]; then
    python scripts/sync_modal_index.py \
        --local-index-dir data/indexes/default \
        --volume-name nlp-videoqa-index \
        --remote-index-subdir indexes/default
else
    echo "    No local index found — build one first with:"
    echo "    python scripts/build_index.py --device cpu"
fi

echo ""
echo "Modal setup complete. Endpoints:"
echo "  8B:  https://<your-modal-workspace>--nlp-videoqa-vllm-8b-serve.modal.run/v1"
echo "  30B: https://<your-modal-workspace>--nlp-videoqa-vllm-30b-serve.modal.run/v1"
echo "  32B: https://<your-modal-workspace>--nlp-videoqa-vllm-32b-serve.modal.run/v1"
