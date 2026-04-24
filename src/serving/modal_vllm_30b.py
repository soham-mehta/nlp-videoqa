from __future__ import annotations

import json

import modal

MODEL_NAME = "Qwen/Qwen3-VL-30B-A3B-Instruct-FP8"
GPU = "A100-80GB"
MAX_MODEL_LEN = 32768
MAX_NUM_SEQS = 8
MAX_IMAGES_PER_PROMPT = 8
PORT = 8000
APP_NAME = "nlp-videoqa-vllm-30b"
HF_SECRET_NAME = "huggingface-secret"

image = (
    modal.Image.from_registry("nvidia/cuda:12.9.0-devel-ubuntu22.04", add_python="3.12")
    .entrypoint([])
    .uv_pip_install(
        "vllm==0.19.1",
        "huggingface_hub==0.36.0",
        "hf_transfer>=0.1.9",
    )
    .env(
        {
            "HF_HUB_ENABLE_HF_TRANSFER": "1",
            "HF_XET_HIGH_PERFORMANCE": "1",
            "VLLM_WORKER_MULTIPROC_METHOD": "spawn",
            "TOKENIZERS_PARALLELISM": "false",
        }
    )
)

hf_cache = modal.Volume.from_name("hf-cache", create_if_missing=True)
vllm_cache = modal.Volume.from_name("vllm-cache", create_if_missing=True)
hf_secret = modal.Secret.from_name(HF_SECRET_NAME)
app = modal.App(APP_NAME)


@app.function(
    image=image,
    gpu=GPU,
    volumes={
        "/root/.cache/huggingface": hf_cache,
        "/root/.cache/vllm": vllm_cache,
    },
    secrets=[hf_secret],
    scaledown_window=900,
    timeout=30 * 60,
)
@modal.concurrent(target_inputs=2, max_inputs=4)
@modal.web_server(port=PORT, startup_timeout=600)
def serve():
    import subprocess

    cmd = [
        "vllm",
        "serve",
        MODEL_NAME,
        "--served-model-name", MODEL_NAME,
        "--host", "0.0.0.0",
        "--port", str(PORT),
        "--uvicorn-log-level", "warning",
        "--gpu-memory-utilization", "0.95",
        "--max-model-len", str(MAX_MODEL_LEN),
        "--max-num-seqs", str(MAX_NUM_SEQS),
        "--tensor-parallel-size", "1",
        "--async-scheduling",
        "--enable-prefix-caching",
        "--no-enforce-eager",
        "--limit-mm-per-prompt", json.dumps({"image": MAX_IMAGES_PER_PROMPT, "video": 0, "audio": 0}),
        "--enable-auto-tool-choice",
        "--tool-call-parser", "hermes",
    ]
    print(*cmd)
    subprocess.Popen(cmd)
