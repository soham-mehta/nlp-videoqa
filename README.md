# Multimodal Video QA

Agentic multimodal RAG system for answering questions about a video corpus. Retrieves
timestamped transcript segments and video frames via a SigLIP2 FAISS index, then
generates answers using Qwen3-VL models served on Modal with vLLM.

## Setup

```bash
bash setup.sh
source .venv/bin/activate
```

Prerequisites: Python 3.14+, `ffmpeg` on PATH.

For Modal GPU infrastructure (vLLM endpoints, remote retrieval):

```bash
bash modal_setup.sh
```

## Project Structure

```
src/
  agentic/          # Tool-calling agent loop (ReAct-style)
    agent.py        # Core agent: tool dispatch, context overflow handling
    answering.py    # AgenticRAGAnsweringService (wraps agent + retrieval)
    tools.py        # Tool schemas and dispatch (semantic_search, get_chunks_by_timestamp, etc.)
    backend.py      # RetrievalBackend protocol
    modal_backend.py        # Modal-backed retrieval backend
    benchmark_backend.py    # Benchmark retrieval backend (pre-indexed)
    mock_backend.py         # Mock backend for testing
  rag/              # Baseline RAG pipeline
    answering.py    # BaselineRAGAnsweringService
    openai_generator.py     # OpenAI-compatible chat generator (vLLM)
    image_utils.py  # Frame encoding (base64) for multimodal prompts
    prompts.py      # System prompts
    schemas.py      # RAGResult, GenerationResult
    generator_base.py       # Abstract generator interface
  retrieval/        # Vector search
    faiss_store.py  # Local FAISS index
    modal_client.py # Remote Modal retrieval client
    service.py      # RetrieverService (embedding + search)
    vector_store.py # Abstract vector store
    schemas.py      # RetrievalResult, ChunkResult
  indexing/
    build.py        # FAISS index builder (text + frame chunks)
  models/
    siglip_embedder.py      # SigLIP2 embedding model
    base.py         # Abstract embedder
  eval/             # Evaluation framework
    benchmark_runner.py     # Concurrent benchmark runner (ThreadPoolExecutor)
    grading.py      # Token F1, exact match, retrieval metrics
    metrics.py      # Metric computation
    reporting.py    # JSON/JSONL output formatting
    prediction_schema.py    # Prediction row schema (v1)
    schemas.py      # BenchmarkRunResult, AnswerMetrics, RetrievalMetrics
  benchmark/        # Benchmark data loading
    io.py           # Load/save benchmark JSON
    schemas.py      # BenchmarkItem schema
  config/
    settings.py     # AppConfig, EmbeddingConfig, GenerationConfig
  data/
    loaders.py      # Video/transcript/frame data loaders
    schemas.py      # Data schemas
  serving/          # Modal deployment configs
    modal_vllm_8b.py        # Qwen3-VL-8B-FP8 on L4
    modal_vllm_30b.py       # Qwen3-VL-30B-A3B-FP8 (MoE) on A100-80GB
    modal_vllm_32b.py       # Qwen3-VL-32B-FP8 on A100-80GB
    modal_vllm.py           # Shared vLLM server (legacy)
    modal_retrieval.py      # Modal retrieval service
    modal_index_builder.py  # Remote index builder
  utils/
    io.py           # File I/O helpers
    logging.py      # Logging setup

  # Data prep (historical, depend on deleted config.py / src/utils.py)
  download_videos.py        # Download YouTube videos via yt-dlp
  transcribe.py             # Transcribe audio with faster-whisper
  extract_frames.py         # Extract frames with OpenCV

scripts/
  # Benchmarking
  run_benchmark.py          # Run baseline RAG benchmark
  run_agentic_benchmark.py  # Run agentic RAG benchmark
  llm_judge_pairwise.py     # LLM-as-judge pairwise evaluation
  grade_predictions.py      # Grade prediction JSONL against gold answers
  validate_benchmark.py     # Validate benchmark JSON constraints

  # Index & retrieval
  build_index.py            # Build local FAISS index
  query_index.py            # Query the index from CLI
  answer_question.py        # Answer a single question (baseline RAG)

  # Infrastructure
  sync_modal_index.py       # Upload FAISS index to Modal volume
  sync_modal_assets.py      # Upload video data to Modal volume
  check_modal_stack.py      # Smoke test deployed Modal services
  test_endpoints.py         # Test vLLM endpoints
  lookup_transcript.py      # Inspect transcript segments at timestamps

tests/
  test_agentic_benchmark_backend.py
  test_agentic_mock_backend.py
  test_benchmark_io.py
  test_data_loaders.py
  test_siglip_embedder.py

data/
  benchmark/                # Benchmark question sets
    multimodal_benchmark_v2.json    # 100 questions, 10 videos, 5 types
    smoke_1q.json                   # 1-question smoke test
    smoke_test_5q.json              # 5-question smoke test
  eval/                     # Evaluation outputs
    run1/                   # Qwen2.5-VL-7B results
    run2/                   # Qwen3-VL model scale comparison (8B/30B/32B)
      run_a/                # max_model_len=8192 (baselines + invalidated agentic)
      run_b/                # max_model_len=32768 (agentic + LLM judge)
  indexes/                  # FAISS index artifacts
  frames/                   # Extracted video frames
  transcripts/              # Whisper transcripts
```

## Running Evaluations

### 1. Build the index

```bash
python scripts/build_index.py --device cpu
```

### 2. Deploy Modal infrastructure

```bash
bash modal_setup.sh
```

Or deploy individual vLLM servers:

```bash
modal deploy src/serving/modal_vllm_8b.py
modal deploy src/serving/modal_vllm_30b.py
modal deploy src/serving/modal_vllm_32b.py
```

### 3. Run baseline benchmark

```bash
python scripts/run_benchmark.py \
  --benchmark-path data/benchmark/multimodal_benchmark_v2.json \
  --retrieval-backend modal \
  --generation-model "Qwen/Qwen3-VL-8B-Instruct-FP8" \
  --generation-base-url "https://<workspace>--nlp-videoqa-vllm-8b-serve.modal.run/v1" \
  --output-json data/eval/run/baseline_benchmark_run.json \
  --predictions-jsonl data/eval/run/baseline_predictions_v1.jsonl \
  --max-concurrent 4
```

### 4. Run agentic benchmark

```bash
python scripts/run_agentic_benchmark.py \
  --benchmark-path data/benchmark/multimodal_benchmark_v2.json \
  --retrieval-backend modal \
  --generation-model "Qwen/Qwen3-VL-8B-Instruct-FP8" \
  --generation-base-url "https://<workspace>--nlp-videoqa-vllm-8b-serve.modal.run/v1" \
  --output-json data/eval/run/agentic_benchmark_run.json \
  --predictions-jsonl data/eval/run/agentic_predictions_v1.jsonl \
  --max-concurrent 4
```

### 5. LLM-as-judge pairwise evaluation

Compares baseline vs agentic outputs with randomized A/B positioning:

```bash
python scripts/llm_judge_pairwise.py \
  --benchmark-path data/benchmark/multimodal_benchmark_v2.json \
  --baseline-predictions data/eval/run/baseline_predictions_v1.jsonl \
  --agentic-predictions data/eval/run/agentic_predictions_v1.jsonl \
  --generation-model "Qwen/Qwen3-VL-8B-Instruct-FP8" \
  --generation-base-url "https://<workspace>--nlp-videoqa-vllm-8b-serve.modal.run/v1" \
  --output-json data/eval/run/llm_judge_report.json \
  --output-jsonl data/eval/run/llm_judge_per_question.jsonl \
  --seed 42
```

### 6. Grade predictions against gold answers

```bash
python scripts/grade_predictions.py \
  --benchmark-path data/benchmark/multimodal_benchmark_v2.json \
  --predictions-jsonl data/eval/run/baseline_predictions_v1.jsonl \
  --system-name baseline_rag
```

## Data Preparation

These scripts were used to build the initial corpus. They depend on `config.py` and
`src/utils.py` which have been removed, but are kept for reference.

```bash
python -m src.download_videos    # Download YouTube videos via yt-dlp
python -m src.transcribe         # Transcribe audio with faster-whisper
python -m src.extract_frames     # Extract frames every 1s with OpenCV
```

The processed dataset is published at [mehta233/videoqa-mvp-data](https://huggingface.co/datasets/mehta233/videoqa-mvp-data)
on HuggingFace (23k frames, 10 videos).

## Results

See [data/eval/run2/README.md](data/eval/run2/README.md) for the full Qwen3-VL model
scale comparison (8B / 30B-A3B / 32B), including baseline vs agentic results, LLM judge
evaluations, and analysis.

**Highlights (Run 2, 32k context):**

| Model | Baseline F1 | Agentic F1 | Delta |
|-------|------------|------------|-------|
| 8B    | 0.276      | 0.418      | +51%  |
| 30B   | 0.279      | 0.358      | +28%  |
| 32B   | 0.246      | 0.408      | +66%  |
