# MVP VideoQA Data Pipeline

Minimal data loading + processing layer for a multimodal Video QA project.

This MVP does **only**:
- download a handful of YouTube videos
- store per-video metadata as JSON
- transcribe audio to timestamped JSON segments (faster-whisper)
- extract frames every 1 second (OpenCV) + store frame metadata as JSON
- provide a small Python dataloader for timestamp queries

## Setup

Create a virtualenv and install deps:

```bash
uv venv .venv
source .venv/bin/activate
uv pip install -r requirements.txt
modal setup
```

Assumptions:
- `ffmpeg` is installed and on your PATH

## Configure URLs

Edit `config.py` and set `YOUTUBE_URLS` to **3–5** YouTube video URLs.

## Run the pipeline

From repo root:

```bash
python -m src.download_videos
python -m src.transcribe
python -m src.extract_frames
```

Outputs:
- Videos: `data/raw_videos/`
- Audio WAVs: `data/audio/`
- Transcripts: `data/transcripts/<video_id>.json`
- Frames: `data/frames/<video_id>/*.jpg`
- Video metadata: `data/metadata/videos/<video_id>.json` and `data/metadata/videos_index.json`
- Frame metadata: `data/metadata/frames/<video_id>.json`

Schema examples:
- `data/metadata/schema_examples/transcript_schema_example.json`
- `data/metadata/schema_examples/frames_schema_example.json`

## Use the dataloader

Example:

```python
from src.dataloader import VideoQADataloader

dl = VideoQADataloader()
video_id = dl.list_video_ids()[0]

frame = dl.get_frame_at_time(video_id, timestamp=42.0)
frames = dl.get_n_frames_at_time(video_id, timestamp=42.0, n=3, step_sec=5)
segments = dl.get_transcript_at_time(video_id, timestamp=42.0, window_sec=10)

print(frame["image_path"])
print([f["timestamp_sec"] for f in frames])
print(" ".join(s["text"] for s in segments))
```

## Notes

- Frame lookup uses **nearest timestamp** matching.
- Transcript lookup returns **all segments overlapping** the requested time window.

## Retrieval scaffold (v1)

New modular scaffold for indexing/retrieval lives under:

- `src/config/`
- `src/data/`
- `src/models/`
- `src/indexing/`
- `src/retrieval/`
- `src/benchmark/`
- `src/eval/`
- `src/utils/`

The current benchmark setup uses:

- local indexing to build the FAISS artifacts under `data/indexes/default/`
- a Modal Volume to store those artifacts for remote retrieval
- a deployed Modal retrieval service for both baseline and agentic evaluation
- a deployed Modal vLLM service for answer generation and agent tool calling

Build index:

```bash
python3 scripts/build_index.py --device cpu
```

Query index:

```bash
python3 scripts/query_index.py "How does the speaker set up the project?" --top-k 5
```

Run baseline multimodal RAG answerer against the shared remote vLLM/Modal model:

```bash
VLLM_BASE_URL=http://localhost:8000/v1 python3 scripts/answer_question.py \
  "What is computer science?" \
  --video-id CxGSnA-RTsA \
  --retrieval-backend modal
```

This saves a JSON run artifact under `data/runs/answers/` by default.

Run baseline benchmark:

```bash
VLLM_BASE_URL=http://localhost:8000/v1 python3 scripts/run_benchmark.py \
  --benchmark-path data/benchmark/example_benchmark_v1.jsonl \
  --retrieval-backend modal
```

While it runs, you should see a **tqdm progress bar** (one step per benchmark question). Use `--no-progress` for CI or plain logs.

This writes:

- `data/eval/benchmark_run.json` — full run (metrics + per-question debug + embedded `prediction_rows`)
- `data/eval/predictions_v1.jsonl` — **strict shared format** for grading (`schema_version: prediction_v1`)

Optional verbose per-question JSONL (metrics + debug only):

```bash
VLLM_BASE_URL=http://localhost:8000/v1 python3 scripts/run_benchmark.py \
  --benchmark-path data/benchmark/example_benchmark_v1.jsonl \
  --retrieval-backend modal \
  --detail-jsonl data/eval/benchmark_questions_detail.jsonl
```

Schema for predictions: `data/benchmark/prediction_schema_v1.json`.  
Helper types and builders: `src/eval/prediction_schema.py` (`build_prediction_row`, `validate_prediction_row`).

Run the agentic benchmark over the same index and benchmark questions:

```bash
VLLM_BASE_URL=http://localhost:8000/v1 python3 scripts/run_agentic_benchmark.py \
  --benchmark-path data/benchmark/example_benchmark_v1.jsonl \
  --retrieval-backend modal
```

This writes:

- `data/eval/agentic_benchmark_run.json` - full agentic run (metrics + per-question debug + prediction rows)
- `data/eval/agentic_predictions_v1.jsonl` - strict shared-format predictions for grading

Serve the shared remote OpenAI-compatible model on Modal:

```bash
.venv/bin/modal serve modal_app.py
```

Deploy a persistent Modal endpoint:

```bash
.venv/bin/modal deploy modal_app.py
```

Upload the local FAISS index into a Modal Volume:

```bash
.venv/bin/python scripts/sync_modal_index.py \
  --local-index-dir data/indexes/default \
  --volume-name nlp-videoqa-index \
  --remote-index-subdir indexes/default
```

Deploy the Modal retrieval service that loads the FAISS index from that volume:

```bash
.venv/bin/modal deploy modal_retrieval_app.py
```

Smoke test both deployed services:

```bash
.venv/bin/python scripts/check_modal_stack.py \
  --generation-base-url https://<your-modal-vllm-url>/v1
```

Compare any system’s **prediction_v1** JSONL against benchmark ground truth:

```bash
python3 scripts/grade_predictions.py \
  --benchmark-path data/benchmark/example_benchmark_v1.jsonl \
  --predictions-jsonl data/eval/predictions_v1.jsonl \
  --system-name baseline_rag
```

Use `--no-strict` only if you must grade legacy rows that omit `schema_version`.
