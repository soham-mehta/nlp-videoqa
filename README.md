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
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
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
