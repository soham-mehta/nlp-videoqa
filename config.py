from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Paths:
    """Centralized project paths (all under repo-root `data/`)."""

    root: Path

    @property
    def data(self) -> Path:
        return self.root / "data"

    @property
    def raw_videos(self) -> Path:
        return self.data / "raw_videos"

    @property
    def audio(self) -> Path:
        return self.data / "audio"

    @property
    def transcripts(self) -> Path:
        return self.data / "transcripts"

    @property
    def frames(self) -> Path:
        return self.data / "frames"

    @property
    def metadata(self) -> Path:
        return self.data / "metadata"

    @property
    def videos_metadata_dir(self) -> Path:
        return self.metadata / "videos"

    @property
    def frames_metadata_dir(self) -> Path:
        return self.metadata / "frames"

    @property
    def schema_examples_dir(self) -> Path:
        return self.metadata / "schema_examples"

    @property
    def videos_index_json(self) -> Path:
        return self.metadata / "videos_index.json"


REPO_ROOT = Path(__file__).resolve().parent
PATHS = Paths(root=REPO_ROOT)


# Download 3–5 YouTube URLs here.
YOUTUBE_URLS: list[str] = [
    "https://www.youtube.com/watch?v=EfKLrSxA_H8"
]


# Download settings
YTDLP_FORMAT = "mp4/bestvideo+bestaudio/best"


# Transcription settings
WHISPER_MODEL_SIZE = "base"  # e.g. "tiny", "base", "small", "medium", "large-v3"
WHISPER_DEVICE = "auto"  # "cpu", "cuda", or "auto"
WHISPER_COMPUTE_TYPE = "auto"  # e.g. "int8", "float16", "auto"
WHISPER_VAD_FILTER = True


# Frame extraction settings
FRAME_INTERVAL_SEC = 1.0
FRAME_IMAGE_EXT = ".jpg"
