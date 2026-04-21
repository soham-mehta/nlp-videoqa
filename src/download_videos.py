from __future__ import annotations

import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from config import PATHS, YTDLP_FORMAT, YOUTUBE_URLS
from src.utils import ensure_dir, read_json, write_json


@dataclass(frozen=True)
class VideoMetadata:
    video_id: str
    title: str
    url: str
    duration: float | None
    local_path: str


def _select_best_url(info: dict[str, Any]) -> str:
    return info.get("webpage_url") or info.get("original_url") or info.get("url") or ""


def _extract_metadata(info: dict[str, Any], local_path: Path) -> VideoMetadata:
    video_id = str(info.get("id") or "")
    title = str(info.get("title") or "")
    url = _select_best_url(info)
    duration = info.get("duration")
    duration_f = float(duration) if isinstance(duration, (int, float)) else None
    return VideoMetadata(
        video_id=video_id,
        title=title,
        url=url,
        duration=duration_f,
        local_path=str(local_path),
    )


def merge_videos_index(
    existing: list[dict[str, Any]], new_results: list[VideoMetadata]
) -> list[dict[str, Any]]:
    """
    Merge new download results into an existing index without dropping prior videos.

    Preserves order of `existing`, appends brand-new ids at the end, and updates
    in-place entries when the same `video_id` is downloaded again.
    """
    merged: dict[str, dict[str, Any]] = {}
    order: list[str] = []
    for item in existing:
        vid = str(item["video_id"])
        merged[vid] = dict(item)
        order.append(vid)
    for md in new_results:
        vid = md.video_id
        if vid not in merged:
            order.append(vid)
        merged[vid] = asdict(md)
    return [merged[vid] for vid in order]


def rebuild_videos_index_from_disk(
    *,
    metadata_dir: Path | None = None,
    index_path: Path | None = None,
) -> int:
    """
    Rebuild `videos_index.json` from every `data/metadata/videos/<video_id>.json`.

    Use this if the index was overwritten by a partial download run.
    """
    metadata_dir = metadata_dir or PATHS.videos_metadata_dir
    index_path = index_path or PATHS.videos_index_json
    paths = sorted(metadata_dir.glob("*.json"))
    items = [read_json(p) for p in paths]
    items.sort(key=lambda x: str(x["video_id"]))
    write_json(index_path, items)
    return len(items)


def download_videos(
    urls: list[str] | None = None,
    *,
    out_dir: Path | None = None,
    metadata_dir: Path | None = None,
) -> list[VideoMetadata]:
    """
    Download videos with yt-dlp and store per-video metadata JSON.
    Also writes a `videos_index.json` list for convenience.
    """
    urls = urls or list(YOUTUBE_URLS)
    out_dir = out_dir or PATHS.raw_videos
    metadata_dir = metadata_dir or PATHS.videos_metadata_dir

    ensure_dir(out_dir)
    ensure_dir(metadata_dir)

    from yt_dlp import YoutubeDL

    ydl_opts = {
        "format": YTDLP_FORMAT,
        "outtmpl": str(out_dir / "%(id)s__%(title)s.%(ext)s"),
        "restrictfilenames": False,
        "noplaylist": True,
        "quiet": True,
        "no_warnings": True,
    }

    results: list[VideoMetadata] = []
    with YoutubeDL(ydl_opts) as ydl:
        for url in urls:
            try:
                info = ydl.extract_info(url, download=True)
            except Exception as e:
                raise RuntimeError(f"yt-dlp failed for url={url!r}: {e}") from e

            # IMPORTANT: don't guess the local path. yt-dlp may apply sanitization
            # rules and/or choose a different extension than `info["ext"]`.
            video_id = str(info.get("id") or "")
            local_path = Path(ydl.prepare_filename(info))
            if not local_path.is_absolute():
                local_path = (Path.cwd() / local_path).resolve()

            md = _extract_metadata(info, local_path)
            write_json(metadata_dir / f"{md.video_id}.json", asdict(md))
            results.append(md)

    existing: list[dict[str, Any]] = []
    if PATHS.videos_index_json.exists():
        existing = read_json(PATHS.videos_index_json)
    write_json(PATHS.videos_index_json, merge_videos_index(existing, results))
    return results


def main() -> None:
    if len(sys.argv) > 1 and sys.argv[1] == "--rebuild-index":
        n = rebuild_videos_index_from_disk()
        print(f"Rebuilt {PATHS.videos_index_json} with {n} videos")
        return
    download_videos()
    print(f"Downloaded and wrote metadata to {PATHS.metadata}")


if __name__ == "__main__":
    main()
